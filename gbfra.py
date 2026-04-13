from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import json
import time
import copy
import asyncio
import shutil
from pathlib import Path
import re
import math
import sys
import traceback
from enum import StrEnum
from base64 import b64encode
from io import BytesIO
import argparse

# Third party
import aiohttp
from PIL import Image, ImageDraw

# Class to manipulate a vector2-type structure (X, Y)
# Call the 'i' property to obtain an integer tuple to use with Pillow
@dataclass(slots=True)
class V():
    x : int|float = 0
    y : int|float = 0
    
    def __init__(self : V, X : int|float, Y : int|float) -> None:
        self.x = X
        self.y = Y
    
    @staticmethod
    def ZERO() -> V:
        return V(0, 0)
    
    def copy(self : V) -> V:
        return V(self.x, self.y)
    
    # operators
    def __add__(self : V, other : V|tuple|list|int|float) -> V:
        if isinstance(other, float) or isinstance(other, int):
            return V(self.x + other, self.y + other)
        else:
            return V(self.x + other[0], self.y + other[1])
    
    def __radd__(self : V, other : V|tuple|list|int|float) -> V:
        return self.__add__(other)

    def __sub__(self : V, other : V|tuple|list|int|float) -> V:
        if isinstance(other, float) or isinstance(other, int):
            return V(self.x - other, self.y - other)
        else:
            return V(self.x - other[0], self.y - other[1])
    
    def __rsub__(self : V, other : V|tuple|list|int|float) -> V:
        return self.__sub__(other)

    def __mul__(self : V, other : V|tuple|list|int|float) -> V:
        if isinstance(other, float) or isinstance(other, int):
            return V(self.x * other, self.y * other)
        else:
            return V(self.x * other[0], self.y * other[1])

    def __rmul__(self : V, other : V|tuple|list|int|float) -> V:
        return self.__mul__(other)

    # for access via []
    def __getitem__(self : V, key : int) -> int|float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Index out of range")

    def __setitem__(self : V, key : int, value : int|float) -> None:
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError("Index out of range")

    # len is fixed at 2
    def __len__(self : V) -> int:
        return 2

    def __str__(self : V) -> str:
        return f"{self.x},{self.y}"

    def __repr__(self : V) -> str:
        return f"V(x={self.x}, y={self.y})"

    # to convert to an integer tuple (needed for pillow)
    @property
    def i(self : V) -> tuple[int, int]:
        return (int(self.x), int(self.y))

# Constants
GBF_SIZE : V = V(640, 654)
CDN : str = "https://prd-game-a-granbluefantasy.akamaized.net/"

# Utility functions
def pexc(e : Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

"""
Handles 2D affine transformations using a 3x3 matrix representation.
The input 'data' is expected to be a flat list of 6 floats: [a, b, c, d, e, f]
representing the transformation:
| a  c  e |
| b  d  f |
| 0  0  1 |
"""
@dataclass(slots=True)
class Matrix3x3:
    data: list[float]

    @classmethod
    def from_state(cls, state):
        x, y = state[0], state[1]
        sx, sy = state[2], state[3]
        rot = math.radians(state[4]) if state[4] % 360 != 0 else 0
        # Note: ignore skewX and skewY
        rx, ry = state[7], state[8]
        
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        a, b = cos_r * sx, sin_r * sx
        c, d = -sin_r * sy, cos_r * sy
        tx = x - (rx * a + ry * c)
        ty = y - (rx * b + ry * d)
        return cls([a, b, c, d, tx, ty])

    def multiply(self, other: Matrix3x3) -> Matrix3x3:
        m1 = [[self.data[0], self.data[2], self.data[4]],
              [self.data[1], self.data[3], self.data[5]], [0, 0, 1]]
        m2 = [[other.data[0], other.data[2], other.data[4]],
              [other.data[1], other.data[3], other.data[5]], [0, 0, 1]]
        res = [[0.0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    res[i][j] += m1[i][k] * m2[k][j]
        return Matrix3x3([res[0][0], res[1][0], res[0][1], res[1][1], res[0][2], res[1][2]])

    def get_pillow_affine(self) -> list[float]:
        # inverts the matrix for Pillow's .transform() method
        m = [[self.data[0], self.data[2], self.data[4]],
             [self.data[1], self.data[3], self.data[5]], [0.0, 0.0, 1.0]]
        try:
            inv = self.invert_matrix(m)
        except ValueError:
            # to avoid crash
            return [1.0, 0, 0, 0, 1.0, 0]
        # Pillow wants: (a, b, c, d, e, f) where x_src = ax_dst + by_dst + c
        return [inv[0][0], inv[0][1], inv[0][2], inv[1][0], inv[1][1], inv[1][2]]

    @staticmethod
    def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
        # inverts a square matrix using Gauss-Jordan elimination.
        n = len(matrix)
        # create an identity matrix of the same size
        inverse = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        # work on a copy to avoid mutating the original
        working_matrix = copy.deepcopy(matrix)
        
        for i in range(n):
            # pivot scaling: make the diagonal element 1.0
            pivot = working_matrix[i][i]
            if abs(pivot) < 1e-9:
                raise ValueError("Matrix is singular and cannot be inverted.")
            scaling_factor = 1.0 / pivot
            for j in range(n):
                working_matrix[i][j] *= scaling_factor
                inverse[i][j] *= scaling_factor
            # make all other elements in this column 0.0
            for k in range(n):
                if k != i:
                    factor = working_matrix[k][i]
                    for j in range(n):
                        working_matrix[k][j] -= factor * working_matrix[i][j]
                        inverse[k][j] -= factor * inverse[i][j]
        return inverse

# Wrapper class to store and manipulate Image objects
# Handle the close() calls on destruction
@dataclass(slots=True)
class IMG():
    image : Image = None
    buffer : BytesIO = None
    
    def __init__(self : IMG, src : str|bytes|IMG|Image, *, auto_convert : bool = True) -> None:
        self.image = None
        self.buffer = None
        match src: # possible types
            case str(): # path to a local file
                self.image = Image.open(src)
                if auto_convert:
                    self.convert("RGBA")
            case bytes(): # bytes (usually received from a network request)
                self.buffer = BytesIO(src) # need a readable buffer for it, and it must stays alive
                self.image = Image.open(self.buffer)
                if auto_convert:
                    self.convert("RGBA")
            case IMG(): # another IMG wrapper
                self.image = src.image.copy()
            case _: # an Image instance. NOTE: I use 'case _' because of how import Pillow, the type isn't loaded at this point
                self.image = src

    @staticmethod
    def new_canvas(size : V) -> IMG:
        i : Image = Image.new('RGB', size.i, "black")
        im_a : Image = Image.new("L", size.i, "black") # Alpha
        i.putalpha(im_a)
        im_a.close()
        return IMG(i)

    def __del__(self : IMG) -> None:
        if self.image is not None:
            self.image.close()
        if self.buffer is not None:
            self.buffer.close()

    def swap(self : IMG, other : IMG) -> None:
        self.image, other.image = other.image, self.image
        self.buffer, other.buffer = other.buffer, self.buffer

    def convert(self : IMG, itype : str) -> None:
        tmp = self.image
        self.image = tmp.convert(itype)
        tmp.close()

    def copy(self : IMG) -> IMG:
        return IMG(self)

    def paste(self : IMG, other : IMG, offset : V|tuple[int, int]) -> None:
        match offset:
            case V():
                self.image.paste(other.image, offset.i, other.image)
            case _:
                self.image.paste(other.image, offset, other.image)

    def paste_transparency(self : IMG, other : IMG, offset : V|tuple[int, int]) -> None:
        alpha : IMG = IMG.new_canvas(V(self.image.size[0], self.image.size[1]))
        alpha.paste(other, offset)
        self.swap(self.alpha(alpha))

    def crop(self : IMG, size : tuple[int, int]|tuple[int, int, int, int]) -> IMG:
        # depending on the tuple size
        if len(size) == 4:
            return IMG(self.image.crop(size))
        elif len(size) == 2:
            return IMG(self.image.crop((0, 0, *size)))
        raise ValueError(f"Invalid size of the tuple passed to IMG.crop(). Expected 2 or 4, received {len(size)}.")

    def resize(self : IMG, size : V|tuple[int, int]) -> IMG:
        match size:
            case V():
                return IMG(self.image.resize(size.i, Image.Resampling.LANCZOS))
            case tuple():
                return IMG(self.image.resize(size, Image.Resampling.LANCZOS))
        raise TypeError(f"Invalid type passed to IMG.resize(). Expected V or tuple[int, int], received {type(size)}.")

    def rotate(self : IMG, angle : int, center : V|tuple[int, int]|None = None) -> IMG:
        match center:
            case V():
                return IMG(self.image.rotate(angle, center=center.i, resample=Image.BICUBIC))
            case tuple():
                return IMG(self.image.rotate(angle, center=center, resample=Image.BICUBIC))
            case None:
                return IMG(self.image.rotate(angle, resample=Image.BICUBIC))
        raise TypeError(f"Invalid type passed to IMG.rotate(). Expected V or tuple[int, int], received {type(center)}.")

    def thumbnail(self : IMG, size : V|tuple[int, int]) -> IMG:
        match size:
            case V():
                return IMG(self.image.thumbnail(size.i, Image.Resampling.LANCZOS))
            case tuple():
                return IMG(self.image.thumbnail(size, Image.Resampling.LANCZOS))
        raise TypeError(f"Invalid type passed to IMG.thumbnail(). Expected V or tuple[int, int], received {type(size)}.")

    def ninepatch(self : IMG, size : V|tuple[int, int], margin : int) -> IMG:
        iw, ih = self.image.size
        match size:
            case V():
                tw, th = size.i
            case tuple():
                tw, th = size
            case _:
                raise TypeError(f"Invalid type passed to IMG.ninepatch(). Expected V or tuple[int, int], received {type(size)}.")

        # output image
        out : IMG = IMG(Image.new("RGBA", (tw, th)))
        # corners
        out.paste(self.crop((0, 0, margin, margin)), (0, 0)) # TL
        out.paste(self.crop((iw - margin, 0, iw, margin)), (tw - margin, 0)) # TR
        out.paste(self.crop((0, ih - margin, margin, ih)), (0, th - margin)) # BL
        out.paste(self.crop((iw - margin, ih - margin, iw, ih)), (tw - margin, th - margin)) # BR
        # edges
        out.paste(self.crop((margin, 0, iw - margin, margin)).resize((tw - margin - margin, margin)), (margin, 0)) # margin
        out.paste(self.crop((margin, ih - margin, iw - margin, ih)).resize((tw - margin - margin, margin)), (margin, th - margin)) # margin
        out.paste(self.crop((0, margin, margin, ih - margin)).resize((margin, th - margin - margin)), (0, margin)) # margin
        out.paste(self.crop((iw - margin, margin, iw, ih - margin)).resize((margin, th - margin - margin)), (tw - margin, margin)) # margin
        # center
        center_w = tw - margin - margin
        center_h = th - margin - margin
        if center_w > 0 and center_h > 0:
            out.paste(self.crop((margin, margin, iw - margin, ih - margin)).resize((center_w, center_h)), (margin, margin))
        return out

    def transpose(self : IMG, i : int) -> None:
        tmp = self.image.transpose(i)
        self.image.close()
        self.image = tmp

    def transform(self : IMG, m : Matrix3x3) -> IMG:
        return IMG(
            self.image.transform(
                self.image.size,
                Image.Transform.AFFINE,
                m.get_pillow_affine(),
                resample=Image.Resampling.BILINEAR
            )
        )

    def text(self : IMG, *args, **kwargs) -> IMG:
        ImageDraw.Draw(self.image, 'RGBA').text(*args, **kwargs)
        return self

    def alpha(self : IMG, layer : IMG) -> IMG:
        return IMG(Image.alpha_composite(self.image, layer.image))

    def show(self : IMG) -> None:
        self.image.show()

    def save(self : IMG, path : str, dry : bool = False) -> None:
        if not dry:
            self.image.save(path, "PNG")

# Classes to parse CreateJS raid_appear animations and generate an image
@dataclass
class TweenStep:
    type: str # 'to' or 'wait'
    props: dict
    duration: int

@dataclass
class Instance:
    name: str
    symbol_name: str
    transform: list[float] # x, y, scaleX, scaleY, rotation, skewX, skewY, regX, regY
    tweens: list[TweenStep]
    initial_props: dict = None

@dataclass
class Symbol:
    name: str
    type: str # 'Bitmap' or 'MovieClip'
    source_rect: list[int] = None # For Bitmaps: x, y, w, h
    instances: list[Instance] = None # For MovieClips
    total_frames: int = 1
    stop_frame: int | None = None

class CreateJSTimelineParser:
    def __init__(self : CreateJSTimelineParser, name : str, js_data : str, atlas : IMG) -> None:
        self.name = name
        self.js_data = js_data
        self.atlas = atlas
        self.symbols: dict[str, Symbol] = {}
        self._parse()

    def _parse(self : CreateJSTimelineParser) -> None:
        # parse sub-rectangles, i.e. bitmaps
        bitmap_re = re.compile(r"\(a\.(\w+)=function\(\)\{this\.sourceRect=new c\.Rectangle\((\d+),(\d+),(\d+),(\d+)\),this\.initialize\(b\.\w+\)\}\)\.prototype=(?:[a-z]|lib)=new c\.Bitmap")
        for match in bitmap_re.finditer(self.js_data):
            name, x, y, w, h = match.groups()
            self.symbols[name] = Symbol(name=name, type="Bitmap", source_rect=[int(x), int(y), int(w), int(h)])

        # parse MovieClips
        mc_re = re.compile(r"\(a\.(\w+)=function\(.*?\)\{(.*?)\}\)\.prototype=(?:(?:[a-z]|lib)=new c\.(?:MovieClip|Container)|d\(a\.\1,.*?\)|(?:[a-z]|lib)=new c\.Bitmap)")
        for match in mc_re.finditer(self.js_data):
            name, body = match.groups()
            if name not in self.symbols:
                # don't overwrite bitmap if already parsed
                self.symbols[name] = self._parse_movieclip(name, body)

    def _parse_movieclip(self : CreateJSTimelineParser, name : str, body : str) -> Symbol:
        # use a list to preserve Z-index
        instances: list[Instance] = []
        inst_map: dict[str, Instance] = {}

        # search stop frame
        stop_frame = None
        stop_match = re.search(r"this\.frame_(\d+)\s*=\s*function\s*\(\)\s*\{\s*this\.stop\(\)\s*\}", body)
        if stop_match:
            stop_frame = int(stop_match.group(1))
            
        # parse instances and their symbols
        # Example: this.instance=new a.raid_appear_9102383_vs_b
        inst_re = re.compile(r"this\.(\w+)=new a\.(\w+)")
        for inst_match in inst_re.finditer(body):
            inst_name, sym_name = inst_match.groups()
            if not sym_name:
                sym_name = inst_name 
            inst = Instance(
                name=inst_name,
                symbol_name=sym_name,
                transform=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                tweens=[],
                initial_props={}
            )
            instances.append(inst)
            inst_map[inst_name] = inst

        # parse initial property assignments
        # Example: this.instance.alpha=.1289; this.instance._off=!0;
        prop_re = re.compile(r"this\.(\w+)\.(\w+)=([^,;]+)")
        for prop_match in prop_re.finditer(body):
            inst_name, prop, val = prop_match.groups()
            if inst_name in inst_map and prop not in ("setTransform", "timeline"):
                val = val.strip()
                if val == "!0":
                    val = True
                elif val == "!1":
                    val = False
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        # strip quotes from strings
                        val = val.strip("'\"")
                inst_map[inst_name].initial_props[prop] = val

        # parse setTransform
        # Example: this.instance.setTransform(162,134,1,1,0,0,0,12,4)
        trans_re = re.compile(r"this\.(\w+)\.setTransform\((.*?)\)")
        for trans_match in trans_re.finditer(body):
            inst_name, params_str = trans_match.groups()
            if inst_name in inst_map:
                params = []
                for p in params_str.split(","):
                    try:
                        params.append(float(p))
                    except ValueError:
                        params.append(0.0)
                # CreateJS setTransform: x, y, scaleX, scaleY, rotation, skewX, skewY, regX, regY
                # pad to 9
                full_params = params + [0.0] * (9 - len(params))
                # default scaleX/scaleY to 1.0 if not provided
                if len(params) < 3:
                    full_params[2] = 1.0
                if len(params) < 4:
                    full_params[3] = 1.0
                inst_map[inst_name].transform = full_params

        # parse Tweens
        # Example: this.timeline.addTween(c.Tween.get(this.instance_7).wait(10).to({_off:!1},0)...)
        tween_re = re.compile(r"this\.timeline\.addTween\(c\.Tween\.get\(this(?:\.(\w+))?\)(.*?)\)(?=[,;}])")
        max_duration = 1
        for tween_match in tween_re.finditer(body):
            inst_name, actions_str = tween_match.groups()
            if inst_name is None:
                inst_name = "this"
            
            tweens = []
            current_duration = 0
            
            # parse .to({props}, duration) and .wait(duration)
            action_re = re.compile(r"\.(to|wait)\((.*?)\)")
            for action_match in action_re.finditer(actions_str):
                atype, params = action_match.groups()
                if atype == "to":
                    prop_match = re.search(r"\{(.*?)\}", params)
                    props = {}
                    if prop_match:
                        prop_str = prop_match.group(1)
                        for p in prop_str.split(","):
                            if ":" in p:
                                k, v = p.split(":", 1)
                                k, v = k.strip(), v.strip()
                                if v == "!0":
                                    v = True
                                elif v == "!1":
                                    v = False
                                else:
                                    try: v = float(v)
                                    except:
                                        v = v.strip("'\"")
                                props[k] = v
                    # duration is the number after the properties block
                    dur_match = re.search(r"\},(\d+)", params)
                    duration = int(dur_match.group(1)) if dur_match else 0
                    tweens.append(TweenStep(type="to", props=props, duration=duration))
                    current_duration += duration
                else:
                    # wait(10) or wait(1).call(...)
                    dur_match = re.match(r"^(\d+)", params)
                    duration = int(dur_match.group(1)) if dur_match else 0
                    tweens.append(TweenStep(type="wait", props={}, duration=duration))
                    current_duration += duration
            
            if inst_name != "this" and inst_name in inst_map:
                inst_map[inst_name].tweens = tweens
            
            max_duration = max(max_duration, current_duration)

        return Symbol(name=name, type="MovieClip", instances=instances, total_frames=max_duration, stop_frame=stop_frame)

    def _get_instance_state(self : CreateJSTimelineParser, instance : Instance, frame : int) -> dict:
        # calculates the state of an instance at a specific frame
        state : dict = { # initial state
            'x': instance.transform[0],
            'y': instance.transform[1],
            'scaleX': instance.transform[2],
            'scaleY': instance.transform[3],
            'rotation': instance.transform[4],
            'skewX': instance.transform[5],
            'skewY': instance.transform[6],
            'regX': instance.transform[7],
            'regY': instance.transform[8],
            'alpha': 1.0,
            '_off': False
        }
        
        # override with initial properties
        if instance.initial_props:
            state.update(instance.initial_props)

        if not instance.tweens:
            return state

        elapsed : int = 0
        for i, step in enumerate(instance.tweens):
            if frame < elapsed:
                break
                
            if step.type == "wait":
                elapsed += step.duration
                if frame < elapsed:
                    # during wait, state remains
                    break
            elif step.type == "to":
                start_frame = elapsed
                end_frame = elapsed + step.duration
                
                if frame >= end_frame:
                    # step is finished, apply all properties
                    state.update(step.props)
                    elapsed = end_frame
                else:
                    # interpolate
                    t = (frame - start_frame) / step.duration if step.duration > 0 else 1.0
                    for prop, end_val in step.props.items():
                        if prop in state:
                            if isinstance(end_val, (int, float)) and isinstance(state[prop], (int, float)):
                                state[prop] = state[prop] + (end_val - state[prop]) * t
                            else:
                                state[prop] = end_val
                        else:
                            state[prop] = end_val
                    elapsed = end_frame
                    break
        return state

    def render(self : CreateJSTimelineParser, target_frame: int = -1) -> IMG:
        # priority for target symbol: mc_{name}_set, then mc_{name}, then {name}
        # latter two are untested, they're here for fallback purpose
        target_name = None
        candidates = [f"mc_{self.name}_set", f"mc_{self.name}", self.name]
        for candidate in candidates:
            if candidate in self.symbols:
                target_name = candidate
                break
        
        # fallback: find any symbol that ends with _set and contains the ID
        if not target_name:
            boss_id = self.name.split('_')[-1]
            for sym_name in self.symbols:
                if sym_name.endswith('_set') and boss_id in sym_name:
                    target_name = sym_name
                    break
        
        if not target_name:
            # keep this print for debug
            print(f"Warning: Could not find target symbol for {self.name}. Available symbols: {list(self.symbols.keys())[:5]}...")
            return None

        symbol = self.symbols[target_name]

        if target_frame == -1:
            # Start with finding the frame with the maximum number of visible bitmaps
            max_visible = -1
            best_frames = []
            
            # Search all frames in the target symbol
            bitmaps_per_frame = []
            for f in range(symbol.total_frames):
                visible_count = self._count_visible_bitmaps(symbol, f)
                if visible_count > max_visible:
                    max_visible = visible_count
                    best_frames = [f]
                elif visible_count == max_visible:
                    best_frames.append(f)
                bitmaps_per_frame.append(visible_count)
            
            if best_frames:
                # prefer frames that are also stop frames
                stop_frames_in_best = [f for f in best_frames if f == symbol.stop_frame]
                if stop_frames_in_best:
                    target_frame = stop_frames_in_best[0]
                else:
                    # else try to ballpark from the list of bitmap count per frame
                    highest = max(bitmaps_per_frame)
                    start = bitmaps_per_frame.index(highest)
                    for i in range(start, len(bitmaps_per_frame)):
                        if bitmaps_per_frame[i] < highest:
                            target_frame = i + 1 # pick next one after
                            start = None
                            break
                    if start is not None:
                        # fallback
                        target_frame = symbol.total_frames - 1
            else:
                target_frame = symbol.total_frames - 1

        # output canvas
        canvas = IMG.new_canvas(GBF_SIZE) # gbf resolution
        self._render_recursive(canvas, symbol, target_frame, Matrix3x3([1.0, 0, 0, 1.0, 0, 0]), 1.0)
        return canvas

    def _render_recursive(
        self : CreateJSTimelineParser,
        canvas : IMG, symbol : Symbol,
        frame : int, parent_matrix : Matrix3x3,
        parent_alpha : float, composite : str = None
    ) -> None:
        if symbol.type == "Bitmap":
            rect = symbol.source_rect
            cropped : IMG = self.atlas.crop((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]))
            
            temp : IMG = IMG.new_canvas(GBF_SIZE)
            temp.paste(cropped, (0, 0))
            
            # apply alpha
            if parent_alpha < 1.0:
                r, g, b, a = temp.image.split()
                a = a.point(lambda p: int(p * parent_alpha))
                temp.image.putalpha(a)
            
            transformed : IMG = temp.transform(parent_matrix)
            # Note: I ignore direct "lighter" composite and additive blending
            canvas.swap(canvas.alpha(transformed))
        elif symbol.type == "MovieClip":
            # renders instances in the order they were added to the timeline,
            # which usually corresponds to the order in the code.
            for instance in reversed(symbol.instances):
                state = self._get_instance_state(instance, frame)
                if state.get('_off', False):
                    continue
                alpha = state.get('alpha', 1.0) * parent_alpha
                if alpha <= 0:
                    continue
                inst_state_list = [
                    state['x'], state['y'],
                    state['scaleX'], state['scaleY'],
                    state['rotation'],
                    state['skewX'], state['skewY'],
                    state['regX'], state['regY']
                ]
                inst_matrix = Matrix3x3.from_state(inst_state_list)
                combined_matrix = parent_matrix.multiply(inst_matrix)
                child_symbol = self.symbols[instance.symbol_name]
                child_frame = frame % child_symbol.total_frames
                # pass down compositeOperation if set on instance
                child_composite = state.get('compositeOperation', composite)
                self._render_recursive(canvas, child_symbol, child_frame, combined_matrix, alpha, child_composite)

    def _count_visible_bitmaps(self : CreateJSTimelineParser, symbol : Symbol, frame : int, alpha_threshold : float = 0.1) -> int:
        # recursively counts the number of visible Bitmaps at a specific frame
        if symbol.type == "Bitmap":
            return 1
        count = 0
        if symbol.type == "MovieClip":
            for instance in symbol.instances:
                state = self._get_instance_state(instance, frame)
                if state.get('_off', False) or state.get('alpha', 1.0) < alpha_threshold:
                    continue
                child_symbol = self.symbols[instance.symbol_name]
                child_frame = frame % child_symbol.total_frames
                count += self._count_visible_bitmaps(child_symbol, child_frame, alpha_threshold)
        return count

async def get(client, path) -> bytes:
    response : aiohttp.Response = await client.get(
        CDN + path,
        headers={'connection':'keep-alive'}
    )
    async with response:
        if response.status != 200:
            raise Exception(f"HTTP Error code {response.status} for path: {path}")
        return await response.read()

def sanitize_output(output : str) -> str:
    path = Path(output)
    # existing folder
    if path.is_dir():
        return (path / "output.png").as_posix()
    # existing file
    if path.is_file():
        if path.suffix.lower() != ".png":
            return path.with_name(path.name + ".png").as_posix()
        return path.as_posix()
    return path.as_posix()

async def run() -> None:
    parser = argparse.ArgumentParser(description="Generate a Granblue Fantasy Raid Appear image.")
    parser.add_argument(
        '-i', '--id', 
        type=str, 
        required=True, 
        help="The enemy ID (Required)."
    )
    parser.add_argument(
        '-v', '--variation', 
        type=str, 
        required=False, 
        help="The variation string (Optional)."
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=False, 
        help="The path of the output image (Required)."
    )
    parser.add_argument(
        '-jp', '--japanese',
        action='store_true',
        help="Enable the use of Japanese files (Optional)."
    )
    args : argparse.Namespace = parser.parse_args()
    eid : str = args.id
    var : str = ("_" + args.variation) if args.variation is not None else ""
    out : str = sanitize_output(args.output) if args.output is not None else f"{eid}{var}.png"
    folder : str = "assets" if args.japanese else "assets_en"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as client:
        try:
            javascript : str = (await get(client, f"{folder}/js/cjs/raid_appear_{eid}{var}.js")).decode('utf-8')
            spritesheet : IMG = IMG(await get(client, f"{folder}/img/sp/cjs/raid_appear_{eid}{var}.png"))
            parser = CreateJSTimelineParser(f"raid_appear_{eid}{var}", javascript, spritesheet)
            render : IMG = parser.render()
            render.save(out)
            print(f"Image saved to {out}")
        except Exception as e:
            print(pexc(e))
            print("The above exception occured.")

if __name__ == "__main__":
    asyncio.run(run())