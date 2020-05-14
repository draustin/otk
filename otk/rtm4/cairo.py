from typing import Sequence
import cairocffi as cairo
from . import *
from ..draw2.draw2_cairo import draw

__all__ = ['draw_ray_trace', 'draw_elements']

def draw_ray_trace(r0: Sequence[float], elements: Sequence[Element], ctx: cairo.Context, scale: float):
    ctx.save()
    r = r0
    for element in elements:
        ctx.move_to(0, r[0]*scale)
        ctx.translate(element.thickness*scale, 0)
        rp = element.matrix.dot(r)
        ctx.line_to(0, rp[0]*scale)
        ctx.stroke()
        r = rp
    ctx.restore()

def draw_elements(elements: Sequence[Element], ctx: cairo.Context, scale: float):
    ctx.save()
    for element in elements:
        if element.drawing is not None:
            draw(element.drawing, ctx, scale)
        ctx.translate(element.thickness*scale, 0)
    ctx.restore()