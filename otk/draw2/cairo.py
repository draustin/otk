from functools import singledispatch
import cairocffi as cairo
from . import *

__all__ = ['draw']

@singledispatch
def draw(d: Drawing, ctx: cairo.Context, scale: float):
    raise NotImplementedError(d)

@draw.register
def _(d: Translation, ctx: cairo.Context, scale: float):
    ctx.save()
    ctx.translate(d.x*scale, d.y*scale)
    draw(d.child, ctx, scale)
    ctx.restore()

@draw.register
def _(d: Scaling, ctx: cairo.Context, scale: float):
    ctx.save()
    ctx.scale(d.x, d.y)
    draw(d.child, ctx, scale)
    ctx.restore()

@draw.register
def _(d: Line, ctx: cairo.Context, scale: float):
    ctx.save()
    ctx.move_to(d.x0*scale, d.y0*scale)
    ctx.line_to(d.x1*scale, d.y1*scale)
    ctx.stroke()
    # ctx.fill()
    ctx.restore()

@draw.register
def _(d: Arc, ctx: cairo.Context, scale: float):
    ctx.save()
    ctx.arc(d.xc*scale, d.yc*scale, d.radius*scale, d.theta0, d.theta1)
    ctx.stroke()
    ctx.restore()

@draw.register
def _(d: Sequence, ctx: cairo.Context, scale: float):
    for child in d.children:
        draw(child, ctx, scale)