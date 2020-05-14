import numpy as np
from contextlib import contextmanager
import cairocffi as cairo
from otk import trains, ri, otk_cairo

lamb = 860e-9
f = 100e-3
radius = 12.5e-3
train = trains.Train.design_singlet(ri.fused_silica, f, 1, 5e-3, radius, lamb, ri.air).pad_to_transform(lamb)

def cairo_svg_frame(file, width, height, points_per_unit: float, center_x: float=None, left: float=None, center_y: float=0):
    if left is None:
        if center_x is None:
            center_x = width/2
    else:
        assert center_x is None
        center_x = left + width/2

    width_points = int(round(width*points_per_unit))
    height_points = int(round(height*points_per_unit))
    surface = cairo.SVGSurface(file, width_points, height_points)
    context = cairo.Context(surface)
    context.translate(width_points/2, height_points/2)
    context.scale(points_per_unit, points_per_unit)
    context.translate(-center_x, -center_y)
    return context

broken_spaces = 10e-3, None, 10e-3
length = sum(otk_cairo.break_spaces(train.spaces, broken_spaces)[0])

with open(f'f{f*1e3:.0f}mm_singlet.svg', 'wb') as file:
    ctx = cairo_svg_frame(file, 10*radius, length, 10e3, left=-radius*1.5, center_y=length/2)
    ctx.set_line_width(0.1e-3)
    ctx.set_font_size(1e-3)
    ctx.set_source_rgba(0, 0, 0, 1)
    otk_cairo.make_train_interfaces(ctx, train, broken_spaces)
    otk_cairo.make_train_spaces(ctx, train, broken_spaces)
    otk_cairo.label_train_interfaces(ctx, train, broken_spaces=broken_spaces)
    otk_cairo.label_train_spaces(ctx, train, broken_spaces=broken_spaces)
    ctx.get_target().finish()