"""Drawing using pycairo or cairocffi as backend."""
import numpy as np
from typing import Sequence
import cairocffi as cairo
from . import trains, functions

index_colors = {'air': (0.9, 0.9, 0.9, 1), 'fused_silica': (0, 0.5, 1, 1)}

def draw_polyline(ctx: cairo.Context, xys):
    ctx.move_to(*xys[0])
    for x, y in xys[1:]:
        ctx.line_to(x, y)

def make_interface(ctx: cairo.Context, face:trains.Interface, num_points=32):
    draw_polyline(ctx, face.get_points(num_points))
    ctx.stroke()

def label_interface(ctx: cairo.Context, face:trains.Interface, radius_factor=1):
    """Draw text saying ROC, and a line from the edge of the interface to the text.

    Args:
        radius_factor (scalar): Factor by which interface radius is multiplied to get text x position.
    """
    x = radius_factor*face.radius
    y = functions.calc_sphere_sag(face.roc, face.radius)
    string = 'ROC %.3f mm'%(face.roc*1e3)
    ctx.set_source_rgb(0, 0, 0)
    ctx.save()
    ctx.translate(x, y)
    ctx.show_text(string)
    ctx.new_path()
    ctx.restore()
    if radius_factor > 1:
        draw_polyline(ctx, ((face.radius, y), (x, y)))
        ctx.stroke()

def break_spaces(spaces, broken_spaces):
    if broken_spaces is None:
        broken_spaces = [None]*len(spaces)

    spaces, brokens = zip(*[(space, False) if broken_space is None else (broken_space, True) for space, broken_space in zip(spaces, broken_spaces)])

    return spaces, brokens

def make_train_interfaces(ctx: cairo.Context, train:trains.Train, broken_spaces:Sequence=None):
    """
    Propagation is along positive y axis, starting from origin.

    Args:
        size (scalar): Full transverse dimension of interfaces.
    """
    spaces, brokens = break_spaces(train.spaces, broken_spaces)

    ctx.save()
    ctx.translate(0, spaces[0])
    for interface, space in zip(train.interfaces, spaces[1:]):
        make_interface(ctx, interface)
        ctx.translate(0, space)
    ctx.restore()

def make_train_spaces(ctx: cairo.Context, train:trains.Train, broken_spaces:Sequence=None, frac:float=1/4):
    spaces, brokens = break_spaces(train.spaces, broken_spaces)

    ctx.save()

    radius = train.interfaces[0].radius  # if len(self.interfaces) > 0 else self.spaces[0]
    y_next = 0
    xys_last = np.asarray([[-radius, 0], [radius, 0]])
    n = train.interfaces[0].n1
    for space, next_interface, broken in zip(spaces, train.interfaces + (None,), brokens):
        ctx.set_source_rgba(*index_colors.get(n.name, (0, 0, 1, 1)))

        y_next += space

        # Generate next interface points.
        if next_interface is None:
            xys_next = np.asarray([[-radius, 0], [radius, 0]])
        else:
            xys_next = next_interface.get_points()

        xys_next += [0, y_next]
        if broken:
            num = 8
            n = np.arange(num + 1)
            jagged = np.c_[2*(n/num - 0.5)*radius, (n%2)*space*frac + y_next - space/2]

            xys = np.r_[xys_last, jagged[::-1, :] - (0, space*frac)]
            draw_polyline(ctx, xys)
            ctx.fill()

            xys = np.r_[jagged, xys_next[::-1, :]]
            draw_polyline(ctx, xys)
            ctx.fill()
        else:
            xys = np.r_[xys_last, xys_next[::-1, :]]
            draw_polyline(ctx, xys)
            ctx.fill()

        xys_last = xys_next
        if next_interface is not None:
            n = next_interface.n2
    ctx.restore()

def label_train_interfaces(ctx: cairo.Context, train:trains.Train, radius_factors=1, broken_spaces:Sequence=None):
    spaces, brokens = break_spaces(train.spaces, broken_spaces)
    radius_factors = np.broadcast_to(radius_factors, (len(train.interfaces),))

    ctx.save()
    ctx.translate(0, spaces[0])
    for interface, space, radius_factor in zip(train.interfaces, spaces[1:], radius_factors):
        label_interface(ctx, interface, radius_factor)
        ctx.translate(0, space)
    ctx.restore()

def label_train_spaces(ctx: cairo.Context, train: trains.Train, radius_factors=1, broken_spaces:Sequence=None):
    """Draw material name and thickness of the spaces.

    Args:
        radius_factors (scalar or sequence): Factor by which the average interface radii is multiplied to place the text. If
            a scalar is given it is broadcast.
    """
    spaces, brokens = break_spaces(train.spaces, broken_spaces)
    radius_factors = np.broadcast_to(radius_factors, (len(spaces),))
    radius_last = train.interfaces[0].radius
    y_last = 0
    h_last = 0
    n = train.interfaces[0].n1
    for space, next_interface, radius_factor in zip(spaces, train.interfaces + (None,), radius_factors):
        if next_interface is None:
            radius_next = radius_last
            h_next = 0
        else:
            radius_next = next_interface.radius
            h_next = functions.calc_sphere_sag(next_interface.roc, radius_next)
        y_next = y_last + space
        if space != 0:
            string = '%.3f mm %s'%(space*1e3, n.name)
            radius = (radius_last + radius_next)/2
            x = radius_factor*radius
            y = (y_next + h_next + y_last + h_last)/2

            ctx.save()
            ctx.translate(x, y)
            ctx.show_text(string)
            ctx.new_path()
            ctx.restore()

            if radius_factor > 1:
                draw_polyline(ctx, ((radius, y), (x, y)))
                ctx.stroke()

        y_last = y_next
        h_last = h_next
        if next_interface is not None:
            n = next_interface.n2
            radius_last = next_interface.radius
