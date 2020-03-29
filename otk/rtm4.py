from dataclasses import dataclass
from typing import Callable
import cairo
import numpy as np
from . import abcd, paraxial

def transform_limit(limit, inv_matrix):
    """Transform a limit through a ray transfer matrix.

    Args:
        limit: x, theta, constant coefficients.
        inv_matrix: Inverse matrix of the element
    """
    return np.r_[limit[:2].dot(inv_matrix), limit[2]]


@dataclass
class Element:
    """
    A limit is an inequality satisifed by valid rays i.e. a limit (alpha, beta, gamma) means that a valid ray (x, theta) satisfies
        x*alpha + theta*beta + gamma >= 0.
    This applies *before* the element.
    """
    matrix: np.ndarray
    thickness: float
    draw_func: Callable
    limits: tuple

    def __post_init__(self):
        self.limits = tuple(np.asarray(l) for l in self.limits)
        assert all(limit.shape == (3, ) for limit in self.limits)

    def reverse(self):
        # Transform each limit through self, then negate theta as we are reversing.
        limits = tuple(transform_limit(limit, np.linalg.inv(self.matrix))*[1, -1, 1] for limit in self.limits)
        if self.draw_func is None:
            draw_func = None
        else:
            def draw_func(ctx: cairo.Context):
                ctx.save()
                ctx.translate(self.thickness, 0)
                ctx.scale(-1, 1)
                self.draw_func(ctx)
                ctx.restore()
        return Element(abcd.reverse(self.matrix), self.thickness, draw_func, limits)

    def __mul__(self, other):
        # The limits apply before the element, and transforms are applied right to left. So transform
        # self's limits through other.
        limits = tuple(transform_limit(limit, other.matrix) for limit in self.limits) + other.limits

        def draw_func(ctx: cairo.Context):
            ctx.save()
            if  self.draw_func is not None:
                self.draw_func(ctx)
            ctx.translate(self.thickness, 0)
            if other.draw_func is not None:
                other.draw_func(ctx)
            ctx.translate(other.thickness, 0)
            ctx.restore()

        return Element(self.matrix.dot(other.matrix), self.thickness + other.thickness, draw_func, limits)

def make_thin_lens(f, d:float=None):
    if d is None:
        draw_func = None
    else:
        def draw_func(ctx: cairo.Context):
            ctx.save()
            #ctx.set_source_rgb(0, 0.5, 1)
            ctx.scale(0.1, 1)
            ctx.arc(0, 0, d/2, 0, 2*np.pi)
            ctx.restore()
            #ctx.restore()
            #ctx.fill()
            ctx.stroke()

    return Element(abcd.thin_lens(f), 0, draw_func, ((1, 0, d/2), (-1, 0, d/2)))

def make_propagation(d):
    return Element(abcd.propagation(d), d, None, ())

def make_curved_interface(n1, n2, roc, d:float=None):
    if d is None:
        draw_func = None
    else:
        def draw_func(ctx:cairo.Context):
            ctx.save()
            theta = np.arcsin(d/(2*abs(roc)))
            if roc > 0:
                thetas = np.pi - theta, np.pi + theta
            else:
                thetas = -theta, theta
            ctx.move_to(roc + abs(roc)*np.cos(thetas[0]), abs(roc)*np.sin(thetas[0]))
            ctx.arc(roc, 0, abs(roc), *thetas)
            ctx.stroke()
            #ctx.fill()
            ctx.restore()
    return Element(abcd.curved_interface(n1, n2, roc), 0, draw_func, ((1, 0, d/2), (-1, 0, d/2)))

def make_thick_lens(roc1, n, thickness, roc2, d:float=None, n_ext=1):
    return make_curved_interface(n_ext, n, roc1, d), make_propagation(thickness), make_curved_interface(n, n_ext, roc2, d)

def make_thick_spherical_transform_lens(n, w, f, d:float=None):
    roc, thickness = paraxial.design_thick_spherical_transform_lens(n, w, f)
    return make_thick_lens(roc, n, thickness, -roc, d)

def make_thick_spherical_transform(n, w, f, d:float=None):
    return (make_propagation(w), ) + make_thick_spherical_transform_lens(n, w, f, d) + (make_propagation(w), )

def make_thin_lens_fractional_transform(num_lenses:int, ft:float, theta:float, diameter:float=None):
    if num_lenses == 1:
        fl = ft/np.sin(theta)
        d = ft*np.tan(theta/2)
        return make_propagation(d), make_thin_lens(fl, diameter), make_propagation(d)
    elif num_lenses == 2:
        fl = ft/np.tan(theta/2)
        d = ft*np.sin(theta)
        return make_thin_lens(fl, diameter), make_propagation(d), make_thin_lens(fl, diameter)
    else:
        raise ValueError('Number of lenses must be 1 or 2.')

def reverse_elements(elements):
    return tuple(e.reverse() for e in elements[::-1])

def draw_ray_trace(r0, elements, ctx: cairo.Context):
    ctx.save()
    r = r0
    for element in elements:
        ctx.move_to(0, r[0])
        ctx.translate(element.thickness, 0)
        rp = element.matrix.dot(r)
        ctx.line_to(0, rp[0])
        ctx.stroke()
        r = rp
    ctx.restore()

def draw_elements(elements, ctx: cairo.Context):
    ctx.save()
    for element in elements:
        if element.draw_func is not None:
            element.draw_func(ctx)
        ctx.translate(element.thickness, 0)
    ctx.restore()