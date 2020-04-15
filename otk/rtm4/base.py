from dataclasses import dataclass
from typing import Callable, Sequence, Tuple
import numpy as np
from .. import abcd, paraxial, draw2

__all__ = ['transform_limit', 'Element', 'make_thin_lens', 'make_propagation', 'make_curved_interface', 'make_interface', 'make_thin_lens_fractional_transform']

def transform_limit(limit: Sequence[float], inv_matrix: np.ndarray) -> np.ndarray:
    """Transform a limit through a ray transfer matrix.

    Args:
        limit: x, theta, constant coefficients.
        inv_matrix: Inverse matrix of the element
    """
    return np.r_[np.dot(limit[:2], inv_matrix), limit[2]]

@dataclass
class Element:
    """
    A limit is an inequality satisifed by valid rays i.e. a limit (alpha, beta, gamma) means that a valid ray (x, theta) satisfies
        x*alpha + theta*beta + gamma >= 0.
    This applies *before* the element.
    """
    matrix: np.ndarray
    thickness: float
    drawing: draw2.Drawing
    limits: Tuple[np.ndarray]

    @classmethod
    def make(cls, matrix, thickness, draw_func, limits):
        matrix = np.array(matrix, float)
        assert matrix.shape == (2, 2)
        thickness = float(thickness)
        def to_limit(l):
            l = np.array(l, float)
            assert l.shape == (3,)
            return l
        limits = tuple(to_limit(l) for l in limits)
        return cls(matrix, thickness, draw_func, limits)

    def reverse(self):
        # Transform each limit through self, then negate theta as we are reversing.
        limits = tuple(transform_limit(limit, np.linalg.inv(self.matrix))*[1, -1, 1] for limit in self.limits)
        if self.drawing is None:
            drawing = None
        else:
            drawing = draw2.Translation(self.thickness, 0, draw2.Scaling(-1, 1, self.drawing))
        return Element.make(abcd.reverse(self.matrix), self.thickness, drawing, limits)

    def __mul__(self, other: 'Element'):
        # The limits apply before the element, and transforms are applied right to left. So transform
        # self's limits through other.
        limits = tuple(transform_limit(limit, other.matrix) for limit in self.limits) + other.limits

        children = []
        if self.drawing is not None:
            children.append(self.drawing)
        if other.drawing is not None:
            children.append(draw2.Translation(self.thickness, 0, other.drawing))
        if len(children) > 0:
           drawing = draw2.Sequence(children)
        else:
           drawing = None

        return Element.make(self.matrix.dot(other.matrix), self.thickness + other.thickness, drawing, limits)

def make_thin_lens(f, d:float=None):
    if d is None:
        draw_func = None
        limits = ()
    else:
        drawing = draw2.Scaling(0.1, 1, draw2.Arc(0, 0, d/2, 0, 2*np.pi))
        limits = ((1, 0, d/2), (-1, 0, d/2))

    return Element.make(abcd.thin_lens(f), 0, drawing, limits)

def make_propagation(d):
    return Element(abcd.propagation(d), d, None, ())

def make_curved_interface(n1, n2, roc, d:float=None):
    if d is None:
        draw_func = None
        limits = ()
    else:
        theta = np.arcsin(d/(2*abs(roc)))
        if roc > 0:
            thetas = np.pi - theta, np.pi + theta
        else:
            thetas = -theta, theta
        drawing = draw2.Arc(roc, 0, abs(roc), *thetas)
        # def draw_func(ctx:cairo.Context):
        #     ctx.save()
        #
        #     ctx.move_to(roc + abs(roc)*np.cos(thetas[0]), abs(roc)*np.sin(thetas[0]))
        #     ctx.arc(roc, 0, abs(roc), *thetas)
        #     ctx.stroke()
        #     #ctx.fill()
        #     ctx.restore()
        limits = ((1, 0, d/2), (-1, 0, d/2))
    return Element.make(abcd.curved_interface(n1, n2, roc), 0, drawing, limits)

def make_interface(n1, n2, d:float=None):
    if d is None:
        drawing = None
        limits = ()
    else:
        drawing = draw2.Line(0, -d/2, 0, d/2)
        limits = ((1, 0, d/2), (-1, 0, d/2))
    return Element.make(abcd.interface(n1, n2), 0, drawing, limits)

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
