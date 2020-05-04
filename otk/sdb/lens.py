"""Functions for making lens-shaped surfaces."""
from dataclasses import dataclass
from typing import Sequence, Tuple, Callable
import numpy as np
from .. import functions
from ..types import Sequence2, Sequence3
from ..sdb import *

__all__ = ['make_spherical_singlet', 'make_toroidal_singlet', 'make_spherical_singlet_square_array', 'make_circle', 'make_rectangle']

@dataclass
class LensShape:
    max_radius: float
    make_edge: Callable[[Sequence2[float]], Surface]
    make_bound: Callable[[Sequence3[float]], Primitive]

def make_circle(radius: float) -> LensShape:
    def make_edge(vertex0):
        return InfiniteCylinder(radius, vertex0)
    def make_bound(vertex0, z0, z1):
        # TODO make FiniteCylinder primitive and use it instead.
        return Box((radius, radius, (z1 - z0)/2), (vertex0[0], vertex0[1], vertex0[2] + (z1 + z0)/2))
    return LensShape(radius, make_edge, make_bound)

def make_rectangle(width: float, height: float) -> LensShape:
    max_radius = (width**2 + height**2)**0.5/2
    def make_edge(vertex0):
        return InfiniteRectangularPrism(width, height, center=vertex0)
    def make_bound(vertex0, z0, z1):
        return Box((width/2, height/2, (z1 - z0)/2), (vertex0[0], vertex0[1], vertex0[2] + (z1 + z0)/2))
    return LensShape(max_radius, make_edge, make_bound)

def make_spherical_singlet(roc0: float, roc1: float, thickness: float, shape: LensShape, vertex0: Sequence3[float]) -> Surface:
    """Make circular or rectangular spherical singlet with given radii of curvatures and vertex coordinate.

    Radii of curvatures obey normal optics convention i.e. biconvex is roc0 > 0, roc1 < 0.
    """
    vertex0 = np.asarray(vertex0, float)
    s0 = SphericalSag(roc0, 1, vertex0)
    s1 = SphericalSag(roc1, -1, vertex0 + (0, 0, thickness))
    s2 = shape.make_edge(vertex0[:2])
    z0 = min(functions.calc_sphere_sag(roc0, shape.max_radius), 0)
    z1 = thickness + max(functions.calc_sphere_sag(roc1, shape.max_radius), 0)
    bound = shape.make_bound(vertex0, z0, z1)
    return IntersectionOp((s0, s1, s2), bound)


def make_toroidal_singlet(rocs0, rocs1, thickness, radius: float, vertex0:Sequence[float]=(0,0,0), shape:str='circle'):
    vertex0 = np.asarray(vertex0)
    s0 = ToroidalSag(rocs0, 1, vertex0)
    s1 = ToroidalSag(rocs1, -1, vertex0 + (0, 0, thickness))
    if shape == 'circle':
        s2 = InfiniteCylinder(radius, vertex0[:2])
    elif shape == 'square':
        side_length = radius*2**0.5
        s2 = InfiniteRectangularPrism(side_length, center=vertex0[:2])
    else:
        raise ValueError(f'Unknown shape {shape}.')
    return IntersectionOp((s0, s1, s2))

def make_spherical_singlet_square_array(roc0, roc1, thickness, pitch, size, face_center=None):
    pitch = np.asarray(pitch)
    assert pitch.shape == (2,)
    size = np.asarray(size)
    assert size.shape == (2,)
    if face_center is None:
        face_center = 0, 0, 0
    face_center = np.asarray(face_center)
    assert face_center.shape == (3,)
    s0 = FiniteRectangularArray(pitch, size, SphericalSag(roc0, 1, (0, 0, face_center[2])), center=face_center[:2])
    s1 = FiniteRectangularArray(pitch, size, SphericalSag(roc1, -1, (0, 0, face_center[2] + thickness)), center=face_center[:2])
    s2 = InfiniteRectangularPrism(*(pitch*size), face_center[:2])
    radius = sum(pitch**2)**0.5/2
    z0 = min(functions.calc_sphere_sag(roc0, radius), 0)
    z1 = thickness + max(functions.calc_sphere_sag(roc1, radius), 0)
    bound = Box((size[0]*pitch[0]/2, size[1]*pitch[1]/2, (z1 - z0)/2), (face_center[0], face_center[1], face_center[2] + (z1 + z0)/2))
    return IntersectionOp((s0, s1, s2), bound)


