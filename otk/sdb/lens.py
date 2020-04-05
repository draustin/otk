from typing import Sequence, Tuple
import numpy as np
from .. import math as omath
from ..sdb import *

__all__ = ['make_spherical_singlet', 'make_toroidal_singlet', 'make_spherical_singlet_square_array']

def make_spherical_singlet(roc0, roc1, thickness, vertex0:Sequence[float]=(0,0,0), shape:str='circle', radius:float=None, side_lengths:Tuple[float,float]=None):
    vertex0 = np.asarray(vertex0)
    s0 = SphericalSag(roc0, 1, vertex0)
    s1 = SphericalSag(roc1, -1, vertex0 + (0, 0, thickness))
    if shape == 'circle':
        assert radius is not None
        s2 = InfiniteCylinder(radius, vertex0[:2])
        z0 = min(omath.calc_sphere_sag(roc0, radius), 0)
        z1 = thickness + max(omath.calc_sphere_sag(roc1, radius), 0)
        bound = Box((radius, radius, (z1 - z0)/2), (vertex0[0], vertex0[1], vertex0[2] + (z1 + z0)/2))
    elif shape == 'rectangle':
        assert len(side_lengths) == 2
        s2 = InfiniteRectangularPrism(*side_lengths, center=vertex0[:2])
        radius = (side_lengths[0]**2 + side_lengths[1]**2)**0.5/2
        z0 = min(omath.calc_sphere_sag(roc0, radius), 0)
        z1 = thickness + max(omath.calc_sphere_sag(roc1, radius), 0)
        bound = Box((side_lengths[0]/2, side_lengths[1]/2, (z1 - z0)/2), (vertex0[0], vertex0[1], vertex0[2] + (z1 + z0)/2))
    else:
        raise ValueError(f'Unknown shape {shape}.')

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
    s0 = FiniteRectangularArray(pitch, size, SphericalSag(roc0, 1, (0, 0, face_center[2])), origin=face_center[:2])
    s1 = FiniteRectangularArray(pitch, size, SphericalSag(roc1, -1, (0, 0, face_center[2] + thickness)), origin=face_center[:2])
    s2 = InfiniteRectangularPrism(*(pitch*size), face_center[:2])
    radius = sum(pitch**2)**0.5/2
    z0 = min(omath.calc_sphere_sag(roc0, radius), 0)
    z1 = thickness + max(omath.calc_sphere_sag(roc1, radius), 0)
    bound = Box((size[0]*pitch[0]/2, size[1]*pitch[1]/2, (z1 - z0)/2), (face_center[0], face_center[1], face_center[2] + (z1 + z0)/2))
    return IntersectionOp((s0, s1, s2), bound)


