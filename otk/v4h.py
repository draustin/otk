"""Definitions and operations for homogeneous vectors in 3D space."""
from typing import Sequence
import numpy as np
try:
    import numba
except ImportError:
    numba = None

if numba is None:
    def cross(a, b):
        return np.r_[np.cross(a[:3], b[:3]), 0]
else:
    @numba.njit
    def cross(a, b):
        return np.array((a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0], 0.))

def is_point(x:np.ndarray):
    return (x.shape == (4,)) and (x[3] == 1.)


def is_vector(v:np.ndarray):
    return (v.shape == (4,)) and (v[3] == 0.)

def to_point(x: Sequence[float]) -> np.ndarray:
    x = np.array(x, float)
    assert x.ndim == 1
    if len(x) == 3:
        x = np.r_[x, 1.]
    assert is_point(x)
    return x

def to_vector(x: Sequence[float]) -> np.ndarray:
    x = np.array(x, float)
    assert x.ndim == 1
    if len(x) == 3:
        x = np.r_[x, 0.]
    assert is_vector(x)
    return x

xhat = to_vector((1, 0, 0))
yhat = to_vector((0, 1, 0))
zhat = to_vector((0, 0, 1))
origin = to_point((0, 0, 0))
unit_vectors = xhat, yhat, zhat
