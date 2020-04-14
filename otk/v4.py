"""Operations on spatial 4-vectors."""
from typing import Sequence
import numpy as np
try:
    import numba
except ImportError:
    numba = None

if numba is None:
    def dot(a, b):
        return np.dot(a[:3], b[:3])

    def norm_squared(x):
        return dot(x, x)

    def norm(x):
        return dot(x, x)**0.5

    def normalize(x):
        return x/norm(x)

    def cross(a, b):
        return np.r_[np.cross(a[:3], b[:3]), 0]
else:
    @numba.njit("f8(f8[:], f8[:])")
    def dot(x, y):
        s = 0
        for i in range(len(x)):
            s += x[i]*y[i]
        return s


    @numba.njit("f8(f8[:])")
    def norm_squared(x):
        return dot(x, x)


    @numba.njit("f8(f8[:])")
    def norm(x):
        return norm_squared(x)**0.5


    @numba.njit
    def normalize(x):
        return x/norm(x)

    @numba.njit
    def cross(a, b):
        return np.array((a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0], 0.))
    # def cross(a, b):
    #     # http://numba.pydata.org/numba-doc/0.15.1/arrays.html
    #     # Arrays can be passed in to a function in nopython mode, but not returned. Arrays can only be returned in object mode.
    #     return np.asarray(_cross(a, b))

# if numba is not None:
#     dot  = numba.njit("f8(f8[:], f8[:])")(dot)
#     norm_squared = numba.njit(norm_squared)
#     norm = numba.njit(norm)
#     normalize = numba.njit(normalize)
#     cross = numba.njit(cross)


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
