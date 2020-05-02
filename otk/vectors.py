"""Operations and definitions on vectors."""
from typing import Sequence
import numpy as np

try:
    import numba
except ImportError:
    numba = None

if numba is None:
    dot = np.dot # for convenience

    def norm_squared(x: Sequence[float]):
        return dot(x, x)

    def norm(x: Sequence[float]):
        return dot(x, x)**0.5

    def normalize(x: Sequence[float]):
        return x/norm(x)
else:
    @numba.njit("f8(f8[:], f8[:])") # TODO function signature necessary?
    def dot(x, y):
        # TODO can use np.dot?
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

# 2D vector (2 floats).
Sequence2V = Sequence[float]
# 3D vector (3 floats).
Sequence3V = Sequence[float]
# 2D homogeneous vector (3 floats).
Sequence2VH = Sequence[float]
# 3D homogeneous vector (4 floats).
Sequence3VH = Sequence[float]


