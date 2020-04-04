"""Operations on spatial 4-vectors."""
from typing import Sequence
import numpy as np

xhat = np.asarray([1, 0, 0, 0])
yhat = np.asarray([0, 1, 0, 0])
zhat = np.asarray([0, 0, 1, 0])
unit_vectors = xhat, yhat, zhat

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
