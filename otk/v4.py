"""Operations on spatial 4-vectors."""
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

