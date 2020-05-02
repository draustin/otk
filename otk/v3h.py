"""Definitions and operations for homogeneous vectors in 2D space."""
import numpy as np

def is_point(x: np.ndarray) -> bool:
    return (x.shape == (3,)) and (x[2] == 1.)

def is_vector(v: np.ndarray):
    return (v.shape == (3,)) and (v[2] == 0.)