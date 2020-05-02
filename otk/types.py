"""Define type aliases purely for documentation purposes."""
import numpy as np
from typing import Sequence

# Sequences of a certain length.
Sequence2 = Sequence
Sequence3 = Sequence
Sequence4 = Sequence

# Numpy arrays of a certain shape.
Vector2 = np.ndarray # (2,)
Vector3 = np.ndarray # (3,)
Vector4 = np.ndarray # (4,)
Matrix3 = np.ndarray # (3, 3)
Matrix4 = np.ndarray # (4, 4)

# Numpy arrays with final dimension of certain length (for broadcasted operations).
Scalars = np.ndarray # (..., 1)
Vectors2 = np.ndarray # (..., 2)
Vectors3 = np.ndarray # (..., 3)
Vectors4 = np.ndarray # (..., 4)
