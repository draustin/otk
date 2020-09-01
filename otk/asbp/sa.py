from typing import Union, Tuple
import numpy as np

from ..asbp1 import calc_r_unroll_indices, calc_q_unroll_indices, calc_q, calc_r


def to_scalar_pair(x: Union[float, Tuple[float, float]]) -> np.ndarray:
    """Produce pair of scalars with type checking.

    Returns:
         tuple of scalars
    """
    try:
        assert len(x) == 2
        x = np.asarray(x)
    except TypeError:
        x = np.asarray((x, x))
    assert np.issubdtype(x.dtype, np.number)
    return x


def is_scalar_pair(x):
    """Test if argument is a pair of scalar.s"""
    return len(x) == 2 and all(np.isscalar(e) for e in x)


def unroll_r(rs_support, Er, rs_center=(0, 0)):
    rs_support = to_scalar_pair(rs_support)
    x, y = calc_xy(rs_support, Er.shape, rs_center, True)
    uix = calc_r_unroll_indices(rs_support[0], Er.shape[0], rs_center[0])
    uiy = calc_r_unroll_indices(rs_support[1], Er.shape[1], rs_center[1])
    return x, y, Er[uix[:, None], uiy]


def unroll_q(rs_support, Eq, qs_center=(0, 0)):
    rs_support = to_scalar_pair(rs_support)
    kx, ky = calc_kxky(rs_support, Eq.shape, qs_center, True)
    uix = calc_q_unroll_indices(rs_support[0], Eq.shape[0], qs_center[0])
    uiy = calc_q_unroll_indices(rs_support[1], Eq.shape[1], qs_center[1])
    return kx, ky, Eq[uix[:, None], uiy]


def unroll_along_r(rs_support, fr, rs_center=(0, 0)):
    uix = calc_r_unroll_indices(rs_support[0], fr.shape[0], rs_center[0])
    uiy = calc_r_unroll_indices(rs_support[1], fr.shape[1], rs_center[1])
    return fr[uix[:, None], uiy]


def calc_xy(rs_support, num_pointss, rs_center=(0, 0), unroll=False, broadcast=True):
    rs_support = to_scalar_pair(rs_support)
    num_pointss = to_scalar_pair(num_pointss)
    if broadcast:
        axes = (-2, -1)
    else:
        axes = (-1, -1)
    return [calc_r(*args, unroll) for args in zip(rs_support, num_pointss, rs_center, axes)]


def calc_kxky(rs_support, num_pointss, qs_center=(0, 0), unroll=False, broadcast=True):
    rs_support = to_scalar_pair(rs_support)
    num_pointss = to_scalar_pair(num_pointss)
    if broadcast:
        axes = (-2, -1)
    else:
        axes = (-1, -1)
    return [calc_q(*args, unroll) for args in zip(rs_support, num_pointss, qs_center, axes)]



