"""Functions and definitions related to beam sampling axes."""
from collections import namedtuple
import numpy as np
import mathx

AbsPhase = namedtuple('AbsPhase', ('abs', 'phase'))
RQ = namedtuple('RQ', ('r', 'q'))

def arange_centered(num, axis = -1):
    return mathx.reshape_vec(np.arange(num)-(num-1)/2, axis)

def to_scalar_pair(x):
    """Produce pair of scalars with type checking.

    Returns:
         tuple of scalars
    """
    try:
        assert len(x) == 2
        x = np.asarray(x)
    except TypeError:
        x = np.asarray((x, x))
    assert all(np.isscalar(e) for e in x)
    return x

def is_scalar_pair(x):
    """Test if argument is a pair of scalar.s"""
    return len(x) == 2 and all(np.isscalar(e) for e in x)

def center_samples(num_points, n0):
    """Returns vector n+m[n]*num_points where n = arange(num_points),  with range centered on n0."""
    hn = int(num_points/2)
    nb = np.arange(num_points)
    n = (nb-n0+hn)%num_points+n0-hn
    return n

def calc_unroll_indices(num_points, n0):
    hn = int(num_points/2)
    indices = (n0+np.arange(-hn, hn))%num_points
    return indices

def calc_r_unroll_indices(r_support, num_points, r_center = 0):
    return calc_unroll_indices(num_points, int(r_center/r_support*num_points))

def calc_q_unroll_indices(r_support, num_points, q_center = 0):
    delta_q = 2*np.pi/r_support
    return calc_unroll_indices(num_points, int(q_center/delta_q))

def unroll_r_1d(r_support, Er, r_center = 0, axis = -1):
    num_points = Er.shape[axis]
    r = calc_r(r_support, num_points, r_center, axis, True)
    ui = calc_r_unroll_indices(r_support, num_points, r_center)
    return r, Er[ui]

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

def calc_q(r_support, num_points, q_center = 0, axis = -1, unroll = False):
    delta_q = 2*np.pi/r_support
    n0 = int(q_center/delta_q)
    if unroll:
        hn = int(num_points/2)
        n = mathx.reshape_vec(np.arange(n0-hn, n0+hn), axis)
    else:
        n = center_samples(num_points, n0)
    q = mathx.reshape_vec(n, axis)*delta_q
    return q

def calc_r(r_support, num_points, r_center = 0, axis = -1, unroll = False):
    delta_r = r_support/num_points
    n0 = int(r_center/delta_r)
    if unroll:
        hn = int(num_points/2)
        n = mathx.reshape_vec(np.arange(n0-hn, n0+hn), axis)
    else:
        n = center_samples(num_points, n0)
    r = mathx.reshape_vec(n, axis)*delta_r
    return r

def calc_xy(rs_support, num_pointss, rs_center = (0, 0), unroll = False, broadcast = True):
    rs_support = to_scalar_pair(rs_support)
    num_pointss = to_scalar_pair(num_pointss)
    if broadcast:
        axes = (-2, -1)
    else:
        axes = (-1, -1)
    return [calc_r(*args, unroll) for args in zip(rs_support, num_pointss, rs_center, axes)]

def calc_kxky(rs_support, num_pointss, qs_center = (0, 0), unroll = False, broadcast = True):
    rs_support = to_scalar_pair(rs_support)
    num_pointss = to_scalar_pair(num_pointss)
    if broadcast:
        axes = (-2, -1)
    else:
        axes = (-1, -1)
    return [calc_q(*args, unroll) for args in zip(rs_support, num_pointss, qs_center, axes)]

def calc_Eq_factor(r_support, num_points):
    """Factor by which Eq should be multiplied to approximate continuous Fourier transform (1D)..

    Derivation p130 Dane's logbook 2. Our DFT and IDFT satisfies Parseval's relation expressed as a sum over samples with
    no scaling. But the physically relevant quantity is the integral which we approximate by the sum times the sampling
    period. This function returns the factor by which the numerical Eq should be multiplied to give a quantity with
    the same square integral as the real space field.
    """
    return r_support/(2*np.pi*num_points)**0.5
