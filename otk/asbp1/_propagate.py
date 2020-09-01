from typing import Tuple
import numpy as np
import mathx
import opt_einsum
from ..types import Array1D
from ._functions import fft, ifft, make_fft_matrix, calc_curved_propagation
from ._sa import calc_q, calc_r
from ._fe import prepare_plane_to_plane_sst_1d

def propagate_plane_to_plane_flat_1d(k: float, r_support: float, Er: np.ndarray, z: float, q_center: float = 0.,
                                     axis: int = -1, paraxial: bool = False) -> np.ndarray:
    """Propagate 1D field Er a distance z between planes.

    k is the wavevector length, r_support the spatial support, q_center the transverse angular domain center, and
    axis the spatial axis of Er along which to work. If paraxial is True then paraxial propagation is used; otherwise
    the exact propagator is used.
    """
    num_points = Er.shape[axis]
    Eq = fft(Er, axis)
    q = calc_q(r_support, num_points, q_center, axis)
    if paraxial:
        Eq *= mathx.expj((k - q ** 2 / (2 * k)) * z)
    else:
        Eq *= mathx.expj((k ** 2 - q ** 2) ** 0.5 * z)
    Er = ifft(Eq, axis)
    return Er


def propagate_plane_to_plane_sst_1d(
        k: float, r_support: float, Er: np.ndarray, z: float, m: float, r_center: float = 0., q_center: float = 0,
        axis: int = -1) -> Tuple[np.ndarray, float]:
    """"Propagate 1D field Er a distance z between planes using Sziklas-Siegman transform with domain magnification m.

    The input Er is changed. k is the wavevector length, r_support the spatial support, r_center the real-space domain
    center, q_center the transverse angular domain center, and axis the spatial axis of Er along which to work.

    Sziklas-Siegman transform is used.

    Returns: the propagated field and the propagated real-space domain center.
    """
    # Not sure why I factored the preparation.
    propagator, post_factor, r_center_z = prepare_plane_to_plane_sst_1d(k, r_support, Er, z, m, r_center,
                                                                        q_center, axis)
    Ak = fft(Er, axis)
    Ak *= propagator
    Er = ifft(Ak, axis) * post_factor
    return Er, r_center_z


def propagate_plane_to_curved_sst_1d(k: float, r_support: float, Eri: Array1D, z: Array1D, m: float,
                                     r_center: float = 0., q_center: float = 0.) -> Array1D:
    """Propagate 1D field Eri from plane to sampled curved surface using Sziklas-Siegman transform.

    kz is included. z is 1D array of propagation distances of the same size as Eri.

    Limitations: all output points have common magnification. The output domain is simply m times the input domain i.e.
    it doesn't track the beam according to r_center and q_center.
    """
    assert Eri.ndim == 1
    num_points = len(Eri)
    ri = calc_r(r_support, num_points, r_center)
    qi = calc_q(r_support, num_points, q_center)
    roc = z / (m - 1)
    Qir = mathx.expj(-k * ri ** 2 / (2 * roc[:, None]))
    T = make_fft_matrix(num_points)
    P = mathx.expj((k - qi ** 2 / (2 * k * m)) * z[:, None])
    invT = T.conj()
    ro = ri * m
    Qor = mathx.expj(k * ro ** 2 / (2 * (roc + z))) / m ** 0.5
    # ro: i,  qi: j,  ri: k
    Ero = opt_einsum.contract('i, ij, ij, jk, ik, k->i', Qor, invT, P, T, Qir, Eri)
    return Ero


def propagate_plane_to_waist_spherical_paraxial_1d(k, r_support, Er, z, r_center, q_center, roc, axis=-1):
    num_points = Er.shape[axis]
    m_flat, z_flat, m = calc_curved_propagation(k, r_support, num_points, roc, z)
    if m_flat == 1:
        Er_flat = Er
        r_center_flat = r_center
    else:
        Er_flat, r_center_flat = propagate_plane_to_plane_sst_1d(k, r_support, Er, -z_flat, 1 / m_flat,
                                                                 r_center, q_center, axis)
    return Er_flat, z_flat, m_flat, m, r_center_flat


def propagate_plane_to_plane_spherical_1d(k, r_support, Er, z, r_center=0, q_center=0, roc=np.inf, axis=-1):
    num_points = Er.shape[axis]
    Er_flat, z_flat, m_flat, m, r_center_flat = propagate_plane_to_waist_spherical_paraxial_1d(k, r_support, Er, z,
                                                                                               r_center, q_center, roc)
    r_support_flat = r_support / m_flat
    z_paraxial = z / (1 - (q_center / k) ** 2) ** 1.5
    q_flat = calc_q(r_support_flat, num_points, q_center, axis)
    extra_propagator = mathx.expj((k ** 2 - q_flat ** 2) ** 0.5 * z - (k - q_flat ** 2 / (2 * k)) * z_paraxial)

    Eq_flat = fft(Er_flat, axis)
    Eq_flat *= extra_propagator
    Er_flat = ifft(Eq_flat, axis)

    Er, _ = propagate_plane_to_plane_sst_1d(k, r_support_flat, Er_flat, z_paraxial + z_flat, m * m_flat,
                                            r_center_flat, q_center)

    rp_center = r_center + z * q_center / (k ** 2 - q_center ** 2) ** 0.5

    return Er, m, rp_center


def propagate_plane_to_curved_sst_arbitrary_1d(
        k: float, r_support: float, Eri: Array1D, z: Array1D, xo: Array1D, roc: Array1D, r_center: float = 0.,
        q_center: float = 0, ro_center: float = None, kz_mode: str = 'local_xy') -> Tuple[Array1D, Array1D]:
    """Sziklas-Siegman propagate from uniformly sampled plane to arbitrarily sampled curved surface.

    The input beam is defined by wavenumber k, support along (x, y) rs_support, sampled amplitudes Eri, (x, y) support
    center rs_support, (kx, ky) support center qs_center.

    The distance to the curved surface is given by z of shape (M, N), which can be different to the shape of Eri. The
    sampled positions are (M,1) array xo and (N,) array yo. The radius of curvature for the Sziklas-Siegman transform
    are given by (M, N)-shaped arrays, roc_x and roc_y.

    The longitudinal wavevector mode is defined by kz_mode, either 'paraxial' or 'local_xy'.

    The output field and its partial derivatives w.r.t. x and y are returned as (M,N)-shaped arrays.
    """
    assert Eri.ndim == 1
    M, gradM = calc_plane_to_curved_sst_arbitrary_factors_1d(
        k, r_support, len(Eri), z, xo, roc, r_center, q_center, ro_center, kz_mode)
    Ero = M @ Eri
    gradEro = gradM @ Eri
    return Ero, gradEro