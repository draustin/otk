from typing import Tuple
import numpy as np
from ..types import Array1D
import mathx
from ._functions import fft, ifft, calc_quadratic_phase, calc_propagator_quadratic
from ._sa import calc_r, calc_q


def shift_r_center_1d(r_support, num_points, Er, delta_r_center, q_center, k_on_roc=0, axis=-1):
    r = calc_r(r_support, num_points, axis=axis)
    q = calc_q(r_support, num_points, q_center, axis=axis)
    # Remove quadratic phase and tilt.
    Er *= mathx.expj(-(q[0] * r + k_on_roc * r ** 2 / 2))
    Ek = fft(Er, axis)
    Ek *= mathx.expj(delta_r_center * q)
    Er = ifft(Ek, axis)
    Er *= mathx.expj(q[0] * r + k_on_roc * (r + delta_r_center) ** 2 / 2)
    return Er


def prepare_plane_to_plane_sst_1d(
        k: float, r_support: float, Er: Array1D, z: float, m: float, r_center: float = 0.,
        q_center: float = 0., axis: int = -1, carrier: bool = True) -> Tuple[Array1D, Array1D, float]:
    """Calculate propagator and post-factor for Sziklas-Siegman transform propagation from plane to plane.

    Modifies Er."""
    assert not (np.isclose(z, 0) ^ np.isclose(m, 1))
    if np.isclose(z, 0):
        return 1, 1, r_center
    num_points = Er.shape[axis]
    roc = z / (m - 1)
    r = calc_r(r_support, num_points, r_center, axis)
    q = calc_q(r_support, num_points, q_center, axis)  # -q_center
    Er *= calc_quadratic_phase(-k, r - r_center, roc)
    propagator = calc_propagator_quadratic(k * m, q - k * r_center / roc, z)

    ro_center = r_center + q_center / k * z
    ro = calc_r(r_support * m, num_points, ro_center, axis)
    post_factor = calc_quadratic_phase(k, ro, roc + z) * mathx.expj(
        k * (r_center ** 2 / (2 * roc) - r_center * ro / (roc * m))) / m ** 0.5
    if carrier:
        post_factor *= mathx.expj(k * z)

    return propagator, post_factor, ro_center