import mathx
import pyfftw
import numpy as np
from numba import njit
from ._sa import calc_q, calc_r

fft = lambda Ar, axis=-1: pyfftw.interfaces.numpy_fft.fft(Ar, norm='ortho', axis=axis)
ifft = lambda Ak, axis=-1: pyfftw.interfaces.numpy_fft.ifft(Ak, norm='ortho', axis=axis)

def make_fft_matrix(num_points):
    eye = pyfftw.byte_align(np.eye(num_points))
    matrix = fft(eye)
    return matrix


def make_ifft_arbitrary_matrix(r_support0, num_points0, q_center0, r1):
    """
    Warning: The matrix does not enforce a particular subset of the infinite periodic r1 domain. So if the range of r1
    is greater than 2*pi/r_support0, then the output will be repeated.


    Args:
        r_support0:
        num_points0:
        q_center0:
        r1:

    Returns:

    """
    assert r1.ndim == 1
    q0 = calc_q(r_support0, num_points0, q_center0)
    return mathx.expj(q0 * r1[:, None]) / len(q0) ** 0.5


# def make_ifft_arbitrary_matrix(r_support0, num_points0, q_center0, r_support1, num_points1, r_center1):
#     q0 = sa.calc_q(r_support0, num_points0, q_center0)
#     r1 = sa.calc_r(r_support1, num_points1, r_center1)[:, None]
#     return mathx.expj(q0*r1)/num_points0**0.5


def expand_kz(k, kxc):
    """Expand z component of wavenumber to second order.

    Args:
        k: Wavenumber.
        kxc: Expansion kx coordinate.

    Returns:
        kz: Zeroth order term.
        gxkz: First derivative w.r.t x.
        gxxkz: Second derivative w.r.t. x.

    """
    kz = (k ** 2 - kxc ** 2) ** 0.5
    gxkz = -kxc / kz
    denominator = (k ** 2 - kxc ** 2) ** (3 / 2)
    gxxkz = -(k ** 2) / denominator
    return kz, gxkz, gxxkz


def calc_quadratic_kz_correction(k, kxc):
    """Calculate correction that needs to be applied to propagation with quadratic dispersion relation.

    The propagator that should be applied is -(fx*kx**2)*z/(2*k) (no zeroth order term)

    Args:
        kxc: central x wavenumber

    Returns:
        fx: factor by which propagation distance should be multiplied to give quadratic x propagation distance
        delta_kz: Zero order correction factor. The corresponding additional phase is delta_kz*z.
        delta_gxkz: first order x correction factor i.e. result should have phase of delta_gxkz*kx*z applied. Negative of
            the corresponding translation.
    """
    # Get zeroth, first and second order (nonmixed) derivatives of k at (kxc, kyc).
    kz, gxkz, gykz, gxxkz, gyykz, gxykz = expand_kz(k, kxc)

    # Calculate first and second order derivatives of the quadratic approximation at (kxc, kyc).
    gxkz_p = -kxc / k
    gxxkz_p = -1 / k
    gyykz_p = -1 / k

    # Correction factors are ratios of correct second derivatives to the quadratic approximation ones.
    fx = gxxkz / gxxkz_p
    fy = gyykz / gyykz_p

    # Shift corrections are difference between correct first order terms and quadratic approximation ones.
    delta_gxkz = gxkz - fx * gxkz_p

    # Evaluate zeroth order kz for x and y quadratic approximation.
    kz_px = -kxc ** 2 / (2 * k)

    # Zeroth order correction is what is required to match k_z at (kxc, kyc) after quadratic approximation propagation
    # (with factors fx and fy) and shift correction has been applied.
    delta_kz = kz - fx * kz_px - delta_gxkz * kxc

    return fx, fy, delta_kz, delta_gxkz


@njit
def calc_kz_quadratic_1d(k, q):
    """kz not included."""
    return -q ** 2 / (2 * k)


@njit
def calc_propagator_quadratic(k, q, l):
    """kz not included."""
    return mathx.expj(calc_kz_quadratic_1d(k, q) * l)


@njit
def calc_quadratic_phase(k, r, roc):
    return mathx.expj(k * r ** 2 / (2 * roc))


def calc_flat_plane(k, r_support, num_points, roc):
    r_typical = r_support / (np.pi * num_points) ** 0.5
    if np.isfinite(roc):
        num_rayleighs_flat = k * r_typical ** 2 / (2 * roc)
        m_flat = (1 + num_rayleighs_flat ** 2) ** 0.5
    else:
        m_flat = 1
        num_rayleighs_flat = 0
    r_typical_flat = r_typical / m_flat
    z_R = r_typical_flat ** 2 * k / 2
    z_flat = num_rayleighs_flat * z_R
    return m_flat, z_flat, z_R


def calc_curved_propagation(k, r_support, num_points, roc, z):
    m_flat, z_flat, z_R = calc_flat_plane(k, r_support, num_points, roc)
    num_rayleighs = (z + z_flat) / z_R
    m = (1 + num_rayleighs ** 2) ** 0.5 / m_flat
    return m_flat, z_flat, m


def calc_curved_propagation_m(k, r_support, num_points, roc, z):
    return calc_curved_propagation(k, r_support, num_points, roc, z)[2]


def calc_propagation_m_1d(k, r_support, var_r0, phi_c, var_q, z, num_points):
    """Calculate sensible magnification for Sziklas-Siegman propagation.

    The magnification is chosen so that the ratio of the RMS radius to the support will be equal in real and angular space.

    This has gone through various versions - basically the right answer depends a lot on context. The current version
    aims to keep it simple.

    The operations are all elementwise, so broadcasting any argument is allowed.

    Args:
        k: Wavenumber.
        r_support:
    """
    q_support = 2 * np.pi * num_points / r_support

    # Calculate real-space variance at z. From this infer a lower limit on the magnification.
    var_rz, phi_cz, _ = bvar.calc_propagated_variance_1d(k, var_r0, phi_c, var_q, z)
    m_lower = var_rz ** 0.5 * 12 / r_support

    # Calculate minimum angular variance at z. This will not change with propagation. From this infer an upper
    # limit on the magnification.
    var_q_min = bvar.calc_minimum_angular_variance_1d(var_rz, phi_cz, var_q)
    m_upper = q_support / (var_q_min ** 0.5 * 12)

    if np.any(m_lower > m_upper):
        logger.warning('Magnification lower bound greater than upper bound.')

    m = m_lower

    return m

def calc_plane_to_curved_sst_arbitrary_factors_1d(
        k: float, r_support: float, num_points: int, z: Array1D, xo: Array1D, roc_x: Array1D, ri_center: float = 0.,
        q_center: float = 0., ro_center: float = None, kz_mode: str = 'local_xy') -> Tuple[Array2D, Array2D]:
    """Calculate factors for 2D Sziklas-Siegman propagation (paraxial) from flat to curved surface(s).

    The input beam is defined by wavenumber k, real space support r_support with num_points sampling points centered on
    ri_center, transverse wavevector center of support q_center.

    The output curved surface is defined longitudinal distance from the start plane z at position ro. The Sziklas-Siegman
    propagation to each point has different assumed radius of curvature roc.

    The zero-oth order phase (k_z) is (arbitrarily) included in QinvTP.

    Compared to regular Sziklas-Siegman procedure, the factors are complicated by a transformation that allows nonzero
    real and angular space centers.  See derivation page 114 Dane's logbook 2.

    kz_mode is 'local_xy' or 'paraxial'.

    TODO change local_xy to name that works in 1D as well as 2D.
    """
    assert kz_mode in ('local_xy', 'paraxial')
    assert z.ndim == 1
    assert z.shape == roc_x.shape
    assert z.shape == xo.shape

    if ro_center is None:
        z_center = np.mean(z)
        ro_center = adjust_r(k, ri_center, z_center, q_center, kz_mode)

    # If local_xy mode is requested then propagation distances must be scaled. Calculate scaled propagation distances
    # and required kz component.
    if kz_mode == 'paraxial':
        zx = z
        zy = z
        delta_kz = k
    else:
        fx, fy, delta_kz, delta_gxkz, delta_gykz = calc_quadratic_kz_correction(k, q_center, 0)
        zx = fx * z

    mx = zx / roc_x + 1

    xi = calc_r(r_support, num_points, ri_center)
    kxi = calc_q(r_support, num_points, q_center)  # +q_curvatures)

    # Quadratic phase is centered at origin. Numerically, this requires evaluation of complex exponentials of size N^3.
    # Significant fraction of total cost of this function, but unavoidable.
    Qix = calc_quadratic_phase(-k, xi[:, 0] - ri_center[0], roc_x[:, :, None])

    Tx = make_fft_matrix(num_points)

    roc_xo = roc_x + zx

    # If in local_xy mode, then the result of the Sziklas-Siegman step needs to be translated. This means that the
    # factors which depend on xo and yo need to be translated too. (Note that the x and y factors in the inverse DFT
    # are not translated.) We define xop and yop as the translated output coordinates.
    if kz_mode == 'local_xy':
        xop = xo + delta_gxkz * z
    else:
        xop = xo

    # Effective kx and ky for propagation takes into account center of curvature.
    kxip = kxi[:, 0] - k * ri_center / roc_x[:, :, None]

    r2_support_x = mx * r_support
    valid_x = abs(xo - ro_center) < r2_support_x * 0.5

    # Calculate product of propagator and inverse transform along x axis. The x axis (arbitrarily) includes delta_kz.
    # Could combine all exponents before exponentiation? Won't help much because the time will be dominated by the propagator
    # which is N^3 - the others are N^2.
    phi = delta_kz * z + k * (ri_center ** 2 / (2 * roc_x) - ri_center * xop / (roc_x * mx))
    QinvTPx = (calc_propagator_quadratic(k * mx[:, None], kxip,
                                            zx[:, None]) *  # Propagation phase scaled by magnification.
               calc_quadratic_phase(k, xop, roc_xo)[:, None] *  # Normal Sziklas-Siegman final quadratic phase.
               mathx.expj(phi[:, None] + kxi[:, 0] * (xo / mx)[:,
                                                     None]) / num_points ** 0.5 /  # Correction phases and inverse DFT.
               abs(mx[:, None]) ** 0.5 *  # Magnification correction to amplitude.
               valid_x[:, None])

    # If local_xy mode, need translation correction factor. Could combine this with above to reduce number of N^3 expj
    # calls but this would only give a marginal performance improvement.
    if kz_mode == 'local_xy':
        QinvTPx *= mathx.expj(delta_gxkz * kxi * (z / mx)[:, None])

    # Evaluate derivatives of the invTP factors with respect to output x and y (but constant z - not along the curved
    # surface).
    gradxQinvTPx = 1j * (
            k * (xop / roc_xo - ri_center / (roc_x * mx))[:, None] + kxi / mx[:, None]) * QinvTPx

    # Dot product over kx.
    return (QinvTPx @ Tx) * Qix, (gradxQinvTPx @ Tx) * Qix