import logging
import numpy as np
import scipy
import numba
import opt_einsum
import pyfftw
import mathx
from .. import bvar
from . import sa

logger = logging.getLogger(__name__)

fft = lambda Ar, axis=-1:pyfftw.interfaces.numpy_fft.fft(Ar, norm='ortho', axis=axis)
ifft = lambda Ak, axis=-1:pyfftw.interfaces.numpy_fft.ifft(Ak, norm='ortho', axis=axis)
fft2 = lambda Ar:pyfftw.interfaces.numpy_fft.fft2(Ar, norm='ortho')
ifft2 = lambda Ak:pyfftw.interfaces.numpy_fft.ifft2(Ak, norm='ortho')
empty = lambda shape:pyfftw.empty_aligned(shape, complex)


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
    q0 = sa.calc_q(r_support0, num_points0, q_center0)
    return mathx.expj(q0*r1[:, None])/len(q0)**0.5

# def make_ifft_arbitrary_matrix(r_support0, num_points0, q_center0, r_support1, num_points1, r_center1):
#     q0 = sa.calc_q(r_support0, num_points0, q_center0)
#     r1 = sa.calc_r(r_support1, num_points1, r_center1)[:, None]
#     return mathx.expj(q0*r1)/num_points0**0.5


def expand_kz(k, kxc, kyc):
    """Expand z component of wavenumber to second order.

    Args:
        k: Wavenumber.
        kxc: Expansion kx coordinate.
        kyc: Expansion ky coordinate.

    Returns:
        kz: Zeroth order term.
        gxkz: First derivative w.r.t x.
        gykz: First derivative w.r.t y.
        gxxkz: Second derivative w.r.t. x.
        gyykz: Second derivative w.r.t. y.

    """
    kz = (k**2 - kxc**2 - kyc**2)**0.5
    gxkz = -kxc/kz
    gykz = -kyc/kz
    denominator = (k**2 - kxc**2 - kyc**2)**(3/2)
    gxxkz = -(k**2 - kyc**2)/denominator
    gyykz = -(k**2 - kxc**2)/denominator
    gxykz = -kxc*kyc/denominator
    return kz, gxkz, gykz, gxxkz, gyykz, gxykz


def calc_quadratic_kz_correction(k, kxc, kyc):
    """Calculate correction that needs to be applied to propagation with quadratic dispersion relation.

    The propagator that should be applied is -(fx*kx**2 + fy*ky**2)*z/(2*k) (no zeroth order term)

    Args:
        kxc: central x wavenumber
        kyc: central y wavenumber

    Returns:
        fx: factor by which propagation distance should be multiplied to give quadratic x propagation distance
        fy: same for y
        delta_kz: Zero order correction factor. The corresponding additional phase is delta_kz*z.
        delta_gxkz: first order x correction factor i.e. result should have phase of delta_gxkz*kx*z applied. Negative of
            the corresponding translation.
        delta_gykz: same for y.
    """
    # Get zeroth, first and second order (nonmixed) derivatives of k at (kxc, kyc).
    kz, gxkz, gykz, gxxkz, gyykz, gxykz = expand_kz(k, kxc, kyc)

    # Calculate first and second order derivatives of the quadratic approximation at (kxc, kyc).
    gxkz_p = -kxc/k
    gykz_p = -kyc/k
    gxxkz_p = -1/k
    gyykz_p = -1/k

    # Correction factors are ratios of correct second derivatives to the quadratic approximation ones.
    fx = gxxkz/gxxkz_p
    fy = gyykz/gyykz_p

    # Shift corrections are difference between correct first order terms and quadratic approximation ones.
    delta_gxkz = gxkz - fx*gxkz_p
    delta_gykz = gykz - fy*gykz_p

    # Evaluate zeroth order kz for x and y quadratic approximation.
    kz_px = -kxc**2/(2*k)
    kz_py = -kyc**2/(2*k)

    # Zeroth order correction is what is required to match k_z at (kxc, kyc) after quadratic approximation propagation
    # (with factors fx and fy) and shift correction has been applied.
    delta_kz = kz - fx*kz_px - fy*kz_py - delta_gxkz*kxc - delta_gykz*kyc

    return fx, fy, delta_kz, delta_gxkz, delta_gykz


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def calc_kz_exact(k, kx, ky):
    return (k**2 - kx**2 - ky**2)**0.5


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def calc_kz_paraxial(k, kx, ky):
    return k - (kx**2 + ky**2)/(2*k)


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def calc_kz_quadratic(k, kx, ky):
    return -(kx**2 + ky**2)/(2*k)


@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def calc_kz_quadratic_1d(k, q):
    """kz not included."""
    return -q**2/(2*k)


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64, numba.float64)])
def calc_propagator_exact(k, kx, ky, l):
    return mathx.expj(calc_kz_exact(k, kx, ky)*l)


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64, numba.float64)])
def calc_propagator_paraxial(k, kx, ky, l):
    return mathx.expj(calc_kz_paraxial(k, kx, ky)*l)


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64, numba.float64)])
def calc_propagator_quadratic(k, kx, ky, l):
    return mathx.expj(calc_kz_quadratic(k, kx, ky)*l)


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64)], nopython=True)
def calc_propagator_quadratic_1d(k, q, l):
    """kz not included."""
    return mathx.expj(calc_kz_quadratic_1d(k, q)*l)


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64)])
def calc_quadratic_phase_1d(k, r, roc):
    return mathx.expj(k*r**2/(2*roc))


# def prepare_curved_propagation_1d_tilted_shifted(k, r_support, Er, z, m, r_center = 0, q_center = 0, axis = -1):
#     num_points = Er.shape[axis]
#     r = calc_r(r_support, num_points, r_center)
#     Er *= np.exp(-1j*q_center*r)
#     propagator, post_factor = prepare_curved_propagation_1d(k, r_support, Er, z, m, axis)
#     r_shift = q_center/k*z
#     rp_center = r_center+r_shift
#     rp = calc_r(r_support*m, num_points, rp_center)
#     post_factor *= np.exp(1j*q_center*rp-1j*q_center**2*z/(2*k))
#     return propagator, post_factor, rp_center

def calc_flat_plane(k, r_support, num_points, roc):
    r_typical = r_support/(np.pi*num_points)**0.5
    if np.isfinite(roc):
        num_rayleighs_flat = k*r_typical**2/(2*roc)
        m_flat = (1 + num_rayleighs_flat**2)**0.5
    else:
        m_flat = 1
        num_rayleighs_flat = 0
    r_typical_flat = r_typical/m_flat
    z_R = r_typical_flat**2*k/2
    z_flat = num_rayleighs_flat*z_R
    return m_flat, z_flat, z_R


def calc_curved_propagation(k, r_support, num_points, roc, z):
    m_flat, z_flat, z_R = calc_flat_plane(k, r_support, num_points, roc)
    num_rayleighs = (z + z_flat)/z_R
    m = (1 + num_rayleighs**2)**0.5/m_flat
    return m_flat, z_flat, m


def calc_curved_propagation_m(k, r_support, num_points, roc, z):
    return calc_curved_propagation(k, r_support, num_points, roc, z)[2]

def calc_kz(k, kx, ky, kz_mode:str='exact'):
    if kz_mode == 'paraxial':
        kz = k
    elif kz_mode in ('local_xy', 'local', 'exact'):
        kz = (k**2 - kx**2 - ky**2)**0.5
    else:
        raise ValueError('Unknown kz_mode %s.', kz_mode)
    return kz

def adjust_r(k, ri, z, qs, kz_mode='local_xy'):
    qs = sa.to_scalar_pair(qs)
    kz = calc_kz(k, *qs, kz_mode)
    ro = ri + qs/kz*z
    return ro


# def refract_field(normal, k1, Er, gradEr, k2):
#     I = mathx.abs_sqd(Er)
#     Ik = [(component*Er.conj()).imag for component in gradEr]
#     # Our k-vector is gradient of the eikonal (with scalar k included). In the short wavelength approximation (geometrical optics)
#     # the length of the eikonal is k. (See  Born & Wolf eq. 15b in sec. 3.1.)
#     Ik[2] = np.maximum((I*k1)**2-mathx.dot(Ik[:2]), 0)**0.5
#     Ik_tangent = mathx.project(Ik, normal)
#     Ik_normal = np.maximum((I*k2)**2-mathx.dot(Ik_tangent), 0)**0.5
#     Ik2 = [tc+nc*Ik_normal for tc, nc in zip(Ik_tangent, normal)]
#     return Ik2, I


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
    q_support = 2*np.pi*num_points/r_support

    # Calculate real-space variance at z. From this infer a lower limit on the magnification.
    var_rz, phi_cz, _ = bvar.calc_propagated_variance_1d(k, var_r0, phi_c, var_q, z)
    m_lower = var_rz**0.5*12/r_support

    # Calculate minimum angular variance at z. This will not change with propagation. From this infer an upper
    # limit on the magnification.
    var_q_min = bvar.calc_minimum_angular_variance_1d(var_rz, phi_cz, var_q)
    m_upper = q_support/(var_q_min**0.5*12)

    if np.any(m_lower > m_upper):
        logger.warning('Magnification lower bound greater than upper bound.')

    m = m_lower

    return m


def calc_propagation_ms(k, rs_support, var_r0s, phi_cs, var_qs, zs, num_pointss, f_nexts=(np.inf, np.inf), qfds=(1, 1)):
    rs_support = sa.to_scalar_pair(rs_support)
    var_r0s = sa.to_scalar_pair(var_r0s)
    phi_cs = sa.to_scalar_pair(phi_cs)
    var_qs = sa.to_scalar_pair(var_qs)
    zs = sa.to_scalar_pair(zs)
    num_pointss = sa.to_scalar_pair(num_pointss)
    f_nexts = sa.to_scalar_pair(f_nexts)
    qfds = sa.to_scalar_pair(qfds)
    ms = []
    for r_support, var_r0, phi_c, var_q, z, num_points, f_next, qfd in zip(rs_support, var_r0s, phi_cs, var_qs, zs,
            num_pointss, f_nexts, qfds):
        ms.append(calc_propagation_m_1d(k, r_support, var_r0, phi_c, var_q, z, num_points, f_next, qfd))
    return np.asarray(ms)


def calc_spherical_post_factor(k, rs_support, num_pointss, z, ms, rs_center=(0, 0), qs_center=(0, 0), ro_centers=None,
                               kz_mode='local_xy'):
    assert np.isscalar(z)
    ro_supports = rs_support*ms
    qs_center = sa.to_scalar_pair(qs_center)
    if kz_mode == 'paraxial':
        zx = z
        zy = z
        kz_center = k
        delta_kz = k
    elif kz_mode == 'local_xy':
        fx, fy, delta_kz, delta_gxkz, delta_gykz = calc_quadratic_kz_correction(k, *qs_center)
        zx = fx*z
        zy = fy*z
        kz_center = (k**2 - (qs_center**2).sum())**0.5
    else:
        raise ValueError('Unknown kz_   mode %s.', kz_mode)
    if ro_centers is None:
        ro_centers = rs_center + qs_center/kz_center*z
    xo, yo = sa.calc_xy(ro_supports, num_pointss, ro_centers)
    if kz_mode == 'local_xy':
        xo += delta_gxkz*z
        yo += delta_gykz*z
    roc_x = zx/(ms[0] - 1)
    roc_y = zy/(ms[1] - 1)
    roc_xo = roc_x + zx
    roc_yo = roc_y + zy
    # See derivation page 114 Dane's logbook 2.
    Qo = calc_quadratic_phase_1d(k, xo, roc_xo)*calc_quadratic_phase_1d(k, yo, roc_yo)*mathx.expj(delta_kz*z + k*(
                rs_center[0]**2/(2*roc_x) + rs_center[1]**2/(2*roc_y) - rs_center[0]*xo/(roc_x*ms[0]) - rs_center[
            1]*yo/(roc_y*ms[1])))/(ms[0]*ms[1])**0.5
    return Qo


def calc_plane_to_curved_flat_factors(k, rs_support, num_pointss, z, qs_center=(0, 0), kz_mode='local_xy'):
    """Calculate factors for propagation from plane to uniformly sampled curved surface.

    Args:
        k:
        rs_support:
        num_pointss:
        z:
        qs_center:
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        invTx (2D array):
        gradxinvTx (2D array):
        invTy (2D array):
        gradyinvTy (2D array):
        Px (3D array):
        Py (3D array):
        Tx (2D array):
        Ty (2D array):
    """
    assert kz_mode in ('local_xy', 'paraxial')
    Tx = make_fft_matrix(num_pointss[0])
    if num_pointss[1] == num_pointss[0]:
        Ty = Tx
    else:
        Ty = make_fft_matrix(num_pointss[1])
    if kz_mode == 'local_xy':
        fx, fy, delta_kz, delta_gxkz, delta_gykz = calc_quadratic_kz_correction(k, *qs_center)
        zx = z*fx
        zy = z*fy
    else:
        zx = z
        zy = z
        delta_kz = k
    kx, ky = sa.calc_kxky(rs_support, num_pointss, qs_center)
    # kx & ky are the right-most indices of the propagators.
    Px = calc_propagator_quadratic_1d(k, kx[:, 0], zx[:, :, None])*mathx.expj(delta_kz*z[:, :, None])
    Py = calc_propagator_quadratic_1d(k, ky, zy[:, :, None])
    if kz_mode == 'local_xy':
        Px *= mathx.expj(delta_gxkz*kx[:, 0]*z[:, :, None])
        Py *= mathx.expj(delta_gykz*ky*z[:, :, None])
    invTx = Tx.conj()
    invTy = Ty.conj()
    gradxinvTx = 1j*kx[:, 0]*invTx
    gradyinvTy = 1j*ky*invTy
    return invTx, gradxinvTx, invTy, gradyinvTy, Px, Py, Tx, Ty


def calc_plane_to_curved_flat_arbitrary_factors(k, rs_support, num_pointss, z, xo, yo, qs_center=(0, 0), kz_mode='local_xy'):
    """Calculate factors for propagation from plane to arbitrarily sampled curved surface.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair of): Input aperture along x and y.
        num_pointss (int or tuple of): Number of input samples along x and y. Referred to as K and L below.
        z (M*N array): Propagation distances.
        xo (Mx1 array): Output x values.
        yo (M array): Output y values.
        qs_center (tuple or pair of): Center of transverse wavenumber support.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        invTx (M*K array): Inverse transform factor.
        gradxinvTx (M*K array): X-derivative of inverse transform factor.
        invTy (L*N array): Y inverse transform factor.
        gradyinvTy (L*N array): Y-derivative of inverse transform factor.
        Px (M*N*K array): X propagation.
        Py (M*N*L array): Y propagation.
        Tx (K*K array): X transform factor.
        Ty (L*L array): Y transform factor.
    """
    assert kz_mode in ('local_xy', 'paraxial')
    Tx = make_fft_matrix(num_pointss[0])
    if num_pointss[1] == num_pointss[0]:
        Ty = Tx
    else:
        Ty = make_fft_matrix(num_pointss[1])
    if kz_mode == 'local_xy':
        fx, fy, delta_kz, delta_gxkz, delta_gykz = calc_quadratic_kz_correction(k, *qs_center)
        zx = z*fx
        zy = z*fy
    else:
        zx = z
        zy = z
        delta_kz = k
    kx, ky = sa.calc_kxky(rs_support, num_pointss, qs_center)
    # kx & ky are the right-most indices of the propagators.
    Px = calc_propagator_quadratic_1d(k, kx[:, 0], zx[:, :, None])*mathx.expj(delta_kz*z[:, :, None])
    Py = calc_propagator_quadratic_1d(k, ky, zy[:, :, None])
    if kz_mode == 'local_xy':
        Px *= mathx.expj(delta_gxkz*kx[:, 0]*z[:, :, None])
        Py *= mathx.expj(delta_gykz*ky*z[:, :, None])
    invTx = make_ifft_arbitrary_matrix(rs_support[0], num_pointss[0], qs_center[0], xo[:, 0])
    invTy = make_ifft_arbitrary_matrix(rs_support[1], num_pointss[1], qs_center[1], yo)
    gradxinvTx = 1j*kx[:, 0]*invTx
    gradyinvTy = 1j*ky*invTy
    return invTx, gradxinvTx, invTy, gradyinvTy, Px, Py, Tx, Ty

class Propagator:
    def __init__(self, x_factor, y_factor):
        assert x_factor.shape[:2] == y_factor.shape[:2]
        self.x_factor = x_factor
        self.y_factor = y_factor
        # Profile indicated that repeated conjugation has a small cost.
        self.x_factor_conj = x_factor.conj()
        self.y_factor_conj = y_factor.conj()
        self.input_shape = (x_factor.shape[2], y_factor.shape[2])
        self.output_shape = x_factor.shape[:2]
        self.matrix_shape = np.prod(self.output_shape), np.prod(self.input_shape)

    def apply(self, Eri):
        Ero = opt_einsum.contract('ijm, ijn, mn->ij', self.x_factor, self.y_factor, Eri)
        return Ero

    def apply_vector(self, Eriv):
        Eri = Eriv.reshape(self.input_shape)
        Ero = self.apply(Eri)
        Erov = Ero.ravel()
        return Erov


    def apply_transpose_vector(self, Erov):
        Ero = Erov.reshape(self.output_shape)
        Eri = self.apply_transpose(Ero)
        Eriv = Eri.ravel()
        return Eriv


    def apply_transpose(self, Ero):
        Eri = opt_einsum.contract('ijm, ijn, ij->mn', self.x_factor_conj, self.y_factor_conj, Ero)
        return Eri


    def invert(self, Ero, max_iterations=None, tol=None, method='lsqr'):
        """Calculate input field given output field and propagator.

        Args:
            Ero (2D array): The output field.
            max_iterations (int): Maximum number of iterations for algorithm.
            tol (scalar): Convergence tolerance. Default value (None) depends on the algorithm.
            method (str): One of 'bicgstab', 'lgmres', 'lsqr'. The first two are mainly for historical reasons; they do OK
                but become unstable as the surface gets more tilted. lsqr seems to work well in all cases.

        Returns:
            2D array: The input field.

        * Notes on algorithm choice *
        Tried CG first - it doesn't work, which is not surprising since it wants symmetric matrices. Spent a few weeks with
        bicgstab. It worked quite well for the low f number collimation problem but struggled as the source went off-axis.
        Lgmres was slower, but converged to lower residual. But this gave a noisier result!
        """
        assert method in ('bicgstab', 'lgmres', 'lsqr')
        if tol is None:
            tol = {'bicgstab':1e-8, 'lgmres':1e-8, 'lsqr':1e-6}[method]
        if max_iterations is None:
            max_iterations = 10
        function = scipy.sparse.linalg.__dict__[method]
        if method in ('bicgstab', 'lgmres'):
            A = scipy.sparse.linalg.LinearOperator(self.matrix_shape, matvec=self.apply_vector)
            M = scipy.sparse.linalg.LinearOperator(self.matrix_shape, matvec=self.apply_reverse_vector)
            x0 = self.apply_reverse(Ero).reshape(Ero.size)
            Eriv, info = function(A, Ero.ravel(), x0, tol, max_iterations, M=M)
            if info != 0:
                logger.warning('Numerical solver did not converge after %d iterations.', info)
        else:
            A = scipy.sparse.linalg.LinearOperator(self.matrix_shape, matvec=self.apply_vector,
                rmatvec=self.apply_transpose_vector)
            Eriv, istop, itn, r1norm = scipy.sparse.linalg.lsqr(A, Ero.ravel(), atol=tol, btol=tol,
                iter_lim=max_iterations)[:4]
            if istop != 1:
                logger.log(30, 'Found least squares solution after %d iterations, but not within tolerance. Residual norm %g.', itn,
                    r1norm)
            logger.log(13, 'lsqr number of iterations was %d. Residual norm %g.', itn, r1norm)
        Eri = Eriv.reshape(self.input_shape)
        Erop = self.apply(Eri)
        rel_rms_res = (mathx.sum_abs_sqd(Ero - Erop)/mathx.sum_abs_sqd(Ero))**0.5
        logger.log(16, 'RMS residual/norm = %g.'%rel_rms_res)
        return Eri

# class PlaneToCurvedFlatParaxialPropagator(Propagator):
#     def __init__(self, Fx, Fy):
#         self.Fx = Fx
#         self.Fy = Fy
#         self.Fx_conj = Fx.conj()
#         self.Fy_conj = Fy.conj()
#         self.shape = Fx.shape[:2]
#
#     def apply(self, Eri):
#         Ero = opt_einsum.contract('ijm, ijn, mn->ij', self.Fx, self.Fy, Eri)
#         return Ero
#
#     def apply_vector(self, Eriv):
#         Eri = Eriv.reshape(self.shape)
#         Ero = self.apply(Eri)
#         Erov = Ero.ravel()
#         return Erov
#
#     def apply_reverse(self, Ero):
#         Eri = self.apply_transpose(Ero)
#         # Eri = opt_einsum.contract('ijm, ijn, ij->mn', self.Fx.conj(), self.Fy.conj(), Ero)
#         return Eri
#
#     def apply_reverse_vector(self, Erov):
#         Ero = Erov.reshape(self.shape)
#         Eri = self.apply_reverse(Ero)
#         Eriv = Eri.ravel()
#         return Eriv
#
#     def apply_transpose_vector(self, Erov):
#         Ero = Erov.reshape(self.shape)
#         Eri = self.apply_transpose(Ero)
#         Eriv = Eri.ravel()
#         return Eriv
#
#     def apply_transpose(self, Ero):
#         Eri = opt_einsum.contract('ijm, ijn, ij->mn', self.Fx_conj, self.Fy_conj, Ero)
#         return Eri


def prepare_plane_to_curved_flat(k, rs_support, num_pointss, z, qs_center=(0, 0), kz_mode='local_xy'):
    """Prepare propagator from uniformly sampled plane to uniformly sampled curved surface.

    The zero-th order component (k_z) is included (arbitrarily) in Px.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair of): Input aperture along x and y.
        num_pointss (int or tuple of): Number of input samples along x and y. Referred to as M and N below.
        z (M*N array): Propagation distances.
        qs_center (tuple or pair of): Center of transverse wavenumber support.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        PlaneToCurvedFlatParaxialPropagator object.
    """
    invTx, _, invTy, _, Px, Py, Tx, Ty = calc_plane_to_curved_flat_factors(k, rs_support, num_pointss, z, qs_center,
        kz_mode)
    # xo=i, yo=j, kx=k, ky=l, xi=m, yi=n.
    Fx = opt_einsum.contract('ik, ijk, km->ijm', invTx, Px, Tx)
    Fy = opt_einsum.contract('jl, ijl, ln->ijn', invTy, Py, Ty)
    return Propagator(Fx, Fy)


def prepare_plane_to_curved_flat_arbitrary(k, rs_support, num_pointss, z, xo, yo, qs_center=(0, 0), kz_mode='local_xy'):
    """Prepare propagator from uniformly sampled plane to arbitrarily sampled curved surface.
    The zero-th order component (k_z) is included (arbitrarily) in Px.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair of): Input aperture along x and y.
        num_pointss (int or tuple of): Number of input samples along x and y. Referred to as K and L below.
        z (M*N array): Propagation distances.
        xo (Mx1 array): Output x values.
        yo (M array): Output y values.
        qs_center (tuple or pair of): Center of transverse wavenumber support.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        PlaneToCurvedFlatParaxialPropagator object.
    """
    invTx, _, invTy, _, Px, Py, Tx, Ty = calc_plane_to_curved_flat_arbitrary_factors(k, rs_support, num_pointss, z, xo, yo,
        qs_center, kz_mode)
    # xo=i, yo=j, kx=k, ky=l, xi=m, yi=n.
    Fx = opt_einsum.contract('ik, ijk, km->ijm', invTx, Px, Tx)
    Fy = opt_einsum.contract('jl, ijl, ln->ijn', invTy, Py, Ty)
    return Propagator(Fx, Fy)


def calc_plane_to_curved_spherical_factors(k, rs_support, num_pointss, z, ms, ri_centers=(0, 0), qs_center=(0, 0),
                                           ro_centers=(0, 0), kz_mode='local_xy'):
    """Calculate factors for 2D Sziklas-Siegman propagation (paraxial) from flat to curved surface(s).

    The magnifications must be the same for all output points, but can be different for the x and y planes.

    The zero-oth order phase (k_z) is (arbitrarily) included in QinvTPx.

    Compared to regular Sziklas-Siegman procedure, the factors are complicated by a transformation that allows nonzero
    real and angular space centers.  See derivation page 114 Dane's logbook 2.

    Possible optimizations: could probably shave 50% off by more aggressive use of numba to avoid calculation of intermediates.
    But typically the time is spent using (tensor contract) rather than calculating the factors, so it won't bring
    a great speedup.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        num_pointss (scalar int or pair): Number of sampling points.
        z (array of size num_pointss): Propagation for distance.
        ms (scalar or pair): Magnification(s).
        ri_centers (pair of scalars): Center of initial real-space aperture.
        qs_center (pair of scalars): Center of angular space aperture.
        ro_centers (pair of scalars): Center of final real-space aperture. If not given, the shift from ri_centers is
            inferred from the propagation distance and qs_center.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        QinvTPx (3D array): Product propagator, inverse DFT and quadratic output factor along x. Axes are (xo, yo, kx).
        gradxQinvTPx (3D array): Derivative of above w.r.t output x. Note that z is constant - the derivative is along
            a transverse plane rather than the surface.
        QinvTPy (3D array): Product propagator, inverse DFT and quadratic output factor along x. Axes are (xo, yo, kx).
        gradyQinvTPy (3D array): Derivative of above w.r.t output x. As with gradxQinvTPx, z is constant.
        Tx (2D array): DFT factor. Axes are (kx, xi).
        Ty (2D array): DFT factor. Axes are (ky, yi).
        Qix (3D array): Input x quadratic phase factor vs x. Axes are (xo, yo, xi).
        Qiy (3D array): Input y quadratic phase factor. Axes are (xo, yo, yi).
    """
    assert kz_mode in ('local_xy', 'paraxial')
    num_pointss = sa.to_scalar_pair(num_pointss)
    ms = sa.to_scalar_pair(ms)
    if np.isclose(ms, 1).any():
        logger.debug('At least one magnification (%g, %g) close to 1.', ms)
    qs_center = sa.to_scalar_pair(qs_center)
    # TODO (maybe) factor out adjust_r.
    if kz_mode == 'paraxial':
        kz_center = k
    else:
        kz_center = (k**2 - (qs_center**2).sum())**0.5
    z_center = np.mean(z)
    if ro_centers is None:
        ro_centers = ri_centers + qs_center/kz_center*z_center

    # If local_xy mode is requested then propagation distances must be scaled. Calculate scaled propagation distances
    # and required kz component.
    if kz_mode == 'paraxial':
        zx = z
        zy = z
        delta_kz = k
    else:
        fx, fy, delta_kz, delta_gxkz, delta_gykz = calc_quadratic_kz_correction(k, *qs_center)
        zx = fx*z
        zy = fy*z

    ro_supports = rs_support*ms
    xi, yi = sa.calc_xy(rs_support, num_pointss, ri_centers)
    kxi, kyi = sa.calc_kxky(rs_support, num_pointss, qs_center)  # +q_curvatures)
    roc_x = mathx.divide0(zx, ms[0] - 1, np.inf)
    roc_y = mathx.divide0(zy, ms[1] - 1, np.inf)
    # Quadratic phase is centered at origin. Numerically, this requires evaluation of complex exponentials of size N^3.
    # Significant fraction of total cost of this function, but unavoidable.
    Qix = calc_quadratic_phase_1d(-k, xi[:, 0] - ri_centers[0], roc_x[:, :, None])
    Qiy = calc_quadratic_phase_1d(-k, yi - ri_centers[1], roc_y[:, :, None])

    Tx = make_fft_matrix(num_pointss[0])
    if num_pointss[1] == num_pointss[0]:
        Ty = Tx
    else:
        Ty = make_fft_matrix(num_pointss[1])

    if 0:
        # This was an experiment - we calculate where a given pencil beam lands based on initial position and
        # momentum. It didn't improve the stability of the algorithm. In a ray picture, clip factor should be 0.5. But
        # I found that anything below 3 gave unacceptable artefacts.
        clip_factor = 3
        mean_rocs = z_centers/(ms - 1)
        xo_kx_xi = xi[:, 0]*ms[0] + (kxi - k*ri_centers[0]/mean_rocs[0])*z_centers[0]/k
        yo_ky_yi = yi*ms[1] + (kyi[:, None] - k*ri_centers[1]/mean_rocs[1])*z_centers[1]/k
        in_xo = abs(xo_kx_xi - ro_centers[0]) < ro_supports[0]*clip_factor
        in_yo = abs(yo_ky_yi - ro_centers[1]) < ro_supports[1]*clip_factor
        Txp = Tx*in_xo
        Typ = Ty*in_yo

    xo, yo = sa.calc_xy(ro_supports, num_pointss, ro_centers)
    roc_xo = roc_x + zx
    roc_yo = roc_y + zy

    # If in local_xy mode, then the factors which depend on xo and yo use transformed quantities, which we subsitute here.
    if kz_mode == 'local_xy':
        xo = xo + delta_gxkz*z
        yo = yo + delta_gykz*z

    # Effective kx and ky for propagation takes into account center of curvature.
    kxip = kxi[:, 0] - k*ri_centers[0]/roc_x[:, :, None]
    kyip = kyi - k*ri_centers[1]/roc_y[:, :, None]

    # Calculate product of propagator and inverse transform along x axis. The x axis (arbitrarily) includes delta_kz.
    # Could combine all exponents before exponentiation? Won't help much because the time will be dominated by the propagator
    # which is N^3 - the others are N^2.
    phi = delta_kz*z + k*(ri_centers[0]**2/(2*roc_x) - ri_centers[0]*xo/(roc_x*ms[0]))
    QinvTPx = (calc_propagator_quadratic_1d(k*ms[0], kxip, zx[:, :, None])*  # Propagation phase scaled by magnification.
               calc_quadratic_phase_1d(k, xo, roc_xo)[:, :, None]*  # Normal Sziklas-Siegman final quadratic phase.
               mathx.expj(phi[:, :, None])*
               Tx.conj()[:, None, :]/  # Inverse DFT.
               abs(ms[0])**0.5)  # Magnification correction to amplitude.

    # Calculate product of propagator and inverse transform along x axis. Result depends on
    phi = k*(ri_centers[1]**2/(2*roc_y) - ri_centers[1]*yo/(roc_y*ms[1]))
    QinvTPy = (calc_propagator_quadratic_1d(k*ms[1], kyip, zy[:, :, None])*  # Propagation phase scaled by magnification.
               calc_quadratic_phase_1d(k, yo, roc_yo)[:, :, None]*  # Normal Sziklas-Siegman final quadratic phase.
               mathx.expj(phi[:, :, None])*
               Ty.conj()/  # Inverse DFT.
               abs(ms[1])**0.5)  # Magnification correction to amplitude.

    # If local_xy mode, need translation correction factor. Could combine this with above to reduce number of N^3 expj
    # calls.
    if kz_mode == 'local_xy':
        QinvTPx *= mathx.expj(delta_gxkz*kxi[:, 0]/ms[0]*z[:, :, None])
        QinvTPy *= mathx.expj(delta_gykz*kyi/ms[1]*z[:, :, None])

    # Evaluate derivatives of the invTP factors with respect to output x and y (but constant z - not along the curved
    # surface).
    gradxQinvTPx = 1j*(k*(xo/roc_xo)[:, :, None] - ri_centers[0]*k/(roc_x[:, :, None]*ms[0]) + kxi[:, 0]/ms[0])*QinvTPx
    gradyQinvTPy = 1j*(k*(yo/roc_yo)[:, :, None] - ri_centers[1]*k/(roc_y[:, :, None]*ms[1]) + kyi/ms[1])*QinvTPy

    # Won't use z gradients for now - will keep for future.
    # gradzPx=1j*calc_kz_paraxial_1d(k*ms[0], kxip)*Px

    factors = QinvTPx, gradxQinvTPx, QinvTPy, gradyQinvTPy, Tx, Ty, Qix, Qiy
    return factors

def calc_plane_to_curved_spherical_arbitrary_factors(k, rs_support, num_pointss, z, xo, yo, roc_x, roc_y, ri_centers=(0, 0),
                                                     qs_center=(0, 0), ro_centers=None, kz_mode='local_xy'):
    """Calculate factors for 2D Sziklas-Siegman propagation (paraxial) from flat to curved surface(s).

    The zero-oth order phase (k_z) is (arbitrarily) included in QinvTPx.

    Compared to regular Sziklas-Siegman procedure, the factors are complicated by a transformation that allows nonzero
    real and angular space centers.  See derivation page 114 Dane's logbook 2.

    Possible optimizations: could probably shave 50% off by more aggressive use of numba to avoid calculation of intermediates.
    But typically the time is spent using (tensor contract) rather than calculating the factors, so it won't bring
    a great speedup.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        num_pointss (scalar int or pair): Number of sampling points in input.
        z (M*N array): Propagation for distance.
        xo (M*1 array): Output x sample values.
        yo (N array): Output y sample values.
        roc_x (M*N array): Radius of curvature in x, at the input, but sampled at the output points.
        roc_y (M*N array): Radius of curvature in y, at the input, but sampled at the output points.
        ri_centers (pair of scalars): Center of initial real-space aperture.
        qs_center (pair of scalars): Center of angular space aperture.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        QinvTPx (3D array): Product propagator, inverse DFT and quadratic output factor along x. Axes are (xo, yo, kx).
        gradxQinvTPx (3D array): Derivative of above w.r.t output x. Note that z is constant - the derivative is along
            a transverse plane rather than the surface.
        QinvTPy (3D array): Product propagator, inverse DFT and quadratic output factor along x. Axes are (xo, yo, kx).
        gradyQinvTPy (3D array): Derivative of above w.r.t output x. As with gradxQinvTPx, z is constant.
        Tx (2D array): DFT factor. Axes are (kx, xi).
        Ty (2D array): DFT factor. Axes are (ky, yi).
        Qix (3D array): Input x quadratic phase factor vs x. Axes are (xo, yo, xi).
        Qiy (3D array): Input y quadratic phase factor. Axes are (xo, yo, yi).
    """
    assert np.isscalar(k)
    rs_support = sa.to_scalar_pair(rs_support)
    assert kz_mode in ('local_xy', 'paraxial')
    assert np.ndim(z) == 2
    assert np.shape(z) == np.shape(roc_x)
    assert np.shape(z) == np.shape(roc_y)
    assert np.shape(xo) == (np.shape(z)[0], 1)
    assert np.shape(yo) == (np.shape(z)[1], )
    num_pointss = sa.to_scalar_pair(num_pointss)
    qs_center = sa.to_scalar_pair(qs_center)

    if ro_centers is None:
        z_center = np.mean(z)
        ro_centers = adjust_r(k, ri_centers, z_center, qs_center, kz_mode)

    # If local_xy mode is requested then propagation distances must be scaled. Calculate scaled propagation distances
    # and required kz component.
    if kz_mode == 'paraxial':
        zx = z
        zy = z
        delta_kz = k
    else:
        fx, fy, delta_kz, delta_gxkz, delta_gykz = calc_quadratic_kz_correction(k, *qs_center)
        zx = fx*z
        zy = fy*z

    mx = zx/roc_x + 1
    my = zy/roc_y + 1

    xi, yi = sa.calc_xy(rs_support, num_pointss, ri_centers)
    kxi, kyi = sa.calc_kxky(rs_support, num_pointss, qs_center)  # +q_curvatures)

    # Quadratic phase is centered at origin. Numerically, this requires evaluation of complex exponentials of size N^3.
    # Significant fraction of total cost of this function, but unavoidable.
    Qix = calc_quadratic_phase_1d(-k, xi[:, 0] - ri_centers[0], roc_x[:, :, None])
    Qiy = calc_quadratic_phase_1d(-k, yi - ri_centers[1], roc_y[:, :, None])

    Tx = make_fft_matrix(num_pointss[0])
    if num_pointss[1] == num_pointss[0]:
        Ty = Tx
    else:
        Ty = make_fft_matrix(num_pointss[1])

    roc_xo = roc_x + zx
    roc_yo = roc_y + zy

    # If in local_xy mode, then the result of the Sziklas-Siegman step needs to be translated. This means that the
    # factors which depend on xo and yo need to be translated too. (Note that the x and y factors in the inverse DFT
    # are not translated.) We define xop and yop as the translated output coordinates.
    if kz_mode == 'local_xy':
        xop = xo + delta_gxkz*z
        yop = yo + delta_gykz*z
    else:
        xop = xo
        yop = yo[None, :]

    # Effective kx and ky for propagation takes into account center of curvature.
    kxip = kxi[:, 0] - k*ri_centers[0]/roc_x[:, :, None]
    kyip = kyi - k*ri_centers[1]/roc_y[:, :, None]

    r2_support_x = mx*rs_support[0]
    r2_support_y = my*rs_support[1]
    valid_x = abs(xo - ro_centers[0]) < r2_support_x*0.5
    valid_y = abs(yo - ro_centers[1]) < r2_support_y*0.5

    # Calculate product of propagator and inverse transform along x axis. The x axis (arbitrarily) includes delta_kz.
    # Could combine all exponents before exponentiation? Won't help much because the time will be dominated by the propagator
    # which is N^3 - the others are N^2.
    phi = delta_kz*z + k*(ri_centers[0]**2/(2*roc_x) - ri_centers[0]*xop/(roc_x*mx))
    QinvTPx = (calc_propagator_quadratic_1d(k*mx[:, :, None], kxip, zx[:, :, None])*  # Propagation phase scaled by magnification.
               calc_quadratic_phase_1d(k, xop, roc_xo)[:, :, None]*  # Normal Sziklas-Siegman final quadratic phase.
               mathx.expj(phi[:, :, None] + kxi[:, 0]*(xo/mx)[:, :, None])/num_pointss[0]**0.5/ # Correction phases and inverse DFT.
               abs(mx[:, :, None])**0.5*  # Magnification correction to amplitude.
               valid_x[:, :, None])

    # Calculate product of propagator and inverse transform along x axis.
    phi = k*(ri_centers[1]**2/(2*roc_y) - ri_centers[1]*yop/(roc_y*my))
    QinvTPy = (calc_propagator_quadratic_1d(k*my[:, :, None], kyip, zy[:, :, None])*  # Propagation phase scaled by magnification.
               calc_quadratic_phase_1d(k, yop, roc_yo)[:, :, None]*  # Normal Sziklas-Siegman final quadratic phase.
               mathx.expj(phi[:, :, None] + kyi*(yo/my)[:, :, None])/num_pointss[1]**0.5/ # Correction phases and inverse DFT.
               abs(my[:, :, None])**0.5*  # Magnification correction to amplitude.
               valid_y[:, :, None])

    # If local_xy mode, need translation correction factor. Could combine this with above to reduce number of N^3 expj
    # calls but this would only give a marginal performance improvement.
    if kz_mode == 'local_xy':
        QinvTPx *= mathx.expj(delta_gxkz*kxi[:, 0]*(z/mx)[:, :, None])
        QinvTPy *= mathx.expj(delta_gykz*kyi*(z/my)[:, :, None])

    # Evaluate derivatives of the invTP factors with respect to output x and y (but constant z - not along the curved
    # surface).
    gradxQinvTPx = 1j*(k*(xop/roc_xo - ri_centers[0]/(roc_x*mx))[:, :, None] + kxi[:, 0]/mx[:, :, None])*QinvTPx
    gradyQinvTPy = 1j*(k*(yop/roc_yo - ri_centers[1]/(roc_y*my))[:, :, None] + kyi/my[:, :, None])*QinvTPy

    factors = QinvTPx, gradxQinvTPx, QinvTPy, gradyQinvTPy, Tx, Ty, Qix, Qiy
    return factors


class MagnifyingPropagator(Propagator):
    def __init__(self, x_factor, y_factor, ms):
        Propagator.__init__(self, x_factor, y_factor)
        self.ms = ms

    def apply_reverse(self, Ero):
        Eri = self.apply_transpose(Ero)*np.prod(self.ms)
        return Eri

    def apply_reverse_vector(self, Erov):
        Ero = Erov.reshape(self.output_shape)
        Eri = self.apply_reverse(Ero)
        Eriv = Eri.ravel()
        return Eriv


def prepare_plane_to_curved_spherical(k, rs_support, num_pointss, z, ms, rs_center=(0, 0), qs_center=(0, 0),
                                      ro_centers=(0, 0), kz_mode='local_xy'):
    """Prepare spherical wavefront propagator from uniformly sampled plane to curved surface.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        num_pointss (scalar int or pair): Number of sampling points.
        z (array of size num_pointss): Propagation for distance.
        ms (scalar or pair): Magnification(s).
        rs_center (pair of scalars): Center of initial real-space aperture.
        qs_center (pair of scalars): Center of angular space aperture.
        ro_centers (pair of scalars): Center of final real-space aperture. If not given, the shift from ri_centers is
            inferred from the propagation distance and qs_center.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        PlaneToCurvedSphericalParaxialPropagator object
    """
    #744 ms for num_pointss = 256, 256.
    ms = sa.to_scalar_pair(ms)
    invTPx, gradxphiinvTPx, invTPy, gradyphiinvTpy, Tx, Ty, Qix, Qiy = calc_plane_to_curved_spherical_factors(k,
        rs_support, num_pointss, z, ms, rs_center, qs_center, ro_centers, kz_mode)
    x_factor = opt_einsum.contract('ijk, km, ijm -> ijm', invTPx, Tx, Qix)
    y_factor = opt_einsum.contract('ijl, ln, ijn -> ijn', invTPy, Ty, Qiy)
    return MagnifyingPropagator(x_factor, y_factor, ms)


def prepare_plane_to_curved_spherical_arbitrary(k, rs_support, num_pointss, z, xo, yo, roc_xo, roc_yo, rs_center=(0, 0),
                                                qs_center=(0, 0), ro_centers=None, kz_mode='local_xy'):
    """Prepare spherical wavefront propagator from uniformly sampled plane to arbitrarily sampled curved surface.

    744 ms for num_pointss = 256, 256.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        num_pointss (scalar int or pair): Number of sampling points in input.
        z (M*N array): Propagation for distance.
        xo (M*1 array): Output x sample values.
        yo (N array): Output y sample values.
        roc_xo (M*N array): Radius of curvature along x at the input, sampled at the output points.
        roc_y (M*N array): Radius of curvature along y at the input, sampled at the output points.
        rs_center (pair of scalars): Center of initial real-space aperture.
        qs_center (pair of scalars): Center of angular space aperture.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        MagnifyingPropagator object.
    """
    invTPx, gradxphiinvTPx, invTPy, gradyphiinvTpy, Tx, Ty, Qix, Qiy = calc_plane_to_curved_spherical_arbitrary_factors(k,
        rs_support, num_pointss, z, xo, yo, roc_xo, roc_yo, rs_center, qs_center, ro_centers, kz_mode)
    x_factor = opt_einsum.contract('ijk, km, ijm -> ijm', invTPx, Tx, Qix)
    y_factor = opt_einsum.contract('ijl, ln, ijn -> ijn', invTPy, Ty, Qiy)
    return MagnifyingPropagator(x_factor, y_factor, (roc_xo, roc_yo))
