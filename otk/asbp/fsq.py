"""Functions of a sampled quantities."""
import logging
import numpy as np
import opt_einsum
import mathx
from mathx import matseq
from . import sa, math
from .. import bvar
logger = logging.getLogger(__name__)

def shift_r_center_1d(r_support, num_points, Er, delta_r_center, q_center, k_on_roc = 0, axis = -1):
    r = sa.calc_r(r_support, num_points, axis = axis)
    q = sa.calc_q(r_support, num_points, q_center, axis = axis)
    # Remove quadratic phase and tilt.
    Er *= mathx.expj(-(q[0]*r+k_on_roc*r**2/2))
    Ek = math.fft(Er, axis)
    Ek *= mathx.expj(delta_r_center*q)
    Er = math.ifft(Ek, axis)
    Er *= mathx.expj(q[0]*r+k_on_roc*(r+delta_r_center)**2/2)
    return Er


def prepare_curved_paraxial_propagation_1d(k, r_support, Er, z, m, r_center = 0, q_center = 0, axis = -1, carrier = True):
    """Modifies Er."""
    assert not (np.isclose(z, 0) ^ np.isclose(m, 1))
    if np.isclose(z, 0):
        return 1, 1, r_center
    num_points = Er.shape[axis]
    roc = z/(m-1)
    r = sa.calc_r(r_support, num_points, r_center, axis)
    q = sa.calc_q(r_support, num_points, q_center, axis)#-q_center
    Er *= math.calc_quadratic_phase_1d(-k, r - r_center, roc)
    propagator = math.calc_propagator_quadratic_1d(k*m, q - k*r_center/roc, z)

    ro_center = r_center + q_center/k*z
    ro = sa.calc_r(r_support*m, num_points, ro_center, axis)
    post_factor = math.calc_quadratic_phase_1d(k, ro, roc+z)*mathx.expj(k*(r_center**2/(2*roc) - r_center*ro/(roc*m)))/m**0.5
    if carrier:
        post_factor *= mathx.expj(k*z)

    return propagator, post_factor, ro_center


def calc_gradxyE(rs_support, Er, qs_center=(0, 0)):
    Ek = math.fft2(Er)
    kx, ky = sa.calc_kxky(rs_support, Ek.shape, qs_center)
    gradxE = math.ifft2(Ek*1j*kx)
    gradyE = math.ifft2(Ek*1j*ky)
    return gradxE, gradyE

def calc_gradxyE_spherical(k, rs_support, Er, rocs, rs_center=(0, 0), qs_center=(0, 0)):
    rs_support = sa.to_scalar_pair(rs_support)
    rocs = sa.to_scalar_pair(rocs)
    x, y = sa.calc_xy(rs_support, Er.shape, rs_center)
    xp = x - rs_center[0]
    yp = y - rs_center[1]
    Qx = math.calc_quadratic_phase_1d(k, xp, rocs[0])
    Qy = math.calc_quadratic_phase_1d(k, yp, rocs[1])
    Erp = Er*Qx.conj()*Qy.conj()
    Eqp = math.fft2(Erp)
    kx, ky = sa.calc_kxky(rs_support, Er.shape, qs_center)
    gradxE = math.ifft2(Eqp*1j*kx)*Qx*Qy + Er*1j*k*xp/rocs[0]
    gradyE = math.ifft2(Eqp*1j*ky)*Qx*Qy + Er*1j*k*yp/rocs[1]
    return gradxE, gradyE

def calc_Igradphi(k, Er, gradxyEr, Ir=None) -> list:
    if Ir is None:
        Ir = mathx.abs_sqd(Er)

    # Calculate phase gradient times intensity.
    Igradxyphi = [(component*Er.conj()).imag for component in gradxyEr]

    # The phase gradient is the eikonal (with scalar k included). In the short wavelength approximation (geometrical optics)
    # the length of the eikonal is k. (See  Born & Wolf eq. 15b in sec. 3.1.)
    Igradzphi = np.maximum((Ir*k)**2 - matseq.dot(Igradxyphi), 0)**0.5
    Igradphi = Igradxyphi + [Igradzphi]

    return Igradphi

def refract_field_gradient(normal, k1, Er, gradxyEr1, k2, Ir=None):
    """Apply local Snell's law at an interface given derivatives of the field.

    Args:
        normal (3-tuple): arrays of surface normal components. Should have positive z component.
        k1: wavenumber in initial medium.
        Er: field.
        gradxyEr1 (2-tuple): derivatives of Er w.r.t x and y.
        k2: wavenumber in final medium.

    Returns:
        Igradphi2 (3-tuple): product of intensity |Er|^2 and gradient of phase w.r.t. coordinate axes.
        Ir: |Er|^2 (for reuse).
    """
    assert np.all(normal[2]>0)
    assert len(gradxyEr1) == 2
    if Ir is None:
        Ir = mathx.abs_sqd(Er)

    # Calculate intensity-scaled phase gradient (3 vector).
    Igradphi=calc_Igradphi(k1, Er, gradxyEr1, Ir)

    Igradphi_tangent = mathx.project_onto_plane(Igradphi, normal)[0]
    Igradphi_normal = np.maximum((Ir*k2)**2-mathx.dot(Igradphi_tangent), 0)**0.5
    Igradphi2 = [tc+nc*Igradphi_normal for tc, nc in zip(Igradphi_tangent, normal)]

    return Igradphi2



def calc_beam_properties(rs_support, z, Er, Igradphi, rs_center=(0, 0)):
    """Calculate real space and angular centroids, and curvature.

    Assumes that propagation is to +z. Positive ROC means diverging i.e. center of curvature is to left.

    Args:
        k:
        x:
        y:
        Er:
        Igradphi (tuple): I times gradient of phase along the x, y and z.
        rs_center:
        Ir:

    Returns:

    """
    x, y = sa.calc_xy(rs_support, Er.shape, rs_center)
    Ir = mathx.abs_sqd(Er)
    sumIr = Ir.sum()

    # Calculate centroid. Improve the input estimate rs_center.
    rs_center[0], rs_center[1], varx, vary, _ = mathx.mean_and_variance2(x, y, Ir, sumIr)

    # Mean transverse wavenumber is intensity-weighted average of transverse gradient of phase.
    qs_center = np.asarray([component.sum() for component in Igradphi[:2]])/sumIr

    meanz=mathx.moment(z, Ir, 1, sumIr)

    # Correct spatial phase for surface curvature - approximate propagation from z to meanz.
    Er=Er*mathx.expj((meanz-z)*mathx.divide0(Igradphi[2], Ir))

    # Do this twice, since on the first pass we are using qs_center from Igradphi which is just an estimate.
    for _ in range(2):
        # Calculate phase of quadratic component at RMS distance from center. Proportional to A in Siegman IEE J. Quantum Electronics Vol. 27
        # 1991. Positive means diverging.
        phi_cx = 0.5*((x-rs_center[0])*(Igradphi[0]-Ir*qs_center[0])).sum()/sumIr
        phi_cy = 0.5*((y-rs_center[1])*(Igradphi[1]-Ir*qs_center[1])).sum()/sumIr

        # Fourier transform Er with quadratic phase removed.
        Ek = math.fft2(Er*mathx.expj(-(x-rs_center[0])**2*phi_cx/varx)*mathx.expj(-(y-rs_center[1])**2*phi_cy/vary))
        kx, ky = sa.calc_kxky(rs_support, Er.shape, qs_center)

        # Calculate mean square sizes of Fourier transform with quadratic phase removed.
        #qs_center[0], qs_center[1], varkxp, varkyp, _ = mathx.mean_and_variance2(kx, ky, mathx.abs_sqd(Ek))
        qs_center[0] = mathx.moment(kx, abs(Ek), 1)
        qs_center[1] = mathx.moment(ky, abs(Ek), 1)

        Ik=mathx.abs_sqd(Ek)
        sumIk = Ik.sum()
        varkxp = mathx.moment(kx - qs_center[0], Ik, 2, sumIk)
        varkyp = mathx.moment(ky - qs_center[1], Ik, 2, sumIk)

        # Calculate angular variance.
        varkx = bvar.infer_angular_variance_spherical(varx, phi_cx, varkxp)
        varky = bvar.infer_angular_variance_spherical(vary, phi_cy, varkyp)

    return meanz, rs_center, np.asarray((varx, vary)), qs_center, np.asarray((varkx, varky)), np.asarray((phi_cx, phi_cy))



def calc_propagation_spherical(k, r_support, zi, Er, z, Igradphi=None, rs_center=(0, 0), qs_center=(0, 0), f_nexts=(np.inf, np.inf), qfds=(1, 1)):
    assert np.isscalar(z)

    Ir = mathx.abs_sqd(Er)

    if Igradphi is None:
        Ek=math.fft2(Er)
        kx, ky = sa.calc_kxky(r_support, Ek.shape, qs_center)
        gradxEx = math.ifft2(Ek*1j*kx)
        gradyEy = math.ifft2(Ek*1j*ky)
        Igradphi = calc_Igradphi(k, Er, (gradxEx, gradyEy), Ir)

    zi_mean, rs_center, var_rs, qs_center, var_qs, phi_cs=calc_beam_properties(k, r_support, zi, Er, Igradphi, rs_center)

    if isinstance(z,str) and z=='waist':
        return_z = True
        z, var_r0, Msqd, z_R = bvar.calc_waist(k, var_rs, phi_cs, var_qs)
        z=np.mean(z)
    else:
        return_z = False

    m = bvar.calc_propagation_ms(k, r_support, var_rs, phi_cs, var_qs, z-zi_mean, Er.shape, f_nexts, qfds)

    if return_z:
        return m, z, rs_center, qs_center, zi_mean, rs_center + qs_center/k*(z - zi_mean)
    else:
        return m, rs_center, qs_center, zi_mean, rs_center + qs_center/k*(z - zi_mean)

def calc_refracted_propagation_spherical(r_support, z12, normal12, k1, Er1, gradEr1, k2, z, r1_centers=(0, 0), qs_center=(0,), f_nexts=(np.inf, np.inf), qfds=(1, 1)):
    """Refract"""
    Ir1 = mathx.abs_sqd(Er1)
    Igradphi1 = refract_field_gradient(normal12, k1, Er1, gradEr1, k2, Ir1)
    return calc_propagation_spherical(k2, r_support, z12,  Er1, z, Igradphi1, r1_centers, qs_center, f_nexts, qfds)



def calc_kxky_moment(rs_support, Iq, qs_center = (0, 0)):
    kxu, kyu, Iqu = sa.unroll_q(rs_support, Iq, qs_center)
    return mathx.moment(kxu, Iqu, 1), mathx.moment(kyu, Iqu, 1)


def propagate_plane_to_plane_flat_1d(k, r_support, Er, z, q_center=0, axis=-1, paraxial=False):
    num_points = Er.shape[axis]
    Eq = math.fft(Er, axis)
    q = sa.calc_q(r_support, num_points, q_center, axis)
    if paraxial:
        Eq *= mathx.expj((k-q**2/(2*k))*z)
    else:
        Eq *= mathx.expj((k**2-q**2)**0.5*z)
    Er = math.ifft(Eq, axis)
    return Er


def propagate_plane_to_plane_spherical_paraxial_1d(k, r_support, Er, z, m, r_center=0, q_center=0, axis=-1):
    """"Propagate field with spherical wavefront from plane to plane.

    Args:
        k:
        r_support:
        Er: changed!
    """
    propagator, post_factor, r_center_z = prepare_curved_paraxial_propagation_1d(k, r_support, Er, z, m, r_center, q_center, axis)
    Ak = math.fft(Er, axis)
    Ak *= propagator
    Er = math.ifft(Ak, axis)*post_factor
    return Er, r_center_z

def propagate_plane_to_plane_spherical_paraxial_1dE(k, r_support, Eri, z, m, r_center = 0, q_center = 0):
    """Propagate field with spherical wavefront from plane to plane.

    kz is included.

    Args:
        k:
        rs_support:
        Eri:
        z (2D array): propagation distance vs (x, y)
        m:
        rs_center:
        qs_center:

    Returns:

    """
    assert Eri.ndim == 1
    num_points = len(Eri)
    ri = sa.calc_r(r_support, num_points, r_center)
    qi = sa.calc_q(r_support, num_points, q_center)
    roc = z/(m-1)
    Qir = mathx.expj(-k*ri**2/(2*roc[:, None]))
    T = math.make_fft_matrix(num_points)
    P = mathx.expj((k-qi**2/(2*k*m))*z[:, None])
    invT = T.conj()
    ro = ri*m
    Qor = mathx.expj(k*ro**2/(2*(roc+z)))/m**0.5
    # ro: i,  qi: j,  ri: k
    Ero = opt_einsum.contract('i, ij, ij, jk, ik, k->i', Qor, invT, P, T, Qir, Eri)
    return Ero


def propagate_plane_to_waist_spherical_paraxial_1d(k, r_support, Er, z, r_center, q_center, roc, axis = -1):
    num_points = Er.shape[axis]
    m_flat, z_flat, m = math.calc_curved_propagation(k, r_support, num_points, roc, z)
    if m_flat == 1:
        Er_flat = Er
        r_center_flat = r_center
    else:
        Er_flat, r_center_flat = propagate_plane_to_plane_spherical_paraxial_1d(k, r_support, Er, -z_flat, 1/m_flat, r_center, q_center, axis)
    return Er_flat, z_flat, m_flat, m, r_center_flat

def propagate_plane_to_plane_spherical_1d(k, r_support, Er, z, r_center = 0, q_center = 0, roc = np.inf, axis = -1):
    num_points = Er.shape[axis]
    Er_flat, z_flat, m_flat, m, r_center_flat = propagate_plane_to_waist_spherical_paraxial_1d(k, r_support, Er, z, r_center, q_center, roc)
    r_support_flat = r_support/m_flat
    z_paraxial = z/(1-(q_center/k)**2)**1.5
    q_flat = sa.calc_q(r_support_flat, num_points, q_center, axis)
    extra_propagator = mathx.expj((k**2-q_flat**2)**0.5*z-(k-q_flat**2/(2*k))*z_paraxial)

    Eq_flat = math.fft(Er_flat, axis)
    Eq_flat *= extra_propagator
    Er_flat = math.ifft(Eq_flat, axis)

    Er, _ = propagate_plane_to_plane_spherical_paraxial_1d(k, r_support_flat, Er_flat, z_paraxial+z_flat, m*m_flat, r_center_flat, q_center)

    rp_center = r_center+z*q_center/(k**2-q_center**2)**0.5

    return Er, m, rp_center

def propagate_plane_to_plane_flat(k, rs_support, Er, z, qs_center=(0, 0), kz_mode='local_xy'):
    """Regularly (non-Sziklas-Siegman) propagate a field between two flat planes.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        Er (2D array): Initial field.
        z (scalar): Propagation distance.
        qs_center (pair): Center of transverse wavenumber space.
        kz_mode (str): 'paraxial', 'local_xy', or 'exact'.

    Returns:
        2D array: Propagated field.
    """
    rs_support = sa.to_scalar_pair(rs_support)
    qs_center = sa.to_scalar_pair(qs_center)
    Eq = math.fft2(Er)
    kx, ky = sa.calc_kxky(rs_support, Eq.shape, qs_center)
    if kz_mode == 'paraxial':
        propagator = math.calc_propagator_paraxial(k, kx, ky, z)
    elif kz_mode in ('local_xy', 'local'):
        kxp = kx - qs_center[0]
        kyp = ky - qs_center[1]
        kz, gxkz, gykz, gxxkz, gyykz, gxykz = math.expand_kz(k, *qs_center)
        propagator = mathx.expj((kz + gxkz*kxp + gykz*kyp + gxxkz*kxp**2/2 + gyykz*kyp**2/2)*z)
        if kz_mode == 'local':
            propagator *= mathx.expj(gxykz*kxp*kyp*z)
    elif kz_mode == 'exact':
        propagator = math.calc_propagator_exact(k, kx, ky, z)
    else:
        raise ValueError('Unknown kz_mode %s.'%kz_mode)
    Eq *= propagator
    Er = math.ifft2(Eq)
    return Er


def propagate_plane_to_plane_spherical(k, rs_support, Er, z, ms, rs_center=(0, 0), qs_center=(0, 0), ro_centers=None, kz_mode='local_xy'):
    """Propagate from one plane to another using Sziklas-Siegman.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space full support in x and y.
        Er (2D array): Input field.
        z (scalar): Propagation distance.
        ms (scalar or pair): Sziklas-Siegman magnification in x and y.
        rs_center (pair): Center of domain and center of wavefront curvature.
        qs_center (pair): Center of transverse wavenumber domain.
        ro_centers (pair): Center of output domain. Default is what follows from rs_center, qs_center and z.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        2D array: Output field.
    """
    assert kz_mode in ('local_xy', 'paraxial')
    rs_support = sa.to_scalar_pair(rs_support)
    ms = sa.to_scalar_pair(ms)
    if kz_mode == 'paraxial':
        zx = z
        zy = z
    elif kz_mode == 'local_xy':
        fx, fy, delta_kz, delta_gxkz, delta_gykz = math.calc_quadratic_kz_correction(k, *qs_center)
        zx = fx*z
        zy = fy*z
    else:
        raise ValueError('Unknown kz_mode %s.'%kz_mode)
    x, y = sa.calc_xy(rs_support, Er.shape, rs_center)
    roc_x = zx/(ms[0] - 1)
    roc_y = zy/(ms[1] - 1)
    kx, ky = sa.calc_kxky(rs_support, Er.shape, qs_center)
    kxp = kx - k*rs_center[0]/roc_x
    kyp = ky - k*rs_center[1]/roc_y
    Er = Er*math.calc_quadratic_phase_1d(-k, x - rs_center[0], roc_x)*math.calc_quadratic_phase_1d(-k, y - rs_center[1], roc_y)
    Eq = math.fft2(Er)
    Eq *= math.calc_propagator_quadratic_1d(k*ms[0], kxp, zx)*math.calc_propagator_quadratic_1d(k*ms[1], kyp, zy)
    # If local_xy mode, need translation correction factor.
    if kz_mode == 'local_xy':
        Eq *= mathx.expj(delta_gxkz*kx/ms[0]*z + delta_gykz*ky/ms[1]*z)

    Er = math.ifft2(Eq)*math.calc_spherical_post_factor(k, rs_support, Er.shape, z, ms, rs_center, qs_center, ro_centers, kz_mode)
    return Er


def propagate_plane_to_curved_flat(k, rs_support, Eri, z, qs_center=(0, 0), kz_mode='local_xy'):
    """ """
    invTx, gradxinvTx, invTy, gradyinvTy, Px, Py, Tx, Ty = math.calc_plane_to_curved_flat_factors(k, rs_support, Eri.shape, z, qs_center, kz_mode)
    # xo=i, yo=j, kx=k, ky=l, xi=m, yi=n.
    Ero = opt_einsum.contract('ik, jl, ijk, ijl, km, ln, mn -> ij', invTx, invTy, Px, Py, Tx, Ty, Eri)
    gradxEo = opt_einsum.contract('ik, jl, ijk, ijl, km, ln, mn -> ij', gradxinvTx, invTy, Px, Py, Tx, Ty, Eri)
    gradyEo = opt_einsum.contract('ik, jl, ijk, ijl, km, ln, mn -> ij', invTx, gradyinvTy, Px, Py, Tx, Ty, Eri)
    return Ero, (gradxEo, gradyEo)

def propagate_plane_to_curved_flat_arbitrary(k, rs_support, Eri, z, xo, yo, qs_center=(0, 0), kz_mode='local_xy'):
    """Propagate from uniformly sampled plane to arbitrarily sampled curved surface.

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
        Ero (MxN array): Output field.
        gradxyEo (tuple of MxN arrays): Partial derivatives of output field w.r.t. x and y at constant z (not along
            the surface).
    """
    factors = math.calc_plane_to_curved_flat_arbitrary_factors(k, rs_support, Eri.shape, z, xo, yo, qs_center, kz_mode)
    invTx, gradxinvTx, invTy, gradyinvTy, Px, Py, Tx, Ty = factors
    # xo=i, yo=j, kx=k, ky=l, xi=m, yi=n.
    Ero = opt_einsum.contract('ik, jl, ijk, ijl, km, ln, mn -> ij', invTx, invTy, Px, Py, Tx, Ty, Eri)
    gradxEo = opt_einsum.contract('ik, jl, ijk, ijl, km, ln, mn -> ij', gradxinvTx, invTy, Px, Py, Tx, Ty, Eri)
    gradyEo = opt_einsum.contract('ik, jl, ijk, ijl, km, ln, mn -> ij', invTx, gradyinvTy, Px, Py, Tx, Ty, Eri)
    return Ero, (gradxEo, gradyEo)


def invert_plane_to_curved_flat(k, rs_support, Ero, z, qs_center=(0, 0), kz_mode='local_xy', max_iterations=None, tol=None):
    propagator = math.prepare_plane_to_curved_flat(k, rs_support, Ero.shape, z, qs_center, kz_mode)
    Eri = propagator.invert(Ero, max_iterations, tol)
    return Eri, propagator

def propagate_curved_to_plane_flat(k, rs_support, Eri, z, qs_center=(0, 0), kz_mode='local_xy', max_iterations=None, tol=None):
    Eri, propagator = invert_plane_to_curved_flat(k, rs_support, Eri, -z, qs_center, kz_mode, max_iterations, tol)
    return Eri



def propagate_plane_to_curved_spherical(k, rs_support, Eri, z, ms, rs_center=(0, 0), qs_center=(0, 0), ro_centers=None, kz_mode='local_xy'):
    """
    The kz term uses zx.

    Args:
        k:
        rs_support:
        Eri: not changed
        z (2D array): propagation distance vs (x, y)
        m:
        rs_center:
        qs_center:

    Returns:


    cProfile.run('asbp.propagate_curved_paraxial_surface(k, rs_support, Er1, z2, m, rs_center, qs_center)', sort = 'cumtime')
    May 14 2018 1530 - cProfile.run sorted by cumtime gives for (2**8, 2**8) gives:

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    4.110    4.110 {built-in method builtins.exec}
        1    0.041    0.041    4.110    4.110 <string>:1(<module>)
        1    2.048    2.048    4.069    4.069 asbp.py:138(propagate_curved_paraxial_surface)
        1    0.095    0.095    2.017    2.017 contract.py:253(contract)
        3    0.000    0.000    1.026    0.342 blas.py:99(tensor_blas)
        3    1.026    0.342    1.026    0.342 {built-in method numpy.core.multiarray.dot}
        7    0.000    0.000    0.895    0.128 einsumfunc.py:703(einsum)

    Half the time is the tensor contraction. Currently using mathx.expj(numpy expression) for factors. For iteration,
    lowest hanging fruit is precomputing the factors. Could also evaluate them using numba.

    Replacing quadratic phase expressions with numba. Initial %timeit 4.74 s per loop. After Qix,  Qiy to numba - down to 4.1 s.
    After Px, Py - down to 3.8 s.

    Factored out x_factor and y_factor. Now 4.5 s for one off,  but expect it to be much faster for multiple calculations.

    Numerical instability. 2**6, 2**7. Did 20 iterations,  at which point it was clearly exploding (RMS error 10^10). Calculated
    propagator,  applied to resulting Er1,  then applied in reverse. Result was 12 times bigger i.e. eigenvalue of A^t*A
    is 12 where A is the matrix which transforms from Eri to Ero.
    """
    assert Eri.ndim == 2
    ms = sa.to_scalar_pair(ms)
    factors = math.calc_plane_to_curved_spherical_factors(k, rs_support, Eri.shape, z, ms, rs_center, qs_center, ro_centers, kz_mode)
    #Qo, gradxphiQo, gradyphiQo,
    invTPx, gradxinvTPx, invTPy, gradyinvTPy, Tx, Ty, Qix, Qiy = factors
    #(gradxphiQo, gradyphiQo, gradzphiQo), (gradxinvTx, gradyinvTy, (gradzPx, gradzPy)), Qo, invTx, invTy, Px, Py, Tx, Ty, Qix, Qiy = factors

    # xo: i,  yo: j,  kx: k,  ky: l,  xi: m,  yi: n.
    Ero = opt_einsum.contract('ijk, ijl, km, ln, ijm, ijn, mn -> ij', invTPx, invTPy, Tx, Ty, Qix, Qiy, Eri)
    gradxEro = opt_einsum.contract('ijk, ijl, km, ln, ijm, ijn, mn -> ij', gradxinvTPx, invTPy, Tx, Ty, Qix, Qiy, Eri)
    gradyEro = opt_einsum.contract('ijk, ijl, km, ln, ijm, ijn, mn -> ij', invTPx, gradyinvTPy, Tx, Ty, Qix, Qiy, Eri)
    # This one isn't correct - see (partial) discussion page 117 Dane's logbook 2. For now we can assume geometrical optics
    # applies (aproximately),  so the length of the phase gradient is k.
    #gradzEro = 1j*gradzphiQo*Ero+opt_einsum.contract('ij, ik, jl, ijk, ijl, km, ln, ijm, ijn, mn->ij', Qo, invTx, invTy, gradzPx, Py, Tx, Ty, Qix, Qiy, Eri)+opt_einsum.contract('ij, ik, jl, ijk, ijl, km, ln, ijm, ijn, mn->ij', Qo, invTx, invTy, Px, gradzPy, Tx, Ty, Qix, Qiy, Eri)
    return Ero, (gradxEro, gradyEro)


def propagate_plane_to_curved_spherical_arbitrary(k, rs_support, Eri, z, xo, yo, roc_x, roc_y, rs_center=(0, 0),
                                                  qs_center=(0, 0), ro_centers=None, kz_mode='local_xy'):
    """Propagate from uniformly sampled plane to arbitrarily sampled curved surface.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        Eri (2D array): Input field.
        z (M*N array): Propagation for distance.
        xo (M*1 array): Output x sample values.
        yo (N array): Output y sample values.
        roc_x (M*N array): Input radius of curvature along x.
        roc_y (M*N array): Input radius of curvature along y.
        rs_center (pair of scalars): Center of initial real-space aperture.
        qs_center (pair of scalars): Center of angular space aperture.
        kz_mode (str): 'paraxial' or 'local_xy'.

    Returns:
        Ero (M*N array): Output field
        gradxyEro (tuple of M*N arrays): Partial derivatives of output field w.r.t x and y.
    """
    assert Eri.ndim == 2
    invTPx, gradxinvTPx, invTPy, gradyinvTPy, Tx, Ty, Qix, Qiy = \
        math.calc_plane_to_curved_spherical_arbitrary_factors(k, rs_support, Eri.shape, z, xo, yo, roc_x, roc_y, rs_center,
                                                        qs_center, ro_centers, kz_mode)

    # xo: i,  yo: j,  kx: k,  ky: l,  xi: m,  yi: n.
    Ero = opt_einsum.contract('ijk, ijl, km, ln, ijm, ijn, mn -> ij', invTPx, invTPy, Tx, Ty, Qix, Qiy, Eri)
    gradxEro = opt_einsum.contract('ijk, ijl, km, ln, ijm, ijn, mn -> ij', gradxinvTPx, invTPy, Tx, Ty, Qix, Qiy, Eri)
    gradyEro = opt_einsum.contract('ijk, ijl, km, ln, ijm, ijn, mn -> ij', invTPx, gradyinvTPy, Tx, Ty, Qix, Qiy, Eri)

    return Ero, (gradxEro, gradyEro)


def invert_plane_to_curved_spherical(k, rs_support, Ero, z, ms, rs_center=(0, 0), qs_center=(0, 0), ro_centers=(0, 0),
                                     kz_mode='local_xy', max_iterations=None, tol=None):
    """
    kz term uses zx.

    Args:
        k:
        rs_support:
        Ero:
        zx:
        ms:
        rs_center:
        qs_center:
        zy:
        ro_centers:
        max_iterations:
        tol:



    Returns:

    """
    ms = sa.to_scalar_pair(ms)
    num_pointss = Ero.shape
    propagator = math.prepare_plane_to_curved_spherical(k, rs_support, num_pointss, z, ms, rs_center, qs_center, ro_centers, kz_mode)
    if 0:
        # Old fixed iteration method,  based on intuition that propagation from curved surface to plane and back
        # is approximately identity. Starts to converge,  but it is unstable. Can be made stable by reducing the change
        # size,  but becomes slow. Requires rms_tol and alpha.
        rms_tol = 1e-6
        alpha = 0.1

        Erop = np.zeros(Ero.shape, dtype = complex)
        Eri = np.zeros(Ero.shape, dtype = complex)
        num_iterations = 0
        sum_sqd_Ero = mathx.sum_abs_sqd(Ero)
        while True:
            delta_Ero = Ero-Erop
            rms_error = (mathx.sum_abs_sqd(delta_Ero)/sum_sqd_Ero)**0.5
            if rms_error <= 1e-6 or num_iterations >= max_iterations:
                break
            print('Iteration %d: RMS error %g.'%(num_iterations, rms_error))
            Eri += propagator.apply_reverse(delta_Ero)*alpha
            #Eri *= (sum_sqd_Ero*np.prod(ms)/mathx.sum_abs_sqd(Eri))**0.5
            Erop = propagator.apply(Eri)
            num_iterations += 1
        if rms_error>rms_tol:
            print('Warning - after %d iterations,  RMS error was %g but tolerance was %g.'%(num_iterations, rms_error, rms_tol))
    else:
        Eri = propagator.invert(Ero, max_iterations, tol)
    return Eri, propagator

def invert_plane_to_curved_spherical_arbitrary(k, rs_support, num_pointss, Ero, z, xo, yo, roc_xo, roc_yo, rs_center=(0, 0), qs_center=(0, 0),
                                               ro_centers=None, kz_mode='local_xy', invert_kwargs=None):
    """Invert propagation from a plane to a curved arbitrarily sampled surface.

    Args:
        k (scalar): Wavenumber.
        rs_support (scalar or pair): Real space support.
        num_pointss (int or tuple of ints): Number
        Ero (2D array): Output field (at  z).
        z (M*N array): Propagation for distance.
        xo (M*1 array): Output x sample values.
        yo (N array): Output y sample values.
        roc_xo (M*N array): Radius of curvature at the input along x, sampled at output points.
        roc_yo (M*N array): Radius of curvature at the input along y, sampled at output points.
        rs_center (pair of scalars): Center of initial real-space aperture.
        qs_center (pair of scalars): Center of angular space aperture.
        kz_mode (str): 'paraxial' or 'local_xy'.
        invert_kwargs (dict): Passed on to Propagator.invert method.

    Returns:
        Eri (K*L array): Field at input plane.
        Propagator object
    """
    assert Ero.shape == z.shape
    assert xo.shape == (Ero.shape[0], 1)
    assert yo.shape == (Ero.shape[1], )
    assert roc_xo.shape == z.shape
    assert roc_yo.shape == z.shape
    if invert_kwargs is None:
        invert_kwargs = {}

    propagator = math.prepare_plane_to_curved_spherical_arbitrary(k, rs_support, num_pointss, z, xo, yo, roc_xo, roc_yo,
        rs_center, qs_center, ro_centers, kz_mode)
    Eri = propagator.invert(Ero, **invert_kwargs)
    return Eri, propagator

def propagate_curved_to_plane_spherical(k, rs_support, Eri, z, ms, ri_centers=(0, 0), qs_center = (0, 0),
                                        ro_centers=None, kz_mode='local_xy', max_iterations=None, tol=None):
    """

    Args:
        k:
        rs_support:
        Eri:
        zx:
        ms:
        ri_centers:
        qs_center:
        zy:
        ro_centers:
        max_iterations:
        tol:

    Returns:

    """
    assert kz_mode in ('local_xy', 'paraxial')
    if ro_centers is None:
        Iri = mathx.abs_sqd(Eri)
        sumIri = Iri.sum()
        z_center = mathx.moment(z, Iri, 1, sumIri)
        ro_centers = math.adjust_r(k, ri_centers, z, qs_center, kz_mode) # TODO should be z_center?
    Ero, propagator = invert_plane_to_curved_spherical(k, rs_support*ms, Eri, -z, 1/ms, ro_centers, qs_center, ri_centers, kz_mode, max_iterations, tol)
    return Ero

def propagate_arbitrary_curved_to_plane_spherical(k, xi, yi, Eri, roc_xi, roc_yi, zi, ro_supports, num_pointsso, ri_centers=(0, 0), qs_center = (0, 0),
                                                  ro_centers=None, kz_mode='local_xy', invert_kwargs=None):
    """

    Args:
        k:
        xi (Mx1 array):
        yi (N array):
        Eri (MxN array):
        roc_x (M*N array): Input radius of curvature along x.
        roc_y (M*N array): Input radius of curvature along y.
        zi (MxN array):
        ro_supports:
        num_pointsso:
        ri_centers:
        qs_center:
        ro_centers:
        kz_mode:
        invert_kwargs:

    Returns:
        Ero (array of size num_pointsso):
    """
    assert xi.shape == (Eri.shape[0], 1)
    assert yi.shape == (Eri.shape[1], )
    assert Eri.ndim == 2
    assert roc_xi.shape == Eri.shape
    assert roc_yi.shape == Eri.shape
    assert zi.shape == Eri.shape

    if ro_centers is None:
        z_center = mathx.moment(zi, mathx.abs_sqd(Eri), 1)
        ro_centers = math.adjust_r(k, ri_centers, z_center, qs_center, kz_mode)

    z = -zi
    roc_x = roc_xi + zi
    roc_y = roc_yi + zi
    Ero, propagator = invert_plane_to_curved_spherical_arbitrary(k, ro_supports, num_pointsso, Eri, z, xi, yi, roc_x,
        roc_y, ro_centers, qs_center, ri_centers, kz_mode, invert_kwargs)

    return Ero

# def propagate_plane_to_waist_spherical_paraxial(k, rs_support, Er, z, rs_center = (0, 0), qs_center = (0, 0), rocs = np.inf):
#     rs_support = to_pair(rs_support)
#     rocs = to_pair(rocs)
#     rs_center = np.asarray(rs_center)
#     qs_center = np.asarray(qs_center)
#     num_pointss = Er.shape[-2:]
#     ms_flat, zs_flat, ms = np.asarray([calc_curved_propagation(k, r_support, num_points, roc, z) for r_support, num_points, roc in zip(rs_support, num_pointss, rocs)]).T
#     Er_flat, r_centers_flat = propagate_plane_to_plane_spherical_paraxial(k, rs_support, Er, -zs_flat, 1/ms_flat, rs_center, qs_center, carrier = False)
#     return Er_flat, zs_flat, ms_flat, ms, r_centers_flat
#
# def propagate_plane_to_plane_spherical(k, rs_support, Er, z, rs_center = (0, 0), qs_center = (0, 0), rocs = np.inf):
#     rs_support = to_pair(rs_support)
#     rs_center = np.asarray(rs_center)
#     qs_center = np.asarray(qs_center)
#     rocs = to_pair(rocs)
#     num_pointss = Er.shape[-2:]
#     Er_flat, zs_flat, ms_flat, ms, r_centers_flat = propagate_plane_to_waist_spherical_paraxial(k, rs_support, Er, z, rs_center, qs_center, rocs)
#     r_supports_flat = rs_support/ms_flat
#
#     # See Dane's notebook 2 p105.
#     denominator = (k**2-qs_center[0]**2-qs_center[1]**2)**(3/2)
#     z_paraxial_x = z*k*(k**2-qs_center[1]**2)/denominator
#     z_paraxial_y = z*k*(k**2-qs_center[0]**2)/denominator
#     zs_paraxial = np.asarray((z_paraxial_x, z_paraxial_y))
#
#     kx_flat, ky_flat = [calc_q(r_support_flat, num_points, q_center, axis) for r_support_flat, num_points, q_center, axis in zip(r_supports_flat, num_pointss, qs_center, (-2, -1))]
#     extra_propagator = mathx.expj(calc_kz(k, kx_flat, ky_flat)*z+z_paraxial_x*kx_flat**2/(2*k)+z_paraxial_y*ky_flat**2/(2*k))
#
#     Eq_flat = fft2(Er_flat)
#     Eq_flat *= extra_propagator
#     Er_flat = ifft2(Eq_flat)
#
#     Er, _ = propagate_plane_to_plane_spherical_paraxial(k, r_supports_flat, Er_flat, zs_paraxial+zs_flat, ms*ms_flat, r_centers_flat, qs_center, carrier = False)
#     rp_centers = rs_center+z*qs_center/(k**2-qs_center**2)**0.5
#
#     return Er, ms, rp_centers