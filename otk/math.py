"""Optics-specific math functions."""
import cmath
import numba
from typing import Tuple
import numpy as np

def calc_sphere_sag_normal_xy(roc, x, y, clip=False):
    """Calculate the sag of a spherical surface.

    The surface is tangent to the z = 0 plane. The center of curvature is at (0, 0, roc). So positive ROC is convex
    (viewed from negative z) and has positive sag. The normal always points in the positive z direction.

    Args:
        roc: Radius of curvature
        x, y: Transverse coordinates.
        clip (bool): If True, then the hemisphere is extended across the rest of the z = roc plane. For any (x, y) points that
            are further out than the ROC (and therefore don't intersect with the sphere), the reported sag is 0 and the
            normal is (0, 0, 1).

    Returns:
        sag: Z value of surface.
        normal: Normal vector.
    """
    zsqd = roc**2 - x**2 - y**2
    if clip:
        zsqd = np.maximum(zsqd, 0)
    else:
        zsqd_min = np.min(zsqd)
        if zsqd_min < 0:
            raise ValueError('Surface goes %g mm beyond hemisphere.'%((-zsqd_min)**0.5*1e3))
    z = zsqd**0.5
    sag = roc - np.sign(roc)*z
    nx = -x/roc
    ny = -y/roc
    nz = z/abs(roc)
    if clip:
        valid = zsqd > 0
        nx = np.choose(valid, (0, nx))
        ny = np.choose(valid, (0, ny))
        nz = np.choose(valid, (1, nz))
        return sag, (nx, ny, nz), valid
    else:
        return sag, (nx, ny, nz)


def calc_sphere_sag_xy(roc, x, y, clip=False):
    zsqd = roc**2 - x**2 - y**2
    if clip:
        zsqd = np.maximum(zsqd, 0)
    else:
        zsqd_min = np.min(zsqd)
        if zsqd_min < 0:
            raise ValueError('Surface goes %g mm beyond hemisphere.'%((-zsqd_min)**0.5*1e3))
    return roc - np.sign(roc)*zsqd**0.5


def calc_sphere_sag(roc, r, derivative:bool=False):
    """Calculate sag of spherical surface.

    Args:
        roc: Same sign as the sag.
        r: Distance from center.
        derivative: If True, derivative w.r.t r is returned instead.
    """
    if np.isinf(roc):
        return np.zeros(np.shape(r))
    else:
        zsqd = roc**2 - r**2
        if derivative:
            return np.sign(roc)*r/zsqd**0.5
        else:
            return roc - np.sign(roc)*zsqd**0.5

# Want it to work for array ROC.
calc_sphere_sag = np.vectorize(calc_sphere_sag, excluded=['r', 'derivative'])


def calc_conic_sag(roc, kappa, alphas, rho, derivative:bool=False):
    """Calculate sag of a conic asphere surface.

    Args:
        roc (scalar): Radius of curvature.
        kappa (scalar): Conic parameter. Special values:
            kappa < 0: Hyperboloid.
            kappa = 0: Paraboloid.
            0 < kappa < 1: Elipsoid of revolution about major axis.
            kappa = 1: Sphere
            kappa > 1: Elipsoid of revolution about minor axis.
        alphas (sequence): Second and higher order coefficients.
    """
    if derivative:
        alphaps = np.arange(2, 2 + len(alphas))*alphas
        return rho/(roc*(1 - kappa*(rho/roc)**2)**0.5) + np.polyval(alphaps[::-1], rho)*rho
    else:
        return rho ** 2/(roc*(1 + (1 - kappa*(rho/roc) ** 2) ** 0.5)) + np.polyval(alphas[::-1], rho)*rho ** 2



@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64)], cache=True)
def calc_ideal_lens_phase_paraxial(x, y, k_on_f):
    return cmath.exp(-0.5j*(x**2 + y**2)*k_on_f)




@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64, numba.float64)], cache=True)
def calc_ideal_lens_phase(x, y, k, f):
    return cmath.exp(-1j*k*((x**2 + y**2 + f**2)**0.5 - f))


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64, numba.float64)], cache=True)
def calc_ideal_square_lens_array_phase_paraxial(x, y, pitch, k_on_f):
    half_pitch = pitch/2
    u = (x%pitch) - half_pitch
    v = (y%pitch) - half_pitch
    return cmath.exp(-0.5j*(u**2 + v**2)*k_on_f)


@numba.vectorize([numba.complex64(numba.float64, numba.float64, numba.float64, numba.float64, numba.float64)],
    cache=True)
def calc_ideal_square_lens_array_phase(x, y, pitch, k, f):
    half_pitch = pitch/2
    u = (x%pitch) - half_pitch
    v = (y%pitch) - half_pitch
    return cmath.exp(-1j*k*((u**2 + v**2 + f**2)**0.5 - f))


def calc_fresnel_coefficients(n1, n2, cos_theta1, cos_theta2 = None) -> Tuple:
    """

    Args:
        n1: Refractive index in first medium.
        n2: Refractive index in second medium
        cos_theta1: Cosine of angle of incidence.

    Returns:

    """
    assert np.all(cos_theta1>=0)
    if cos_theta2 is None:
        sin_theta1 = (1 - cos_theta1**2)**0.5
        sin_theta2 = n1*sin_theta1/n2
        cos_theta2 = (1 - sin_theta2**2)**0.5

    r_s = (n1*cos_theta1 - n2*cos_theta2)/(n1*cos_theta1 + n2*cos_theta2)
    r_p = (n2*cos_theta1 - n1*cos_theta2)/(n2*cos_theta1 + n1*cos_theta2)
    t_s = 2*n1*cos_theta1/(n1*cos_theta1 + n2*cos_theta2)
    t_p = 2*n1*cos_theta1/(n2*cos_theta1 + n1*cos_theta2)
    return (r_p, r_s), (t_p, t_s)

@numba.vectorize([numba.float64(numba.complex128, numba.complex128, numba.complex128, numba.float64)], nopython=True)
def unwrap_quadratic_phase(a, b, c, x):
    """Evaluate phase of quadratic on same Riemann surface as constant te

    Args:
        a: Qudratic coefficient.
        b: Linnar coefficient.
        c: Constant.
        x (real): Value at which to evaluate quadratic.

    Returns:
        Phase of ax^2 + bx + c with no branch cuts on trajectory from x=0.
    """
    root_det = cmath.sqrt(b**2 - 4*a*c)
    x1 = (-b + root_det)/(2*a)
    x2 = (-b - root_det)/(2*a)
    return np.angle(x - x1) - np.angle(-x1) + np.angle(x - x2) - np.angle(-x2) + np.angle(c)
