import numpy as np
import pyfftw
import mathx
from scipy import special
from . import sa


def calc_gaussian_1d(k, r, waist0, z=0, r0=0, q0=0, carrier=True, gradr=False):
    """Calculate 1D Gaussian beam profile.

    The result is normalized to unity amplitude at the waist (not on the axis).

    Args:
        k: Wavenumber.
        r: Sampling points.
        waist0: Beam waist.
        z: Axial position relative to waist.
        r0: Real-space offset of waist.
        q0: Wavenumber-space offset of waist.
        carrier (bool): Whether to include kz carrier phase.
        gradr (bool): Whether to return partial derivative w.r.t r of the field.

    Returns:
        Er, or gradrEr.
    """
    z_R = waist0**2*k/2
    waist = waist0*(1 + (z/z_R)**2)**0.5
    roc = z + mathx.divide0(z_R**2, z, np.inf)
    # if z == 0:
    #    roc = np.inf
    # else:
    #    roc = z+z_R**2/z
    r1 = r0 + q0*z/k
    # In 1D,  Gouy phase shift is halved. See
    # Physical origin of the Gouy phase shift,  Feng and Winful,  Optics Letters Vol. 27 No. 8 2001.
    phi = (r - r1)**2*k/(2*roc) + q0*(r - r1) - 0.5*np.arctan(z/z_R) + q0**2*z/(2*k)
    if carrier:
        phi += k*z
    Er = (waist0/waist)**0.5*np.exp(-((r - r1)/waist)**2 + 1j*phi)
    if gradr:
        gradrphi = (r - r1)*k/roc + q0
        result = Er*(-2*(r - r1)/waist**2 + 1j*gradrphi)
    else:
        result = Er
    return pyfftw.byte_align(result)


def calc_gaussian(k, x, y, waist0s, z=0, r0s=(0, 0), q0s=(0, 0), gradr=False):
    """Sample 2D paraxial Gaussian beam.

    The normalization is such that the analytic (true, not sampled) field has unity integrated |E(r)|^2 across a transverse
    plane.

    Args:
        k (scalar): Wavenumber.
        x (N*1 array): X values to sample.
        y (Length M array): Y values to sample.
        waist0s (scalar or pair of): Waist in x and y planes.
        z (numeric that broadcasts with x and y): Z values relative to the waist to sample.
        r0s (pair of scalars): Coordinates of real-space center of Gaussian.
        q0s (pair of scalars): Coordinates of angular-space center of Gaussian.
        gradr (bool): If True, return transverse partial derivatives.

    Returns:
        If gradr: Er, (gradxEr, gradyEr). Otherwise just Er.
    """
    waist0s = sa.to_scalar_pair(waist0s)

    # 1D functions are normalized to unity amplitude on the waist. We want result to have normalized power.
    factor = (np.pi*np.prod(waist0s)/2)**0.5

    Ex = calc_gaussian_1d(k, x, waist0s[0], z, r0s[0], q0s[0], False)
    Ey = calc_gaussian_1d(k, y, waist0s[1], z, r0s[1], q0s[1], True)
    Er = pyfftw.byte_align(Ex*Ey)/factor

    if gradr:
        gradxEx = calc_gaussian_1d(k, x, waist0s[0], z, r0s[0], q0s[0], False, True)
        gradyEy = calc_gaussian_1d(k, y, waist0s[1], z, r0s[1], q0s[1], True, True)
        gradxEr = pyfftw.byte_align(gradxEx*Ey)/factor
        gradyEr = pyfftw.byte_align(Ex*gradyEy)/factor
        return Er, (gradxEr, gradyEr)
    else:
        return Er

def calc_bessel(x, y, radius, gradr=False):
    rho = (x**2 + y**2)**0.5

    zero = special.jn_zeros(0, 1)

    # See RT2 p75 for derivation
    norm_fac = np.pi**0.5*special.jv(1, zero)*radius
    z = rho*zero/radius
    in_beam = rho <= radius
    E = pyfftw.byte_align(special.jv(0, z)*in_beam/norm_fac)

    if gradr:
        gradrhoE = special.jvp(0, z, 1)*in_beam/norm_fac*zero/radius
        gradxE = pyfftw.byte_align(mathx.divide0(gradrhoE*x, rho))
        gradyE = pyfftw.byte_align(mathx.divide0(gradrhoE*y, rho))
        return E, (gradxE, gradyE)
    else:
        return E


