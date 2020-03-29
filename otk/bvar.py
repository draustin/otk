"""Functions for computing the low-order properties of beams during propagation and/or passing through lenses."""
import numpy as np


def transform_angular_variance_lens_1d(k, var_r, phi_c, var_q, f):
    """Calculate the angular variance of a beam that has passed through a lens.

    The lens phase is -k*r**2/(2*f). The formula used is (15) of Siegman IEEE J. Quantum Electronics,  vol. 27 1991,  with
    k = 2*pi/lambda and opposite sign convention for f. It is valid for any refractive index.

    Args:
        var_r (scalar): real space variance.
        phi_c (scalar): real-space curvature
        var_q (scalar): angular variance of the beam before the lens.
        f (scalar): focal length of the lens - positive = converging lens.

    Returns:
        var_qp (scalar): angular variance of the beam after the lens.
    """
    return var_q - 4*k*phi_c/f + var_r*k**2/f**2


def calc_correction_lens_range_1d(k, var_r, phi_c, var_q, var_q_max):
    """Calculate range of lenses which correct wavefront curvature so that a beam is well sampled.

    Args:
        k: wavenumber.
        var_r: real-space variance.
        phi_c: effective wavefront curvature - equal to of phase of quadratic phase at RMS radius.
        var_q: angular variance of beam without lens.
        var_q_max: maximum allowed angular variance - should be somewhat less than the support.

    Returns:
        power_range (array of length 2): range of lens powers in which wavenumber variance is less than var_q_max.
    """
    power_range = np.sort(np.roots((var_r*k**2, -4*k*phi_c, var_q - var_q_max)))
    if np.any(power_range.imag/power_range.real > 1e-6) or len(power_range) != 2:
        return None
    power_range = power_range.real
    return power_range


def infer_angular_variance_spherical(var_r, phi_c, var_q_min):
    """Calculate angular variance given properties of a beam with quadratic phase corrected by a lens.

    The lens which removes the spherical component has focal length f = -k*var_r/(2*phi_c). This means that -phi_c is the
    phase of the quadratic component at r  =  var_r**0.5.

    The formula is (19) of Siegman IEEE J. Quantum Electronics,  vol. 27 1991,  with k = 2*pi/lambda. For consistency with
    the rest of the program,  I use slightly different definitions. The derivation is on p120 of Dane's Fathom notebook 2.
    It is valid for any refractive index.

    Args:
        var_r (scalar): real space variance.
        phi_c (scalar): real-space curvature - see above.
        var_q_min (scalar): angular variance of the beam after its curvature has been removed.

    Returns:
        var_q (scalar): angular variance of the beam after the lens.
    """
    var_q = var_q_min + 4*phi_c**2/var_r
    return var_q


def calc_minimum_angular_variance_1d(var_r, phi_c, var_q):
    """Calculate minimum possible angular variance of a beam achievable with a correction lens.

    Args:
        var_r (scalar): real space variance.
        phi_c (scalar): real-space curvature - see above.
        var_q (scalar): angular variance of the beam.

    Returns:
        var_q_min (scalar): minimum possible angular variance of the beam.
    """
    var_q_min = var_q - 4*phi_c**2/var_r
    return var_q_min


def calc_propagated_variance_1d(k, var_r0, phi_c0, var_q, z):
    """Calculate real space variance of a beam after propagation.

    Args:
        k: Wavenumber.
        var_r0: Real-space variance at z = 0.
        var_q: Angular variance.
        phi_c0: Effective wavefront curvature at z = 0 - equal to phase of quadratic phase at RMS radius. Positive means
            diverging.
        z: Propagation distance.

    Returns:
        var_rz: Real-space variance at z = z.
        phi_cz: Effective wavefront curvature at z.
        rocz: Corresponding radius of curvature at z.
    """
    zw, var_rw, Msqd, z_R = calc_waist(k, var_r0, phi_c0, var_q)

    # var_rz = var_r0 + 4*phi_c0*z/k + (z/k)**2*var_q
    var_rz = var_rw*(1 + ((z - zw)/z_R)**2)
    rocz = (z - zw) + z_R**2/(z - zw)
    phi_cz = k*var_rz/(2*rocz)
    return var_rz, phi_cz, rocz

def calc_sziklas_siegman_roc(k, var_r0, phi_c0, var_q, z):
    """Calculate appropriate input radius of curvature for Sziklas-Siegman propagation.

    If the beam is at the waist, then ROC will be infinity.

    Args:
        k: Wavenumber.
        var_r0: Real-space variance of input.
        phi_c0: Phase of effective wavefront curvature at input at RMS radius.
        var_q: Transverse wavenumber space variance at input.
        z: Distance to propagate.

    Returns:
        roc: Radius of curvature. The required magnification is z/roc + 1.
    """
    # Calculate properties of waist.
    zw, var_rw, Msqd, z_R = calc_waist(k, var_r0, phi_c0, var_q)
    return calc_sziklas_siegman_roc_from_waist(z_R, zw, z)

def calc_sziklas_siegman_roc_from_waist(z_R, zw, z):
    """Get ROC for Sziklas-Siegman propagation given location of waist and Rayleigh range.

    Propagation is from the origin to z.

    Args:
        z_R: Rayleigh range.
        zw: Location of waist (relative to origin).
        z: Distance to propagate.

    Returns:
        radius of curvature
    """
    z_R, zw, z = np.broadcast_arrays(z_R, zw, z)
    # See p131 Dane's logbook p131.
    m = ((z_R**2 + (z - zw)**2)/(z_R**2 + zw**2))**0.5
    m_eq_1 = np.isclose(m, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        roc = np.asarray(z/(m - 1)) # Need roc to be an array for next line to work.
        roc[m_eq_1] = (-(z_R**2 + zw**2)/zw)[m_eq_1]
    return roc

def calc_waist(k, var_r, phi_c, var_q):
    """Caclulate waist position, waist size, beam quality facator and Rayleigh range."""
    z0 = -2*phi_c*k/var_q
    var_r0 = var_r - 4*phi_c**2/var_q
    Msqd = 2*(var_r0*var_q)**0.5
    z_R = 2*k*var_r0/Msqd
    return z0, var_r0, Msqd, z_R
