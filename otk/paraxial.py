"""Basic paraxial optics calculations."""
from typing import Tuple
import numpy as np
from . import abcd, functions

def calc_image_distance(object_distance, focal_length):
    """Calculate image distance using imaging equation."""
    return 1/(1/focal_length - 1/object_distance)


def calc_combined_lens(f1, f2, d):
    """Calculate focal length and principal planes of a compound lens.

    Args:
        f1: focal length of first lens
        f2: focal length of second lens
        d: distance

    Returns:
        f: effective focal length of combined lens
        pp1: distance from first lens to principal plane. Positive is away from second lens.
        pp2: distance from second lens to principal plane. Positive is away from first lens.
    """
    f = 1/(1/f1 + 1/f2 - d/(f1*f2))
    pp1 = d*f1/(d - f1 - f2)
    pp2 = d*f2/(d - f1 - f2)
    return f, pp1, pp2


def infer_combined_lens_separation(f1, f2, f):
    """Infer the separation of a combined lens from the individual focal lengths and the effective focal length.

    Args:
        f1: focal length of first lens
        f2: focal length of second lens
        f: effective focal length

    Returns:
        separation between lenses
    """
    # Dane's logbook #1,  page 92
    return f1 + f2 - f1*f2/f


def calc_object_image_distances(f, m):
    """

    Args:
        f: focal length
        m: magnification

    Returns:
        u: object distance
        v: image distance
    """
    u = f*(1 - 1/m)
    v = -m*u
    return u, v


def design_thick_spherical_transform_lens(n, w, f):
    """Choose radius of curvature and thickness for a Fourier transform with given focal length and working distance.

    Args:
        n: refractive index
        w: working distance
        f: transform focal length

    Returns:
        roc: radius of curvature
        d: center thickness
    """
    # Derivation p68 Dane's Fathom logbook #2
    roc = f*(n - 1) + w*(n - 1)
    d = (f - w)*n*roc/(f*(n - 1))
    return roc, d

def design_singlet_transform(n: float, ws: Tuple[float, float], f: float, ne:float=1):
    """Choose curvature and thickness for a singlet Fourier transform given focal length and working distances.

    Args:
        n: Refractive index
        ws: Working distances - distance from focal plane to glass.
        f: Transform focal length.

    Returns:
        rocs (2-tuple): Radii of curvature. Biconvex means rocs[0] > 0 and rocs[1] < 0.
        d: Center thickness.
    """
    # Derivation p1 Dane's notebook 3.
    w1, w2 = ws
    h1 = f - w1
    h2 = w2 - f
    d = n*(h1 - h2 + h1*h2/f)
    roc1 = np.divide(d*f*(n - ne), -h2*n)
    roc2 = np.divide(d*f*(n - ne), -h1*n)
    rocs = roc1, roc2
    return rocs, d

def design_singlet(n: float, f: float, s: float, d: float, ne: float=1):
    """Design a singlet given refractive index, focal length, shape factor, and center thickness.

    Not vectorized.

    Args:
        n: Refractive index.
        f: Effective focal length - distance to focus of parallel incident rays, divided by external refractive index.
        s: Coddington shape factor. 0 = symmetric. +1 is convex-plano, -1 is plano-convex.
        d: Center thickness.
        ne: External refractive index.

    Returns:
        Radii of curvature of surfaces.
    """
    # Derivation page 1 of Dane's logbook 3, generalized to nonunity external refractive index on p5.
    if np.isclose(s, -1):
        swap = True
        s = -s
    else:
        swap = False

    a = n*(s + 1)/f
    b = -2*n*(n - ne)
    c = -(n - ne)**2*d*(s - 1)

    r1 = np.divide(-b + (b**2 - 4*a*c)**0.5, 2*a)
    r2 = np.divide(r1*(s + 1), s - 1)

    if swap:
        r1, r2 = -r2, -r1

    return r1, r2

def design_spherical_inverse_lens(n, w, f, thickness):
    """Design a spherical inverse lens with given working distance and thickness.

    An inverse lens is (defined here as) a compound lens consisting of two plano-convex lenses with the plane sides
    facing out.

    Args:
        n: Refractive index.
        w: Working distance - the distance (in air) from a plane side to the focal plane.
        f: Focal length in air.
        thickness: Center thickness of one element.

    Returns:
        roc: Radius of curvature of inner focal_surfaces. Positive for positive f.
        d: Length of air gap between (inner) vertices.
    """
    wp = thickness + w*n
    roc, d = design_thick_spherical_transform_lens(1/n, wp, f)
    return -roc, d

def design_symmetric_compound_lens(f, w):
    """Design a compound lens with a given focal length and working distance.

    Args:
       f: Desired focal length.
       w: Desired working distance, defined here as distance from focal planes to the outer principal planes of the singlets.

    Returns:
        fl: Focal length of singlet.
        d: Separation of inner principal planes of singlets.
    """
    # This turns out to be true... I derived it algebraically. There should be geometric argument but I can't think of one
    # right now.
    fl = w + f
    d = (f**2 - w**2)/f
    return fl, d

def calc_thick_spherical_lens(n, roc1, roc2, d, ne=1):
    """Calculate the properties of a thick spherical lens.

    Positive radius of curvature means center is further along the direction of travel. For a biconvex positive lens,
    roc1>0,  roc2<0. The principal plane distances are from the vertex to the principal plane. For the first principal plane,
    the positive means moving into the lens. For the second,  positive means moving away from the lens. So for a biconvex
    lens, h1>0 and h2<0. The working distances are f - h1 and f + h2.

    For a nice diagram see e.g. http://www.physics.purdue.edu/~jones105/phys42200_Spring2013/notes/Phys42200_Lecture32.pdf

    Args:
        n: refractive index
        roc1: radius of curvature of first surface
        roc2: radius of curvature of second surface
        d: center thickness of the lens
        ne: External refractive index.

    Returns:
        f: focal length
        h1: principal plane position 1
        h2: principal plane position 2
    """
    # Focal length.
    f = 1/((n - ne)*(1/roc1 - 1/roc2 + (n - ne)*d/(n*roc1*roc2)))
    # Distance to first principal plane from input vertex.
    h1 = -f*(n - ne)*d/(n*roc2)
    # Distance to second principal plane from output vertex.
    h2 = -f*(n - ne)*d/(n*roc1)
    return f, h1, h2



def infer_spherical_lens_thickness(n, f, roc1, roc2):
    d = (1/(f*(n-1)) - 1/roc1 + 1/roc2)*n*roc1*roc2/(n - 1)
    return d

def calc_defocus_object_image_distances(subarray_pitch, mla_pitch, f_mla):
    """Compute object distance for quasi-plane waves incident upon MLA to give required subarray pitch.

    No defocus would mean infinite object distance. Positive object distance means diverging wavefronts.

    Args:
        subarray_pitch (scalar): distance between subarrays on the dense modulator
        mla_pitch (scalar): pitch of microlens array
        f_mla (scalar): effective focal length of microlens array (pair or bulk)

    Returns:
        mla_object_distance (scalar): required object distance
        mla_image_distance (scalar): required image distance
    """
    # When we move the object of a given MLA pair lens by a MLA pitch in one direction,
    # the image should move by the difference between our coarse spacing and the MLA pitch in the other direction.
    mla_magnification = -(subarray_pitch - mla_pitch)/mla_pitch
    mla_object_distance, mla_image_distance = calc_object_image_distances(f_mla, mla_magnification)
    return mla_object_distance, mla_image_distance

def calc_multi_interface_abcd(ns, rocs, ds):
    assert len(ns) == len(rocs) + 1
    assert len(ds) == len(rocs) - 1
    m = abcd.curved_interface(ns[0], ns[1], rocs[0])
    for n1, n2, roc, d in zip(ns[1:], ns[2:], rocs[1:], ds):
        # Propagate to next surface.
        m = np.matmul(abcd.propagation(d), m)
        # Apply interface.
        m = np.matmul(abcd.curved_interface(n1, n2, roc), m)
    return m

def calc_multi_element_lens_abcd(ns, rocs, ds):
    assert len(ns) == len(rocs) + 1
    assert len(ds) == len(rocs) + 1
    m = abcd.propagation(ds[0])
    for n1, n2, roc, d in zip(ns[:-1], ns[1:], rocs, ds[1:]):
        # Apply interface.
        m = np.matmul(abcd.curved_interface(n1, n2, roc), m)
        # Propagate to next surface (or exit surface).
        m = np.matmul(abcd.propagation(d), m)
    return m

def calc_multi_element_lens_petzval_sum(ns, rocs):
    """Calculate Petzval sum of a multi-element lens.

    Args:
        ns (sequence): Refractive indices of all media.
        rocs (sequence): Length is one less than ns.

    Returns:
        Petzval sum.
    """
    assert len(ns) == len(rocs) + 1
    ps = 0
    for n1, n2, roc in zip(ns[:-1], ns[1:], rocs):
        ps += (n2 - n1)/(roc*n1*n2)
    return ps

def print_multi_element_lens(ns, rocs, ds):
    print('n thickness (mm) ROC (mm)')
    print('%.3f %.3f -'%(ns[0], ds[0]*1e3))
    for n, roc, d in zip(ns[1:], rocs, ds[1:]):
        print('%.3f %.3f %.3f'%(n, d*1e3, roc*1e3))

def calc_rocs(curvature, shape_factor):
    """Convert lens surface curvature and shape factor to ROCs.

    See Dane's logbook 2 p143.

    Args:
        curvature: Difference of curvatures i.e 1/ROC1 - 1/ROC2. Positive = converging.
        shape_factor: Coddington shape factor. 0 = symmetric. +1 is convex-plano, -1 is plano-convex.

    Returns:
        roc1, roc2
    """
    #ratio =  (shape_factor + 1)/(shape_factor - 1) # Ratio of roc2 to roc1.
    #roc1 = (1 - 1/ratio)/curvature
    #roc2 = (ratio - 1)/curvature
    with np.errstate(divide='ignore'):
        roc1 = np.divide(2., curvature*(shape_factor + 1))
        roc2 = np.divide(2, curvature*(shape_factor - 1))
    return roc1, roc2

def calc_center_thickness(rocs: tuple, radius: float, min_thickness: float):
    """Calc. center thickness of lens given ROCs, radius, and minimum thickness."""
    sags = tuple(functions.calc_sphere_sag(roc, radius) for roc in rocs)
    return max(sags[0] - sags[1], 0) + min_thickness

