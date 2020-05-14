"""Miscellaneous optics-specific functions.

TODO better distinguishing and handling of (i) numba not installed and (ii) numba not installed but not wanted due to
compilation time. I don't remember why I used numba.vectorize for some routines below... maybe can replace with numba.njit
everywhere? Could also make a decorator that applies njit if numba is installed.

TODO move stuff in mathx.
"""
import cmath
from typing import Tuple, Sequence, Callable
import numpy as np
from otk import v4hb, v4h

try:
    import numba
except ImportError:
    numba = None

from .types import Sequence3, Vector3, Vectors4, Scalars, Matrix4

def abs_sqd(x):
    """Element-wise absolute value squared."""
    return x.real**2 + x.imag**2

if numba is None:
    def dot(a: Sequence, b: Sequence):
        """Dot product with second argument conjugated."""
        return np.dot(a, np.conj(b))

    def norm_squared(x: Sequence):
        return dot(x, x)

    def norm(x: Sequence):
        return dot(x, x)**0.5

    def normalize(x: Sequence):
        return np.asarray(x)/norm(x)
else:
    abs_sqd = numba.vectorize(cache=True)(abs_sqd)

    @numba.njit
    def dot(x: Sequence, y: Sequence):
        """Dot product with second argument conjugated."""
        s = 0.
        for i in range(len(x)):
            s += x[i]*np.conj(y[i])
        return s

    @numba.njit
    def norm_squared(x):
        return dot(x, x)


    @numba.njit
    def norm(x):
        return norm_squared(x)**0.5


    @numba.njit
    def normalize(x):
        return np.asarray(x)/norm(x)

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


    @numba.vectorize([numba.float64(numba.complex128, numba.complex128, numba.complex128, numba.float64)],
        nopython=True)
    def unwrap_quadratic_phase(a, b, c, x):
        """Evaluate phase of quadratic on same Riemann surface as constant term.

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


def intersect_planes(n1: Sequence3, p1: Sequence3, n2: Sequence3, p2: Sequence3) -> Vector3:
    """Calculate a point common to two planes.

    Each plane is defined by dot(n1, (x - p1)) = 0.

    Broadcasting not supported.

    Args:
        n1: Normal vector of plane 1.
        p1: Point on plane 1.
        n2: Normal vector of plane 1.
        p2: Point on plane 1.

    Returns:
        A point on both planes. TODO how is defined?
    """
    n1 = np.asarray(n1, float)
    n2 = np.asarray(n2, float)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    a = (((dot(n1, n1), dot(n1, n2))), (dot(n2, n1), dot(n2, n2)))
    b = dot(n1, p1), dot(n2, p2)
    c1, c2 = np.linalg.solve(a, b)
    point = c1*n1 + c2*n2
    return point


def intersect_spherical_surface(ox, oy, oz, vx, vy, vz, roc):
    """Intersect rt(s) with a spherical surface.

    The surface is tangent to the z = 0 plane. The center of sphere is (0, 0, roc). So roc > 0 means convex as seen
    from the negative z side.

    Args:
        ox, oy, oz: Ray origin coordinates.
        vx, vy, vz: Ray direction coordinates.
        roc (scalar): radius of curvature.

    Returns:
        Distance to intersection in units of (vx, vy, vz).
    """
    a = vx**2 + vy**2 + vz**2
    b = 2*(vx*ox + vy*oy + vz*(oz - roc))
    c = ox**2 + oy**2 + oz*(oz - 2*roc)
    determinant = b**2 - 4*a*c
    root_determinant = np.maximum(determinant, 0)**0.5
    d1 = (-b + root_determinant)/(2*a)
    d2 = (-b - root_determinant)/(2*a)
    iz1 = oz + vz*d1
    iz2 = oz + vz*d2
    # Pick the solution with the smallest |z|.
    d = np.choose(abs(iz2) < abs(iz1), (d1, d2))
    d = np.asarray(d)
    d[determinant < 0] = np.nan
    return d


def refract_vector(incident: Vectors4, normal: Vectors4, n_ratio: Scalars) -> Vectors4:
    """Calculate refracted wave vector given incident, normal and ratio of refractive indices.

    Broadcasting fully supported.

    Surface normal must be normalized.

    Result has length n_ratio times that of incident vector.
    """
    incident_normal_component = v4hb.dot(incident, normal)
    projected = incident - normal*incident_normal_component
    refracted_normal_component = np.sign(incident_normal_component)*(n_ratio**2*v4hb.dot(incident) - v4hb.dot(projected))**0.5
    refracted = refracted_normal_component*normal + projected
    return refracted


def reflect_vector(incident: Vectors4, normal: Vectors4) -> Vectors4:
    """Reflect incident vector given mirror normal.

    Mirror normal must be normalized.
    """
    incident_normal_component = v4hb.dot(incident, normal)
    reflected = incident - 2*normal*incident_normal_component
    return reflected


def calc_mirror_matrix(matrix: Matrix4) -> Matrix4:
    """Calculate matrix which reflects about local z=0 plane given local-to-parent matrix."""
    return np.linalg.inv(matrix) @ np.diag((1, 1, -1, 1)) @ matrix
#    return np.matmul(np.linalg.inv(matrix), np.matmul(np.diag((1, 1, -1, 1)), matrix))


def make_perpendicular(u: Vectors4, v: Vectors4) -> Vectors4:
    """Make unit vector perpendicular to a pair of unit vectors, handling degeneracy and broadcasting."""
    w = v4hb.cross(u, v)
    m = v4hb.dot(w)
    zero = np.isclose(m[...,0], 0, atol=1e-15)

    uzero = u[zero]
    if uzero.size > 0:
        # Degenerate case (u & v parallel). Calculate cross product of u with each unit vector.
        uzerocs = [v4hb.cross(uzero, chat) for chat in v4h.unit_vectors]
        mzerocs = [v4hb.dot(v) for v in uzerocs]
        mzeros = np.concatenate(mzerocs, -1)
        index = mzeros.argmax(-1)[..., None]

        wzero = np.choose(index, uzerocs)
        mzero = np.choose(index, mzerocs)

        w[zero] = wzero
        m[zero] = mzero

    w /= m**0.5

    return w

def calc_zemax_conic_lipschitz(radius: float, roc: float, kappa: float = 1., alphas: Sequence[float] = ()):
    # Compute bound on second derivative of sag.
    ns = np.arange(2, len(alphas) + 2)
    l1 = sum(abs(alpha)*n*(n - 1)*radius**(n - 2) for n, alpha in zip(ns, alphas))
    if np.isfinite(roc):
        l1 += roc**2/(roc**2 - kappa*radius**2)**1.5
    def absf1(rho):
        return abs(rho/(roc**2 - kappa*rho**2)**0.5 + sum(alpha*n*rho**(n - 1) for n, alpha in zip(ns, alphas)))
    return bound_upper1d(absf1, l1, 0., radius, 2.)


def bound_upper1d(f: Callable[[float], float], lipschitz: float, a: float, b: float, alpha: float = 2., beta: float = 1e-3) -> float:
    """Get upper bound on f(x) for x in [a, b] given Lipschitz constant.

    The returned bound is no looser than max(alpha*max(f), beta*lipschitz*(b - a)).
    """
    assert lipschitz >= 0
    assert alpha > 1.
    assert beta > 0.
    x = a
    fx = f(x)
    assert fx >= 0
    bound = max(alpha*fx, beta*lipschitz*(b - a))
    while x < b:
        # Take maximum step such that Lipschitz condition doesn't increase bound.
        dx = (bound - fx)/lipschitz
        x = min(b, x + dx) # Want x==b exactly if at end of domain.
        fx = f(x)
        assert fx >= 0
        bound = max(fx*alpha, bound)
    return bound