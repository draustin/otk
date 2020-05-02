"""3D geometry functions building on v4b.py.

TODO move to functions
"""
import numpy as np
from . import v4h, v4hb
# TODO distinguish between broadcasted & not versions
from .v4hb import dot, normalize, to_xy

__all__ = ['refract_vector', 'reflect_vector', 'make_perpendicular']

class NoIntersectionError(Exception):
    pass

def intersect_planes(n1, p1, n2, p2):
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
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    a = ((np.asscalar(dot(n1, n1)), np.asscalar(dot(n1, n2))), (np.asscalar(dot(n2, n1)), np.asscalar(dot(n2, n2))))
    b = np.asscalar(dot(n1, p1)), np.asscalar(dot(n2, p2))
    c1, c2 = np.linalg.solve(a, b)
    point = [0, 0, 0, 1] + c1*n1 + c2*n2
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


def refract_vector(incident, normal, n_ratio):
    """Calculate refracted wave vector.

    TODO complex refractive indices?

    Args:
        incident (...x4 array): Incident vector.
        normal (...x4 array): Surface normal (normalized).

    Returns:
        ...x4 array: Refracted vector, with length n_ratio times that of incident vector.
    """
    incident_normal_component = dot(incident, normal)
    projected = incident - normal*incident_normal_component
    refracted_normal_component = np.sign(incident_normal_component)*(n_ratio**2*dot(incident) - dot(projected))**0.5
    refracted = refracted_normal_component*normal + projected
    #refracted = normalize(refracted_unnorm)
    return refracted


def reflect_vector(incident, normal):
    """Reflect incident vector given mirror normal.

    Args:
        incident (...x4 array): Incident vector(s).
        normal (...x4 array): Normal vector(s).

    Returns:
        ...x4 array: Reflected vector(s).
    """
    incident_normal_component = dot(incident, normal)
    reflected = incident - 2*normal*incident_normal_component
    return reflected


# def apply_4x4_point(matrix, point):
#     assert len(point) in (3, 4)
#     assert np.ndim(point) in (1, 2)
#     if len(point) == 3:
#         if np.ndim(point) == 1:
#             point = point[0], point[1], point[2], 1
#         else:
#             point = np.concatenate((point, np.ones((1, np.shape(point)[1]))), 0)
#     return np.matmul(matrix, point)




def calc_mirror_matrix(matrix):
    return np.matmul(np.linalg.inv(matrix), np.matmul(np.diag((1, 1, -1, 1)), matrix))


def make_perpendicular(u, v):
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