import math
from functools import singledispatch
from typing import Sequence, Tuple
import numpy as np
import mathx
import otk.functions

from .. import functions
from .. import v4hb
from .. import trains
from . import boundaries

class Profile:
    def intersect_line(self, line):
        d = self.intersect(line.origin, line.vector)
        return line.advance(d)

    def intersect(self, origin: np.ndarray, vector: np.ndarray) -> float:
        raise NotImplementedError()

    def calc_normal(self, point):
        raise NotImplementedError()

    def is_planar(self):
        """If True, then profile must be local z=0."""
        return False

    def calc_z(self, x, y):
        """Calculate surface height (z).

        Default implementation uses intersect method, but subclasses can redefine with optimization.

        Returns:
            array of shape of np.broadcast(x, y).
        """
        z = self.intersect(v4hb.stack_xyzw(x, y, 0, 1), np.asarray([0, 0, 1, 0]))[..., 0]
        assert z.shape == np.broadcast(x, y).shape
        return z


class PlanarProfile(Profile):
    """A flat plane surface profile. """

    def __str__(self):
        return 'planar'

    def __repr__(self):
        return 'PlanarProfile()'

    def intersect(self, origin, vector):
        """Intersect rt specified in local frame with surface.

        The arguments must be mutually broadcastable,  but can otherwise be any shape.

        Args:
            origin (...x4 array): Ray origin point.
            vector (...x4 array): Ray direction.

        Returns:
            (...x1 array): Number of vectors to intersection.

        It is the responsibility of higher level to ensure that only one intersection occurs in the physical system. For
        a lens array this means the angle between incoming rays and surface normal does not exceed 90 degrees.
        """
        return -origin[..., [2]]/vector[..., [2]]

    def calc_normal(self, point):
        """Calculate normal vector in local coordinates.

        Args:
            point ((...,4) array): Point at which to calculate normal in local coordinates.

        Returns:
            (...,4) array: Normal vector. Always has positive z component in local coordinates.
        """
        # Since the normal is independent of point I tried returning an array of shape (4), but its convenient elsewhere
        # to assume that normal has the same shape as point.
        normal = np.broadcast_to([0, 0, 1, 0], point.shape)
        return normal

    def is_planar(self):
        return True

    def calc_z(self, x, y):
        return np.zeros(np.broadcast(x, y).shape)


class SphericalProfile(Profile):
    """A spherical surface profile, tangent at the origin to the z = 0 plane (in the local frame).

    The center of curvature is at (0, 0, roc) in local coordinates. So positive roc means a convex surface as seen from
    the negative z side.
    """

    def __init__(self, roc):
        self.roc = roc

    def __repr__(self):
        return 'SphericalProfile(roc=%.3f mm)'%(self.roc*1e3)

    def intersect(self, origin, vector):
        """Intersect rt specified in local frame with surface.

        Args:
            origin (...x4 array): Ray origin point.
            vector (...x4 array: Ray direction.

        Returns:
            d (...x1 array): Number of directions to intersection.
        """
        d = \
        otk.functions.intersect_spherical_surface(origin[..., 0], origin[..., 1], origin[..., 2], vector[..., 0], vector[..., 1],
            vector[..., 2], self.roc)[..., None]
        return d

    def calc_z(self, x, y):
        return functions.calc_sphere_sag(self.roc, (x ** 2 + y ** 2) ** 0.5)

    def calc_normal(self, point):
        """Calculate surface normal.

        Args:
            point (...x4 array): Point to calculate normal at. Must lie on surface.

        Returns:
            normal (...x4 array): Normal vector. Has positive z component.
        """
        normal = -point/self.roc
        normal[..., 2] += 1
        normal[..., 3] = 0
        return normal  # nx = -point[..., 0]/self.roc  # ny = -point[..., 1]/self.roc  # nz = (self.roc - point[..., 2])/self.roc  # return v4hb.stack_xyzw(nx, ny, nz, 0)

    def __str__(self):
        return 'spherical with ROC = %.3f mm'%(self.roc*1e3)

class BinaryProfile(Profile):
    """profiles[0] for points in boundary. Otherwise profiles[1]."""
    def __init__(self, profiles:Tuple[Profile, Profile], boundary:boundaries.Boundary):
        self.profiles = profiles
        self.boundary = boundary

    def __repr__(self):
        return 'BinaryProfile(%r, %r)'%(self.profiles, self.boundary)

    def intersect(self, origin, vector):
        ds = [p.intersect(origin, vector) for p in self.profiles]
        point0 = origin + ds[0]*vector
        is_inside0 = self.boundary.is_inside(point0[...,[0]],point0[...,[1]])
        # If inside, return ds[0]. Otherwise return ds[1].
        d = np.where(is_inside0, *ds)
        return d

    def calc_normal(self, point):
        is_inside0 = self.boundary.is_inside(point[...,[0]],point[...,[1]])
        normals = [p.calc_normal(point) for p in self.profiles]
        normal = np.where(is_inside0, *normals)
        return normal


class ArbitraryRotationallySymmetricProfile(Profile):
    def __init__(self):
        self.max_num_iterations = 100
        self.intersect_tolerance = 1e-10

    def intersect(self, origin, vector):
        """Intersect rt with surface.

        Args:
            origin (...x4 array): Starting point(s).
            vector (...x4 array): Normalized rt vectors.

        Returns:
            ...x1 array: Distance(s) to intersection.
        """
        # Perform an approximate intersection as a starting guess.
        d = self.intersect_approx(origin, vector)
        num_iterations = 0
        while num_iterations < self.max_num_iterations:
            # Calculate current trial intersection point.
            point = origin + d*vector
            # Compute radial distance.
            rho = v4hb.dot(point[..., :2]) ** 0.5
            # Calculate derivative of radial distance with respect to d.
            drhodd = 2*(v4hb.dot(point[..., :2], vector[..., :2]) + d*v4hb.dot(vector[..., :2]))
            # drhodd = 2*((point[..., 0]*vector[..., 0] + point[..., 1]*vector[..., 1]) +
            #            d*(vector[..., 0]**2 + vector[..., 1]**2))
            # Error is point z minus surface height.
            h = point[..., [2]] - self.calc_z(rho)
            # Terminate based on worst case.
            max_error = np.max(abs(h))
            if max_error <= self.intersect_tolerance:
                break
            # Calculate derivative of error w.r.t d.
            dhdd = vector[..., [2]] - self.calc_dzdrho(rho)*drhodd
            # Perform one step of Newton's method.
            d += -h/dhdd
            num_iterations += 1
        if max_error > self.intersect_tolerance:
            raise ValueError('Intersection error %g did not reach tolerance %g in %d iterations.'%(
            max_error, self.intersect_tolerance, num_iterations))
        return d

    def calc_normal(self, point):
        rho = (point[..., 0] ** 2 + point[..., 1] ** 2) ** 0.5
        dzdrho = self.calc_dzdrho(rho)
        factor = mathx.divide0(dzdrho, rho, 0)
        return v4hb.normalize(v4hb.stack_xyzw(-point[..., 0]*factor, -point[..., 1]*factor, 1,
            0))  # return v4hb.normalize(v4hb.stack_xyzw(-point[0]*factor, -point[1]*factor, 1, 0))

    def calc_dzdrho(self, rho:float) -> float:
        raise NotImplementedError()

    def intersect_approx(self, origin, vector) -> float:
        raise NotImplementedError()

class ConicProfile(ArbitraryRotationallySymmetricProfile):
    def __init__(self, roc, kappa=0, alphas=None):
        """Setup with given radius of curvature, conic constant and higher order terms.

        The sag is given by calc_z.

        We take kappa as defined in Spencer and Murty, JOSA 52(6) p 672, 1962. Note that in some other contexts,
        k = kappa - 1 is used as the conic constant. This is the case in Zemax i.e. kappa here equals Zemax conic constant
        plus 1.

        Useful links:
        https://www.iap.uni-jena.de/iapmedia/de/Lecture/Advanced+Lens+Design1393542000/ALD13_Advanced+Lens+Design+7+_+Aspheres+and+freeforms.pdf

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
        ArbitraryRotationallySymmetricProfile.__init__(self)
        self.roc = roc
        self.kappa = kappa
        if alphas is None:
            alphas = []
        self.alphas = np.asarray(alphas)

    def __repr__(self):
        return 'ConicProfile(roc=%r, kappa=%r, alphas=%r)'%(self.roc, self.kappa, self.alphas)

    def calc_z(self, rho):
        return rho ** 2/(self.roc*(1 + (1 - self.kappa*(rho/self.roc) ** 2) ** 0.5)) + np.polyval(self.alphas[::-1],
            rho)*rho ** 2

    def calc_dzdrho(self, rho):
        alphaps = np.arange(2, 2 + len(self.alphas))*self.alphas
        return rho/(self.roc*(1 - self.kappa*(rho/self.roc) ** 2) ** 0.5) + np.polyval(alphaps[::-1], rho)*rho

    def intersect_approx(self, origin, vector):
        """

        Args:
            origin (...x4 array): Ray starting point.
            vector (...x4 array): Ray vector.

        Returns:
            ...x1 array: Distance to intersection.
        """
        if np.isfinite(self.roc):
            # Starting guess is sphere.
            d = otk.functions.intersect_spherical_surface(origin[..., 0], origin[..., 1], origin[..., 2], vector[..., 0],
                vector[..., 1], vector[..., 2], self.roc)[..., None]
        else:
            # Starting guess is plane.
            d = -origin[..., [2]]/vector[..., [2]]

        return d


# def __str__(self):
#     return 'SphericalProfile(roc = %.3f mm,  vertex = (%.3f, %.3f, %.3f) mm,  normal = (%.3f, %.3f, %.3f))'%(
#         self.roc*1e3, *self.matrix[:3, 3]*1e3, *self.matrix[:3, 2])

def make_spherical_profile(roc, kappa=1, alphas=()):
    """If ROC is infinity, returns planar profile. Otherwise returns SphericalProfile."""
    if np.isfinite(roc):
        if kappa == 1 and np.allclose(alphas, 0):
            return SphericalProfile(roc)
        else:
            return ConicProfile(roc, kappa, alphas)
    else:
        assert np.allclose(alphas,
            0), 'Aspheric terms with infinite ROC not allowed (no means of calculating starting guess).'
        return PlanarProfile()

# class RegularSquareLatticeLattice(SquareLattice):
#     """Regular square lattice of regular square lattices"""
#
#     def __init__(self, fine_pitch: float, fine_size: int, coarse_pitch: float, coarse_size: int):
#         SquareLattice.__init__(self, fine_pitch, coarse_pitch*coarse_size, fine_size*coarse_size)
#         self.fine_pitch = fine_pitch
#         self.fine_size = fine_size
#         self.coarse_pitch = coarse_pitch
#         self.coarse_size = coarse_size
#         self.fine_lattice = RegularSquareLattice(fine_pitch, fine_size)
#         self.coarse_lattice = RegularSquareLattice(coarse_pitch, coarse_size)
#
#     def __repr__(self):
#         return '%s(fine_pitch=%r, fine_size=%r, coarse_pitch=%r, coarse_size=%r)'%(
#         type(self).__name__, self.fine_pitch, self.fine_size, self.coarse_pitch, self.coarse_size)
#
#     def split_indices(self, ix, iy):
#         cix = np.floor(ix/self.fine_size).astype(int)
#         ciy = np.floor(iy/self.fine_size).astype(int)
#         fix = ix - cix*self.fine_size
#         fiy = iy - ciy*self.fine_size
#         return cix, ciy, fix, fiy
#
#     def combine_indices(self, cix, ciy, fix, fiy):
#         """Inverse function of split indices."""
#         ix = cix*self.fine_lattice.size + fix
#         iy = ciy*self.fine_lattice.size + fiy
#         return ix, iy
#
#     def calc_point(self, ix, iy):
#         ix = np.asarray(ix)
#         iy = np.asarray(iy)
#         cix, ciy, fix, fiy = self.split_indices(ix, iy)
#         xc, yc = self.coarse_lattice.calc_point(cix, ciy)
#         xo, yo = self.fine_lattice.calc_point(fix, fiy)
#         return xc + xo, yc + yo
#
#     def calc_fractional_indices(self, x, y):
#         x = np.asarray(x)
#         y = np.asarray(y)
#         cix, ciy = self.coarse_lattice.calc_indices(x, y)
#         xc, yc = self.coarse_lattice.calc_point(cix, ciy)
#         xo = x - xc
#         yo = y - yc
#         fix, fiy = self.fine_lattice.calc_fractional_indices(xo, yo)
#         ix, iy = self.combine_indices(cix, ciy, fix, fiy)
#         return ix, iy
#
#     def scale(self, sx: float, sy: float = None) -> 'Lattice':
#         """Return scaled version of self."""
#         if sy is None:
#             sy = sx
#         assert sx == sy, 'Rectangular scaling not implemented yet.'
#         return RegularSquareLatticeLattice(self.fine_pitch*sx, self.fine_size, self.coarse_pitch*sx, self.coarse_size)


class LatticeProfile(Profile):
    def __init__(self, lattice, profile):
        self.lattice = lattice
        self.profile = profile

    def __repr__(self):
        return 'LatticeProfile(%r, %r)'%(self.lattice, self.profile)

    def intersect(self, origin, vector):
        ox = origin[..., 0]
        oy = origin[..., 1]
        oz = origin[..., 2]

        vx = vector[..., 0]
        vy = vector[..., 1]
        vz = vector[..., 2]

        # Only approximate - if a ray crosses a cell boundary between origin and intersection, will give error.
        d_plane = -oz/vz
        x_plane = ox + d_plane*vx
        y_plane = oy + d_plane*vy
        nx, ny = self.lattice.calc_indices(x_plane, y_plane)
        cx, cy = self.lattice.calc_point(nx, ny)

        center = v4hb.stack_xyzw(cx, cy, 0, 1)
        return self.profile.intersect(origin - center, vector)

    def __str__(self):
        return 'LatticeProfile(%s, %s)'%(self.lattice, self.profile)

    def calc_normal(self, point):
        nx, ny = self.lattice.calc_indices(point[..., 0], point[..., 1])
        cx, cy = self.lattice.calc_point(nx, ny)
        center = v4hb.stack_xyzw(cx, cy, 0, 1)
        normal = self.profile.calc_normal(point - center)
        return normal

    def is_planar(self):
        return self.profile.is_planar()

    def partition(self, origin, vector, delta):
        """

        Args:
            origin (pair of scalar):
            vector  (pair of scalar):
            delta:

        Returns:

        """
        partitions = self.lattice.partition_along_line(origin, vector)
        return self.make_sections(partitions, origin, vector, delta)

    def make_sections(self, partitions: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]], origin: np.ndarray,
            vector: np.ndarray, delta: float) -> Sequence[np.ndarray]:
        """Make one cross section for each partition.
        """
        sections = []
        for (ix, iy), (d1, d2) in partitions:
            num_points = int(math.ceil((d2 - d1)/delta))
            d = np.linspace(d1, d2, num_points)
            x, y = origin[:, None] + vector[:, None]*d
            cx, cy = self.lattice.calc_point(ix, iy)
            z = self.profile.calc_z(x - cx, y - cy)
            section = v4hb.stack_xyzw(x, y, z, 1)
            sections.append(section)
        return sections


class SquareArrayProfile(Profile):
    def __init__(self, pitch, profile):
        self.pitch = pitch
        self.profile = profile

    def intersect(self, origin, vector):
        ox = origin[..., 0]
        oy = origin[..., 1]
        oz = origin[..., 2]

        vx = vector[..., 0]
        vy = vector[..., 1]
        vz = vector[..., 2]

        d_plane = -oz/vz
        x_plane = ox + d_plane*vx
        y_plane = oy + d_plane*vy
        nx, ny = self.calc_index(x_plane, y_plane)
        cx, cy = self.calc_center(nx, ny)

        center = v4hb.stack_xyzw(cx, cy, 0, 1)
        return self.profile.intersect(origin - center, vector)

    def calc_index(self, x, y):
        nx = np.round(x/self.pitch - 0.5).astype(int)
        ny = np.round(y/self.pitch - 0.5).astype(int)
        return nx, ny

    def calc_center(self, nx, ny):
        return (nx + 0.5)*self.pitch, (ny + 0.5)*self.pitch

    def calc_normal(self, point):
        nx, ny = self.calc_index(point[..., 0], point[..., 1])
        cx, cy = self.calc_center(nx, ny)
        center = v4hb.stack_xyzw(cx, cy, 0, 1)
        normal = self.profile.calc_normal(point - center)
        return normal

    def __str__(self):
        return 'SquareArrayProfile(pitch = %.3f mm, offset = (%.3f, %.3f) mm, %s)'%(
        self.pitch*1e3, *(self.offset*1e3), self.profile)


class SphericalSquareArrayProfile(SquareArrayProfile):
    def __init__(self, roc, pitch):
        self.roc = roc
        # self.pitch = pitch
        self.edge_sag = functions.calc_sphere_sag_xy(roc, pitch/2, 0, clip=True)
        self.corner_sag = functions.calc_sphere_sag_xy(roc, pitch/2, pitch/2, clip=True)
        profile = SphericalProfile(roc)
        SquareArrayProfile.__init__(self, pitch, profile)

    # def intersect(self, origin, vector):
    #     ox = origin[..., 0]
    #     oy = origin[..., 1]
    #     oz = origin[..., 2]
    #
    #     vx = vector[..., 0]
    #     vy = vector[..., 1]
    #     vz = vector[..., 2]
    #
    #     d_plane = -oz/vz
    #     x_plane = ox + d_plane*vx
    #     y_plane = oy + d_plane*vy
    #     nx = np.floor(x_plane/self.pitch)
    #     ny = np.floor(y_plane/self.pitch)
    #     cx, cy = self.calc_center(nx, ny)
    #     return v4hb.intersect_spherical_surface(ox - cx, oy - cy, oz, vx, vy, vz, self.roc)[..., None]
    #
    # def calc_center(self, nx, ny):
    #     return (nx + 0.5)*self.pitch, (ny + 0.5)*self.pitch
    #
    # def calc_normal(self, point):
    #     nx = np.floor(point[..., 0]/self.pitch)
    #     ny = np.floor(point[..., 1]/self.pitch)
    #     cx, cy = self.calc_center(nx, ny)
    #     normal_x = -(point[..., 0] - cx)/self.roc
    #     normal_y = -(point[..., 1] - cy)/self.roc
    #     normal_z = (self.roc - point[..., 2])/self.roc
    #     return v4hb.stack_xyzw(normal_x, normal_y, normal_z, 0)

    def __str__(self):
        return 'SphericalSquareArrayProfile(roc = %.3f mm,  pitch = %.3f mm)'%(self.roc*1e3, self.pitch*1e3)

@singledispatch
def to_profile(obj) -> Profile:
    raise NotImplementedError(obj)

@to_profile.register
def _(self: trains.Interface):
    if np.isfinite(self.roc):
        return SphericalProfile(self.roc)
    else:
        return PlanarProfile()

@to_profile.register
def _(self: trains.ConicInterface):
    return ConicProfile(self.roc, self.kappa, self.alphas)