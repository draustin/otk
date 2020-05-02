from typing import Iterable, Sequence, List

import otk.functions
import otk.h4t
import otk.rt1.lines
import scipy.optimize
import numpy as np
from .. import ri
from . import profiles, boundaries, interfaces
from .. import v4hb, trains
import mathx
from .lines import Line


class MutableTransformable:
    def translate(self, x, y, z, frame='global'):
        self.transform(otk.h4t.make_translation(x, y, z), frame)

    def scale(self, x, y, z, frame='global'):
        self.transform(otk.h4t.make_scaling(x, y, z), frame)

    def rotate_y(self, theta, frame='global'):
        """Rotate around y axis.

        Positive theta is given by right hand rule (thumb points along y, rotation is fingers)."""
        self.transform(otk.h4t.make_y_rotation(theta), frame)

    def translate_z(self, z, frame='global'):
        self.transform(otk.h4t.make_translation(0, 0, z), frame)

    def transform(self, transformation:np.ndarray, frame='global'):
        raise NotImplementedError()

class MutableTransform(MutableTransformable):
    """Mutable object with transformation matrix.

    Attributes:
        matrix (4x4 array): Applied *on the right* to coordinate row vectors.
    """

    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.eye(4)
        self.set_matrix(matrix)

    def set_matrix(self, matrix):
        # TODO maybe copy the matrix to prevent accidental (or deliberate?) overwrite?
        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(self.matrix)
        self.reflection_matrix = otk.functions.calc_mirror_matrix(
            self.matrix)  # np.matmul(self.inverse_matrix, np.matmul(np.diag((1, 1, -1, 1)), self.matrix))

    def transform(self, transformation, frame='global'):
        if frame == 'local':
            matrix = np.matmul(transformation, self.matrix)
        elif frame == 'global':
            matrix = np.matmul(self.matrix, transformation)
        else:
            raise ValueError('Unknown frame %s.'%frame)
        self.set_matrix(matrix)

    def to_global(self, r):
        return v4hb.transform(r, self.matrix)

    def to_local(self, r):
        return v4hb.transform(r, self.inverse_matrix)

class CompoundMutableTransform(MutableTransformable):
    def __init__(self, elements: Iterable[MutableTransform]):
        self.elements = elements

    def transform(self, transformation, frame:str='global'):
        for e in self.elements:
            e.transform(transformation, frame)

class Surface(MutableTransform):
    """Mutable object representing a surface in 3D space that rays can intercept rays and beams and modify them.

    All constituent objects are immutable. The surface is the "real-world" object.

    Attributes:
        boundary (Boundary): Defines edge of surface.
        interface: Defines how the surface affects light.
    """

    boundary_clip_r_support_factor = 1.05

    def __init__(self, profile: profiles.Profile, matrix: np.array = None, name='', boundary=None, mask=None,
            interface:interfaces.Interface=None):
        """A surface in 3D space.

        Args:
            matrix:
            reflects:
            boundary: A two dimensional region that specifies the extent of the surface in its local z=0 plane.
        """
        MutableTransform.__init__(self, matrix)
        self.profile = profile
        self.name = name
        if boundary is None:
            boundary = boundaries.InfiniteBoundary()
        self.boundary = boundary
        self.mask = mask
        self.interface = interface

    def __str__(self):
        return 'surface %s, %s, %s, %s, origin = (%.3f, %.3f, %.3f) mm,  normal = (%.3f, %.3f, %.3f), %s'%(
        self.name, self.profile, self.boundary, self.mask, *self.matrix[3, :3]*1e3, *self.matrix[2, :3], self.interface)

    def __repr__(self):
        return 'Surface(%r, %r, %r, %r, %r, %r)'%(
        self.profile, v4hb.repr_transform(self.matrix), self.name, self.boundary, self.mask, self.interface)

    def intersect_global_curve(self, calc_point, d0):
        """Intersect parameterized curve with surface.

        Args:
            calc_point: Accepts d, returns 4 array in global coordinates..
            d0: Starting guess.

        Returns:
            scalar: Solution.
        """

        def calc_delta_z(d):
            point_global = calc_point(d)
            point_local = self.to_local(point_global)
            delta_z = point_local[2] - self.profile.calc_z(*point_local[:2])
            return delta_z

        d = scipy.optimize.fsolve(calc_delta_z, d0)
        return d

    def intersect_global(self, line):
        """Intersect rt bundle specified in parent frame with surface.

        This is just a wrapper around intersect_local which performs the necessary coordinate transformations.

        The arguments must be mutually broadcastable,  but can otherwise be any shape.

        Args:
            line (Line): Line, specified in parent coordinates.

        Returns:
            d (...x1): number of directions to intersection
        """
        origin_local = self.to_local(line.origin)
        vector_local = self.to_local(line.vector)
        return self.profile.intersect(origin_local, vector_local)

    def intersect_other(self, matrix, origin, vector):
        origin_global = np.matmul(origin, matrix)
        vector_global = np.matmul(vector, matrix)
        bundle = Line(origin_global, vector_global)
        return self.intersect_global(bundle)

    def is_inside_global(self, point):
        """Check is point is inside boundary.

        Args:
            point (...x4 array): Point(s) to check.

        Returns:
            ...x1 arrray of bool: True if point is inside.
        """
        point_local = self.to_local(point)
        return self.boundary.is_inside(point_local[..., [0]], point[..., [1]])

    def calc_normal(self, point):
        """Calculate surface normal in global coordinates.

        Args:
            point: Point at which to calculate normal, in global coordinates.

        Returns:
            normal: Normal vector in global coordinates.
        """
        point_local = self.to_local(point)
        normal_local = self.profile.calc_normal(point_local)
        normal = self.to_global(normal_local)
        return normal

    def set_profile(self, profile):
        """Set the surface profile and perform necessary internal updates.

        Args:
            profile:
        """
        # Trivial at the moment, but for e.g. plotting may need sampled surface caches etc.
        self.profile = profile

    def set_boundary(self, boundary):
        # Trivial at the moment, but for e.g. plotting may need sampled surface caches etc.
        self.boundary = boundary

    def set_mask(self, mask):
        # Trivial at the moment, but for e.g. plotting may need sampled surface caches etc.
        self.mask = mask

    def set_interface(self, interface):
        self.interface = interface

    def make_section(self, matrix, num_points=16):
        inverse_matrix = np.linalg.inv(matrix)

        # Create normalized vector which lies on the surface plane and the cross section plane.
        vector = v4hb.normalize(v4hb.cross(self.matrix[2], matrix[2]))

        # Get a point on both planes.
        origin = v4hb.intersect_planes(self.matrix[2], self.matrix[3], matrix[2], matrix[3])

        vector_local = self.to_local(vector)[:2]
        origin_local = self.to_local(origin)[:2]
        interval = self.boundary.get_interval(origin_local, vector_local)
        d = np.linspace(*interval, num_points)
        x, y = origin_local[:, None] + vector_local[:, None]*d
        z = self.profile.calc_z(x, y)
        points_local = v4hb.stack_xyzw(x, y, z, 1)
        points_projected = v4hb.transform(points_local, np.matmul(self.matrix, inverse_matrix))
        assert np.allclose(points_projected[..., 2], 0)
        return points_projected[..., :2]

    # def get_n_next(self, current_n, ray_dot_z, lamb):
    #     """Get refractive index of a ray passing through the surface.
    #
    #     Args:
    #         current_n (scalar): Current refractive index.
    #         ray_dot_z:
    #         lamb (scalar): Wavelength.
    #
    #     Returns:
    #         scalar: Next refractive index at lamb.
    #     """
    #     n = self.n_nexts[int(ray_dot_z.ravel()[0] > 0)]
    #     return current_n if n is None else n(lamb)


# class OpticalSurface(Surface):
#
#     def __init__(self, profile, n_nexts=None, matrix=None, reflects=False, name='', boundary=None, boundary_mode='mask',
#                  samples_beam=False):
#         """A Surface_ object that can modify rays.
#
#         Args:
#             n_nexts (pair of ri.Index): refractive indices upon moving through surface in (negative z,  positive z) directions.
#                 None means no change.
#             matrix:
#             reflects (bool): Whether or not it reflects.
#             boundary: A two dimensional region that specifies the extent of the surface in its local z=0 plane.
#             boundary_mode (str): 'off', 'mask', or 'clip'.
#             samples_beam (bool): If true, then beams are sampled on the surface in tracing even if this is not actually
#                 needed.
#         """
#         Surface.__init__(self, profile, matrix, name, boundary, n_nexts)
#         self.reflects = bool(reflects)
#         # self.boundary_clip_num_points_factor = 1.1 # Unused
#         self.set_boundary_mode(boundary_mode)
#         self.samples_beam = bool(samples_beam)
#         self.calc_k_filter = None
#
#     def __repr__(self):
#         matrix = v4hb.repr_transform(self.matrix)
#         s = 'OpticalSurface(%r, n_next=(%r, %r), %s, reflects=%r, %s, %r, %s, samples_beam=%r, %r)'
#         return s%(self.profile, *self.n_nexts, matrix, self.reflects, self.name, self.boundary, self.boundary_mode,
#                   self.samples_beam, self.calc_k_filter)
#
#     def modulates(self):
#         return self.boundary.finite and self.boundary_mode in ('clip', 'mask') or self.calc_k_filter is not None
#
#     def set_boundary_mode(self, mode):
#         assert mode in ('off', 'mask', 'clip')
#         self.boundary_mode = mode
#
#     def set_k_filter(self, calc_k_filter):
#         self.calc_k_filter = calc_k_filter

def make_spherical_lens_surfaces(roc1, roc2, thickness, n, n_out=ri.vacuum, boundary=None):
    """Make a pair of surfaces representing a spherical lens.

    Args:
        roc1 (scalar): Radius of curvature of first surface. Positive means convex.
        roc2 (scalar): Radius of curvature of second surface. Positive means concave.
        thickness (scalar): Center thickness.
        n (scalar): Refractive index.
        n_out (scalar): External refractive index.
        boundary (Boundary): Boundary of both surfaces.

    Returns:
        2-tuple of OpticalSurface: The first has its vertex at (0, 0, -thickness/2) and the second on the other side.
    """
    interface = interfaces.FresnelInterface(n_out, n)
    s1 = Surface(profiles.SphericalProfile(roc1), otk.h4t.make_translation(0, 0, -thickness/2), '', boundary, None, interface)
    s2 = Surface(profiles.SphericalProfile(roc2), otk.h4t.make_translation(0, 0, thickness/2), '', boundary, None, interface.flip())
    return s1, s2

def make_surface(interface: trains.Interface, shape:str, matrix:np.ndarray=None, name:str=None) -> Surface:
    if matrix is None:
        matrix = np.eye(4)
    profile = profiles.to_profile(interface)
    if shape == 'circle':
        boundary = boundaries.CircleBoundary(interface.radius/2)
    elif shape == 'square':
        boundary = boundaries.SquareBoundary(interface.radius*2**0.5)
    else:
        boundary = None
    surface = Surface(profile, matrix, name, boundary, None, interfaces.FresnelInterface(interface.n1, interface.n2))
    return surface

def make_surfaces_lattice(self:trains.Train, lattice: mathx.Lattice, boundary: boundaries.Boundary = None, names: Sequence[str] = None):
    """Make optical surface for each interface with lattice surface profiles.

    The surfaces are placed along the z axis.

    Args:
        lattice: Surface profile is given by train interfaces repeated on the lattice.
        boundary: Boundary for all surfaces.
        names (sequence of str): Names to give to surfaces.


    Returns:
        list of surfaces
    """
    if names is None:
        names = [None]*len(self.interfaces)
    assert len(names) == len(self.interfaces)
    surfaces = []
    z = self.spaces[0]
    for num, (interface, space, name) in enumerate(zip(self.interfaces, self.spaces[1:], names)):
        profile = interface.make_profile()
        if lattice is not None:
            profile = profiles.LatticeProfile(lattice, profile)
        matrix = otk.h4t.make_translation(0, 0, z)
        surface = Surface(profile, matrix, name, boundary, None, interfaces.FresnelInterface(interface.n1, interface.n2))
        surfaces.append(surface)
        z += space
    return surfaces

def make_analysis_surfaces(train: trains.Train):
    lens_surfaces = make_surfaces(train)
    image_surface = Surface(profiles.PlanarProfile(), otk.h4t.make_translation(0, 0, train.length))
    keys = ['transmitted']*len(lens_surfaces) + ['incident']
    surfaces = lens_surfaces + [image_surface]
    return surfaces, keys

def make_surfaces(train: trains.Train, names: Sequence[str] = None, shape: str = 'circle') -> List[Surface]:
    """Make optical surface for each interface.

    The surfaces are placed along the z axis.

    Args:
        names (sequence of str): Names to give to surfaces.
        shape (str): Determines surface boundary. Can be 'circle', 'square', or None. If None, then surfaces are
            created with default boundary.

    Returns:
        list of surfaces
    """
    if names is None:
        names = [None]*len(train.interfaces)
    assert len(names) == len(train.interfaces)
    assert shape in ('circle', 'square', None)
    surfaces = []
    z = train.spaces[0]
    for num, (interface, space, name) in enumerate(zip(train.interfaces, train.spaces[1:], names)):
        matrix = otk.h4t.make_translation(0, 0, z)
        # profile = interface.make_profile()
        # if shape == 'circle':
        #     boundary = rt.CircleBoundary(interface.radius/2)
        # elif shape == 'square':
        #     boundary = rt.SquareBoundary(interface.radius*2**0.5)
        # else:
        #     boundary = None
        # surface = rt.Surface(profile, matrix, name, boundary, None, rt.FresnelInterface(interface.n1, interface.n2))
        surface = make_surface(interface, shape, matrix, name)
        surfaces.append(surface)
        z += space
    return surfaces

    # def make_analysis_surfaces(self, stop_z: float = 0):
    #     # Create stop and image surfaces.
    #     stop_surface = rt.Surface(rt.PlanarProfile(), rt.make_translation(0, 0, stop_z))
    #     image_surface = rt.Surface(rt.PlanarProfile(), rt.make_translation(0, 0, self.length))
    #
    #     # Create lens surfaces.
    #     lens_surfaces = self.make_surfaces()
    #
    #     keys = ['transmitted']*len(lens_surfaces) + ['incident']
    #     surfaces = lens_surfaces + [image_surface]
    #     trace_fun = lambda ray: ray.trace_surfaces(surfaces, keys)
    #
    #     return stop_surface, image_surface, trace_fun, lens_surfaces

    # TODO resurrect
    # def trace_distortion(self, lamb: float, stop_size: float, num_rays: int, thetas: Sequence[float],
    #         stop_shape: str = 'circle') -> np.ndarray:
    #     surfaces, keys = self.make_analysis_surfaces()
    #     stop_surface = rt1.Surface(rt1.PlanarProfile())
    #     trace_fun = lambda ray: ray.trace_surfaces(surfaces, keys)[0]
    #     xs = rt1.trace_distortion(stop_surface, surfaces[-1], trace_fun, lamb, stop_size, num_rays, thetas, stop_shape)
    #     return xs