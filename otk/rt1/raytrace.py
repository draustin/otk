from dataclasses import dataclass
from typing import Iterable, Tuple
from typing import Sequence

import numpy as np
from otk.h4t import make_rotation

try:
    import gizeh
except ImportError:
    pass
from . import interfaces
from .surfaces import Surface
from ..v4hb import *
from .lines import Line

@dataclass
class Ray:
    """Immutable object representing light ray(s).

    Multiple rays can be represented, with their properties running over leading dimensions of the internal arrays.
    The final dimension is reserved for spatial coordinates.

    It may seem redundant to have separate polarization and flux variables, since the power could be encoded
    in the length of the polarization vector. However it is convenient to be able to set the flux to zero e.g. if
    the ray misses an aperture, but still propagate it through the rest system, which requires a polarization to
    be defined.

    Args:
        line: Contains origin and direction vector of ray(s).
        polarization: Unit vector (possibly complex) of polarization at the origin.
        flux: Flux carried by ray(s).
        phase_origin: Phase of each ray at the origin.
        lamb: Wavelength of ray(s).
        n: Refractive index that ray(s) propagate with.
    """

    line: Line
    polarization: np.ndarray
    flux: np.ndarray
    phase_origin: np.ndarray
    lamb: np.ndarray
    n: np.ndarray = 1

    def __post_init__(self):
        self.polarization = np.asarray(self.polarization)
        self.flux = np.atleast_1d(self.flux)
        self.phase_origin = np.atleast_1d(self.phase_origin)
        self.n = np.atleast_1d(self.n)
        self.lamb = np.atleast_1d(self.lamb)
        self.k = 2*np.pi*self.n/self.lamb

        # Check polarization.
        assert self.polarization.shape[-1] == 4
        assert np.allclose(dot(self.line.vector, self.polarization), 0)
        assert np.allclose(dot(self.polarization), 1)

        self.shape = np.broadcast(self.line.origin, self.polarization, self.flux, self.phase_origin, self.lamb,
            self.n).shape

        # Non-spatial component quantities should not span last dimension.
        for array in (self.flux, self.phase_origin, self.lamb, self.n):
            assert array.shape[-1] == 1

        self.num_rays = np.prod(self.shape[:-1])

    def __getitem__(self, item):
        return Ray(self.line[item], self.polarization[item], self.phase_origin[item], self.lamb[item], self.n[item])

    def advance(self, d: float):
        """Copy self advancing a distance d."""
        line = self.line.advance(d)
        phase_origin = self.phase_origin + d*self.k
        return Ray(line, self.polarization, self.flux, phase_origin, self.lamb, self.n)

    def transform(self, matrix: np.ndarray):
        """Copy self applying transformation matrix."""
        line = self.line.transform(matrix)
        polarization = self.polarization.dot(matrix)
        return Ray(line, polarization, self.flux, self.phase_origin, self.lamb, self.n, )

    def flip(self):
        """Copy self flipping direction."""
        return Ray(self.line.flip(), self.polarization, self.flux, self.phase_origin, self.lamb, self.n)

    def make_gizeh_wavefront(self, phase: float, inverse_matrix: np.ndarray, color=(1, 0, 0)) -> gizeh.Element:
        """Make gizeh polyline representing a wavefront.

        Advances origin by phase and plots x, y components.
        """
        d = phase/self.k
        points = np.matmul(self.line.advance(d).origin, inverse_matrix)[..., :2].reshape((-1, 2))
        return gizeh.polyline(points, stroke=color, stroke_width=1)

    def to_fan(self, normal, half_angle, num_rays):
        """Given a central ray, make a ray fan formed by rotating the ray vector through a uniformly spaced set of angles.

        Args:
            self (Ray): Starting ray.
            normal (1d array): Vector about which to rotate central ray. Must be normalized.
            half_angle (scalar): Half angle of fan.
            num_rays (int): Number of rays in fan.

        Returns:
            Ray
        """
        # Define uniformly spaced angles.
        thetas = np.linspace(-half_angle, half_angle, num_rays)
        # Rotate central ray vector around each angle.
        vectors = []
        polarizations = []
        for theta in thetas:
            matrix = make_rotation(normal, theta)
            vector = self.line.vector.dot(matrix)
            vectors.append(vector)
            polarization = self.polarization.dot(matrix)
            polarizations.append(polarization)
        # Make ray fan.
        line = Line(self.line.origin, np.stack(vectors))

        ray_fan = Ray(line, polarizations, self.flux, self.phase_origin, self.lamb, self.n)
        return ray_fan

    def propagate_to_surface(self, surface: Surface) -> Tuple[np.ndarray, 'Ray']:
        d = surface.intersect_global(self.line)
        advanced = self.advance(d)
        return d, advanced

    def apply_boundary(self, surface: Surface):
        # TODO implement - set flux to zero if outside
        return self

    def apply_mask(self, surface):
        # TODO implement - modify phase_origin and flux
        return self

    def to_local_interface_mode(self, mode: interfaces.InterfaceMode) -> 'Ray':
        """Project self onto precalculated mode of an interface.

        Self must be in surface local coordinates.
        """
        polarization = np.einsum('...i, ...ij', self.polarization, mode.matrix)
        polarization_factor = dot(polarization)
        polarization /= polarization_factor**0.5
        line = Line(self.line.origin, mode.vector)
        ray = Ray(line, polarization, self.flux*polarization_factor*mode.n/self.n, self.phase_origin, self.lamb, mode.n)
        return ray

    def calc_local_interface_modes(self, surface: Surface) -> dict:
        """Self must be in surface local coordinates."""
        if surface.interface is None:
            return {}
        normal = surface.profile.calc_normal(self.line.origin)
        modes = surface.interface.calc_modes(self.line.origin, normal, self.lamb, self.line.vector, self.n)
        return modes

    def trace_surfaces(self, surfaces: Iterable[Surface], mode_keys: Iterable[str]) -> Tuple[
        Sequence['RaySegment'], Sequence['Deflection']]:
        """Trace over a sequence of surfaces.

        Args:
            surfaces: The surfaces.
            mode_keys: The interface mode to use at each surface.

        Returns:
            The first segment is open ended at the start. The last may or may not be open ended, depending
                on if the final surface yields a ray.
        """
        ray = self
        last_surface = None
        segments = []
        deflections = []

        for surface, mode_key in zip(surfaces, mode_keys):
            length, incident_ray = ray.propagate_to_surface(surface)
            segments.append(RaySegment((last_surface, surface), ray, length))

            last_surface = surface

            if mode_key is None:
                ray = None
                break
            elif mode_key == 'incident':
                ray = incident_ray
                deflection = None
            else:
                # TODO mask, boundary etc.
                incident_ray_local = incident_ray.transform(surface.inverse_matrix)
                modes = incident_ray_local.calc_local_interface_modes(surface)
                deflected_ray_local = incident_ray_local.to_local_interface_mode(modes[mode_key])
                ray = deflected_ray_local.transform(surface.matrix)
                deflection = Deflection(surface, incident_ray_local, deflected_ray_local)
            deflections.append(deflection)

        if ray is not None:
            segments.append(RaySegment((last_surface, None), ray, None))

        return segments, deflections

    @classmethod
    def make_filled_cone(cls, half_angle, lamb: float, num_radial: int = 8, num_azimuthal: int = 8):
        """Make filled cone of rays at origin pointing along z axis."""
        phi = np.arange(num_azimuthal)/num_azimuthal*2*np.pi
        theta = np.arange(1, num_radial + 1)[:, None]/num_radial*half_angle
        vx = np.cos(phi)*np.sin(theta)
        vy = np.sin(phi)*np.sin(theta)
        vz = np.cos(theta)
        return cls(Line([0, 0, 0, 1], stack_xyzw(vx, vy, vz, 0)), [1, 0, 0, 0], 1, 0, lamb)

    @classmethod
    def make_filled_pyramid(cls, half_angle, lamb: float, num_x: int = 8, num_y: int = None, pol: np.ndarray = None):
        """Make filled cone of rays at origin pointing along z axis."""
        if num_y is None:
            num_y = num_x
        if pol is None:
            pol = [1, 0, 0, 0]
        vy = np.sin(np.linspace(-half_angle, half_angle, num_y))
        vx = np.sin(np.linspace(-half_angle, half_angle, num_x))[:, None]
        vz = (1 - vx**2 - vy**2)**0.5
        vector = stack_xyzw(vx, vy, vz, 0)
        pol = normalize(cross(vector, pol))
        return cls(Line([0, 0, 0, 1], vector), pol, 1, 0, lamb)

    @classmethod
    def make_single(cls, lamb: float, pol: np.ndarray = None):
        """Single ray pointing along z axis."""
        if pol is None:
            pol = [1, 0, 0, 0]
        vector = [0, 0, 1, 0]
        pol = normalize(cross(vector, pol))
        return cls(Line([0, 0, 0, 1], vector), pol, 1, 0, lamb)

    @classmethod
    def make_filled_extended_pyramid(cls, side_length: float, half_angle: float, lamb: float, num_x: int = 4,
        num_y: int = None, num_vx: int = None, num_vy: int = None, pol: np.ndarray = None):
        if num_y is None:
            num_y = num_x
        if num_vx is None:
            num_vx = num_y
        if num_vy is None:
            num_vy = num_vx
        if pol is None:
            pol = [1, 0, 0, 0]

        x = np.linspace(-side_length/2, side_length/2, num_x)[:, None, None, None]
        y = np.linspace(-side_length/2, side_length/2, num_y)[:, None, None]
        origin = stack_xyzw(x, y, 0, 1)

        vx = np.sin(np.linspace(-half_angle, half_angle, num_vx))[:, None]
        vy = np.sin(np.linspace(-half_angle, half_angle, num_vy))
        vz = (1 - vx**2 - vy**2)**0.5
        vector = stack_xyzw(vx, vy, vz, 0)

        pol = normalize(cross(vector, pol))

        return cls(Line(origin, vector), pol, 1, 0, lamb)

@dataclass
class Deflection:
    surface: Surface
    incident_ray: Ray
    deflected_ray: Ray

# def trace_surface_path(surfaces: Sequence[Surface], keys, ray: Ray) -> List[RaySegment]:
#
#     nodes, edges = og.trace_surface_path(surfaces, keys, ray)
#     last_surface = None
#     segments = []
#     for (ray, ray_data), edge in zip_longest(nodes, edges):
#         if edge is None:
#             # Should be final surface.
#             surface = None
#             length = None
#         elif edge['transformer'].operation == 'propagate':
#             surface = edge['transformer'].surface
#             length = edge['distance']
#         else:
#             continue
#         segment = RaySegment((last_surface, surface), ray, length)
#         segments.append(segment)
#         last_surface = surface
#     return segments

# def apply_interface(self, surface: Surface):
#     if surface.interface is None:
#         return []
#
#     line_local = self.line.transform(surface.inverse_matrix)
#     normal_local = surface.profile.calc_normal(line_local.origin)
#     modes = surface.interface.calc_modes(line_local.origin, normal_local, self.lamb, line_local.vector, self.n)
#     polarization_local = geo3.transform(self.polarization, surface.inverse_matrix)
#
#     deflecteds = []
#     for key, mode in modes.items():
#         new_polarization_local = np.einsum('...i, ...ij', polarization_local, mode.matrix)
#         polarization_factor = geo3.dot(new_polarization_local)
#         new_polarization_local /= polarization_factor**0.5
#         new_polarization = geo3.transform(new_polarization_local, surface.matrix)
#         new_line_local = geo3.Line(line_local.origin, mode.vector)
#         new_line = new_line_local.transform(surface.matrix)
#         ray = Ray(new_line, new_polarization, self.flux*polarization_factor*mode.n/self.n, self.phase_origin,
#                   self.lamb, mode.n)
#         deflecteds.append((key, ray))
#
#     return deflecteds

# Dp we need this?
# @dataclass
# class Intersection:
#     ray: Ray
#     surface: Surface
#
#     def __post_init__(self):
#         self.ray_local = self.ray.transform(self.surface.inverse_matrix)


def join_segments(*args):
    """Connect sequences of segments.

    Args:
        *args: Sequences of segments.

    Returns:
        list of segments
    """
    segments = args[0]
    for other_segments in args[1:]:
        if len(segments) > 0 and len(other_segments) > 0:
            # Connect last segment of segments to first segment of other_segments.
            segments[-1] = segments[-1] + other_segments[0]
            segments += other_segments[1:]
        else:
            segments += other_segments
    return segments


@dataclass
class RaySegment:
    surfaces: Tuple[Surface, Surface]
    ray: Ray
    length: np.ndarray

    def __post_init__(self):
        assert (self.length is None) == (self.surfaces[1] is None)
        if self.length is None:
            self.phase_end = None
        else:
            self.phase_end = self.ray.phase_origin + self.ray.k*self.length

    def __str__(self):
        def get_name(surface):
            if surface is None:
                return 'None'
            else:
                return surface.name

        return '(surface %s to %s), %s, length %.3f mm'%(
            get_name(self.surfaces[0]), get_name(self.surfaces[1]), self.ray, self.length*1e3)

    def connect(self, other):
        assert self.surfaces[1] is None
        assert self.length is None
        assert other.surfaces[0] is None
        assert self.ray is other.ray
        return RaySegment((self.surfaces[0], other.surfaces[1]), self.ray, other.length)

    def __add__(self, other):
        return self.connect(other)

    def make_gizeh_ray(self, inverse_matrix, stroke_width=1):
        """Make Gizeh element containing a line for each ray.

        Args:
            inverse_matrix (4x4 array): Transforms from segment parent coordinate system to Gizeh element coordinates.

        Returns:
            gizeh.Group
        """
        assert self.length is not None  # todo more elegant way of dealing with open segments
        starts = np.matmul(self.ray.line.origin, inverse_matrix)[..., :2].reshape((-1, 2))
        stops = np.matmul(self.ray.line.advance(self.length).origin, inverse_matrix)[..., :2].reshape((-1, 2))
        elements = []
        for start, stop in zip(starts, stops):
            elements.append(gizeh.polyline(np.vstack((start, stop)), stroke_width=stroke_width, stroke=(1, 0, 0)))
        return gizeh.Group(elements)

    @classmethod
    def make_lines(cls, segments):
        """

        Args:
            segments:

        Returns:
            sequence of size x n x 4 arrays, where size is number of rays and n is number of segments in this continuous
                subsequence.
        """
        arrays = []
        stop = None
        array = None
        for segment in segments:
            if segment.length is None:
                stop = None
                continue
            start = segment.ray.line.origin
            if array is None or stop is not None and not np.allclose(start, stop):
                array = []
                arrays.append(array)
                array.append(start)
            stop = segment.ray.line.advance(segment.length).origin
            array.append(stop)

        pointss = []
        for array in arrays:
            points = np.concatenate([point.reshape((-1, 1, 4)) for point in array], 1)
            pointss.append(points)

        return pointss

    @classmethod
    def make_multi_gizeh_ray(cls, segments, inverse_matrix, stroke_width=1):
        liness = cls.make_lines(segments)
        elements = []
        for lines in liness:
            lines_projected = np.matmul(lines, inverse_matrix)[..., :2]
            for line in lines_projected:
                element = gizeh.polyline(line, stroke_width=stroke_width, stroke=(1, 0, 0))
                elements.append(element)
        group = gizeh.Group(elements)
        return group

    @classmethod
    def make_multi_gizeh_wavefront(cls, segments, inverse_matrix, phase, color=(1, 0, 0), stroke_width=2):
        """Draw wavefront in first segment that contains phase."""
        points_parent = cls.connect_iso_phase(segments, phase)
        points = np.matmul(points_parent, inverse_matrix)[..., :2].reshape((-1, 2))
        element = gizeh.polyline(points, stroke_width=stroke_width, stroke=color)
        return element

    @classmethod
    def connect_iso_phase(cls, segments, phase):
        shape = segments[0].ray.line.shape
        points = np.full(shape + (4,), np.nan)
        for segment in segments:
            # Compute all points on this segment. Inefficient, but not a bottleneck.
            d = (phase - segment.ray.phase_origin)/segment.ray.k
            points_segment = segment.ray.line.advance(d).origin

            # Compute phase interval of segment.
            if segment.surfaces[0] is None:
                phase0 = np.full(shape + (1,), -np.inf)
            else:
                phase0 = segment.ray.phase_origin

            if segment.surfaces[1] is None:
                phase1 = np.full(shape + (1,), np.inf)
            else:
                phase1 = segment.ray.phase_origin + segment.ray.k*segment.length

            in_segment = ((phase0 <= phase) & (phase <= phase1))[..., 0]

            points[in_segment, :] = points_segment[in_segment, :]
        assert not np.isnan(points).any()
        return points

# class SequentialTrace:
#     def trace_ray(self, ray: Ray) -> List[RaySegment]:
#         """Trace a ray through the system.
#
#         Returns:
#             The first segment is open ended at the start.
#
#         Args:
#             ray:
#
#         Returns:
#
#         """
#         raise NotImplementedError()
#
#
# class NullSequentialTrace(SequentialTrace):
#     def trace_ray(self, ray: Ray) -> List[RaySegment]:
#         return [RaySegment((None, None), ray, None)]
#
#
# class SurfaceSequentialTrace(SequentialTrace):
#     """Something which can trace rays."""
#
#     def __init__(self, surface):
#         self.surface = surface
#
#     def trace_ray(self, ray: Ray) -> List[RaySegment]:
#         segment0, deflected_ray = RaySegment.intersect_ray(ray, self.surface)
#         segment1 = RaySegment((self.surface, None), deflected_ray, None)
#         return [segment0, segment1]
#
#
# class SurfacesSequentialTrace(SequentialTrace):
#     def __init__(self, surfaces, keys=None):
#         self.surfaces = surfaces
#         if keys is None:
#             keys = ['transmitted']*len(surfaces)
#         self.keys = keys
#
#     def trace_ray(self, ray: Ray) -> List[RaySegment]:
#         return trace_surface_path(self.surfaces, ray, self.keys)
#
#
# class CompoundSequentialTrace(SequentialTrace):
#     def __init__(self, subsystems):
#         self.subsystems = subsystems
#
#     def trace_ray(self, ray: Ray) -> List[RaySegment]:
#         segmentss = []
#         for subsystem in self.subsystems:
#             segments = subsystem.trace_ray(ray)
#             segmentss.append(segments)
#             ray = segments[-1].ray
#
#         all_segments = join_segments(*segmentss)
#
#         return all_segments


# class BundleTrace:
#     def __init__(self, initial, n_start, intersections):
#         self.initial = initial
#         self.n_start = n_start
#         self.intersections = intersections
#         self.shape = initial.shape
#
#     def __str__(self):
#         return 'initial: %s, num. intersections %d'%(self.initial, len(self.intersections))
#
#     def is_all_inside(self):
#         inside = np.full(self.shape + (1,), True)
#         for intersection in self.intersections:
#             inside &= intersection.is_inside()
#         return inside
#
#     def lookup_incident_vector(self, segment_index):
#         if segment_index == 0:
#             return self.initial.vector
#         else:
#             return self.intersections[segment_index - 1].deflected_vector
#
#     def get_bundle(self, index):
#         if index < 0:
#             index += len(self.intersections) + 1
#         if index == 0:
#             return self.initial
#         else:
#             i = self.intersections[index-1]
#             return geo3.Line(i.point, i.deflected_vector)
#
#     def index_bundles(self, item):
#         intersections = [i.index_bundle(item) for i in self.intersections]
#         return BundleTrace(self.initial[item], self.n_start, intersections)
#
#     def get_segment_n(self, index):
#         if index < 0:
#             index += len(self.intersections) + 1
#         if index == 0:
#             return self.n_start
#         else:
#             return self.intersections[index - 1].n_next

# def trace(focal_surfaces, initial_bundle, n_start=1):
#     #x = initial.origin
#     #v = initial.vector
#     current_bundle = initial_bundle
#     n = n_start
#     intersections = []
#     for surface in focal_surfaces:
#         d = surface.intersect_global(current_bundle)
#         #xi = x + v*d
#         intersection_bundle = current_bundle.advance(d)
#         # Include mask somehow?
#         vr, n_next = surface.deflect(intersection_bundle.origin, intersection_bundle.vector, n)
#         intersections.append(Intersection(surface, intersection_bundle.origin, vr, n_next))
#
#         #x = xi
#         #v = vr
#         current_bundle = geo3.Line(intersection_bundle.origin, vr)
#         n = n_next
#
#     return BundleTrace(initial_bundle, n_start, intersections)
