"""Parabasal Gaussian beams

Parabasal Gaussian beams are useful approximate solutions to the wave equation. A PBG mode is defined in 3D space
by a base ray, which defines its propagation direction, and two parabasal rays, which define its size and divergence.
The parabasal rays are said to be 'complex', a nomenclature that I found confusing initially. In terms of
regular ray tracing, if 'complex' ray is actually two regular rays, called the 'real' and the 'imaginary' ray. So
a parabasal Gaussian is actually defined by five regular rays - the base and the two pairs of parabasal regular rays.
The five rays are traced through the optical system as normal. The complex nature of the parabasal rays becomes
evident when we evaluate the field. Defining z as distance along the base ray, each parabasal ray is expressed as
h(z) = h(0) + u(z) where h and u are 2D complex vectors in the plane perpendicular to the base ray. The real and imaginary
parts of h(z) are the intersections of the parabasal ray real and imaginary parts with this plane.

There are some good writeups, especially by Greynolds. See the references.

Coordinate geometry and broadcasting follows the rt package convention i.e. coordinates run along the -1 axis. Other
axes are available for broadcasting i.e. multiple parabasal Gaussians.

References:
    Greynolds 1986:  A. W. Greynolds, “Vector Formulation Of The Ray-Equivalent Method For General Gaussian Beam
        Propagation,” Proc.SPIE , vol. 679. p. 679, 1986.
"""
import logging
from collections import namedtuple

import otk.rt1.lines
import numpy as np
import pyqtgraph_extended as pg
from typing import Sequence
from .functions import abs_sqd 
try:
    import gizeh
except ImportError:
    pass
import otk.rt1.raytrace
from . import rt1
from . import asbp

logger = logging.getLogger(__name__)

ParaxialRayPair = namedtuple('ParaxialRayPair', ('h1', 'u1', 'h2', 'u2'))
ParaxialRayPair.__doc__ = """
A pair of paraxial rays defining a parabasal Gaussian w.r.t an optical axis.

Attributes:
    h1 (...x4 array): Parabasal bundle 1 transverse coordinates.
    u1 (...x4 array): Parabasal bundle 1 vector.
    h2 (...x4 array): Parabasal bundle 2 transverse coordinates.
    u2 (...x4 array): Parabasal bundle 2 vector.
"""


class Mode:
    """A base ray and a pair of parabasal rays.



    Attributes:
        line_base (rt.Line):
        line1 (rt.ComplexLine):
        line2 (rt.ComplexLine):
    """

    def __init__(self, line_base: otk.rt1.lines.Line, line1: otk.rt1.lines.ComplexLine, line2: otk.rt1.lines.ComplexLine):
        self.shape = line_base.shape
        assert self.shape == line1.shape
        assert self.shape == line2.shape
        self.line_base = line_base
        self.line1 = line1
        self.line2 = line2

        self.prp = calc_pbg_transverse(line_base, line1, line2)
        self.lagrange_invariant = calc_lagrange_invariant(*self.prp)

    def __repr__(self):
        return 'Mode(line_base=%r, line1=%r, line2=%r)'%(self.line_base, self.line1, self.line2)

    def __str__(self):
        return 'mode lines: base %s, 1 %s, 2 %s'%(self.line_base, self.line1, self.line2)

    def reshape(self, shape: Sequence[int]) -> 'Mode':
        """

        Args:
            shape: Array shape, not including final (spatial) dimension.

        Returns:
            Copy of self, reshaped.
        """
        return Mode(self.line_base.reshape(shape), self.line1.reshape(shape), self.line2.reshape(shape))

    def stack(self):
        arrays = (self.line_base.origin, self.line1.real.origin, self.line1.imag.origin, self.line2.real.origin,
                  self.line2.imag.origin)
        origin = np.stack(arrays, 0)

        arrays = (self.line_base.vector, self.line1.real.vector, self.line1.imag.vector, self.line2.real.vector,
                  self.line2.imag.vector)
        vector = np.stack(arrays, 0)

        return otk.rt1.lines.Line(origin, vector)

    @classmethod
    def unstack(cls, line):
        """Unstack elements of a rt tracing bundle.

        Args:
            line (rt.Line): Laid out in same format as returned from the stack method i.e. the zeroth dimension
                should have length 5.

        Returns:
            ModeBundle object
        """
        assert line.shape[0] == 5
        bundle_base = otk.rt1.lines.Line(line.origin[0, ...], line.vector[0, ...])
        bundle1 = otk.rt1.lines.ComplexLine(otk.rt1.lines.Line(line.origin[1, ...], line.vector[1, ...]),
                                 otk.rt1.lines.Line(line.origin[2, ...], line.vector[2, ...]))
        bundle2 = otk.rt1.lines.ComplexLine(otk.rt1.lines.Line(line.origin[3, ...], line.vector[3, ...]),
                                 otk.rt1.lines.Line(line.origin[4, ...], line.vector[4, ...]))
        return cls(bundle_base, bundle1, bundle2)

    def calc_field(self, k, point, flux=1, phi_axis_origin=0, calc_grad_phi=False):
        """Point must broadcast with shape."""
        return calc_pbg_field(k, self.line_base, *self.prp, point, flux, phi_axis_origin, calc_grad_phi)

    def calc_total_field(self, k, point, flux=1, phi_axis_origin=0):
        # Broadcast mode coefficients to match self.
        flux = np.broadcast_to(flux, self.shape + (1,))
        phi_axis_origin = np.broadcast_to(phi_axis_origin, self.shape + (1,))

        # Use reshaped form of self running along -3 axis, with points reshaped to run along -2 axis.
        field_ = self.reshape((-1, 1)).calc_field(k, point.reshape((-1, 4)), flux.reshape((-1, 1, 1)),
                                                  phi_axis_origin.reshape((-1, 1, 1)))[0]
        field_ = field_.sum(-3)
        field = field_.reshape(point.shape[:-1] + (1,))
        return field

    def transform(self, matrix):
        return Mode(self.line_base.transform(matrix), self.line1.transform(matrix), self.line2.transform(matrix))

    def project_field(self, k, points, field):
        points_ = points.reshape((-1, 4))
        field_ = field.reshape((-1, 1))
        coefficients_, projected_field_ = self.reshape((-1,))._project_field(k, points_, field_)
        coefficients = coefficients_.reshape(self.shape + (1,))
        projected_field = projected_field_.reshape(field.shape)
        flux = abs_sqd(coefficients)
        phi_axis_origin = np.angle(coefficients)
        return flux, phi_axis_origin, projected_field

    def _project_field(self, k, points, field):
        assert len(self.shape) == 1
        mode_fields = self.calc_field(k, points[..., None, :])[0][..., 0]
        coefficients = np.linalg.lstsq(mode_fields, field, rcond=None)[0]
        projected_field = np.matmul(mode_fields, coefficients)
        return coefficients, projected_field

    def advance(self, k, d):
        """Advance origin of rays and calculate on-axis phase shift.

        Args:
            k (scalar): Wavenumber.
            d (scalar): Distance to advance.

        Returns:
            Mode: Advanced bundle.
            ...x1 array: On-axis phase shift at origin of advanced bundle.
        """
        new_bundle_base = self.line_base.advance(d)
        _, phi_axis_origin = self.calc_field(k, new_bundle_base.origin)
        return Mode(new_bundle_base, self.line1.advance(d), self.line2.advance(d)), phi_axis_origin

    @classmethod
    def make(cls, line_base, axis1, lamb, waist1, d1=0, waist2=None, d2=None):
        """Make parabasal Gaussian rt bundle.

        Args:
            line_base (raytrace.Line): Base rt bundle.
            axis1 (...x4 array): Defines the displacement from base rt.
            waist1 (scalar): Waist size along axis 1.
            d1 (scalar): Propagation distance to axis 1 waist.
            waist2 (scalar): Propagation distance to axis 2 waist.
            d2 (scalar): Propagation distance to axis 2 waist.


        Returns:
            raytrace.Line object
            tuple: The axis1 and axis2 that were used.
        """
        if waist2 is None:
            waist2 = waist1
        if d2 is None:
            d2 = d1
        origin, vector, axis1 = np.broadcast_arrays(line_base.origin, line_base.vector, axis1)
        axis2 = rt1.normalize(rt1.cross(vector, axis1))
        axis1 = rt1.cross(axis2, vector)
        theta1 = lamb/(np.pi*waist1)
        theta2 = lamb/(np.pi*waist2)

        # Define axis 1 divergence ray.
        vector1_real = rt1.normalize(vector + axis1*theta1)
        origin1_real = origin + vector*d1

        # Define axis 1 waist ray.
        origin1_imag = origin + axis1*waist1
        vector1_imag = vector

        # Define axis 2 divergence ray.
        vector2_real = rt1.normalize(vector + axis2*theta2)
        origin2_real = origin + vector*d2

        # Define axis 1 waist ray.
        origin2_imag = origin + axis2*waist2
        vector2_imag = vector

        bundle1 = otk.rt1.lines.ComplexLine(otk.rt1.lines.Line(origin1_real, vector1_real), otk.rt1.lines.Line(origin1_imag, vector1_imag))
        bundle2 = otk.rt1.lines.ComplexLine(otk.rt1.lines.Line(origin2_real, vector2_real), otk.rt1.lines.Line(origin2_imag, vector2_imag))

        return cls(line_base, bundle1, bundle2), (axis1, axis2)

    def project_profile(self, profile):
        points = rt1.stack_xyzw(profile.x, profile.y, profile.z, 1)
        flux, phi_axis_origin, projected_field = self.project_field(profile.k, points, profile.Er)
        beam = Beam(profile.n, profile.lamb, self, flux, phi_axis_origin)
        return beam, projected_field

    @classmethod
    def make_profile_angular_grid(cls, profile, waist, num_rays, num_sigmas=2):
        """Make angular grid of parabasal Gaussian modes covering a sampled beam profile.

        Args:
            profile (asbp.Profile): Source profile.
            waist (scalar): Waist of parabasal Gaussian modes.
            num_rays (int): Number of rays to a side of the square grid.
            num_sigmas (scalar): Number of multiples of profile angular standard deviation to cover.

        Returns:
            Mode: Line of parabasal Gaussian modes, spatially centered on profile, covering its angular spectrum.
        """
        theta_maxs = np.arcsin((profile.qs_center + num_sigmas*profile.var_qs**0.5)/profile.k)
        theta_mins = np.arcsin((profile.qs_center - num_sigmas*profile.var_qs**0.5)/profile.k)
        mode = cls.make_angular_grid(profile.lamb, waist, theta_mins, theta_maxs, num_rays)
        return mode

    @classmethod
    def make_angular_grid(cls, lamb, waist, theta_mins, theta_maxs, num_rays):
        theta_mins = asbp.to_scalar_pair(theta_mins)
        theta_maxs = asbp.to_scalar_pair(theta_maxs)
        thetay = np.linspace(theta_mins[0], theta_maxs[0], num_rays)
        thetax = np.linspace(theta_mins[1], theta_maxs[1], num_rays)[:, None]
        vector = rt1.normalize(rt1.stack_xyzw(thetax, thetay, 1, 0))
        axis1 = rt1.normalize(rt1.stack_xyzw(1, 0, 0, 0))
        origin = 0, 0, 0, 1
        bundle = cls.make(otk.rt1.lines.Line(origin, vector), axis1, lamb, waist)
        return bundle


class Beam:
    """Immutable class representing parabasal Gaussian beam(s).

     Physically, I don't think it's meaningful to assign a different polarization to the parabasal
    rays. However, need to trace them using normal ray tracing, and if we encounter birefringent materials then we
    they need a polarization. So, the PBG mode should have one polarization per base ray, but the stack operation
    needs to generate polarizations for the side rays. If vb is the normalized base ray vector, pb its polarization, and
    vr the normalized parabasal ray vector, then the polarization of the parabasal ray should be (vb x pb) x vr. Note that
    pb's phase comes out unconjugated here.

    Attributes:
        n (scalar): Refractive index.
        lamb (scalar): Free-space wavelength.
        mode (Mode): The mode of the beam(s).
        flux (...x1 array): Flux of beam(s).
        phi_axis_origin (...x1 array): Phase at the base ray origin.
    """

    def __init__(self, n, lamb, mode: Mode, polarization: np.ndarray, flux: np.ndarray, phi_axis_origin=0):
        self.n = n
        self.lamb = lamb
        self.k = 2*np.pi*n/lamb
        self.mode = mode
        self.polarization = np.broadcast_to(polarization, mode.shape + (4,))
        assert np.allclose(rt1.dot(self.mode.line_base.vector, self.polarization), 0)
        assert np.allclose(rt1.dot(self.polarization), 1)
        self.flux = np.broadcast_to(flux, mode.shape + (1,))
        self.phi_axis_origin = np.broadcast_to(phi_axis_origin, mode.shape + (1,))

    def __str__(self):
        return '%s, %.2f nm, %s, flux %g - %g, phi %g - %g'%(
            self.n, self.lamb*1e9, self.mode, self.flux.min(), self.flux.max(), self.phi_axis_origin.min(),
            self.phi_axis_origin.max())

    def __repr__(self):
        return 'Beam(n=%r, lamb=%r, mode=%r, polarization=%r, flux=%r, phi_axis_origin=%r)'%(
            self.n, self.lamb, self.mode, self.polarization, self.flux, self.phi_axis_origin)

    @classmethod
    def unstack(cls, ray, phi_axis_origin=None):
        mode = Mode.unstack(ray.line)
        polarization = ray.polarization[0, ...]
        flux = ray.flux[0, ...]
        if phi_axis_origin is None:
            phi_axis_origin = ray.phase_origin[0]
        return Beam(ray.n, ray.lamb, mode, polarization, flux, phi_axis_origin)

    def stack(self):
        pol_base = self.polarization
        y_base = rt1.cross(self.mode.line_base.vector, self.polarization)
        pol1_real = rt1.cross(y_base, self.mode.line1.real.vector)
        pol1_imag = rt1.cross(y_base, self.mode.line1.imag.vector)
        pol2_real = rt1.cross(y_base, self.mode.line2.real.vector)
        pol2_imag = rt1.cross(y_base, self.mode.line2.imag.vector)
        polarization = rt1.normalize(np.stack((pol_base, pol1_real, pol1_imag, pol2_real, pol2_imag), 0))

        ray = rt1.Ray(self.mode.stack(), polarization, self.flux, self.phi_axis_origin, self.lamb, self.n)
        return ray

    def get_base(self):
        return rt1.Ray(self.mode.line_base, self.phi_axis_origin, self.lamb, self.n)

    def transform(self, matrix):
        return Beam(self.n, self.lamb, self.mode.transform(matrix), self.polarization.dot(matrix), self.flux,
                    self.phi_axis_origin)

    def advance(self, d):
        mode, dphi_axis_origin = self.mode.advance(self.k, d)
        return Beam(self.n, self.lamb, mode, self.polarization, self.flux, self.phi_axis_origin + dphi_axis_origin)

    def calc_field(self, point, calc_grad_phi=False):
        return self.mode.calc_field(self.k, point, self.flux, self.phi_axis_origin, calc_grad_phi)

    def calc_total_field(self, point):
        return self.mode.calc_total_field(self.k, point, self.flux, self.phi_axis_origin)

    @classmethod
    def project_asbp_beam_angular_grid(cls, asbp_beam, waist, num_rays, num_sigmas=2):
        """Project asbp.beam onto angular grid of parabasal Gaussians.

        Args:
            asbp_beam (asbp.Beam): Beam to project.
            waist (scalar): Waist size.
            num_rays (int): Number of rays to a side of the grid.
            num_sigmas (scalar): Half-width of angular grid in beam standard deviations.

        Returns:
            bundle_global (Beam): Projected parabasal Gaussian bundle.
            projected_field (array of same size as beam profile): The projected bundle sampled on the same grid as the
                beam profile.
        """
        mode = Mode.make_profile_angular_grid(asbp_beam.profile, waist, num_rays, num_sigmas)[0]
        beam, projected_field = mode.project_profile(asbp_beam.profile)
        beam_global = beam.transform(asbp_beam.matrix)
        return beam_global, projected_field

    def make_asbp_profile(self, surface_profile=None, rs_support=None, num_pointss=64, rs_center=(0, 0),
            qs_center=(0, 0), num_sigmas=6):
        """Make asbp.Profile on given surface profile.

        Args:
            surface_profile: rt.Profile object.

        Returns:
            asbp.Profile object.
        """
        if surface_profile is None:
            surface_profile = rt1.PlanarProfile()

        if rs_center is None or rs_support is None:
            origin = surface_profile.intersect_line(self.mode.line_base).origin
            xi, yi = rt1.to_xy(origin)
            flux = self.flux[..., 0]
            mean_x, mean_y, var_x, var_y, var_xy = mathx.mean_and_variance2(xi, yi, flux)
            if rs_center is None:
                rs_center = (mean_x, mean_y)
            if rs_support is None:
                x1, y1 = rt1.to_xy(self.mode.prp.h1)
                x2, y2 = rt1.to_xy(self.mode.prp.h2)
                var_x = max(var_x, abs_sqd(x1).max(), abs_sqd(x2).max())
                var_y = max(var_y, abs_sqd(y1).max(), abs_sqd(y2).max())
                rs_support = num_sigmas*np.asarray((var_x, var_y))**0.5

        if qs_center is None:
            vector = self.mode.line_base.vector
            kx, ky = rt1.to_xy(vector)*self.k
            flux = self.flux[..., 0]
            mean_kx, mean_ky, var_kx, var_ky, var_kxky = mathx.mean_and_variance2(kx, ky, flux)
            qs_center = mean_kx, mean_ky

        rs_support = asbp.to_scalar_pair(rs_support)
        num_pointss = asbp.to_scalar_pair(num_pointss)
        x, y = asbp.calc_xy(rs_support, num_pointss, rs_center)
        z = surface_profile.calc_z(x, y)
        points = rt1.stack_xyzw(x, y, z, 1)
        Er = self.calc_total_field(points)[..., 0]
        # This only works if the field is well sampled. The most promising method for getting a correct gradient is to
        # evaluate the derivatives of the Gaussians directly. TODO derive and implement.
        gradxyE = asbp.calc_gradxyE(rs_support, Er)
        if ((z.max() - z.min())*self.k) > 1e-6:
            profile = asbp.CurvedProfile(self.lamb, self.n, surface_profile.calc_z, rs_support, Er, gradxyE, rs_center,
                                         qs_center)
        else:
            z = z.mean()
            profile = asbp.PlaneProfile(self.lamb, self.n, z, rs_support, Er, gradxyE, rs_center, qs_center)
        return profile

    @classmethod
    def make(cls, line_base, axis1, lamb, waist1, polarization, flux, d1=0, waist2=None, d2=None, n=1,
            phi_axis_origin=0):
        mode = Mode.make(line_base, axis1, lamb, waist1, d1, waist2, d2)[0]
        beam = Beam(n, lamb, mode, polarization, flux, phi_axis_origin)
        return beam

    def scale_flux(self, factor):
        """Multiply flux by given factor.

        Returns:
            copy of self with modified flux
        """
        return Beam(self.n, self.lamb, self.mode, self.flux*factor, self.phi_axis_origin)

    def calc_edge_distance(self, d, rhat, exponent=1):
        """Calculate distance of beam edge.

        Args:
            d (...x1 array): Distance along beam.
            rhat (...x4 array): Unit vector along which edge distance from axis is calculated.
            exponent (...x1 array): Value of imaginary part of exponent in amplitude at which edge is defined i.e.
                exponent=1 means the usual e^-2 intensity level.

        Returns:
            ...x1 array: Distance along rhat from beam axis to beam edge.
        """
        r = calc_edge_distance(self.k, self.mode.line_base, *self.mode.prp, d, rhat, exponent)
        return r


class Segment:
    """A parabasal Gaussian beam (bundle) in free space, possibly bounded at either end by surfaces.

    Attributes:
        surfaces (2-tuple): rt.Surface objects. Either can be None.
        beam: All base lines must start on surfaces[0].
        length: Distance along base lines between surfaces, if both are given. Otherwise None.
    """

    def __init__(self, surfaces, beam, length):
        # If both surfaces are given, length must be given. Otherwise length should be None.
        assert (length is None) == (surfaces[1] is None)
        self.surfaces = surfaces
        self.beam = beam
        self.length = length

    def __str__(self):
        if self.length is None:
            length_str = 'length=None'
        else:
            length_str = '%.3f mm <= length <= %.3f mm'%(self.length.min()*1e3, self.length.max()*1e3)

        def get_name(surface):
            if surface is None:
                return None
            else:
                return surface.name

        return 'surface %s to %s, %s, %s'%(*[get_name(s) for s in self.surfaces], self.beam, length_str)

    def __repr__(self):
        return 'Segment(%r, %r, %r)'%(self.surfaces, self.beam, self.length)

    def __add__(self, other):
        assert self.surfaces[1] is None
        assert self.length is None
        assert other.surfaces[0] is None
        assert self.beam is other.beam
        return Segment((self.surfaces[0], other.surfaces[1]), self.beam, other.length)

    @classmethod
    def unstack(cls, ray_segment, phi_axis_origin=None):
        beam = Beam.unstack(ray_segment.ray, phi_axis_origin)
        if ray_segment.length is not None:
            length = ray_segment.length[0]  # Want length of base ray.
        else:
            length = None
        segment = cls(ray_segment.surfaces, beam, length)
        return segment

    @classmethod
    def unstack_sequence(cls, ray_segments, flux=1, phi_axis_origin=None):
        if len(ray_segments) == 0:
            return []
        if phi_axis_origin is None:
            phi_axis_origin = ray_segments[0].ray.phase_origin[0]
        last_base_ray_phase = ray_segments[0].ray.phase_origin[0]
        segments = []
        for ray_segment in ray_segments:
            surfaces = ray_segment.surfaces
            base_ray = ray_segment.ray[0]
            if surfaces[0] is not None:
                flux = flux*surfaces[0].is_inside_global(base_ray.line.origin)
            # If the ray had a phase jump between segments, apply it to the PBG.
            phi_axis_origin = phi_axis_origin + base_ray.phase_origin - last_base_ray_phase
            segment = cls.unstack(ray_segment, phi_axis_origin)
            if segment.length is not None:
                beam = segment.beam
                advanced_base_ray = base_ray.advance(segment.length)
                point = advanced_base_ray.line.origin
                phi_axis_origin = beam.calc_field(point)[1]
                flux = flux*surfaces[1].is_inside_global(point)
                last_base_ray_phase = advanced_base_ray.phase_origin
            segments.append(segment)
        return segments

    def get_base(self):
        return rt1.RaySegment(self.surfaces, self.beam.get_base(), self.length)

    def make_asbp_beam(self, surface_index, rs_support=None, num_pointss=64, rs_center=None, qs_center=None,
            num_sigmas=6):
        surface = self.surfaces[surface_index]
        if surface_index == 1:
            pbg_beam = self.beam.advance(self.length)
        else:
            pbg_beam = self.beam
        # Base rays of beam are now on surface.
        pbg_beam = pbg_beam.transform(surface.inverse_matrix)
        profile = pbg_beam.make_asbp_profile(surface.profile, rs_support, num_pointss, rs_center, qs_center, num_sigmas)
        asbp_beam = asbp.Beam(profile, surface.matrix)
        return asbp_beam, pbg_beam

    def make_profile_widget(self, surface_index, **kwargs):
        asbp_beam, pbg_beam = self.make_asbp_beam(surface_index, **kwargs)
        widget = asbp.PlaneCurvedProfileWidget()
        widget.set_profile(asbp_beam.profile)
        widget.set_line(pbg_beam.mode.line_base)
        widget.show()
        return widget

    def make_gizeh(self, matrix, fill=(1, 0, 0), num_points=16):
        """

        Args:
            matrix (4x4 array): Transforms from coordinate system of segment to 2D scene.
            fill:
            num_points:

        Returns:

        """
        assert all(surface is not None for surface in self.surfaces), 'Segment must be bounded by surfaces'
        assert np.prod(self.beam.mode.shape) == 1, 'Only one PBG allowed (for now)'
        length = np.asscalar(self.length)  # Extra dimensions confuse things.
        origin = self.beam.mode.line_base.origin.reshape((4,))
        vector = self.beam.mode.line_base.vector.reshape((4,))
        rhat = rt1.cross(matrix[2, :], vector)

        def calc_point(d, sign):
            """Calculate beam edge point.

            Args:
                d (scalar): Distance along beam.
                sign (scalar): Direction along rhat.

            Returns:
                ...x4 array: Beam edge point(s).
            """
            r = self.beam.calc_edge_distance(d, sign*rhat)
            point = origin + d*vector + sign*rhat*r
            return point

        dss = []
        for sign in (-1, 1):
            dss.append([])
            for (surface, d_guess) in zip(self.surfaces, (0, length)):
                d_root = surface.intersect_global_curve(lambda d: calc_point(d, sign).reshape((4,)), d_guess)
                dss[-1].append(d_root)

        dm = np.linspace(dss[0][0], dss[0][1], num_points)[:, None]
        pointsm = calc_point(dm, -1).reshape((-1, 4))
        dp = np.linspace(dss[1][1], dss[1][0], num_points)[:, None]
        pointsp = calc_point(dp, 1).reshape((-1, 4))
        to_section = np.linalg.inv(matrix)[:, :2]
        points_section = np.concatenate((pointsm, pointsp)).dot(to_section)
        return gizeh.polyline(points_section, close_path=True, fill=fill)


# def

def rotate_pbg(ray_base, ray1, ray2, theta):
    # Rotate about ray_base by (complex) angle theta.
    pass


def trace_surfaces(surfaces, keys, beam):
    ray = beam.stack()
    ray_segments = ray.trace_surfaces(surfaces, keys)
    pbg_segments = Segment.unstack_sequence(ray_segments)
    return pbg_segments


def follow_pbg_bundle_trace(ray_trace, lamb, phi_axis_origin=None, all_inside=None):
    current_bundle = ray_trace.initial
    n = ray_trace.n_start
    if phi_axis_origin is None:
        phi_axis_origin = np.zeros(ray_trace.shape[1:] + (1,))
    if all_inside is None:
        all_inside = np.full(ray_trace.shape[1:] + (1,), True)
    phi_axis_origins = [phi_axis_origin]
    all_insides = [all_inside]
    for intersection in ray_trace.intersections:
        bundle = Mode.unstack(current_bundle)
        final = intersection.point[0, ...]
        k = n*2*np.pi/lamb
        phi_axis_origin = bundle.calc_field(k, final)[1] + phi_axis_origin
        phi_axis_origins.append(phi_axis_origin)

        all_inside = all_insides[-1] & intersection.is_inside().all(axis=0)
        all_insides.append(all_inside)

        current_bundle = otk.rt1.lines.Line(intersection.point, intersection.deflected_vector)
        n = intersection.n_next
    phi_axis_origins = np.stack(phi_axis_origins)
    return phi_axis_origins, all_insides


def calc_quadratic_roots(a, b, c):
    root_det = b**2 - 4*a*c
    a2 = 2*a
    zd = root_det/a2
    zm = -b/a2
    z1 = zm + zd
    z2 = zm - zd
    return z1, z2


def project_pbg_bundle(base, bundle):
    h_real, u_real = base.project_line(bundle.real)
    h_imag, u_imag = base.project_line(bundle.imag)
    h = h_real - 1j*h_imag
    u = u_real - 1j*u_imag
    return h, u


def calc_pbg_transverse(bundle_base, bundle1, bundle2):
    """Intersect each parabasal rt with a plane perpendicular to and containing to origin of corresponding base rt.

    Args:
        bundle_base (rt.Line): Base rt(s).
        bundle1 (rt.ComplexLine): Parabasal rt bundle 1.
        bundle2 (rt.ComplexLine): Parabasal rt bundle 2.

    Returns:
        tuple: Four (..., 4) arrays: h1, the transverse position of the first parabasal rt, its vector, and likewise h2
            and u2 for the second parabasal rt.
    """
    h1, u1 = project_pbg_bundle(bundle_base, bundle1)
    h2, u2 = project_pbg_bundle(bundle_base, bundle2)
    return ParaxialRayPair(h1, u1, h2, u2)


def calc_lagrange_invariant(h1, u1, h2, u2):
    """Calculate Largrange invariant given transverse rt coordinates.

    The Lagrange invariant is defined in eq. (5) of Greynolds 1986. For well-behaved parabasal Gaussian rays it equals
    zero.

    Args:
        h1, u1, h2, h2: Form a ParaxialRayPair.

    Returns:
        ...x1 complex array: Lagrange invariant.
    """
    return rt1.dot(u1, h2) - rt1.dot(u2, h1)


def calc_pbg_field(k, base, h1origin, u1, h2origin, u2, point, flux=1, phi_axis_origin=0, calc_grad_phi=False):
    """Calculate field and related quantities of parabasal Gaussian beam(s).

    Args:
        k (scalar): Wavenumber.
        base (rt.Line): Base ray.
        h1origin, u1, h2origin, u2(...x4 arrays): Form a ParaxialRayPair.
        point (...x4 array): Points at which to evaluate field.
        flux (...x1 array): Flux(es).
        phi_axis_origin (...x1 array): Phase at base ray origin.
        calc_grad_phi (bool): Whether to calculate phase gradient.

    Returns:
        ...x1 array: Field amplitude at point.
        ...x1 array: Phase on base ray closest to point.
        ...x4 array: Only if calc_grad_phi is True. Gradient of field phase at point.
    """
    v = base.vector
    # Project sampling points onto components perpendicular and parallel to plane containing base ray origin.
    z, r = base.project_point(point)
    h1 = h1origin + u1*z
    h2 = h2origin + u2*z

    # h1 x h2 is a quadratic in z. Calculate coefficients.
    a = rt1.triple(v, u1, u2)  # z**2 coefficient.
    b = rt1.triple(v, u1, h2origin) + rt1.triple(v, h1origin, u2)  # z coefficient
    h1_cross_h2_origin = rt1.triple(v, h1origin, h2origin)  # constant
    h1_cross_h2 = z*(a*z + b) + h1_cross_h2_origin

    u1_dot_r = rt1.dot(u1, r)
    u2_dot_r = rt1.dot(u2, r)
    v_h1_r = rt1.triple(v, h1, r)
    v_h2_r = rt1.triple(v, h2, r)

    # Evaluate 'wavefront' - distance factor in exponent. See (7) of Greynolds 1986.
    w_numerator = v_h1_r*u2_dot_r - v_h2_r*u1_dot_r
    w = w_numerator/(2*h1_cross_h2)

    # Compute on-axis phase as sum of starting phase, carrier and inverse square root of cross product. To unwrap the
    # phase as h1 x h2 crosses the negative real axis, we express (h1 x h2)(z), which is quadratic in z, as the
    # product (z - z1)*(z - z2). We only care about the phase change, so the overall factor does not matter.
    z1, z2 = calc_quadratic_roots(a, b, h1_cross_h2_origin)
    # Final factor is phase of inverse square root of h1 x h2, (6) of Greynolds 1986.
    phi_axis = phi_axis_origin + k*z - 0.5*(np.angle(z - z1) - np.angle(-z1) + np.angle(z - z2) - np.angle(-z2))

    # Evaluate and check exponent. Use threshold for warning on imaginary part its smallest value is usually slightly negative.
    phi = phi_axis + k*w
    phi_imag_min = phi.imag.min()
    level = logging.WARNING if phi_imag_min < -1e-3 else logging.DEBUG
    logger.log(level, 'phi.imag.min = %g', phi_imag_min)

    # Evaluate field.
    field = (2*flux/(np.pi*abs(h1_cross_h2)))**0.5*np.exp(1j*phi)
    field[np.broadcast_to(flux,
                          field.shape) == 0] = 0  # For robustness - dodgy rays give nan field but their flux may be zero.

    if calc_grad_phi:
        # Calculate derivative of numerator of w w.r.t z. The derivative of the v_h1_r triple product is v_u1_r
        # (and likewise for h2).
        dw_numerator_dz = rt1.triple(v, u1, r)*u2_dot_r - rt1.triple(v, u2, r)*u1_dot_r
        dh1_cross_h2_dz = 2*a*z + b  # derivative of h1 x h2
        # Apply quotient rule.
        dw_dz = (dw_numerator_dz*h1_cross_h2 - w_numerator*dh1_cross_h2_dz)/(2*h1_cross_h2**2)
        # Phase angle is equal to imaginary part of logarithm. Derivative follows.
        dphi_axis_dz = k - 0.5*((1/(z - z1)).imag + (1/(z - z2)).imag)
        # Derivative of phase w.r.t z is sum of derivatives of on-axis phase and wavefront.
        dphi_dz = dphi_axis_dz + k*dw_dz.real

        # Calculate gradient of wavefront w.r.t r. This gradient is perpendicular to v (u1 and u2 are perpendicular to v).
        # Note this assumes r is real (dot product conjugates second argument). The cross products are conjugated
        # by convention rt.cross conjugates its output.
        gradr_w_numerator = v_h1_r*u2 + rt1.cross(v, h1).conj()*u2_dot_r - (v_h2_r*u1 + rt1.cross(v, h2).conj()*u1_dot_r)
        gradr_w = gradr_w_numerator/(2*h1_cross_h2)

        # Phase gradient is sum of partial derivatives w.r.t to axial and transverse coordinates.
        grad_phi = dphi_dz*v + k*gradr_w.real

        return field, phi_axis, grad_phi
    else:
        return field, phi_axis


def calc_edge_distance(k, base, h1origin, u1, h2origin, u2, z, rhat, exponent=1):
    """Calculate distance of beam edge.

    Args:
        k:
        base:
        h1origin:
        u1:
        h2origin:
        u2:
        z (...x1 array): Distance along beam.
        rhat (...x4 array): Unit vector along which edge distance from axis is calculated.
        exponent (...x1 array): Value of imaginary part of exponent in amplitude at which edge is defined i.e.
            exponent=1 means the usual e^-2 intensity level.

    Returns:
        ...x1 array: Distance along rhat from beam axis to beam edge.
    """
    h1 = h1origin + u1*z
    h2 = h2origin + u2*z
    # The cross product is a quadratic in z. Calculate coefficients.
    a = rt1.triple(base.vector, u1, u2)  # z**2 coefficient.
    b = rt1.triple(base.vector, u1, h2origin) + rt1.triple(base.vector, h1origin, u2)  # z coefficient.
    cross_origin = rt1.triple(base.vector, h1origin, h2origin)  # Constant coefficient.
    cross = z*(a*z + b) + cross_origin

    # Complex phase is quadratic in displacement along rhat. Get coefficient.
    wk = k*(rt1.triple(base.vector, h1, rhat)*rt1.dot(u2, rhat) - rt1.triple(base.vector, h2, rhat)*rt1.dot(u1, rhat))/(
            2*cross)
    r = (exponent/wk.imag)**0.5
    return r


def plot_projection(profile, projected):
    residual = profile.Er - projected

    glw = pg.GraphicsLayoutWidget()
    field_plot = glw.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Field')
    field_image = asbp.make_Er_image_item(profile.rs_support, profile.Er, profile.rs_center, 'amplitude')
    field_plot.addItem(field_image)
    glw.addHorizontalSpacer()

    projected_plot = glw.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Projected')
    projected_plot.setXYLink(field_plot)
    projected_image = asbp.make_Er_image_item(profile.rs_support, projected, profile.rs_center, 'amplitude')
    projected_plot.addItem(projected_image)
    glw.addHorizontalSpacer()
    glw.addColorBar(images=(field_image, projected_image), rel_row=2)
    glw.addHorizontalSpacer()

    residual_plot = glw.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Residual')
    residual_image = asbp.make_Er_image_item(profile.rs_support, residual, profile.rs_center, 'amplitude')
    residual_plot.addItem(residual_image)
    residual_plot.setXYLink(projected_plot)
    glw.addColorBar(image=residual_image, rel_row=2)

    glw.resize(1200, 360)
    glw.show()
    return glw
