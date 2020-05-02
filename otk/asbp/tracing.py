import copy
import logging

import numpy as np
import mathx
import otk.functions
import otk.h4t

from . import profiles
from .. import rt1

logger = logging.getLogger(__name__)

__all__ = ['Beam']


class ImmutableTransform:
    """Immutable object with transformation matrix.

    Attributes:
        matrix (4x4 array): Applied *on the right* to coordinate row vectors.
    """

    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.eye(4)
        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)

    def transform(self, transformation, frame='global'):
        if frame == 'local':
            matrix = np.matmul(transformation, self.matrix)
        elif frame == 'global':
            matrix = np.matmul(self.matrix, transformation)
        else:
            raise ValueError('Unknown frame %s.'%frame)
        return self._set_matrix(matrix)

    def _set_matrix(self, matrix: np.ndarray):
        other = copy.copy(self)
        other.matrix = matrix
        return other

    def translate(self, x, y, z, frame='global'):
        return self.transform(otk.h4t.make_translation(x, y, z), frame)

    def scale(self, x, y, z, frame='global'):
        return self.transform(otk.h4t.make_scaling(x, y, z), frame)

    def rotate_y(self, theta, frame='global'):
        """Rotate around y axis.

        Positive theta is given by right hand rule (thumb points along y, rotation is fingers)."""
        return self.transform(otk.h4t.make_y_rotation(theta), frame)

    def translate_z(self, z, frame='global'):
        return self.transform(otk.h4t.make_translation(0, 0, z), frame)

    def to_global(self, r):
        return rt1.transform(r, self.matrix)

    def to_local(self, r):
        return rt1.transform(r, self.inverse_matrix)


class Beam(ImmutableTransform):
    """A beam positioned in space sampled on a two dimensional surface."""

    # TODO remove segment_string
    def __init__(self, profile: profiles.Profile, matrix=None, segment_string=''):
        """

        Args:
            profile (asbp.Profile object): the sampled field.
            matrix (4x4 array): Transforms from profile to global coordinates.
            segment_string (str): A string describing the beam in the context of a beam segment e.g. 'incident' or 'refracted'.
        """
        ImmutableTransform.__init__(self, matrix)
        self.profile = profile
        self.segment_string = segment_string
        self.frame = self.to_global(self.profile.frame)
        # Return sampling points as nxmx4 array.
        self.rs = rt1.transform(self.profile.calc_points(), self.matrix)
        self.x, self.y = rt1.to_xy(self.rs)
        self.normalized_wavevector = rt1.transform(self.profile.calc_normalized_wavevector(), self.matrix)

    def _set_matrix(self, matrix):
        return Beam(self.profile, matrix)

    def transform_local_rs(self, matrix=None, ravel=False):
        """Transform profile sampling axes into parent coordinate system.

        The profile's regular sampling grid is transformed by the product of the beam's local-to-global matrix
        and the given matrix.

        Args:
            matrix (4x4 array): Transformation applied after that of self.
            ravel (bool): Whether to broadcast and ravel the sampling grid.

        Returns:
            If ravel:
                nx4 array: Transformed points, where n is profile.Er.size.
            else:
                x, y, z: arrays of same shape as profile.Er.
        """
        profile = self.profile
        if matrix is None:
            matrix = np.eye(4)
        compound_matrix = rt1.transform(self.matrix, matrix)
        if ravel:
            x, y, z = np.broadcast_arrays(profile.x, profile.y, profile.z)
            points = rt1.transform(rt1.stack_xyzw(x.ravel(), y.ravel(), z.ravel(), 1), compound_matrix)
            return points
        else:
            x, y, z, w = mathx.mult_vec_mat((profile.x, profile.y, profile.z, 1), compound_matrix)
            return x, y, z

    def transform_local_qs(self, matrix):
        """Transform profile angular space sampling axes into another coordinate system.

        The profile's regular sampling grid is transformed by the product of the beam's local-to-global matrix
        and the given matrix.

        Args:
            matrix (4x4 array): Transformation applied after that of self.

        Returns:
            kx, ky, kz: arrays of same shape as profile.Er.
        """
        profile = self.profile
        kx, ky, kz, _ = mathx.mult_vec_mat((profile.kx, profile.ky), rt1.transform(self.matrix, matrix))
        return kx, ky, kz

    def fourier_transform(self, f):
        """TODO what exactly does this do?"""
        rs_transform = self.inverse_matrix[3, :2]
        profile = self.profile.fourier_transform(f, rs_transform, 0)
        beam = type(self)(profile, np.eye(4))
        return beam

    # Keep me as will be required at some point.
    # def apply_k_filter(self, surface):
    #     if surface.calc_k_filter is None:
    #         return
    #     kx, ky, kz = self.transform_local_qs(surface.inverse_matrix)
    #     fq = surface.calc_k_filter(kx, ky, kz)
    #     filtered_profile = self.profile.filter(fq)
    #     filtered_beam = Beam(filtered_profile, self.matrix, 'filtered')
    #     return filtered_beam

    def propagate_to_surface(self, surface: rt1.Surface):
        """Propagate beam to surface."""
        if isinstance(self.profile, profiles.PlaneProfile):
            incident_beam = self.propagate_plane_to_surface(surface)
            planarized_beam = None
        else:
            # if rs_support is None:
            #     rs_support = self.profile.rs_support/1.05 # HACK
            # if num_pointss is None:
            #     num_pointss = tuple(int(round(s/1.1)) for s in self.profile.Er.shape) # HACK
            plane_profile = self.profile.planarize()
            planarized_beam = Beam(plane_profile, self.matrix)
            incident_beam = planarized_beam.propagate_plane_to_surface(surface)
        return incident_beam, planarized_beam

    def propagate_plane_to_surface(self, surface: rt1.Surface, curved_r_support_factor: float = 1,
            curved_num_points_support_factor: float = 1, kz_mode='local_xy'):
        """Propagate to a Surface object.

        Args:
            surface: ..rt.Surface object.
            trace (BeamTrace object): For context.
            kz_mode: 'paraxial', 'local_xy' 'local' or 'exact' - but depending on the beam.z and the surface z,
                not all are supported.

        """
        profile = self.profile
        assert isinstance(profile, profiles.PlaneProfile)
        # Intersect ray from self rs_center along vector implied by qs_center with surface. All points/vectors in
        # local coordinates.
        # origin, vector = self.calc_center_ray_local(kz_mode)
        origin = self.profile.frame[3, :]
        vector = self.profile.frame[2, :]
        d = surface.intersect_other(self.matrix, origin, vector)[0]  # Want scalar.
        # Get nominal z value to propagate to.
        center_z = profile.z + d*vector[2]
        logger.debug('Beam center intersects with surface %s at distance %.3f mm.', surface.name, d*1e3)
        point = origin + d*vector
        normal = self.to_local(surface.calc_normal(self.to_global(point)))
        # normal = np.matmul(self.inverse_matrix, surface.calc_normal(np.matmul(self.matrix, point)))
        r_supports_check = profile.calc_propagation_ms(center_z)*profile.rs_support
        to_curved = (abs(normal[..., :2]*r_supports_check*profile.k) > 1e-6).any() or not surface.profile.is_planar()

        if to_curved:
            r_support_factor = curved_r_support_factor
            num_points_factor = curved_num_points_support_factor
        else:
            r_support_factor = 1
            num_points_factor = 1

        rs_support = profile.rs_support*profile.calc_propagation_ms(center_z)*r_support_factor
        num_pointss = tuple(int(round(s*num_points_factor)) for s in profile.Er.shape)

        # Calculate self sampling center after propagation.
        rs_center = profile.adjust_rs_center(center_z)

        if not to_curved and num_pointss == self.profile.Er.shape:
            new_profile = profile.propagate_to_plane(center_z, rs_center, rs_support/profile.rs_support, kz_mode)
            new_beam = Beam(new_profile, self.matrix, 'incident')
        else:
            ## Calculate sampling grid after propagation.
            # x, y = sa.calc_xy(rs_support, num_pointss, rs_center)
            zfun = SurfaceZSampling(surface, self.matrix)
            new_profile = profile.propagate_to_curved(rs_support, num_pointss, rs_center, zfun, kz_mode=kz_mode)
            new_beam = Beam(new_profile, self.matrix, 'incident')

        return new_beam

    def apply_boundary(self, boundary: rt1.Boundary, clip: bool = False):
        inside = boundary.is_inside(self.x, self.y)
        if inside.all():
            return self
        else:
            masked_profile = self.profile.mask_binary(inside)
        if clip:
            clipped_profile = masked_profile.clip_points(inside)
            clipped_beam = Beam(clipped_profile, self.matrix)
            return clipped_beam
        else:
            masked_beam = Beam(masked_profile, self.matrix)
            return masked_beam

    def apply_mask(self, mask: rt1.Mask):
        f, gradxyf = mask.eval(self.x, self.y, True)
        profile = self.profile.mask(f, gradxyf)
        masked_beam = Beam(profile, self.matrix)
        return masked_beam

    def refract(self, normal, n, scale_Er, polarization):
        normal_local = normal.dot(self.inverse_matrix)
        polarization_local = polarization.dot(self.inverse_matrix)
        profile = self.profile.refract(rt1.to_xyz(normal_local), n, scale_Er, polarization_local[:2])
        return Beam(profile, self.matrix)

    def reflect(self, normal, n, scale_Er, polarization):
        normal_local = normal.dot(self.inverse_matrix)
        polarization_local = polarization.dot(self.inverse_matrix)
        profile = self.profile.reflect(rt1.to_xyz(normal_local), n, scale_Er, polarization_local[:2])
        # Reflect about plane z_local = profile.z at rs_center. WTF?
        matrix = rt1.transform(
            otk.functions.calc_mirror_matrix(otk.h4t.make_translation(0, 0, profile.z_center)), self.matrix)
        return Beam(profile, matrix)

    def make_interface_modes(self, surface_profile: rt1.Profile, interface: rt1.Interface):
        """Assumes self.matrix transforms to surface coordinates."""
        # All quantities in surface coordinates.
        normal = surface_profile.calc_normal(self.rs)
        modes = interface.calc_modes(self.rs, normal, self.profile.lamb, self.normalized_wavevector, self.profile.n)

        def to_mode(key):
            mode = modes[key]
            polarization = self.frame[0, :]
            new_polarization = np.einsum('...i, ...ij', polarization, mode.matrix)
            pol_Er_factor = (rt1.dot(new_polarization)**0.5).reshape(self.profile.Er.shape)
            # Take mean polarization.
            new_polarization = rt1.normalize(
                (new_polarization.reshape((-1, 4))*mathx.abs_sqd(self.profile.Er).reshape((-1, 1))).sum(axis=0))
            Er_factor = pol_Er_factor*(mode.n/self.profile.n)**0.5
            if mode.direction == rt1.Directions.TRANSMITTED:
                return self.refract(normal, mode.n, Er_factor, new_polarization)
            else:
                return self.reflect(normal, mode.n, Er_factor, new_polarization)

        return to_mode



    # DON'T DELETE - THIS WAS FIRST ATTEMPT. THEN DECIDED FOR TIME BEING WILL HAVE HOMOGENOUS POLARIZATION IN PROFILE.  # WILL ADD NONUNIFORM POLARIZATION LATER.  #  def _apply_interface(self, surface: rt.Surface, graph: nx.DiGraph):  #     # Transform beam local sampling points into surface local points.  #     point_local = self.transform_local_rs(surface.inverse_matrix, True)  #  #  Calculate surface normal in surface local coordinates.  #     normal_local = surface.profile.calc_normal(point_local)  #  #  Calculate incidence vector.  #     incident_vector = mathx.divide0(self.profile.Igradphi, mathx.abs_sqd(self.profile.Er), 0)  #     incident_vector_local = surface.to_local(incident_vector)  #     # Calculate interface modes at each point.  #     modes = surface.interface.calc_modes(point_local, normal_local, self.profile.lamb, incident_vector_local,  #         self.profile.n)


class SurfaceZSampling:
    def __init__(self, surface, matrix):
        """Callable which returns a z as a function of x and y in local coordinates.

        Args:
            surface (Surface object): Surface to sample.
            matrix (4x4 matrix): Local (x, y, z) to global transform.
        """
        self.surface = surface
        self.matrix = matrix

    def __call__(self, x, y):
        """Cast rays starting at local (x, y, 0) along z axis onto surface.

        Args:
            x, y: Arrays must be mutually broadcastable.

        Results:
            z (array of shape of broadcast of x and y): Distance from (x, y, 0) to surface along z axis.
        """
        # Cast rays along self z axis from center_z plane for each sampling point.
        xv, yv = np.broadcast_arrays(x, y)
        xv = xv.ravel()
        yv = yv.ravel()
        zv = self.surface.intersect_other(self.matrix, rt1.stack_xyzw(xv, yv, 0, 1), rt1.stack_xyzw(0, 0, 1, 0))
        z = zv.reshape(np.broadcast(x, y).shape)
        return z



