from ..v4hb import dot, cross, triple, normalize, \
    stack_xyzw, concatenate_xyzw, transform, to_xyzw, to_xy, to_xyz
from otk.h4t import make_x_rotation, make_y_rotation, make_z_rotation, make_translation, make_rotation, make_scaling, \
    make_frame
from otk.functions import refract_vector, reflect_vector, calc_mirror_matrix
from .raytrace import Ray, Line
from .boundaries import Boundary, InfiniteBoundary, SquareBoundary, RectangleBoundary, CircleBoundary
from .masks import Mask, LatticeMask
from .profiles import Profile, PlanarProfile, SphericalProfile, SphericalSquareArrayProfile, make_spherical_profile, \
    ConicProfile, SquareArrayProfile, LatticeProfile, \
    BinaryProfile
from .interfaces import Interface, FresnelInterface, SampledCoating, Mirror, PerfectRefractor, Directions, FresnelInterface
from .surfaces import MutableTransform, CompoundMutableTransform, make_spherical_lens_surfaces, Surface, make_analysis_surfaces
from .raytrace import Ray, RaySegment
from .analysis import connect_mapped_points
from .analysis import SpotArray, trace_distortion, trace_train_spot_array

# from rt import * isn't recommended, but need to list names here to prevent linter/PyCharm from thinking the imports are unused.
__all__ = ['dot', 'cross', 'triple', 'normalize',
    'stack_xyzw', 'concatenate_xyzw', 'transform', 'to_xyzw', 'to_xy', 'to_xyz', 'NoIntersectionError', 'Boundary', 'InfiniteBoundary', 'SquareBoundary',
    'RectangleBoundary', 'CircleBoundary', 'Mask', 'LatticeMask', 'Profile', 'PlanarProfile', 'SphericalProfile', 'SphericalSquareArrayProfile', 'make_spherical_profile',
    'ConicProfile', 'SquareArrayProfile', 'LatticeProfile',
     'BinaryProfile', 'Interface', 'FresnelInterface', 'SampledCoating', 'Mirror', 'PerfectRefractor', 'Directions', 'FresnelInterface',
    'MutableTransform', 'CompoundMutableTransform', 'make_spherical_lens_surfaces', 'Surface', 'Ray', 'RaySegment', 'connect_mapped_points', 'SpotArray', 'trace_distortion']
