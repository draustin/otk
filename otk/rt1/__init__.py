# Considering not including these...
# from ..v4hb import dot, cross, triple, normalize, \
#     stack_xyzw, concatenate_xyzw, transform, to_xyzw, to_xy, to_xyz
from ._lines import Line, ComplexLine
from ._raytrace import Ray, RaySegment
from ._boundaries import Boundary, InfiniteBoundary, SquareBoundary, RectangleBoundary, CircleBoundary
from ._masks import Mask, LatticeMask
from ._profiles import Profile, PlanarProfile, SphericalProfile, SphericalSquareArrayProfile, make_spherical_profile, \
    ConicProfile, SquareArrayProfile, LatticeProfile, \
    BinaryProfile
from ._interfaces import Interface, FresnelInterface, SampledCoating, Mirror, PerfectRefractor, Directions, FresnelInterface
from ._surfaces import MutableTransform, CompoundMutableTransform, make_spherical_lens_surfaces, Surface, make_analysis_surfaces
from ._analysis import SpotArray, trace_distortion, trace_train_spot_array, connect_mapped_points

# # from rt import * isn't recommended, but need to list names here to prevent linter/PyCharm from thinking the imports are unused.
# __all__ = ['dot', 'cross', 'triple', 'normalize',
#     'stack_xyzw', 'concatenate_xyzw', 'transform', 'to_xyzw', 'to_xy', 'to_xyz', 'NoIntersectionError', 'Boundary', 'InfiniteBoundary', 'SquareBoundary',
#     'RectangleBoundary', 'CircleBoundary', 'Mask', 'LatticeMask', 'Profile', 'PlanarProfile', 'SphericalProfile', 'SphericalSquareArrayProfile', 'make_spherical_profile',
#     'ConicProfile', 'SquareArrayProfile', 'LatticeProfile',
#      'BinaryProfile', 'Interface', 'FresnelInterface', 'SampledCoating', 'Mirror', 'PerfectRefractor', 'Directions', 'FresnelInterface',
#     'MutableTransform', 'CompoundMutableTransform', 'make_spherical_lens_surfaces', 'Surface', 'Ray', 'RaySegment', 'connect_mapped_points', 'SpotArray', 'trace_distortion']
