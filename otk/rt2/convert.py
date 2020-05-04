from functools import  singledispatch
from itertools import accumulate
from typing import Sequence, List
from warnings import warn
import numpy as np
from .. import trains
from ..types import Sequence3
from ..sdb import *
from . import *
from . import scalar

__all__ = ['make_element', 'make_surface', 'make_elements', 'to_rectanglar_array_sag', 'to_rectangular_array_surface', 'to_rectangular_array_element',
    'to_rectangular_array_elements', 'make_sag_function']

# begin make_surface

@singledispatch
def make_surface(obj, *args, **kwargs) -> Surface:
    raise NotImplementedError()

@make_surface.register
def _(obj: trains.SphericalSurface, side: float=1., vertex:Sequence[float]=None):
    return SphericalSag(obj.roc, side, vertex)

@make_surface.register
def _(obj: trains.ConicSurface, side:float = 1, vertex:Sequence[float]=None):
    return ZemaxConic(obj.roc, obj.radius, side, obj.kappa, obj.alphas, vertex)

@make_surface.register
def _(obj: trains.SegmentedSurface, side:float = 1, vertex:Sequence[float]=None):
    radii = list(accumulate(s.radius for s in obj.segments[:-1]))
    surfaces = []
    for segment, sag in zip(obj.segments, obj.sags):
        surfaces.append(make_surface(segment, side=side, vertex=vertex + (0, 0, sag)))
    return SegmentedRadial(surfaces, radii, vertex[:2])

@make_surface.register
def _(obj: trains.Singlet, shape:str='circle', vertex:Sequence[float]=None) -> Surface:
    if vertex is None:
        vertex = np.zeros(3)
    front = make_surface(obj.surfaces[0], 1., vertex)
    back = make_surface(obj.surfaces[1], -1., vertex + (0, 0, obj.thickness))
    z0 = min(obj.surfaces[0].sag_range[0], 0)
    z1 = obj.thickness + max(obj.surfaces[1].sag_range[1], 0)
    if shape == 'circle':
        side = InfiniteCylinder(obj.radius, vertex[:2])
        bound_half_size = obj.radius
    elif shape == 'square':
        side = InfiniteRectangularPrism(obj.radius*2**0.5, center=vertex[:2])
        bound_half_size = obj.radius/2**0.5
    else:
        raise ValueError(f'Unknown shape {shape}.')
    bound = Box((bound_half_size, bound_half_size, (z1 - z0)/2), (vertex[0], vertex[1], vertex[2] + (z1 + z0)/2))
    surface = IntersectionOp((front, back, side), bound)
    return surface

# end make_surface

# begin make_sag function

@singledispatch
def make_sag_function(obj) -> SagFunction:
    raise NotImplementedError(obj)

@make_sag_function.register
def _(obj: trains.SphericalSurface):
    return ZemaxConicSagFunction(obj.roc, obj.radius)

@make_sag_function.register
def _(obj: trains.ConicSurface) -> SagFunction:
    return ZemaxConicSagFunction(obj.roc, obj.radius, obj.kappa, obj.alphas)

@make_sag_function.register
def _(obj: trains.SegmentedSurface):
    warn('No segmented sag function, so using central segment.')
    return make_sag_function(obj.segments[0])

# end make_sag function


# begin make_element

@singledispatch
def make_element(obj, *args, **kwargs) -> Element:
    raise NotImplementedError()

@make_element.register
def  _(obj: trains.Singlet, shape:str='circle', vertex:Sequence[float]=None) -> Element:
    surface = make_surface(obj, shape, vertex)
    return SimpleElement(surface, UniformIsotropic(obj.n), perfect_refractor)

# end make_element

# begin make_elements

@singledispatch
def make_elements(obj, *args, **kwargs) -> List[Element]:
    raise NotImplementedError()

@make_elements.register
def _(obj: trains.SingletSequence, shape:str='circle', start:Sequence[float]=None) -> List[Element]:
    """

    Args:
        obj: Object ot convert.
        shape: 'circle' or 'square'
        start: Start position 3-vector.

    Returns:

    """
    if start is None:
        start = 0, 0, 0
    start = np.array(start, float)
    assert start.shape == (3,)
    elements = []
    z = 0
    for space, singlet in zip(obj.spaces[:-1], obj.singlets):
        z += space
        elements.append(make_element(singlet, shape, start + (0, 0, z)))
        z += singlet.thickness
    return elements

# end make_elements

def to_rectanglar_array_sag(surface: trains.Surface, levels: Sequence[RectangularArrayLevel] = None, side: float=1., center: Sequence[float] = None) -> Surface:
    """Make nested rectangular array sag surface from train surface.

    The geometry of each level of the array is defined in levels, from smallest to largest scale.
    """
    fn = RectangularArraySagFunction.make_multi(make_sag_function(surface), levels)
    return Sag(fn, side, center)

def to_rectangular_array_surface(singlet: trains.Singlet, levels: Sequence[RectangularArrayLevel], center: Sequence3 = (0., 0., 0.)) -> Surface:
    """Make nested rectangular lens array of given singlet.

    The geometry of each level of the array is defined in levels, from smallest to largest scale.
    """
    center = np.array(center, float)
    assert center.shape == (3,)
    # TODO special cases for planar surface
    front = to_rectanglar_array_sag(singlet.surfaces[0], levels, 1., center)
    back = to_rectanglar_array_sag(singlet.surfaces[1], levels, -1., center + (0, 0, singlet.thickness))
    z0 = min(singlet.surfaces[0].sag_range[0], 0)
    z1 = singlet.thickness + max(singlet.surfaces[1].sag_range[1], 0)
    width, height = levels[-1].pitch*levels[-1].size
    side = InfiniteRectangularPrism(width, height, center[:2])
    bound = Box((width/2, height/2, (z1 - z0)/2), (center[0], center[1], center[2] + (z1 + z0)/2))
    surface = IntersectionOp((front, back, side), bound)
    return surface

def to_rectangular_array_element(singlet: trains.Singlet, levels: Sequence[RectangularArrayLevel], center: Sequence3 = (0., 0., 0.)) -> Element:
    surface = to_rectangular_array_surface(singlet, levels, center)
    return SimpleElement(surface, UniformIsotropic(singlet.n), perfect_refractor)

def to_rectangular_array_elements(train: trains.SingletSequence, levels: Sequence[RectangularArrayLevel], center: Sequence3 = (0., 0., 0.)) -> List[Element]:
    center = np.array(center, float)
    assert center.shape == (3, )

    elements = []
    z = 0
    for space, singlet in zip(train.spaces[:-1], train.singlets):
        z += space
        elements.append(to_rectangular_array_element(singlet, levels, center + (0, 0, z)))
        z += singlet.thickness
    return elements