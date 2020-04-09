from functools import  singledispatch
from itertools import accumulate
from typing import Sequence, List
import numpy as np
from .. import trains
from ..sdb import *
from . import *
from . import scalar

__all__ = ['make_element', 'make_surface', 'make_elements', 'make_square_array_surface', 'make_square_array_element',
    'make_square_array_elements', 'make_sag_function']

# begin make_surface

@singledispatch
def make_surface(obj, *args, **kwargs) -> Surface:
    raise NotImplementedError()

@make_surface.register
def _(obj: trains.Surface, side: float=1., vertex:Sequence[float]=None):
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
def make_sag_function(obj: trains.Surface) -> SagFunction:
    return ZemaxConicSagFunction(obj.roc, obj.radius)

@make_sag_function.register
def _(obj: trains.ConicSurface) -> SagFunction:
    return ZemaxConicSagFunction(obj.roc, obj.radius, obj.kappa, obj.alphas)

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

# begin make_square_array_surface

@singledispatch
def make_square_array_surface(obj, *args, **kwargs) -> Surface:
    raise NotImplementedError(obj)

@make_square_array_surface.register
def _(obj: trains.Surface, side: float=1., vertex: Sequence[float] = None) -> Surface:
    fn = RectangularArraySagFunction(make_sag_function(obj), (obj.radius*2**0.5,)*2)
    return Sag(fn, side, vertex)

@make_square_array_surface.register
def _(obj: trains.Singlet, size: Sequence[int], origin: Sequence[float] = None, pitch: float = None) -> Surface:
    size = np.array(size, int)
    assert size.shape == (2,)
    if origin is None:
        origin = np.zeros(3)
    origin = np.array(origin, float)
    assert origin.shape == (3,)
    # TODO special cases for planar surface
    if pitch is None:
        pitch = obj.radius*2**0.5
    front = make_square_array_surface(obj.surfaces[0], 1, origin + (pitch/2, pitch/2, 0))
    back = make_square_array_surface(obj.surfaces[1], -1, origin + (pitch/2, pitch/2, obj.thickness))
    z0 = min(obj.surfaces[0].sag_range[0], 0)
    z1 = obj.thickness + max(obj.surfaces[1].sag_range[1], 0)
    width, height = pitch*size
    side = InfiniteRectangularPrism(width, height, origin[:2])
    bound = Box((width/2, height/2, (z1 - z0)/2), (origin[0], origin[1], origin[2] + (z1 + z0)/2))
    surface = IntersectionOp((front, back, side), bound)
    return surface

# end make_square_array_surface

# begin make_square_array_element
@singledispatch
def make_square_array_element(obj, size: Sequence[int], origin: Sequence[float] = None) -> Element:
    raise NotImplementedError(obj)

@make_square_array_element.register
def _(obj: trains.Singlet, size: Sequence[int], origin: Sequence[float] = None, pitch: float = None) -> Element:
    surface = make_square_array_surface(obj, size=size, origin=origin, pitch=pitch)
    return SimpleElement(surface, UniformIsotropic(obj.n), perfect_refractor)
# end make_square_array_element

# begin make_square_array_elements
@singledispatch
def make_square_array_elements(obj, size: Sequence[int], start: Sequence[float] = None, pitch: float = None) -> List[Element]:
    raise NotImplementedError(obj)

@make_square_array_elements.register
def _(obj: trains.SingletSequence, size: Sequence[int], start: Sequence[float] = None, pitch: float = None) -> List[Element]:
    if start is None:
        start = 0, 0, 0
    start = np.array(start, float)
    assert start.shape == (3,)

    if pitch is None:
        pitch = obj.singlets[0].radius*2**0.5

    elements = []
    z = 0
    for space, singlet in zip(obj.spaces[:-1], obj.singlets):
        z += space
        elements.append(make_square_array_element(singlet, size, start + (0, 0, z), pitch))
        z += singlet.thickness
    return elements
# end make_square_array_elements

