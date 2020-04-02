from functools import  singledispatch
from itertools import accumulate
from typing import Sequence, List
import numpy as np
from .. import trains
from ..sdb import *
from . import *

__all__ = ['make_element', 'make_surface', 'make_elements']

@singledispatch
def make_element(obj, *args, **kwargs) -> Element:
    raise NotImplementedError()

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
    if shape == 'circle':
        side = InfiniteCylinder(obj.radius, vertex[:2])
    elif shape == 'square':
        side = InfiniteRectangularPrism(obj.radius*2**0.5, center=vertex[:2])
    else:
        raise ValueError(f'Unknown shape {shape}.')
    surface = IntersectionOp((front, back, side))
    return surface

@make_element.register
def  _(obj: trains.Singlet, shape:str='circle', vertex:Sequence[float]=None) -> Element:
    surface = make_surface(obj, shape, vertex)
    return SimpleElement(surface, UniformIsotropic(obj.n), perfect_refractor)

@singledispatch
def make_elements(obj, *args, **kwargs) -> List[Element]:
    raise NotImplementedError()

@make_elements.register
def _(obj: trains.SingletSequence, shape:str='circle', start:Sequence[float]=None) -> List[Element]:
    if start is None:
        start = np.zeros(3)
    elements = []
    z = 0
    for space, singlet in zip(obj.spaces[:-1], obj.singlets):
        z += space
        elements.append(make_element(singlet, shape, start + (0, 0, z)))
        z += singlet.thickness
    return elements







