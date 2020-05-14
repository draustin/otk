import numpy as np
from typing import Sequence, Callable
from dataclasses import dataclass
from functools import singledispatch
from ...types import Sequence4
from ...functions import norm, normalize

from ..geometry import *

__all__ = ['getsdb', 'identify', 'spheretrace', 'getnormal', 'traverse', 'ISDB']

def getsdb(surface: Surface, x: Sequence4) -> float:
    """Get signed distance bound.

    Args:
        x: Position 4-vector.
    """
    return surface.getsdb(x)

@singledispatch
def identify(surface: Surface, x: Sequence4) -> ISDB:
    """Identify bounding surface.

    Args:
        x: Position - see getsdb.
    """
    raise NotImplementedError(surface)

# @singledispatch
# def getsag(s, x: Sequence[float]) -> float:
#     """
#
#     Args:
#         s:
#         x: Position 2-vector.
#     """
#     raise NotImplementedError()

@singledispatch
def getnormal(surface:Surface, x, h=1e-9) -> np.ndarray:
    ks = (np.asarray((1, 1, 1, 0)), np.asarray((1, -1, -1, 0)),
        np.asarray((-1, 1, -1, 0)), np.asarray((-1, -1, 1, 0)))
    return normalize(sum(k*getsdb(surface, x + k*h) for k in ks))

@identify.register
def _(s: UnionOp, x):
    isdb0 = identify(s.surfaces[0], x)
    for child in s.surfaces[1:]:
        isdb0 = isdb0.min(identify(child, x))
    return isdb0

# @which.register
# def _(self:UnionOp, isdbs: Sequence[ISDB]) -> int:
#     d0 = isdbs[0].d
#     index0 = 0
#     for index, isdb in zip(range(1, len(isdbs)), isdbs[1:]):
#         if isdb.d < d0:
#             d0 = isdb.d
#             index0 = index
#     return index0

@singledispatch
def traverse(s:Surface, x:Sequence[float]):
    """Depth-first traversal starting at s yielding (surface, signed distance) for each descendent."""
    raise NotImplementedError(s)

@traverse.register
def _(s:Primitive, x:Sequence[float]):
    d = getsdb(s, x)
    yield s, d
    return d

@traverse.register
def _(self:UnionOp, x:Sequence[float]):
    d = yield from traverse(self.surfaces[0], x)
    for child in self.surfaces[1:]:
        d_child = yield from traverse(child, x)
        d = min(d, d_child)
    yield self, d
    return d

@traverse.register
def _(self:DifferenceOp, x:Sequence[float]):
    d0 = yield from traverse(self.surfaces[0], x)
    d1 = yield from traverse(self.surfaces[1], x)
    d = max(d0, -d1)
    yield self, d
    return d

@identify.register
def _(s: IntersectionOp, x):
    isdb0 = identify(s.surfaces[0], x)
    for child in s.surfaces[1:]:
        isdb0 = isdb0.max(identify(child, x))
    return isdb0

# @which.register
# def _(self:IntersectionOp, isdbs: Sequence[ISDB]) -> int:
#     d0 = isdbs[0].d
#     index0 = 0
#     for index, isdb in zip(range(1, len(isdbs)), isdbs[1:]):
#         if isdb.d > d0:
#             d0 = isdb.d
#             index0 = index
#     return index0

@traverse.register
def _(self:IntersectionOp, x:Sequence[float]):
    d = yield from traverse(self.surfaces[0], x)
    for child in self.surfaces[1:]:
        d_child = yield from traverse(child, x)
        d = max(d, d_child)
    yield self, d
    return d

@identify.register
def _(s: DifferenceOp, x):
    isdb0 = identify(s.surfaces[0], x)
    isdb1 = identify(s.surfaces[1], x)
    return isdb0.max(isdb1.negate())

@traverse.register
def _(self:AffineOp, x:Sequence[float]):
    d = yield from traverse(self.surfaces[0], np.dot(x, self.invm))
    return d*self.scale

@identify.register
def _(self: AffineOp, x):
    return identify(self.surfaces[0], np.dot(x, self.invm)).times(self.scale)

@traverse.register
def _(s:SegmentedRadial, x):
    rho = norm(x[:2] - s.vertex)
    d = None
    for ss, r in zip(s.surfaces, s.radii):
        ds = (yield from traverse(ss, x))
        if rho <= r and d is None:
            d = ds
    if d is None:
        d = ds
    return d

@identify.register
def _(s: SegmentedRadial, x):
    rho = norm(x[:2] - s.vertex)
    for ss, r in zip(s.surfaces[:-1], s.radii):
        if rho <= r:
            return identify(ss, x)
    return identify(s.surfaces[-1], x)

@traverse.register
def _(s:FiniteRectangularArray, x):
    return (yield from traverse(s.surfaces[0], s.transform(x)))

@identify.register
def _(s: FiniteRectangularArray, x):
    return identify(s.surfaces[0], s.transform(x))

def sum_weighted(w1, x1s, w2, x2s):
    a1 = w2/(w1 + w2)
    a2 = 1 - a1
    return tuple(x1*a1 + x2*a2 for x1, x2 in zip(x1s, x2s))

@dataclass
class SphereTrace:
    d: float
    t: float
    x: float
    steps: int
    last_d: float
    last_t: float
    last_x: float

    def __post_init__(self):
        self.xm, self.tm = sum_weighted(abs(self.last_d), (self.x, self.t), abs(self.d), (self.last_x, self.last_t))

def _spheretrace(getsdb:Callable, x0:Sequence[float], v:Sequence[float], t_max:float, epsilon:float, max_steps:int, sign:float=None, through:bool=False):
    x0 = np.asarray(x0)
    v = np.asarray(v)
    assert x0.shape == (4,)
    assert v.shape == (4,)
    assert x0[3] == 1.0
    assert v[3] == 0.0
    assert t_max > 0

    t = 0
    steps = 0  # number of steps taken
    x = x0
    last_x = None
    last_dp = None
    last_t = None
    if sign is None:
        d0 = getsdb(x)
        sign = np.sign(d0)
    while True:
        dp = getsdb(x)*sign
        if dp < 0 and last_dp <= epsilon:
            # assert last_dp <= epsilon, f'{last_dp}, {dp}'
            assert through
            assert dp >= -epsilon
            break
        elif dp <= epsilon and not through:
            break
        elif steps == max_steps:
            break
        last_x = x
        last_dp = dp
        last_t = t
        # With infinite precision, small step would be epsilon.
        t += max(dp, epsilon/2)
        x = x0 + t*v
        steps += 1
        if t > t_max:
            break

    return dp*sign, t, x, steps, last_dp*sign, last_t, last_x

def spheretrace(surface:Surface, x0:Sequence[float], v:Sequence[float], t_max:float, epsilon:float, max_steps:int, sign:float=None, through:bool=False):
    """Spheretrace scalar (no broadcasting).

        Args:
            getsdb:
            x0: Initial position.
            v:
            t_max:
            epsilon:
            max_steps:
            through:

        Returns:

        """
    getsdb_surface = lambda x: getsdb(surface, x)
    result = _spheretrace(getsdb_surface, x0, v, t_max, epsilon, max_steps, sign, through)
    return SphereTrace(*result)