import numpy as np
from typing import Sequence
from dataclasses import dataclass
from functools import singledispatch
#from ..vector3 import *
from ..geometry import *

__all__ = ['getsdb', 'identify', 'spheretrace', 'getnormal', 'norm', 'normalize', 'dot', 'cross', 'norm_squared', 'getsag', 'containing']

def dot(a, b):
    return np.dot(a[:3], b[:3])

def norm_squared(x):
    return dot(x, x)

def norm(x):
    return dot(x, x)**0.5

def normalize(x):
    return x/norm(x)

def cross(a, b):
    return np.r_[np.cross(a[:3], b[:3]), 0]

@dataclass
class ISDB:
    d: float
    surface: Surface
    face: int

@singledispatch
def getsdb(surface: Surface, x: Sequence[float]) -> float:
    """Get signed distance bound.

    Args:
        x: Position 4-vector.
    """
    raise NotImplementedError(surface)

@singledispatch
def identify(surface: Surface, x: Sequence[float]) -> ISDB:
    """Identify bounding surface.

    Args:
        x: Position - see getsdb.
    """
    raise NotImplementedError()

@singledispatch
def getsag(s, x: Sequence[float]) -> float:
    """

    Args:
        s:
        x: Position 2-vector.
    """
    raise NotImplementedError()

@singledispatch
def getnormal(surface:Surface, x, h=1e-7) -> np.ndarray:
    ks = (np.asarray((1, 1, 1, 0)), np.asarray((1, -1, -1, 0)),
        np.asarray((-1, 1, -1, 0)), np.asarray((-1, -1, 1, 0)))
    return normalize(sum(k*getsdb(surface, x + k*h) for k in ks))

@identify.register
def _(surface:Primitive, x):
    d = getsdb(surface, x)
    return ISDB(d, surface, 0)

@identify.register
def _(s:Compound, x: np.ndarray) -> ISDB:
    isdbs = [s.identify(x) for s in s.surfaces]
    index = s.which(isdbs)
    return isdbs[index]

@singledispatch
def which(s:Compound, isdbs: Sequence[ISDB]) -> int:
    raise NotImplementedError()

@getsdb.register
def _(self:UnionOp, p):
    d = getsdb(self.surfaces[0], p)
    for surface in self.surfaces[1:]:
        d = np.minimum(d, getsdb(surface, p))
    return d

@which.register
def _(self:UnionOp, isdbs: Sequence[ISDB]) -> int:
    d0 = isdbs[0].d
    index0 = 0
    for index, isdb in zip(range(1, len(isdbs)), isdbs[1:]):
        if isdb.d < d0:
            d0 = isdb.d
            index0 = index
    return index0

@singledispatch
def containing(s:Surface, x:Sequence[float]):
    raise NotImplementedError()

@containing.register
def _(s:Primitive, x:Sequence[float]):
    d = getsdb(s, x)
    if getsdb(s, x) <= 0:
        yield s
    return d

@containing.register
def _(self:UnionOp, x:Sequence[float]):
    d = yield from containing(self.surfaces[0], x)
    for child in self.surfaces[1:]:
        d_child = yield from containing(child, x)
        d = min(d, d_child)
    if d <= 0:
        yield self
    return d

@getsdb.register
def _(self:IntersectionOp, x):
    d = getsdb(self.surfaces[0], x)
    for surface in self.surfaces[1:]:
        d = np.maximum(d, getsdb(surface, x))
    return d

@which.register
def _(self:IntersectionOp, isdbs: Sequence[ISDB]) -> int:
    d0 = isdbs[0].d
    index0 = 0
    for index, isdb in zip(range(1, len(isdbs)), isdbs[1:]):
        if isdb.d > d0:
            d0 = isdb.d
            index0 = index
    return index0

@containing.register
def _(self:IntersectionOp, x:Sequence[float]):
    d = yield from containing(self.surfaces[0], x)
    for child in self.surfaces[1:]:
        d_child = yield from containing(child, x)
        d = max(d, d_child)
    if d <= 0:
        yield self
    return d

@getsdb.register
def _(self:DifferenceOp, p) -> np.ndarray:
    d0 = getsdb(self.surfaces[0], p)
    d1 = getsdb(self.surfaces[1], p)
    return max(d0, -d1)

@getsdb.register
def _(self:AffineOp, p:Sequence[float]) -> int:
    d = getsdb(self.surfaces[0], np.dot(p, self.invm))
    return d*self.scale

@containing.register
def _(self:AffineOp, x:Sequence[float]):
    d = yield from containing(self.surfaces[0], np.dot(x, self.invm))
    return d*self.scale

@getsdb.register
def _(s:SegmentedRadial, x):
    rho = norm(x[:2] - s.vertex)
    for ss, r in zip(s.surfaces[:-1], s.radii):
        if rho <= r:
            return getsdb(ss, x)
    return getsdb(s.surfaces[-1], x)

@singledispatch
def transform(s: Surface, x:Sequence[float]) -> np.ndarray:
    pass

@transform.register
def _(s: FiniteRectangularArray, x: Sequence[float]) -> np.ndarray:
    index = np.clip(np.floor((x[:2] - s.corner)/s.pitch), 0, s.size - 1)
    center = (index + 0.5)*s.pitch + s.corner
    return np.r_[x[:2] - center, x[2:]]

@getsdb.register
def _(s:FiniteRectangularArray, x):
    return getsdb(s.surfaces[0], transform(s, x))

@containing.register
def _(s:FiniteRectangularArray, x):
    return (yield from containing(s.ch))

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
    x0 = np.asarray(x0)
    v = np.asarray(v)
    assert x0.shape == (4,)
    assert v.shape == (4,)
    assert x0[3] == 1.0
    assert v[3] == 0.0
    assert t_max > 0

    t = 0
    steps = 0 # number of steps taken
    x = x0
    last_x = None
    last_dp = None
    last_t = None
    if sign is None:
        d0 = getsdb(surface, x)
        sign = np.sign(d0)
    while True:
        dp = getsdb(surface, x)*sign
        if dp < 0 and last_dp <= epsilon:
            #assert last_dp <= epsilon, f'{last_dp}, {dp}'
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

    return SphereTrace(dp*sign, t, x, steps, last_dp*sign, last_t, last_x)