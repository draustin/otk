import itertools
from dataclasses import dataclass
from functools import singledispatch
from typing import Sequence, Tuple
import numpy as np
from ..types import Vector4
from .. import v4h, v3h

__all__ = ['union', 'intersection', 'difference', 'isclose', 'Interval', 'AABR', 'AABB']

@singledispatch
def union(obj, *args):
    raise NotImplementedError(obj)

@singledispatch
def intersection(obj, *args):
    raise NotImplementedError(obj)

@singledispatch
def difference(a, b):
    raise NotImplementedError((a, b))

@singledispatch
def isclose(a, b) -> bool:
    raise NotImplementedError((a, b))

@dataclass
class Interval:
    a: float
    b: float

    @property
    def empty(self):
        return self.b >= self.a

@union.register
def _(obj: Interval, *args):
    objs = (obj,) + args
    return Interval(min(o.a for o in objs), max(o.b for o in objs))

@intersection.register
def _(obj: Interval, *args):
    objs = (obj,) + args
    return Interval(max(o.a for o in objs), min(o.b for o in objs))


@difference.register
def _(i0: 'Interval', i1: 'Interval'):
    if i1.empty or i0.empty:
        return i0
    elif i1.a <= i0.a:
        return Interval(max(i0.a, i1.b), i0.b)
    elif i1.a <= i0.b:
        if i1.b <= i0.b:
            return i0
        else:
            return Interval(i0.a, i1.a)
    else:
        return i0

@isclose.register
def _(i0: Interval, i1: Interval):
    return np.isclose(i0.a, i1.a) and np.isclose(i0.b, i1.b)

@dataclass
class AABR:
    """Axis-aligned bounding rectangle."""
    corner0: np.ndarray
    corner1: np.ndarray

    def __post_init__(self):
        assert v3h.is_point(self.corner0)
        assert v3h.is_point(self.corner1)

    def split(self) -> Tuple[Interval, Interval]:
        return Interval(self.corner0[0], self.corner1[0]), Interval(self.corner0[1], self.corner1[1])

    @classmethod
    def combine(cls, ix: Interval, iy: Interval):
        return AABR(np.array((ix.a, iy.a, 1)), np.array((ix.b, iy.b, 1)))

    @property
    def empty(self):
        return any(self.corner0 >= self.corner1)

@union.register
def _(b: AABR, *args):
    bs = (b, ) + args
    corner0 = np.minimum.reduce([b.corner0 for b in bs])
    corner1 = np.maximum.reduce([b.corner1 for b in bs])
    return AABR(corner0, corner1)

@intersection.register
def _(b: AABR, *args):
    bs = (b, ) + args
    corner0 = np.maximum.reduce([b.corner0 for b in bs])
    corner1 = np.minimum.reduce([b.corner1 for b in bs])
    return AABR(corner0, corner1)

@difference.register
def _(a: AABR, b: AABR) -> AABR:
    ax, ay = a.split()
    bx, by = b.split()
    dx = difference(ax, bx)
    dy = difference(ay, by)
    if dx.empty:
        ry = dy
    else:
        ry = ay
    if dy.empty:
        rx = dx
    else:
        rx = ax
    return AABR.combine(rx, ry)

@isclose.register
def _(b0: AABR, b1: AABR):
    return np.allclose(b0.corner0, b1.corner0) and np.allclose(b0.corner1, b1.corner1)

@dataclass
class AABB:
    """Axis-aligned bounding box."""
    corners: Tuple[Vector4, Vector4]

    def __post_init__(self):
        assert len(self.corners) == 2
        assert v4h.is_point(self.corners[0])
        assert v4h.is_point(self.corners[1])

    @property
    def empty(self) -> bool:
        return any(self.corners[0] >= self.corners[1])

    @property
    def size(self) -> Vector4:
        return self.corners[1] - self.corners[0]

    @property
    def center(self) -> Vector4:
        return (self.corners[0] + self.corners[1])/2

    def split(self, axis: int) -> Tuple[Interval, AABR]:
        i = Interval(self.corners[0][axis], self.corners[1][axis])
        if axis == 0:
            other_axes = [1, 2, 3]
        elif axis == 1:
            other_axes = [0, 2, 3]
        elif axis == 2:
            other_axes = [1, 2, 3]
        else:
            raise ValueError('Axis should be 0, 1, or 2.')
        return i, AABR(self.corners[0][other_axes], self.corners[1][other_axes])

    @classmethod
    def make(cls, corner0: Sequence[float], corner1: Sequence[float]):
        corner0 = v4h.to_point(corner0)
        corner1 = v4h.to_point(corner1)
        return cls((corner0, corner1))

    @classmethod
    def combine(cls, ix: Interval, iy: Interval, iz: Interval) -> 'AABB':
        return AABB((np.array((ix.a, iy.a, iz.b, 1)), np.array((ix.b, iy.b, iz.b, 1))))

    # @classmethod
    # def union2(cls, b0: 'AABB', b1: 'AABB'):
    #     return AABB(np.minimum(b0.corners[0], b1.corners[0]), np.maximum(b0.corners[1], b1.corners[1]))
    #
    # @classmethod
    # def intersection2(cls, b0: 'AABB', b1: 'AABB'):
    #     return AABB(np.maximum(b0.corners[0], b1.corners[0]), np.minimum(b0.corners[1], b1.corners[1]))

@union.register
def _(b: AABB, *args):
    bs = (b, ) + args
    corner0 = np.minimum.reduce([b.corners[0] for b in bs])
    corner1 = np.maximum.reduce([b.corners[1] for b in bs])
    return AABB((corner0, corner1))

@intersection.register
def _(b: AABB, *args):
    bs = (b, ) + args
    corner0 = np.maximum.reduce([b.corners[0] for b in bs])
    corner1 = np.minimum.reduce([b.corners[1] for b in bs])
    return AABB((corner0, corner1))

@difference.register
def _(b0: AABB, b1: AABB):
    def adjust_axis(axis: int) -> Interval:
        i0, r0 = b0.split(axis)
        i1, r1 = b1.split(axis)
        di = difference(i0, i1)
        dr = difference(r0, r1)
        if dr.empty:
            return di
        else:
            return i0
    intervals = [adjust_axis(axis) for axis in range(3)]
    return AABB.combine(*intervals)

@isclose.register
def _(b0: AABB, b1: AABB):
    return np.allclose(b0.corners[0], b1.corners[0]) and np.allclose(b0.corners[1], b1.corners[1])
