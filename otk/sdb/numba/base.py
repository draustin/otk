import numpy as np
from typing import Sequence
from numba import njit, typeof
from functools import singledispatch
from ...functions import norm, normalize
from ..geometry import *
from ..npscalar.base import SphereTrace

__all__ = ['gen_getsdb', 'gen_getsag', 'spheretrace', 'identify', 'gen_identify', 'getnormal']

@singledispatch
def gen_getsdb(surface):
    raise NotImplementedError(surface)

@singledispatch
def gen_getsag(sagfun):
    raise NotImplementedError(sagfun)

@singledispatch
def gen_identify(surface):
    raise NotImplementedError(surface)

ks_getnormal = (
    np.asarray((1, 1, 1, 0)),
    np.asarray((1, -1, -1, 0)),
    np.asarray((-1, 1, -1, 0)),
    np.asarray((-1, -1, 1, 0))
)

@singledispatch
def getnormal(surface:Surface, x, h=1e-9) -> np.ndarray:
    getsdb = get_cached_getsdb(surface)
    return normalize(sum(k*getsdb(x + k*h) for k in ks_getnormal))

def gen_reduce_onebyone(op, funs):
    fun0 = funs[0]
    if len(funs) > 2:
        oneless = gen_reduce_onebyone(op, funs[1:])
        @njit
        def result(x):
            return op(fun0(x), oneless(x))
    else:
        fun1 = funs[1]
        @njit
        def result(x):
            return op(fun0(x), fun1(x))
    return result

def gen_reduce_pairwise(op, funs):
    if len(funs) == 2:
        f0, f1 = funs
        @njit
        def result(x):
            return op(f0(x), f1(x))
    elif len(funs) == 3:
        f0, f1, f2 = funs
        @njit
        def result(x):
            return op(f0(x), op(f1(x), f2(x)))
    else:
        n = int(len(funs)/2)
        fa = gen_reduce_pairwise(op, funs[:n])
        fb = gen_reduce_pairwise(op, funs[n:])
        @njit
        def result(x):
            return op(fa(x), fb(x))
    return result

@njit
def min_first(a, b):
    if a[0] <= b[0]:
        return a
    else:
        return b

@njit
def max_first(a, b):
    if a[0] >= b[0]:
        return a
    else:
        return b


@gen_getsdb.register
def _(u: UnionOp):
    # List doesn't work for some reason.
    gs = [get_cached_getsdb(child) for child in u.surfaces]
    # @njit("f8(f8[:])")
    # def g(x):
    #     d = gs[0](x)
    #     for child_g in gs[1:]:
    #         d = min(d, child_g(x))
    #     return d
    return gen_reduce_pairwise(min, gs)

@gen_identify.register
def _(u: UnionOp):
    gs = [get_cached_identify(child) for child in u.surfaces]
    return gen_reduce_pairwise(min_first, gs)


@gen_getsdb.register
def _(surface: IntersectionOp):
    gs = tuple(get_cached_getsdb(child) for child in surface.surfaces)
    return gen_reduce_pairwise(max, gs)

@gen_identify.register
def _(surface: IntersectionOp):
    gs = tuple(get_cached_identify(child) for child in surface.surfaces)
    return gen_reduce_pairwise(max_first, gs)

@gen_getsdb.register
def _(surface: DifferenceOp):
    g0 = get_cached_getsdb(surface.surfaces[0])
    g1 = get_cached_getsdb(surface.surfaces[1])
    @njit#("f8(f8[:])")
    def g(x):
        return max(g0(x), -g1(x))
    return g

@gen_identify.register
def _(s: DifferenceOp):
    g0 = get_cached_identify(s)
    g1 = get_cached_identify(s)
    @njit
    def g(x):
        di0 = g0(x)
        di1 = g1(x)
        di1 = (-di1[0],) + (di1[1:])
        return max_first(di0, di1)
    return g

@gen_getsdb.register
def _(s:SegmentedRadial):
    vertex = s.vertex
    radii = s.radii
    getsdbs = tuple(get_cached_getsdb(child) for child in s.surfaces)
    @njit#("f8(f8[:])")
    def g(x):
        rho = norm(x[:2] - vertex)
        for getsdb, r in zip(getsdbs[:-1], radii):
            if rho <= r:
                return getsdb(x)
        return getsdbs[-1](x)
    return g

@gen_identify.register
def _(s:SegmentedRadial):
    vertex = s.vertex
    radii = s.radii
    identifies = tuple(get_cached_identify(child) for child in s.surfaces)
    @njit
    def g(x):
        rho = norm(x[:2] - vertex)
        for identify, r in zip(identifies[:-1], radii):
            if rho <= r:
                return identify(x)
        return identifies[-1](x)

    return g

@gen_getsdb.register
def _(s: FiniteRectangularArray):
    getsdb_child = get_cached_getsdb(s.surfaces[0])
    corner = s.corner
    pitch = s.pitch
    size = s.size
    @njit
    def getsdb(x):
        index = np.minimum(np.maximum(np.floor((x[:2] - corner)/pitch), 0), size - 1)
        center = (index + 0.5)*pitch + corner
        xp = np.array((x[0] - center[0], x[1] - center[1], x[2], x[3]))
        return getsdb_child(xp)
    return getsdb

@gen_identify.register
def _(s: FiniteRectangularArray):
    identify_child = get_cached_identify(s.surfaces[0])
    corner = s.corner
    pitch = s.pitch
    size = s.size
    @njit
    def identify(x):
        index = np.minimum(np.maximum(np.floor((x[:2] - corner)/pitch), 0), size - 1)
        center = (index + 0.5)*pitch + corner
        xp = np.array((x[0] - center[0], x[1] - center[1], x[2], x[3]))
        return identify_child(xp)
    return identify

def lookup_cache(cache: dict, key, calc_value):
    try:
        value = cache[key]
    except KeyError:
        value = calc_value()
        cache[key] = value

    return value

getsdb_cache = {}
spheretrace_cache = {}
identify_cache = {}

class IntLabels:
    def __init__(self):
        self._objects = []
        self._labels = {}

    def get_label(self, obj: object) -> int:
        try:
            return self._labels[obj]
        except KeyError:
            label = len(self._objects)
            self._objects.append(obj)
            self._labels[obj] = label
            return label

    def get_object(self, label: int) -> object:
        return self._objects[label]

surface_labels = IntLabels()

def get_cached_getsdb(surface: Surface):
    return lookup_cache(getsdb_cache, surface, lambda : gen_getsdb(surface))

def get_cached_spheretrace(surface: Surface):
    return lookup_cache(spheretrace_cache, surface, lambda : gen_spheretrace(surface))

def get_cached_identify(surface: Surface):
    return lookup_cache(identify_cache, surface, lambda : gen_identify(surface))

def gen_spheretrace(surface: Surface):
    getsdb = get_cached_getsdb(surface)

    @njit
    def spheretrace(x0:Sequence[float], v:Sequence[float], t_max:float, epsilon:float, max_steps:int, sign:float=None, through:bool=False):
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

    return spheretrace

def spheretrace(surface:Surface, x0:Sequence[float], v:Sequence[float], t_max:float, epsilon:float, max_steps:int, sign:float=None, through:bool=False):
    inner = get_cached_spheretrace(surface)
    result = inner(x0, v, t_max, epsilon, max_steps, sign, through)
    return SphereTrace(*result)

def identify(surface: Surface, x: Sequence[float]) -> ISDB:
    fun = get_cached_identify(surface)
    d, label, face = fun(x)
    id_surface = surface_labels.get_object(label)
    return ISDB(d, id_surface, face)