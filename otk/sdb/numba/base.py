import logging
import numpy as np
from typing import Sequence
from numba import njit, typeof
from functools import singledispatch
from ..geometry import *
from ..npscalar.base import SphereTrace

__all__ = ['norm', 'dot', 'norm_squared', 'gen_getsdb', 'gen_getsag', 'spheretrace']

logger = logging.getLogger(__name__)

@njit("f8(f8[:], f8[:])")
def dot(x, y):
    s = 0
    for i in range(len(x)):
        s += x[i]*y[i]
    return s

@njit("f8(f8[:])")
def norm_squared(x):
    return dot(x, x)

@njit("f8(f8[:])")
def norm(x):
    return norm_squared(x)**0.5

@singledispatch
def gen_getsdb(surface):
    raise NotImplementedError(surface)

@singledispatch
def gen_getsag(sagfun):
    raise NotImplementedError(sagfun)

def gen_reduce(op, funs):
    fun0 = funs[0]
    if len(funs) > 2:
        oneless = gen_reduce(op, funs[1:])
        @njit
        def result(x):
            return op(fun0(x), oneless(x))
    else:
        fun1 = funs[1]
        @njit
        def result(x):
            return op(fun0(x), fun1(x))
    return result

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
    return gen_reduce(min, gs)

@gen_getsdb.register
def _(surface: IntersectionOp):
    gs = tuple(get_cached_getsdb(child) for child in surface.surfaces)
    # @njit("f8(f8[:])")
    # def g(x):
    #     d = gs[0](x)
    #     #for child_g in gs[1:]:
    #     #    d = max(d, child_g(x))
    #     for i in range(1, len(gs)):
    #         d = max(d, gs[i](x))
    #     return d
    return gen_reduce(max, gs)

@gen_getsdb.register
def _(surface: DifferenceOp):
    g0 = get_cached_getsdb(surface.surfaces[0])
    g1 = get_cached_getsdb(surface.surfaces[1])
    @njit("f8(f8[:])")
    def g(x):
        return max(g0(x), -g1(x))
    return g

@gen_getsdb.register
def _(s:SegmentedRadial):
    vertex = s.vertex
    radii = s.radii
    getsdbs = tuple(get_cached_getsdb(child) for child in s.surfaces)
    @njit("f8(f8[:])")
    def g(x):
        rho = norm(x[:2] - vertex)
        for getsdb, r in zip(getsdbs[:-1], radii):
            if rho <= r:
                return getsdb(x)
        return getsdbs[-1](x)
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

def lookup_cache(cache: dict, key, calc_value):
    try:
        return cache[key]
    except KeyError:
        value = calc_value()
        cache[key] = value
        return value

getsdb_cache = {}
spheretrace_cache = {}

def get_cached_getsdb(surface: Surface):
    return lookup_cache(getsdb_cache, surface, lambda : gen_getsdb(surface))

def gen_spheretrace(surface: Surface):
    logger.info(f'Generating spheretrace for {surface}.')
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
    inner = lookup_cache(spheretrace_cache, surface, lambda : gen_spheretrace(surface))
    result = inner(x0, v, t_max, epsilon, max_steps, sign, through)
    return SphereTrace(*result)