import numpy as np
from numba import njit
from functools import singledispatch
from ... import v4
from ..geometry import *

__all__ = ['norm', 'dot', 'norm_squared', 'gen_getsdb', 'gen_getsag']

#norm = njit(v4.norm)


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

@gen_getsdb.register
def _(u: UnionOp):
    # List doesn't work for some reason.
    gs = tuple(gen_getsdb(child) for child in u.surfaces)
    @njit("f8(f8[:])")
    def g(x):
        d = gs[0](x)
        for child_g in gs[1:]:
            d = min(d, child_g(x))
        return d
    return g

@gen_getsdb.register
def _(surface: IntersectionOp):
    gs = tuple(gen_getsdb(child) for child in surface.surfaces)
    @njit("f8(f8[:])")
    def g(x):
        d = gs[0](x)
        for child_g in gs[1:]:
            d = max(d, child_g(x))
        return d
    return g

@gen_getsdb.register
def _(surface: DifferenceOp):
    g0 = gen_getsdb(surface.surfaces[0])
    g1 = gen_getsdb(surface.surfaces[1])
    @njit("f8(f8[:])")
    def g(x):
        return max(g0(x), -g1(x))
    return g

# @gen_getsdb.register
# def _(u: UnionOp):
#     g0 = gen_getsdb(u.surfaces[0])
#     g1 = gen_getsdb(u.surfaces[1])
#     @njit("f8(f8[:])")
#     def g(x):
#         return min(g0(x), g1(x))
#     return g