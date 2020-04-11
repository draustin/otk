import numpy as np
from numba import njit
from ..geometry import *
from .base import  *

@gen_getsdb.register
def _(s: Box):
    half_size = s.half_size
    center = s.center
    radius = s.radius
    @njit("f8(f8[:])")
    def g(x:np.ndarray):
        q = np.abs(x[:3] - center) - (half_size - radius)
        return norm(np.maximum(q, 0.)) + min(max(q[0], max(q[1], q[2])), 0.0) - radius
    return g

@gen_getsdb.register
def _(s: InfiniteCylinder):
    o = s.o
    r = s.r
    @njit("f8(f8[:])")
    def g(x):
        return norm(x[:2] - o) - r
    return g

@gen_getsdb.register
def _(s: InfiniteRectangularPrism):
    center = s.center
    half_size = np.asarray((s.width/2, s.height/2))
    @njit("f8(f8[:])")
    def g(x):
        q = np.abs(x[:2] - center) - half_size
        return norm(np.maximum(q, 0.0)) + min(max(q[0], q[1]), 0.0)
    return g

@gen_getsdb.register
def _(s:Plane):
    n = s.n
    c = s.c
    @njit("f8(f8[:])")
    def g(x):
        # It doesn't like dot???
        #return dot(x[:3], n) + c
        return x[0]*n[0] + x[1]*n[1] + x[2]*n[2] + c
    return g

@gen_getsdb.register
def _(s: Sag):
    getsag = gen_getsag(s.sagfun)
    origin = s.origin
    side = s.side
    lipschitz = s.lipschitz
    @njit("f8(f8[:])")
    def g(x):
        sag = getsag(x[:2] - origin[:2])
        return side*(sag + origin[2] - x[2])/lipschitz
    return g