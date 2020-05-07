import numpy as np
from warnings import warn
from numba import njit
from ... import v4h
from ...functions import norm, norm_squared
from ..geometry import *
from .base import  *
from .base import get_cached_getsdb, surface_labels

@gen_getsdb.register
def _(s: Box):
    half_size = s.half_size
    center = s.center
    radius = s.radius
    @njit#("f8(f8[:])")
    def g(x:np.ndarray):
        q = np.abs(x[:3] - center) - (half_size - radius)
        return norm(np.maximum(q, 0.)) + min(max(q[0], max(q[1], q[2])), 0.0) - radius
    return g

@gen_identify.register(Box)
def _(surface):
    warn('Not identifying box face.')
    getsdb = get_cached_getsdb(surface)
    label = surface_labels.get_label(surface)
    @njit
    def identify(x):
        return getsdb(x), label, 0
    return identify

@gen_getsdb.register
def _(s: InfiniteCylinder):
    o = s.o
    r = s.r
    @njit#("f8(f8[:])")
    def g(x):
        return norm(x[:2] - o) - r
    return g

@gen_getsdb.register
def _(s: InfiniteRectangularPrism):
    center = s.center
    half_size = np.asarray((s.width/2, s.height/2))
    @njit#("f8(f8[:])")
    def g(x):
        q = np.abs(x[:2] - center) - half_size
        return norm(np.maximum(q, 0.0)) + min(max(q[0], q[1]), 0.0)
    return g

@gen_identify.register(InfiniteRectangularPrism)
def _(surface):
    warn('Not identifying prism face.')
    getsdb = get_cached_getsdb(surface)
    label = surface_labels.get_label(surface)
    @njit
    def identify(x):
        return getsdb(x), label, 0
    return identify

@gen_getsdb.register
def _(s:Plane):
    n = s.n
    c = s.c
    @njit#("f8(f8[:])")
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
    @njit#("f8(f8[:])")
    def g(x):
        sag = getsag(x[:2] - origin[:2])
        return side*(sag + origin[2] - x[2])/lipschitz
    return g

@gen_getsdb.register
def _(s: ZemaxConic):
    vertex = s.vertex
    radius_sqd = s.radius**2
    roc = s.roc
    kappa = s.kappa
    alphas = np.asarray(s.alphas)
    side = s.side
    lipschitz = s.lipschitz
    @njit#("f8(f8[:])")
    def getsdb(x):
        xp = x[:3] - vertex
        rho2 = min(norm_squared(xp[:2]), radius_sqd)
        rho = rho2**0.5
        if np.isfinite(roc):
            z = rho2/(roc*(1 + (1 - kappa*rho2/roc**2)**0.5))
        else:
            z = 0
        if len(alphas) > 0:
            h = alphas[-1]
            for alpha in alphas[-2::-1]:
                h = h*rho + alpha
            z += h*rho2
        return side*(z - xp[2])/lipschitz
    return getsdb

@gen_getsdb.register
def _(s: SphericalSag):
    roc = s.roc
    side = s.side
    center = s.center
    vertexz = s.vertex[2]
    inside = side*np.sign(roc)
    @njit#("f8(f8[:])")
    def getsdb(x):
        if np.isfinite(roc):
            a = inside*(norm(x[:3] - center) - abs(roc))
            b = -side*(x[2] - center[2])
            if inside > 0:
                return min(a, b)
            else:
                return max(a, b)
        else:
            return side*(vertexz - x[2])
    return getsdb

# Primitives with only one face.
@gen_identify.register(Sag)
@gen_identify.register(ToroidalSag)
@gen_identify.register(ZemaxConic)
@gen_identify.register(BoundedParaboloid)
@gen_identify.register(SphericalSag)
@gen_identify.register(Hemisphere)
@gen_identify.register(Plane)
@gen_identify.register(InfiniteCylinder)
@gen_identify.register(Sphere)
def _(surface):
    getsdb = get_cached_getsdb(surface)
    label = surface_labels.get_label(surface)
    @njit
    def identify(x):
        return getsdb(x), label, 0
    return identify