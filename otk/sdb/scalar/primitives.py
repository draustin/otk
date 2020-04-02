import numpy as np
from .. import *
from .base import *
from ...v4 import *

__all__ = []

@getsdb.register
def _(s:Sphere, x):
    return norm(x[:3] - s.o) - s.r

@getsdb.register
def _(s:InfiniteCylinder, x):
    return norm(x[:2] - s.o) - s.r

@getsdb.register
def _(s:InfiniteRectangularPrism, x):
    q = abs(x[:2] - s.center) - (s.width/2, s.height/2)
    return norm(np.maximum(q, 0.0)) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)

@getsdb.register
def _(s:Plane, x):
    return np.dot(x[:3], s.n) + s.c

@getsdb.register
def _(s:Hemisphere, x):
    return np.minimum(norm(x[:3] - s.o) - s.r, s.sign*(x[2] - s.o[2]))

@getsdb.register
def _(s:SphericalSag, x):
    if np.isfinite(s.roc):
        inside = s.side*np.sign(s.roc)
        a = inside*(norm(x[:3] - s.center) - abs(s.roc))
        b = -s.side*(x[2] - s.center[2])
        fun = np.minimum if inside > 0 else np.maximum
        return fun(a, b)
    else:
        return s.side*(s.vertex[2] - x[2])

@getsdb.register
def _(s:BoundedParaboloid, x):
    xp = x[:3] - s.vertex
    d = (xp[2] - min(xp[0]**2 + xp[1]**2, s.radius**2)/(2*s.roc))*s.cos_theta
    if not s.side:
        d = -d
    return d

@getsdb.register
def _(s:ZemaxConic, x):
    xp = x[:3] - s.vertex
    rho2 = min(norm_squared(xp[:2]), s.radius**2)
    rho = rho2**0.5
    if np.isfinite(s.roc):
        z = rho2/(s.roc*(1 + (1 - s.kappa*rho2/s.roc**2)**0.5))
    else:
        z = 0
    if len(s.alphas) > 0:
        h = s.alphas[-1]
        for alpha in s.alphas[-2::-1]:
            h = h*rho + alpha
        z += h*rho2
    return s.side*(z - xp[2])/s.lipshitz

@getsdb.register
def _(s:ToroidalSag, x):
    pass

@getsdb.register
def _(s:Sag, x):
    sag = getsag(s.sagfun, x[:2] - s.origin[:2])
    return s.side*(sag + s.origin[2] - x[2])/s.lipschitz
