import numpy as np
from numba import njit

from .base import *
from .. import *
from ...functions import norm_squared

__all__ = []

@gen_getsag.register
def _(s:ZemaxConicSagFunction):
    roc = s.roc
    radius_sqd = s.radius**2
    kappa = s.kappa
    alphas = np.asarray(s.alphas)
    @njit#("f8(f8[:])")
    def g(x):
        rho2 = min(norm_squared(x[:2]), radius_sqd)
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
        return z
    return g

@gen_getsag.register
def _(s:RectangularArraySagFunction):
    pitch = s.pitch
    size = s.size
    clamp = s.clamp
    getsag_unit = gen_getsag(s.unit)
    if size is None:
        @njit#("f8(f8[:])")
        def g(x):
            q = np.abs(np.mod(x + pitch/2, pitch) - pitch/2)
            return getsag_unit(q)
    else:
        @njit#("f8(f8[:])")
        def g(x):
            n = np.minimum(np.maximum(np.floor(x/pitch + size/2), 0), size - 1)
            q = np.abs(x - (n + 0.5 - size/2)*pitch)
            if clamp:
                q = np.minimum(q, pitch/2)
            return getsag_unit(q)
    return g

