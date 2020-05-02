from typing import Sequence
import numpy as np
from .. import *
from .base import *
from ... import v4h

__all__ = []

@getsag.register
def _(s:ZemaxConicSagFunction, x:Sequence[float]) -> float:
    rho2 = min(v4h.norm_squared(x[:2]), s.radius**2)
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
    return z

@getsag.register
def _(s:RectangularArraySagFunction, x: Sequence[float]) -> float:
    if s.size is None:
        q = abs(np.mod(x + s.pitch/2, s.pitch) - s.pitch/2)
    else:
        n = np.clip(np.floor(x/s.pitch + s.size/2), 0, s.size - 1)
        q = abs(x - (n + 0.5 - s.size/2)*s.pitch)
        if s.clamp:
            q = np.minimum(q, s.pitch/2)
    return getsag(s.unit, q)