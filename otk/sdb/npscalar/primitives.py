import numpy as np
from warnings import warn
from .. import *
from .base import *
from ...functions import norm, norm_squared

__all__ = []


@identify.register
def _(s: InfiniteRectangularPrism, x):
    d = getsdb(s, x)
    warn('Face not identified')
    return ISDB(d, s, 0)

@identify.register
def _(s: Box, x):
    d = getsdb(s, x)
    warn('Face not identified')
    return ISDB(d, s, 0)

# Primitives with only one face.
@identify.register(Sag)
@identify.register(ToroidalSag)
@identify.register(ZemaxConic)
@identify.register(BoundedParaboloid)
@identify.register(SphericalSag)
@identify.register(Hemisphere)
@identify.register(Plane)
@identify.register(InfiniteCylinder)
@identify.register(Sphere)
def _(surface, x):
    d = getsdb(surface, x)
    return ISDB(d, surface, 0)

