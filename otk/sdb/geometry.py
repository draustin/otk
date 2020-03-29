from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Callable, Union
import numpy as np
from ..vector3 import *

__all__ = ['Surface', 'Sphere', 'Box', 'Torus', 'Ellipsoid', 'InfiniteCylinder', 'Plane', 'Sag', 'SagFunction', 'UnionOp', 'IntersectionOp',
    'DifferenceOp', 'AffineOp', 'Compound', 'Primitive', 'SphericalSag', 'Hemisphere', 'InfiniteRectangularPrism',
    'FiniteRectangularArray', 'ToroidalSag', 'BoundedParaboloid', 'ZemaxConic', 'SegmentedRadial']

class Surface:
    pass

class Primitive(Surface):
    pass

class Sphere(Primitive):
    def __init__(self, r:float, o:Sequence[float]=None):
        if o is None:
            o = (0.0, 0.0, 0.0)
        assert len(o) == 3
        self.r = r
        self.o = np.asarray(o)

class Box(Primitive):
    def __init__(self, half_size:Sequence[float], center:Sequence[float]=None, radius: float = 0.):
        assert len(half_size) == 3
        self.half_size = np.asarray(half_size)
        if center is None:
            center = 0, 0, 0
        assert len(center) == 3
        self.center = center
        self.radius = radius

class Torus(Primitive):
    def __init__(self, major: float, minor: float, center: Sequence[float]=None):
        if center is None:
            center = 0, 0, 0
        assert len(center) == 3
        self.major = major
        self.minor = minor
        self.center = center

class Ellipsoid(Primitive):
    def __init__(self, radii: Sequence[float], center: Sequence[float]=None):
        assert len(radii) == 3
        if center is None:
            center = 0, 0, 0
        assert len(center) == 3
        self.radii = np.asarray(radii)
        self.center = np.asarray(center)

class InfiniteCylinder(Primitive):
    def __init__(self, r:float, o:Sequence[float]=None):
        if o is None:
            o = 0., 0.
        o = np.array(o, float)
        assert o.shape == (2,)

        self.r = r
        self.o = o

class InfiniteRectangularPrism(Primitive):
    def __init__(self, width:float, height:float=None, center:Sequence[float]=None):
        if height is None:
            height = width
        if center is None:
            center = 0, 0
        self.width = float(width)
        self.height = float(height)
        self.center = np.asarray(center, float)

class Plane(Primitive):
    """Half-space boundaried by infinite plane.

     Signed distance equation:
        d = n.x + c
    """
    def __init__(self, n:Sequence[float], c:float):
        assert len(n) == 3
        self.n = normalize(n)
        self.c = c

class Hemisphere(Primitive):
    def __init__(self, r:float, o:Sequence[float]=None, sign:float=1):
        if o is None:
            o = (0.0, 0.0, 0.0)
        assert len(o) == 3
        self.r = r
        self.o = np.asarray(o)
        self.sign = sign

class SphericalSag(Primitive):
    def __init__(self, roc:float, side:float=1, vertex:Sequence[float]=None):
        """Spherical sag profile.

        Spherical profile applies out to circle of radius |roc| in z=vertex[2] plane beyond which it continues in this
        plane.

        Args:
            roc: Signed radius of curvature. Normal optics convention: positive means sphere center is along +z from
                vertex.
            side: 1 for inside on +z side, -1 for inside on -z side.
            vertex: position of vertex - defaults to origin.
        """
        self.roc = roc
        assert np.isclose(abs(side), 1)
        self.side = side
        if vertex is None:
            vertex = 0, 0, 0
        self.vertex = np.asarray(vertex[:3])

    @property
    def center(self):
        return self.vertex + (0, 0, self.roc)

class SagFunction(ABC):
    @property
    @abstractmethod
    def lipschitz(self) -> float:
        pass

class Sag(Primitive):
    def __init__(self, sagfun: SagFunction, side: float = 1, origin: Sequence[float] = None):
        self.sagfun = sagfun
        self.side = side
        if origin is None:
            origin = 0, 0, 0
        origin = np.array(origin, float)
        assert origin.shape == (3,)
        self.origin = origin

    @property
    def lipschitz(self):
        return (self.sagfun.lipschitz**2 + 1)**0.5

class ZemaxConic(Primitive):
    def __init__(self, roc: float, radius: float, side: float = 1., kappa: float = 1., alphas: Sequence[float] = None, vertex: Sequence[float] = None):
        assert radius > 0
        side = float(side)
        assert side in (-1., 1.)
        if kappa*radius**2 >= roc**2:
            raise ValueError(f'Surface is vertical with radius {radius}, roc {roc} and kappa {kappa}.')
        if alphas is None:
            alphas = []
        ns = np.arange(2, len(alphas) + 2)
        # Calculate Lipschitz bound of the sag function. For now use loose Lipschitz bound equal to sum of bounds.
        sag_lipschitz = radius/(roc**2 - kappa*radius**2)**0.5 + sum(abs(alpha)*n*radius**(n - 1) for n, alpha in zip(ns, alphas))
        if vertex is None:
            vertex = 0, 0, 0
        vertex = np.asarray(vertex)

        self.roc = roc
        self.radius = radius
        self.kappa = kappa
        self.alphas = alphas
        self.lipshitz = (sag_lipschitz**2 + 1)**0.5
        self.vertex = vertex
        self.side = side

class ToroidalSag(Primitive):
    # z-axis is normal at vertex. rocs are in x and y. sign convention is normal optical i.e. positive means curving towards
    # postiive z. side is 1 for inside on +z side. axis of revolution is x.
    # Useful intro: https://spie.org/news/designing-with-toroids?SSO=1
    def __init__(self, rocs:Tuple[float, float], side:float=1, vertex:Sequence[float]=None):
        self.rocs = np.asarray(rocs)
        self.side = side
        if vertex is None:
            vertex = 0, 0, 0
        assert len(vertex) == 3
        self.vertex = np.asarray(vertex)

    @property
    def center(self):
        """Center of the circle of rotation."""
        return self.vertex + (0, 0, self.rocs[1])

    @property
    def inside_x(self):
        return self.side*np.sign(self.rocs[0])

    @property
    def ror(self):
        """Radius of revolution"""
        return abs(self.rocs[0] - self.rocs[1])

class Compound(Surface):
    def __init__(self, surfaces:Sequence[Surface]):
        self.surfaces = surfaces

class FiniteRectangularArray(Compound):
    # surface must be
    def __init__(self, pitch, size, surface:Surface, origin=None, corner=None):
        Compound.__init__(self, [surface])
        pitch = np.asarray(pitch)
        assert pitch.shape == (2,)
        size = np.asarray(size)
        assert size.shape == (2,)

        if origin is None:
            if corner is None:
                corner = -pitch*size/2
        else:
            assert corner is None
            corner = origin - pitch*size/2

        self.pitch = pitch
        self.size = size
        self.corner = corner

class BoundedParaboloid(Primitive):
    def __init__(self, roc: float, radius: float, side:bool, vertex:Sequence[float]=None):
        """Paraboloid out to given radius, then flat.

        Surface equation is
            z = min(rho, roc)**2/(2*roc)
        where rho and z are relative to vertex.

        side True means interior is negative z side.
        """
        self.roc = roc
        self.radius = radius
        self.side = side
        if vertex is None:
            vertex = 0, 0, 0
        self.vertex = np.asarray(vertex)
        self.cos_theta = (1 + (self.radius/self.roc)**2)**-0.5

class UnionOp(Compound):
    pass

class IntersectionOp(Compound):
    pass


class DifferenceOp(Compound):
    def __init__(self, s1:Surface, s2:Surface):
        Compound.__init__(self, (s1, s2))

    #def which(self):

class AffineOp(Compound):
    def __init__(self, s:Surface, m:Sequence[Sequence[float]]):
        """

        Args:
            s:
            m: Child to parent transform.
        """
        Compound.__init__(self, [s])
        m = np.asarray(m)
        assert m.shape == (4, 4)

        m3 = m[:3, :3]
        a = m3.dot(m3.T)
        orthonormal = np.isclose(a[0, 1], 0.0) & np.isclose(a[0, 2], 0.0) & np.isclose(a[1, 2], 0.0) & \
                      np.isclose(a[1, 1], a[0, 0]) & np.isclose(a[2, 2], a[0, 0])
        # TODO currently scale only correct for orthonormal. For not orthonormal want a bound. Geometric mean?
        scale = abs(a[0, 0])**0.5

        self.s = s
        self.m = m
        self.invm = np.linalg.inv(m)
        self.orthonormal = orthonormal
        self.scale = scale

# TODO change to piecewise function of rho with Lipschitz the maximum Lipschitz.
class SegmentedRadial(Compound):
    def __init__(self, segments: Sequence[Surface], radii: Sequence[float], vertex: Sequence[float] = None):
        Compound.__init__(self, segments)
        segments = np.asarray(segments, Surface)
        assert segments.ndim == 1
        radii = np.asarray(radii, float)
        assert radii.shape == (segments.size - 1, )
        if vertex is None:
            vertex = 0 ,0
        vertex = np.asarray(vertex, float)
        assert vertex.shape == (2,)

        self.radii = radii
        self.vertex = vertex

#get_intersection(x)
# normal, interiors on either side (from already known but good to check), interface
