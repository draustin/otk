from typing import List
from  itertools import  product
from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import numpy as np
from .. import v4
from ..h4t import make_translation
from . import bounding

from ..v4b import *

__all__ = ['Surface', 'Sphere', 'Box', 'Torus', 'Ellipsoid', 'InfiniteCylinder', 'Plane', 'Sag', 'SagFunction', 'UnionOp', 'IntersectionOp',
    'DifferenceOp', 'AffineOp', 'Compound', 'Primitive', 'SphericalSag', 'Hemisphere', 'InfiniteRectangularPrism',
    'FiniteRectangularArray', 'ToroidalSag', 'BoundedParaboloid', 'ZemaxConic', 'SegmentedRadial', 'get_root_to_local']

class Surface:
    """
    A parent of None means no parent / parent is root.
    """
    def __init__(self, parent:'Surface'=None):
        self._parent = parent

    def get_ancestors(self) -> List['Surface']:
        """Return list of ancestors starting from self going to root."""
        surface = self
        surfaces = []
        while surface is not None:
            surfaces.append(surface)
            surface = surface._parent
        return surfaces

    def get_parent_to_child(self, x: np.ndarray) -> np.ndarray:
        return np.eye(4)

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        """Get axis-aligned bounding box.

        The box is returned in the form (xa, ya, za, 1), (xb, yb, zb, 1).

        The matrix m transforms from the coordinate system of self to the box axes. If vp = (xa or xb, ya or yb, za or zb, 1)
        is box vertex in axis aligned space and v is the same in self space then vp = v*m.
        """
        raise NotImplementedError(self)

    def descendants(self):
        """Returns generator of descendants in depth-first traversal."""
        raise NotImplementedError(self)

def get_root_to_local(self:Surface, x: np.ndarray) -> np.ndarray:
    ancestors = self.get_ancestors()
    m = np.eye(4)
    for surface0 in ancestors[::-1]:
        m0 = surface0.get_parent_to_child(x)
        m = np.dot(m, m0)
        x = np.dot(x, m0)
    return m

class Primitive(Surface):
    def descendants(self):
        yield self

class Sphere(Primitive):
    def __init__(self, r:float, o:Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
        if o is None:
            o = (0.0, 0.0, 0.0)
        assert len(o) == 3
        self.r = r
        self.o = np.asarray(o)

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        assert np.isclose(np.linalg.det(m), 1)
        op = np.r_[self.o, 1].dot(m)
        rv = self.r, self.r, self.r, 0
        return bounding.AABB.make(op - rv, op + rv)

def get_box_vertices(center: Sequence[float], half_size: Sequence[float]) -> np.ndarray:
    vertices = []
    for signs in product((-1, 1), repeat=3):
        vertex = [center + sign*half_size for center, sign, half_size in zip(center, signs, half_size)] + [1]
        vertices.append(vertex)
    vertices = np.asarray(vertices)
    return vertices

class Box(Primitive):
    def __init__(self, half_size:Sequence[float], center:Sequence[float]=None, radius: float = 0., parent: Surface = None):
        Primitive.__init__(self, parent)
        assert len(half_size) == 3
        self.half_size = np.asarray(half_size)
        if center is None:
            center = 0, 0, 0
        assert len(center) == 3
        self.center = center
        self.radius = radius

    def get_hull_vertices(self) -> np.ndarray:
        return get_box_vertices(self.center, self.half_size)

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        vertices = np.dot(self.get_hull_vertices(), m)
        return bounding.AABB((np.min(vertices, 0), np.max(vertices, 0)))

class Torus(Primitive):
    """Circle of radius minor revolved around z axis."""
    def __init__(self, major: float, minor: float, center: Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
        if center is None:
            center = 0, 0, 0
        assert len(center) == 3
        self.major = major
        self.minor = minor
        self.center = center

    def get_hull_vertices(self) -> np.ndarray:
        xy = self.major + self.minor
        return get_box_vertices(self.center, (xy, xy, self.minor))

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        vertices = np.dot(self.get_hull_vertices(), m)
        return bounding.AABB((np.min(vertices, 0), np.max(vertices, 0)))

class Ellipsoid(Primitive):
    def __init__(self, radii: Sequence[float], center: Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
        assert len(radii) == 3
        if center is None:
            center = 0, 0, 0
        assert len(center) == 3
        self.radii = np.asarray(radii)
        self.center = np.asarray(center)

    def get_hull_vertices(self) -> np.ndarray:
        return get_box_vertices(self.center, self.radii)

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        vertices = np.dot(self.get_hull_vertices(), m)
        return bounding.AABB((np.min(vertices, 0), np.max(vertices, 0)))

class InfiniteCylinder(Primitive):
    def __init__(self, r:float, o:Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
        if o is None:
            o = 0., 0.
        o = np.array(o, float)
        assert o.shape == (2,)

        self.r = r
        self.o = o

class InfiniteRectangularPrism(Primitive):
    def __init__(self, width:float, height:float=None, center:Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
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
    def __init__(self, n:Sequence[float], c:float, parent: Surface = None):
        Primitive.__init__(self, parent)
        assert len(n) == 3
        self.n = normalize(n)
        self.c = c

class Hemisphere(Primitive):
    def __init__(self, r:float, o:Sequence[float]=None, sign:float=1, parent: Surface = None):
        Primitive.__init__(self, parent)
        if o is None:
            o = (0.0, 0.0, 0.0)
        assert len(o) == 3
        self.r = r
        self.o = np.asarray(o)
        self.sign = sign

class SphericalSag(Primitive):
    def __init__(self, roc:float, side:float=1, vertex:Sequence[float]=None, parent: Surface = None):
        """Spherical sag profile.

        Spherical profile applies out to circle of radius |roc| in z=vertex[2] plane beyond which it continues in this
        plane.

        Args:
            roc: Signed radius of curvature. Normal optics convention: positive means sphere center is along +z from
                vertex.
            side: 1 for inside on +z side, -1 for inside on -z side.
            vertex: position of vertex - defaults to origin.
        """
        Primitive.__init__(self, parent)
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
    def __init__(self, sagfun: SagFunction, side: float = 1, origin: Sequence[float] = None, parent: Surface = None):
        Primitive.__init__(self, parent)
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
    def __init__(self, roc: float, radius: float, side: float = 1., kappa: float = 1., alphas: Sequence[float] = None, vertex: Sequence[float] = None, parent: Surface = None):
        Primitive.__init__(self, parent)
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
    def __init__(self, rocs:Tuple[float, float], side:float=1, vertex:Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
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
    def __init__(self, surfaces:Sequence[Surface], parent: Surface = None):
        Surface.__init__(self, parent)
        for s in surfaces:
            assert s._parent is None
            s._parent = self
        self.surfaces = surfaces

    def descendants(self):
        for s in self.surfaces:
            yield from s.descendants()
        yield self

class FiniteRectangularArray(Compound):
    # surface must be
    def __init__(self, pitch, size, surface:Surface, origin=None, corner=None, parent: Surface = None):
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

    def get_center(self, x:Sequence[float]) -> np.ndarray:
        """

        Args:
            x: Position 2-vector.

        Returns:
            Nearest center as position 2-vector.
        """
        index = np.clip(np.floor((x - self.corner)/self.pitch), 0, self.size - 1)
        center = (index + 0.5)*self.pitch + self.corner
        return center

    def transform(self, x:Sequence[float]) -> np.ndarray:
        """

        Args:
            x: Position 4-vector

        Returns:
            Transformed position 4-vector.
        """
        return np.r_[x[:2] - self.get_center(x[:2]), x[2:]]

    def get_parent_to_child(self, x) -> np.ndarray:
        center = self.get_center(x[:2])
        return make_translation(-center[0], -center[1], 0)

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        unit = self.surfaces[0].get_aabb(m)
        corner0 = unit.corners[0] + np.r_[self.corner + self.pitch/2, 0, 0]
        corner1 = unit.corners[1] + np.r_[self.corner + self.pitch*(self.size - 0.5), 0, 0]
        return bounding.AABB((corner0, corner1))

class BoundedParaboloid(Primitive):
    def __init__(self, roc: float, radius: float, side:bool, vertex:Sequence[float]=None, parent: Surface = None):
        """Paraboloid out to given radius, then flat.

        Surface equation is
            z = min(rho, roc)**2/(2*roc)
        where rho and z are relative to vertex.

        side True means interior is negative z side.
        """
        Primitive.__init__(self, parent)
        self.roc = roc
        self.radius = radius
        self.side = side
        if vertex is None:
            vertex = 0, 0, 0
        self.vertex = np.asarray(vertex)
        self.cos_theta = (1 + (self.radius/self.roc)**2)**-0.5

class UnionOp(Compound):
    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        return bounding.union(*[s.get_aabb(m) for s in self.surfaces])

class IntersectionOp(Compound):
    def __init__(self, surfaces:Sequence[Surface], bound:Surface = None, parent: Surface = None):
        Compound.__init__(self, surfaces, parent)
        self.bound = bound

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        if self.bound is None:
            return bounding.intersection(*[s.get_aabb(m) for s in self.surfaces])
        else:
            return self.bound.get_aabb(m)

class DifferenceOp(Compound):
    def __init__(self, s1:Surface, s2:Surface, bound:Surface = None, parent: Surface = None):
        Compound.__init__(self, (s1, s2), parent)
        self.bound = bound

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        if self.bound is None:
            return bounding.difference(self.surfaces[0].get_aabb(m), self.surfaces[1].get_aabb(m))
        else:
            return self.bound.get_aabb(m)

class AffineOp(Compound):
    def __init__(self, s:Surface, m:Sequence[Sequence[float]], parent: Surface = None):
        """

        Args:
            s:
            m: Child to parent transform.
        """
        Compound.__init__(self, [s], parent)
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

    def get_parent_to_child(self, x: np.ndarray) -> np.ndarray:
        return self.invm

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        return self.surfaces[0].get_aabb(self.m.dot(m))

# TODO change to piecewise function of rho with Lipschitz the maximum Lipschitz.
class SegmentedRadial(Compound):
    def __init__(self, segments: Sequence[Surface], radii: Sequence[float], vertex: Sequence[float] = None, parent: Surface = None):
        Compound.__init__(self, segments, parent)
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
