from typing import List
import copy
from dataclasses import dataclass
from  itertools import  product
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union
import numpy as np
from ..types import Sequence2, Vector2, Sequence3, Vector2Int, Vector3, Sequence4, Matrix4
from ..functions import normalize, calc_zemax_conic_lipschitz, norm, norm_squared
from ..h4t import make_translation
from . import bounding

__all__ = ['Surface', 'Sphere', 'Box', 'Torus', 'Ellipsoid', 'InfiniteCylinder', 'Plane', 'Sag', 'SagFunction', 'UnionOp', 'IntersectionOp',
    'DifferenceOp', 'AffineOp', 'Compound', 'Primitive', 'SphericalSag', 'Hemisphere', 'InfiniteRectangularPrism',
    'FiniteRectangularArray', 'ToroidalSag', 'BoundedParaboloid', 'ZemaxConic', 'SegmentedRadial', 'ISDB']

# TODO ABC?
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

    def get_parent_to_child(self, x: Sequence4) -> np.ndarray:
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

    def scale(self, f: float) -> 'Surface':
        """Deep copy of self with coordinate system scaled by f.

        E.g. if the units of all Surface s meters, then the units of s.scale(1000) are mm.
        """
        raise NotImplementedError(self)

    def getsdb(self, x: Sequence4) -> float:
        """Get signed distance bound.

        Args:
            x: Position 4-vector.
        """
        raise NotImplementedError(self)

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        """Return ([active surface at x, parent of active surface at x, ..., self], signed distance)."""
        raise NotImplementedError(self)

@dataclass
class ISDB:
    d: float
    surface: Surface
    face: int

    def max(self, other):
        if self.d >= other.d:
            return self
        else:
            return other

    def min(self, other):
        if self.d <= other.d:
            return self
        else:
            return other

    def negate(self):
        return ISDB(-self.d, self.surface, self.face)

    def times(self, f: float):
        return ISDB(self.d*f, self.surface, self.face)


class Primitive(Surface):
    def descendants(self):
        yield self

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        return [self], self.getsdb(x)


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

    def scale(self, f: float) -> 'Sphere':
        return Sphere(self.r*f, self.o*f)

    def getsdb(self, x: Sequence4) -> float:
        return norm(x[:3] - self.o) - self.r


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
        half_size = np.array(half_size, float)
        assert half_size.shape == (3,)

        if center is None:
            center = 0, 0, 0
        center = np.array(center, float)
        assert center.shape == (3,)

        self.half_size = half_size
        self.center = center
        self.radius = radius

    def get_hull_vertices(self) -> np.ndarray:
        return get_box_vertices(self.center, self.half_size)

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        vertices = np.dot(self.get_hull_vertices(), m)
        return bounding.AABB((np.min(vertices, 0), np.max(vertices, 0)))

    def scale(self, f: float) -> 'Box':
        return Box(self.half_size*f, self.center*f, self.radius*f)

    def getsdb(self, x: Sequence4) -> float:
        q = abs(x[:3] - self.center) - (self.half_size - self.radius)
        return norm(np.maximum(q, 0.)) + min(max(q[0], max(q[1], q[2])), 0.0) - self.radius

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

    def scale(self, f: float) -> 'InfiniteCylinder':
        return InfiniteCylinder(self.r*f, self.o*f)

    def getsdb(self, x: Sequence4) -> float:
        return norm(x[:2] - self.o) - self.r


class InfiniteRectangularPrism(Primitive):
    def __init__(self, width:float, height:float=None, center:Sequence[float]=None, parent: Surface = None):
        Primitive.__init__(self, parent)
        if height is None:
            height = width
        if center is None:
            center = 0, 0
        center = np.asarray(center, float)
        assert center.shape == (2,)
        self.width = float(width)
        self.height = float(height)
        self.center = center

    def getsdb(self, x: Sequence4) -> float:
        q = abs(x[:2] - self.center) - (self.width/2, self.height/2)
        return norm(np.maximum(q, 0.0)) + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)

    def scale(self, f: float) -> 'InfiniteRectangularPrism':
        return InfiniteRectangularPrism(self.width*f, self.height*f, self.center*f)


class Plane(Primitive):
    """Half-space boundaried by infinite plane.

     Signed distance equation:
        d = n.x + c
    """
    # TODO get rid of normalize - make factory method instead
    def __init__(self, n: Sequence[float], c: float, parent: Surface = None):
        Primitive.__init__(self, parent)
        self.n = normalize(n[:3])
        self.c = float(c)

    def scale(self, f: float) -> 'Plane':
        return Plane(self.n, self.c*f)

    def getsdb(self, x: Sequence4) -> float:
        return np.dot(x[:3], self.n) + self.c

class Hemisphere(Primitive):
    def __init__(self, r: float, o: Sequence[float] = None, sign: float = 1, parent: Surface = None):
        Primitive.__init__(self, parent)
        if o is None:
            o = (0.0, 0.0, 0.0)
        assert len(o) == 3
        self.r = r
        self.o = np.asarray(o)
        self.sign = sign

    def getsdb(self, x: Sequence4) -> float:
        return np.minimum(norm(x[:3] - self.o) - self.r, self.sign*(x[2] - self.o[2]))


class SphericalSag(Primitive):
    roc: float
    side: float # +/- 1.
    vertex: Vector3

    def __init__(self, roc: float, side: float = 1., vertex: Sequence3 = (0., 0., 0.), parent: Surface = None):
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
        self.roc = float(roc)
        assert np.isclose(abs(side), 1)
        self.side = float(side)
        vertex = np.array(vertex, float)
        assert vertex.shape == (3,)
        self.vertex = vertex

    @property
    def center(self):
        return self.vertex + (0, 0, self.roc)

    def scale(self, f: float) -> 'SphericalSag':
        return SphericalSag(self.roc*f, self.side, self.vertex*f)

    def getsdb(self, x: Sequence4) -> float:
        if np.isfinite(self.roc):
            inside = self.side*np.sign(self.roc)
            a = inside*(norm(x[:3] - self.center) - abs(self.roc))
            b = -self.side*(x[2] - self.center[2])
            fun = np.minimum if inside > 0 else np.maximum
            return fun(a, b)
        else:
            return self.side*(self.vertex[2] - x[2])

# TODO move to sags.py
class SagFunction(ABC):
    @property
    @abstractmethod
    def lipschitz(self) -> float:
        pass

    @abstractmethod
    def getsag(self, x: Sequence2) -> float:
        pass

class Sag(Primitive):
    """Surface defined by f(r_x - o_x, r_y - o_y) = r_z - o_z."""
    def __init__(self, sagfun: SagFunction, side: float = 1., origin: Sequence3 = None, parent: Surface = None):
        Primitive.__init__(self, parent)
        self.sagfun = sagfun
        self.side = float(side)
        if origin is None:
            origin = 0, 0, 0
        origin = np.array(origin, float)
        assert origin.shape == (3,)
        self.origin = origin

    @property
    def lipschitz(self):
        return (self.sagfun.lipschitz**2 + 1)**0.5

    def getsdb(self, x: Sequence4) -> float:
        sag = self.sagfun.getsag(x[:2] - self.origin[:2])
        return self.side*(sag + self.origin[2] - x[2])/self.lipschitz


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
        # Calculate Lipschitz bound of the sag function.
        sag_lipschitz = calc_zemax_conic_lipschitz(radius, roc, kappa, alphas)
        if vertex is None:
            vertex = 0, 0, 0
        vertex = np.asarray(vertex)

        self.roc = roc
        self.radius = radius
        self.kappa = kappa
        self.alphas = alphas
        self.lipschitz = (sag_lipschitz**2 + 1)**0.5
        self.vertex = vertex
        self.side = side

    def scale(self, f: float) -> 'ZemaxConic':
        ns = np.arange(2, len(self.alphas) + 2)
        alphas = [alpha/f**(n - 1) for alpha, n in zip(self.alphas, ns)]
        return ZemaxConic(self.roc*f, self.radius*f, self.side, self.kappa, alphas, self.vertex*f)

    def getsdb(self, x: Sequence4) -> float:
        xp = x[:3] - self.vertex
        rho2 = min(norm_squared(xp[:2]), self.radius**2)
        rho = rho2**0.5
        if np.isfinite(self.roc):
            z = rho2/(self.roc*(1 + (1 - self.kappa*rho2/self.roc**2)**0.5))
        else:
            z = 0
        if len(self.alphas) > 0:
            h = self.alphas[-1]
            for alpha in self.alphas[-2::-1]:
                h = h*rho + alpha
            z += h*rho2
        return self.side*(z - xp[2])/self.lipschitz

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
    """A finite rectangular array of unit surfaces in the xy plane.

    Position in the xy plane is specified by the center of the array or the lower-left corner
    (assuming pitch is positive). If neither are specified the array is centered at (0, 0).
    """
    pitch: Vector2
    size: Vector2Int
    corner: Vector2

    def __init__(self, pitch: Union[Sequence2, float], size: Union[Sequence2[int], int], unit:Surface, center: Sequence2 = None, corner: Sequence2 = None, parent: Surface = None):
        Compound.__init__(self, [unit], parent)

        pitch = np.broadcast_to(pitch, (2, )).astype(float)
        size = np.broadcast_to(size, (2, )).astype(int)

        if center is None:
            if corner is None:
                corner = -pitch*size/2
        else:
            assert corner is None
            center = np.array(center, float)
            corner = center - pitch*size/2
        corner = np.array(corner, float)
        assert corner.shape == (2,)

        self.pitch = pitch
        self.size = size
        self.corner = corner

    def get_center(self, x: Sequence2) -> Vector2:
        """Return nearest unit center."""
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

    def scale(self, f: float) -> 'FiniteRectangularArray':
        return FiniteRectangularArray(self.pitch*f, self.size, self.surfaces[0].scale(f), self.corner*f)

    def getsdb(self, x: Sequence4) -> float:
        return self.surfaces[0].getsdb(self.transform(x))

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        ancestry, d = self.surfaces[0].get_ancestry_at(self.transform(x))
        ancestry.append(self)
        return ancestry, d

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

    def getsdb(self, x: Sequence4) -> float:
        xp = x[:3] - self.vertex
        d = (xp[2] - min(xp[0]**2 + xp[1]**2, self.radius**2)/(2*self.roc))*self.cos_theta
        if not self.side:
            d = -d
        return d

class UnionOp(Compound):
    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        return bounding.union(*[s.get_aabb(m) for s in self.surfaces])

    def scale(self, f: float) -> 'UnionOp':
        return UnionOp([s.scale(f) for s in self.surfaces])

    def getsdb(self, x: Sequence4) -> float:
        d = self.surfaces[0].getsdb(x)
        for surface in self.surfaces[1:]:
            d = min(d, surface.getsdb(x))
        return d

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        ancestry, d = self.surfaces[0].get_ancestry_at(x)
        for s in self.surfaces[1:]:
            ancestryp, dp = s.get_ancestry_at(x)
            if dp < d:
                ancestry, d = ancestryp, dp
        ancestry.append(self)
        return ancestry, d

# TODO bound should only be required if *no* children have bounds.
class IntersectionOp(Compound):
    def __init__(self, surfaces:Sequence[Surface], bound:Surface = None, parent: Surface = None):
        Compound.__init__(self, surfaces, parent)
        self.bound = bound

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        if self.bound is None:
            return bounding.intersection(*[s.get_aabb(m) for s in self.surfaces])
        else:
            return self.bound.get_aabb(m)

    def scale(self, f: float) -> 'IntersectionOp':
        if self.bound is None:
            bound = None
        else:
            bound = self.bound.scale(f)
        return IntersectionOp([s.scale(f) for s in self.surfaces], bound)

    def getsdb(self, x: Sequence4) -> float:
        d = self.surfaces[0].getsdb(x)
        for surface in self.surfaces[1:]:
            d = max(d, surface.getsdb(x))
        return d

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        ancestry, d = self.surfaces[0].get_ancestry_at(x)
        for s in self.surfaces[1:]:
            ancestryp, dp = s.get_ancestry_at(x)
            if dp > d:
                ancestry, d = ancestryp, dp
        ancestry.append(self)
        return ancestry, d

class DifferenceOp(Compound):
    def __init__(self, s1:Surface, s2:Surface, bound:Surface = None, parent: Surface = None):
        Compound.__init__(self, (s1, s2), parent)
        self.bound = bound

    def get_aabb(self, m: np.ndarray) -> bounding.AABB:
        if self.bound is None:
            return bounding.difference(self.surfaces[0].get_aabb(m), self.surfaces[1].get_aabb(m))
        else:
            return self.bound.get_aabb(m)

    def scale(self, f: float) -> 'DifferenceOp':
        if self.bound is None:
            bound = None
        else:
            bound = self.bound.scale(f)
        return DifferenceOp(self.surfaces[0].scale(f), self.surfaces[1].scale(f), bound)

    def getsdb(self, x: Sequence4) -> float:
        d0 = self.surfaces[0].getsdb(x)
        d1 = self.surfaces[1].getsdb(x)
        return max(d0, -d1)

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        ancestry0, d0 = self.surfaces[0].get_ancestry_at(x)
        ancestry1, d1 = self.surfaces[1].get_ancestry_at(x)
        d1 = -d1
        if d0 >= d1:
            ancestry, d = ancestry0, d0
        else:
            ancestry, d = ancestry1, d1
        ancestry.append(self)
        return ancestry, d

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

    def getsdb(self, x: Sequence4) -> float:
        d = self.surfaces[0].getsdb(np.dot(x, self.invm))
        return d*self.scale

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        ancestry, d = self.surfaces[0].get_ancestry_at(np.dot(x, self.invm))
        d *= self.scale
        ancestry.append(self)
        return ancestry, d


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

    def scale(self, f: float) -> 'SegmentedRadial':
        return SegmentedRadial([s.scale(f) for s in self.surfaces], self.radii*f, self.vertex*f)

    def getsdb(self, x: Sequence4) -> float:
        rho = norm(x[:2] - self.vertex)
        for ss, r in zip(self.surfaces[:-1], self.radii):
            if rho <= r:
                return ss.getsdb(x)
        return self.surfaces[-1].getsdb(x)

    def get_ancestry_at(self, x: Sequence4) -> Tuple[List['Surface'], float]:
        rho = norm(x[:2] - self.vertex)
        for ss, r in zip(self.surfaces[:-1], self.radii):
            if rho <= r:
                ancestry, d = ss.get_ancestry_at(x)
                break
        else:
            ancestry, d = self.surfaces[-1].get_ancestry_at(x)
        ancestry.append(self)
        return ancestry, d


#get_intersection(x)
# normal, interiors on either side (from already known but good to check), interface
