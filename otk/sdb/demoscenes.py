import itertools
import numpy as np
from otk.h4t import make_translation, make_rotation
from . import *
from ..v4 import normalize
from .lens import *

class Centers:
    def __init__(self, num_columns: int, spacing: float):
        self.num_columns = num_columns
        self.spacing = spacing
        self.row = 0
        self.column = 0

    def __next__(self) -> np.ndarray:
        x = np.asarray((self.column*self.spacing, self.row*self.spacing, 0))
        self.column += 1
        if self.column == self.num_columns:
            self.column = 0
            self.row += 1
        return x

    @property
    def center(self) -> np.ndarray:
        return np.asarray(((self.num_columns - 1)*self.spacing/2, self.row*self.spacing/2, 0))

def make_primitives():
    centers = Centers(3, 2.)
    surfaces = []
    colors = []

    # sphere, finite cone, finite cylinder, rectangular prism, torus
    surfaces.append(Sphere(0.5, next(centers)))
    colors.append((1, 0, 0))

    surfaces.append(Box((0.25, 0.5, 1), next(centers)))
    colors.append((0, 1, 0))

    surfaces.append(Box((0.5, 0.5, 1), next(centers), 0.2))
    colors.append((0, 0, 1))

    surfaces.append(Torus(0.5, 0.1, next(centers)))
    colors.append((1, 1, 0))

    surfaces.append(Ellipsoid((0.25, 0.5, 1), next(centers)))
    colors.append((1, 0, 1))
    
    surface = UnionOp(surfaces)

    properties = dict(edge_width = 0.01, edge_color = (0.3, 0.3, 0.3))
    set_properties = {s:dict(surface_color=c) for s, c in zip(surfaces, colors)}
    sdb_glsl = gen_get_all_recursive(surface, set_properties, properties)

    z_eye = max(centers.num_columns - 1, centers.row)*centers.spacing*2
    center = centers.center
    eye = center + (0, 0, z_eye)
    return Scene('primitives', sdb_glsl, 0.01, z_eye*5, eye, center)

def make_spherical_singlets():
    centers = Centers(3, 2)
    surfaces = []
    colors = []

    roc = 1
    thickness = 0.2
    surfaces = []
    for num, (sign0, sign1) in enumerate(itertools.product((-1, np.inf, 1), repeat=2)):
        kwargs = dict(shape='circle', radius=0.4) if num % 2 else dict(shape='rectangle', side_lengths=(0.8, 0.6))
        surfaces.append(make_spherical_singlet(roc*sign0, roc*sign1, thickness, next(centers) - (0, 0, thickness/2), **kwargs))

    surface = UnionOp(surfaces)
    properties = dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0, 0.5, 1))
    sdb_glsl = gen_get_all_recursive(surface, {}, properties)

    z_eye = max(centers.num_columns - 1, centers.row)*centers.spacing*1.5
    center = centers.center
    eye = center + (0, 0, z_eye)
    return Scene('spherical singlets', sdb_glsl, 0.01, z_eye*5, eye, center)

def make_lens_array():
    surface = make_spherical_singlet_square_array(1, -1.5, 0.8, (0.5, 1), (16, 8))
    properties = dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0, 0.5, 1))
    sdb_glsl = gen_get_all_recursive(surface, {}, properties)
    return Scene('lens array', sdb_glsl, 0.01, 64, np.asarray((0, -8, 16)), np.asarray((0, 0, 0)))

def make_sags():
    centers = Centers(2, 3)
    surfaces = []
    sags = [] # will make these red

    fn = SinusoidSagFunction(0.2, (2*np.pi, 2*np.pi))
    center = next(centers)
    sag = Sag(fn, -1, center + (0.5, 0.5, 1))
    sags.append(sag)
    surfaces.append(IntersectionOp([sag, InfiniteCylinder(1, center[:2]), Plane((0, 0, -1), -1)]))

    fn = ZemaxConicSagFunction(-0.5, 1, -20)
    center = next(centers)
    sag = Sag(fn, -1, center + (0., 0., 1))
    sags.append(sag)
    surfaces.append(IntersectionOp([sag, InfiniteCylinder(1, center[:2]), Plane((0, 0, -1), -1)]))

    fn = ZemaxConicSagFunction(np.inf, 1, 1, [0, 0, 0, 0, -1])
    center = next(centers)
    sag = Sag(fn, -1, center + (0., 0., 1))
    sags.append(sag)
    surfaces.append(IntersectionOp([sag, InfiniteCylinder(1, center[:2]), Plane((0, 0, -1), -1)]))

    pitch = 0.4
    unitfn = ZemaxConicSagFunction(-0.3, pitch/2**0.5)
    fn = RectangularArraySagFunction(unitfn, (pitch, pitch))
    center = next(centers)
    sag = Sag(fn, -1, center + (0., 0., 1))
    sags.append(sag)
    surfaces.append(IntersectionOp([sag, InfiniteCylinder(1, center[:2]), Plane((0, 0, -1), -1)]))

    surface = UnionOp(surfaces)

    properties = dict(edge_width=0.01, edge_color=(0, 0, 0), surface_color=(0.5, 0.5, 0.5))
    set_properties = {sag:dict(surface_color=(1, 0, 0)) for sag in sags}
    sdb_glsl = gen_get_all_recursive(surface, set_properties, properties)
    center = centers.center
    return Scene('sag functions', sdb_glsl, 0.01, 16, center + np.asarray((0, -4, 8)), center)

def make_conic_singlets():
    # all combinations of convex/concave/plano on both sides, alternating square and round
    pass

def make_conic_lens_array():
    pass

def make_combinations():
    centers = Centers(2, 3.)
    surfaces = []
    set_properties = {}

    center = next(centers)
    sphere = Sphere(1, center)
    box = Box((0.5**0.5, 0.5**0.5, 0.5**0.5), center)
    surface0 = UnionOp((sphere, box))
    surfaces.append(surface0)
    set_properties[sphere] = dict(surface_color=(1, 0, 0))

    sphere = Sphere(1)
    box = Box((0.5**0.5, 0.5**0.5, 0.5**0.5))
    center = next(centers)
    transform = make_rotation(normalize((1, 1, 1)), np.pi/4).dot(make_translation(*center))
    surfaces.append(AffineOp(UnionOp((sphere, box)), transform))

    center = next(centers)
    torus = Torus(1, 0.4, center)
    box = Box((1.1, 1.1, 0.3), center)
    surfaces.append(IntersectionOp((torus, box)))
    set_properties[torus] = dict(surface_color=(1, 0, 0))

    center = next(centers)
    box = Box((1, 1, 1), center)
    cylinder = InfiniteCylinder(0.5, center[:2])
    surfaces.append(DifferenceOp(box, cylinder))
    set_properties[cylinder] = dict(surface_color=(1, 0, 0))

    surface = UnionOp(surfaces)

    parent_properties = dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0.5, 0.5, 0.5))

    sdb_glsl = gen_get_all_recursive(surface, set_properties, parent_properties)

    z_eye = max(centers.num_columns - 1, centers.row)*centers.spacing*2
    center = centers.center
    eye = center + (0, 0, z_eye)
    return Scene('combinations', sdb_glsl, 0.01, z_eye*5, eye, center)

def make_conic():
    vertex0 = np.asarray((-1, 0, 0))
    radius = 0.8
    front = ZemaxConic(1, radius, 1, -3, [], vertex=vertex0)
    back = ZemaxConic(-1.4, radius, -1, 3, [], vertex0 + (0, 0, 1))
    side = InfiniteCylinder(radius, vertex0[:2])
    surface0 = IntersectionOp((front, back, side))

    vertex1 = np.asarray((1, 0, 0))
    front = ZemaxConic(np.inf, radius, 1, 0, [0, 0, 0, 1], vertex=vertex1)
    back = ZemaxConic(np.inf, radius, -1, 0, [0, 0, 0, 0, -1], vertex1 + (0, 0, 1))
    side = InfiniteCylinder(radius, vertex1[:2])
    surface1 = IntersectionOp((front, back, side))

    surface = UnionOp((surface0, surface1))
    parent_properties = dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0, 0, 1.))
    sdb_glsl = gen_get_all_recursive(surface, {}, parent_properties)
    return Scene('conic', sdb_glsl, 0.01, 10, np.asarray((0, 0, 5)), np.asarray((0, 0, 0)))

def make_all_scenes():
    return [make_primitives(), make_spherical_singlets(), make_lens_array(), make_combinations(), make_conic(), make_sags()]