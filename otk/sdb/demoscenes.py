import itertools
from collections import defaultdict
import numpy as np
from otk.h4t import make_translation, make_rotation
from ..functions import normalize
from . import *
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

    all_properties = {s:dict(edge_width = 0.01, edge_color = (0.3, 0.3, 0.3)) for s in surface.descendants()}
    for s, c in zip(surfaces, colors):
        all_properties[s]['surface_color'] = c

    sdb_glsl = gen_get_all_recursive(surface, all_properties)

    wireframe_models = [make_wireframe(s.get_aabb(np.eye(4)), (0, 0, 0)) for s in surfaces]

    z_eye = max(centers.num_columns - 1, centers.row)*centers.spacing*2
    center = centers.center
    eye = center + (0, 0, z_eye)
    return Scene.make('primitives', sdb_glsl, 0.5, z_eye*5, eye, center, wireframe_models)

def make_spherical_singlets():
    centers = Centers(3, 2)
    surfaces = []
    colors = []

    roc = 1
    thickness = 0.2
    surfaces = []
    for num, (sign0, sign1) in enumerate(itertools.product((-1, np.inf, 1), repeat=2)):
        shape = make_circle(0.4) if num % 2 else make_rectangle(0.8, 0.6)
        surfaces.append(make_spherical_singlet(roc*sign0, roc*sign1, thickness, shape, next(centers) - (0, 0, thickness/2)))

    surface = UnionOp(surfaces)
    all_properties = {s: dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0, 0.5, 1)) for s in surface.descendants()}
    sdb_glsl = gen_get_all_recursive(surface, all_properties)

    wireframe_models = [make_wireframe(s.get_aabb(np.eye(4)), (0, 0, 0)) for s in surfaces]

    z_eye = max(centers.num_columns - 1, centers.row)*centers.spacing*1.5
    center = centers.center
    eye = center + (0, 0, z_eye)
    return Scene.make('spherical singlets', sdb_glsl, 0.01, z_eye*5, eye, center, wireframe_models)

def make_lens_array():
    surface = make_spherical_singlet_square_array(1, -1.5, 0.8, (0.5, 1), (16, 8))
    all_properties = {s:dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0, 0.5, 1)) for s in surface.descendants()}
    sdb_glsl = gen_get_all_recursive(surface, all_properties)
    wireframe_model = make_wireframe(surface.get_aabb(np.eye(4)), (0, 0, 0))
    return Scene.make('lens array', sdb_glsl, 0.01, 64, np.asarray((0, -8, 16)), np.asarray((0, 0, 0)), [wireframe_model])

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

    # Finite array without clamping.
    pitch = 0.4
    fn = RectangularArraySagFunction(unitfn, (pitch, pitch), (2, 2), False)
    center = next(centers)
    sag = Sag(fn, -1, center + (0., 0., 1))
    sags.append(sag)
    surfaces.append(IntersectionOp([sag, InfiniteCylinder(1, center[:2]), Plane((0, 0, -1), -1)]))

    # Finite array with clamping.
    pitch = 0.4
    fn = RectangularArraySagFunction(unitfn, (pitch, pitch), (2, 2), True)
    center = next(centers)
    sag = Sag(fn, -1, center + (0., 0., 1))
    sags.append(sag)
    surfaces.append(IntersectionOp([sag, InfiniteCylinder(1, center[:2]), Plane((0, 0, -1), -1)]))

    surface = UnionOp(surfaces)

    all_properties = {s:dict(edge_width=0.01, edge_color=(0, 0, 0), surface_color=(0.5, 0.5, 0.5)) for s in surface.descendants()}
    for sag in sags:
        all_properties[sag]['surface_color'] = (1, 0, 0)

    sdb_glsl = gen_get_all_recursive(surface, all_properties)
    center = centers.center

    #wireframe_models = [make_wireframe(s.get_aabb(np.eye(4)), (0, 0, 0)) for s in surfaces]

    return Scene.make('sag functions', sdb_glsl, 0.01, 16, center + np.asarray((0, -4, 8)), center, [])

def make_conic_singlets():
    # all combinations of convex/concave/plano on both sides, alternating square and round
    pass

def make_conic_lens_array():
    pass

def make_combinations():
    centers = Centers(2, 3.)
    surfaces = []
    all_properties = defaultdict(dict)

    center = next(centers)
    sphere = Sphere(1, center)
    box = Box((0.5**0.5, 0.5**0.5, 0.5**0.5), center)
    surface0 = UnionOp((sphere, box))
    surfaces.append(surface0)
    all_properties[sphere]['surface_color'] = (1, 0, 0)

    sphere = Sphere(1)
    box = Box((0.5**0.5, 0.5**0.5, 0.5**0.5))
    center = next(centers)
    transform = make_rotation(normalize((1, 1, 1)), np.pi/4).dot(make_translation(*center))
    surfaces.append(AffineOp(UnionOp((sphere, box)), transform))
    all_properties[sphere]['surface_color'] = (1, 0, 0)

    center = next(centers)
    torus = Torus(1, 0.4, center)
    box = Box((1.1, 1.1, 0.3), center)
    surfaces.append(IntersectionOp((torus, box)))
    all_properties[torus]['surface_color'] = (1, 0, 0)

    center = next(centers)
    box = Box((1, 1, 1), center)
    sphere = Sphere(0.5, (center[0], center[1], center[2]+1))
    surfaces.append(DifferenceOp(box, sphere))
    all_properties[sphere]['surface_color'] = (1, 0, 0)

    surface = UnionOp(surfaces)

    default_properties = dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0.5, 0.5, 1))
    for s in surface.descendants():
        for k, v in default_properties.items():
            all_properties[s].setdefault(k, v)

    sdb_glsl = gen_get_all_recursive(surface, dict(all_properties))

    z_eye = max(centers.num_columns - 1, centers.row)*centers.spacing*2
    center = centers.center
    eye = center + (0, 0, z_eye)

    wireframe_models = [make_wireframe(s.get_aabb(np.eye(4)), (0, 0, 0)) for s in surfaces]

    return Scene.make('combinations', sdb_glsl, 0.01, z_eye*5, eye, center, wireframe_models)

def make_conic():
    surfaces = []

    vertex0 = np.asarray((-1, 0, 0))
    radius = 0.8
    front = ZemaxConic(1, radius, 1, -3, [], vertex=vertex0)
    back = ZemaxConic(-1.4, radius, -1, 3, [], vertex0 + (0, 0, 1))
    side = InfiniteCylinder(radius, vertex0[:2])
    surfaces.append(IntersectionOp((front, back, side), Box((radius, radius, 0.5), vertex0 + (0, 0, 0.5))))

    vertex1 = np.asarray((1, 0, 0))
    front = ZemaxConic(np.inf, radius, 1, 0, [0, 0, 0, 1], vertex=vertex1)
    back = ZemaxConic(np.inf, radius, -1, 0, [0, 0, 0, 0, -1], vertex1 + (0, 0, 1))
    side = InfiniteCylinder(radius, vertex1[:2])
    surfaces.append(IntersectionOp((front, back, side), Box((radius, radius, 0.5), vertex1 + (0, 0, 0.5))))

    surface = UnionOp(surfaces)
    all_properties = {s:dict(edge_width=0.01, edge_color=(0.3, 0.3, 0.3), surface_color=(0, 0, 1.)) for s in surface.descendants()}
    sdb_glsl = gen_get_all_recursive(surface, all_properties)

    wireframe_models = [make_wireframe(s.get_aabb(np.eye(4)), (0, 0, 0)) for s in surfaces]

    return Scene.make('conic', sdb_glsl, 0.01, 10, np.asarray((0, 0, 5)), np.asarray((0, 0, 0)), wireframe_models)

def make_all_scenes():
    return [make_primitives(), make_spherical_singlets(), make_lens_array(), make_combinations(), make_conic(), make_sags()]