from dataclasses import dataclass
from functools import singledispatch
from abc import ABC, abstractmethod
import itertools
from typing import Sequence, List
import numpy as np
from ..v4b import *
from .scalar import spheretrace
from . import bounding

__all__ = ['orthographic', 'projection', 'lookat', 'ndc2ray', 'pix2norm', 'shade_distance', 'raster', 'Scene',
    'WireframeModel', 'make_wireframe', 'Projection', 'Orthographic', 'Perspective']

# TODO rename to make_orthographic, and possibly move to h4
"""
https://lmb.informatik.uni-freiburg.de/people/reisert/opengl/doc/glOrtho.html
http://www.songho.ca/opengl/gl_projectionmatrix.html
"""
def orthographic(l:float, r:float, b:float, t:float, n:float, f:float):
    return np.asarray((
        (2/(r - l), 0,          0,           -(r + l)/(r - l)),
        (0,         2/(t - b),  0,           -(t + b)/(t - b)),
        (0,         0,          -2/(f - n),  -(f + n)/(f - n)),
        (0,         0,          0,           1))).T

# TODO rename to make_perspective, and possibly move to h4
"""
https://lmb.informatik.uni-freiburg.de/people/reisert/opengl/doc/glFrustum.html
http://www.songho.ca/opengl/gl_projectionmatrix.html
"""
def projection(l:float, r:float, b:float, t:float, n:float, f:float):
    return np.asarray((
        (2*n/(r - l), 0,           (r + l)/(r - l), 0),
        (0,           2*n/(t - b), (t + b)/(t - b), 0),
        (0,           0,           (f + n)/(n - f), -2*f*n/(f - n)),
        (0,           0,           -1,              0))).T

"""Produce camera to world transform."""
def lookat(eye:Sequence[float], center:Sequence[float], y:Sequence[float]=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye)
    center = np.asarray(center)

    # eye to center is -z axis.
    z = normalize(eye - center)
    x = normalize(np.cross(y, z))
    y = np.cross(z, x)

    return np.asarray((
        (x[0], y[0], z[0], eye[0]),
        (x[1], y[1], z[1], eye[1]),
        (x[2], y[2], z[2], eye[2]),
        (0.0,  0.0,  0.0,  1.0))).T

"""
i=(0,1), r=2 -> n= -0.5, 0.5
i=(0,3), r=4 -> n= -0.75, 0.75
"""
def pix2norm(i, r):
    return (2*i - (r - 1))/r

"""
P is the projection matrix - maps from space  of surface (column vector) to 'normalized device coordinates' (in the OpenGL) sense. Rays are cast from the near plane to the far plane. See e.g. http://www.songho.ca/opengl/gl_projectionmatrix.html
"""
def raster(target, shader):
    resolution = target.shape
    # TODO screen aware ϵ
    # (nx, ny) are normalized device coordinates
    # Inner loop should be y (first index) for speed https://docs.julialang.org/en/v1/manual/performance-tips/#Access-arrays-in-memory-order,-along-columns-1.
    for ix in range(target.shape[1]):
        nx = pix2norm(ix, target.shape[1])
        for iy in range(target.shape[0]):
            ny = pix2norm(iy, target.shape[0])
            target[iy, ix] = shader(nx, ny)

def ndc2ray(nx:float, ny:float, invP:Sequence[Sequence[float]]):
    invP = np.asarray(invP)

    # Define ray intersection point with near and far planes in normalized device coordinates.
    n0 = np.asarray((nx, ny, -1.0, 1.0))
    nf = np.asarray((nx, ny, 1.0,  1.0))

    # Inverse project into world and divide by w component - see e.g.
    # https://stackoverflow.com/questions/1352564/mapping-from-normalized-device-coordinates-to-view-space
    w0p = n0.dot(invP)
    w0 = w0p / w0p[3]
    wfp = nf.dot(invP)
    wf = wfp / wfp[3]

    vp = wf - w0
    d_max = norm(vp)
    v = vp/d_max

    return w0, v, d_max


"""Shade based on distance for first intersection.
"""
def shade_distance(nx, ny, invP, surface, epsilon, max_steps):
    x0, v, d_max = ndc2ray(nx, ny, invP)
    d, steps, key, xi = spheretrace(surface, x0, v, d_max, epsilon, max_steps)
    return d

@dataclass
class WireframeModel:
    vertices: np.ndarray
    edges: np.ndarray
    color: np.ndarray

    def __post_init__(self):
        assert self.vertices.ndim == 2
        assert self.vertices.shape[1] == 4
        assert self.edges.ndim ==  2
        assert self.edges.shape[1] == 2
        assert self.color.shape == (3,)

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_edges(self):
        return self.edges.shape[0]

    @classmethod
    def make(cls, vertices: Sequence[Sequence[float]], edges: Sequence[Sequence[int]], color: Sequence[float]):
        vertices = np.array(vertices, float)
        edges = np.array(edges, int)
        color = np.array(color, float)
        return WireframeModel(vertices, edges, color)

@singledispatch
def make_wireframe(obj, color: Sequence[float]) -> WireframeModel:
    raise NotImplementedError(obj)

@make_wireframe.register
def _(obj: bounding.AABB, color: Sequence[float]):
    # Get 8x4 array of corner vertices.
    vertices = [[obj.corners[index][axis] for axis, index in enumerate(indices)] + [1] for indices in itertools.product((0, 1), repeat=3)]
    edges = (0, 1), (1, 3), (0, 2), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 7), (4, 6), (6, 7)
    return WireframeModel.make(vertices, edges, color)

class Projection(ABC):
    @property
    @abstractmethod
    def eye_to_clip(self, aspect: float) -> np.ndarray:
        pass

@dataclass
class Orthographic(Projection):
    half_width: float
    z_far: float

    def eye_to_clip(self, aspect: float) -> np.ndarray:
        half_height = self.half_width*aspect
        return orthographic(-self.half_width, self.half_width, -half_height, half_height, 0, self.z_far)

@dataclass
class Perspective(Projection):
    fov: float # radians
    z_near: float
    z_far: float

    def eye_to_clip(self, aspect: float) -> np.ndarray:
        half_width = self.z_near*np.tan(self.fov/2)
        half_height = half_width*aspect
        return projection(-half_width, half_width, -half_height, half_height, self.z_near, self.z_far)

# TODO make oblique projection - see https://www.cs.unm.edu/~angel/CS433/LECTURES/CS433_17.pdf. Could be useful.

@dataclass
class Scene:
    name: str
    sdb_glsl: str
    z_near: float
    z_far: float
    eye: np.ndarray
    center: np.ndarray
    wireframe_models: List[WireframeModel]
