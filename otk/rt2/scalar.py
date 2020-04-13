"""
TODOs:
#      * More concise notation.
#      * Anitropic media
#      * Absorptive media
#      * diffraction grating
"""
from warnings import warn
import numpy as np
from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence, Union, List
from .. import math as omath
from .. import sdb
from .. import geo3
#from ..sdb.npscalar import *
#from ..v4 import *
from .. import v4
from .. sdb import npscalar
from . import *

__all__ = ['Line', 'Ray', 'make_line', 'make_ray', 'perfect_refractor', 'Branch', 'get_points',
    'get_deflector', 'intersect', 'Segment', 'nonseq_trace']

@dataclass
class Line:
    origin: np.ndarray
    vector: np.ndarray

    def __post_init__(self):
        assert v4.is_point(self.origin)
        assert v4.is_vector(self.vector)

    def eval(self, t: float) -> np.ndarray:
        return self.origin + t*self.vector

    def advance(self, t: float) -> 'Line':
        return Line(self.eval(t), self.vector)

    def pass_point(self, x:np.ndarray):
        t = v4.dot(self.vector, x - self.origin)/v4.norm_squared(self.vector)
        return t, self.advance(t)

    def transform(self, m: np.ndarray) -> 'Line':
        return Line(np.dot(self.origin, m), np.dot(self.vector, m))

    def advance_to(self, surface: sdb.Surface) -> 'Line':
        t = intersect(surface, self)
        return self.advance(t)

@singledispatch
def intersect(surface: sdb.Surface, line: Line) -> float:
    """First intersection of line with surface."""
    raise NotImplementedError()

@intersect.register
def _(s: sdb.Plane, l: Line) -> float:
    d = v4.dot(l.vector[:3], s.n)
    try:
        t = -(s.c + v4.dot(l.origin[:3], s.n))/d
    except ZeroDivisionError:
        t = float('inf')
    return t

def make_line(ox, oy, oz,  vx, vy, vz):
    return Line(np.array((ox, oy, oz, 1.)), np.array((vx, vy, vz, 0.)))

@dataclass
class Ray:
    line: Line
    k: np.ndarray
    polarization: np.ndarray
    flux: float
    phase_origin: float
    lamb: float

    def __post_init__(self):
        assert v4.is_vector(self.k)
        assert v4.is_vector(self.polarization)
        assert np.isscalar(self.flux)

    def advance(self, t: float):
        phi = self.phase_origin + np.dot(self.k, self.line.vector)*t
        return Ray(self.line.advance(t), self.k, self.polarization, self.flux, phi, self.lamb)

@singledispatch
def make_ray(*args, **kwargs) -> Ray:
    raise NotImplementedError(args[0])

@make_ray.register
def _(ox: float, oy, oz, vx, vy, vz, px, py, pz, n, flux, phase_origin, lamb):
    """Make ray in isotropic medium.

    Args:
        ox, oy, oz: Ray origin.
        vx, vy, vz: Unnormalized ray vector components.
        px, py, pz: Unnormalized polarization.
        n: Refractive index.
        flux: Ray flux.
        phase_origin: Phase at ray  origin.
        lamb: Wavelength.
    """
    vector = v4.normalize((vx, vy, vz, 0))
    line = Line(np.asarray((ox, oy, oz, 1.)), vector)
    # Make polarization perpendicular to vector.
    y = v4.cross(vector, (px, py, pz, 0))
    pol = v4.normalize(v4.cross(y, vector))

    k = line.vector*n*2*np.pi/lamb
    return Ray(line, k, pol, flux, phase_origin, lamb)

@singledispatch
def get_deflector(e: Element, x: Sequence[float]) -> Deflector:
    raise NotImplementedError()

@get_deflector.register
def _(r:SimpleElement, x:Sequence[float]) -> Deflector:
    return r.deflector

@singledispatch
def get_dielectric_tensor(self:Medium, lamb: float, x: Sequence[float]) -> np.ndarray:
    raise NotImplementedError(self)

@get_dielectric_tensor.register
def _(self:UniformIsotropic, lamb: float, x: Sequence[float]) -> np.ndarray:
    eps = self.n(lamb)**2
    return np.diag((eps, eps, eps))

@calc_amplitudes.register
def _(self:ConstantInterface, n0: float, n1:float, cos_theta0: float, lamb: float) -> tuple:
    return (self.rp, self.rs), (self.tp, self.ts)

@calc_amplitudes.register
def _(self:FresnelInterface, n0: float, n1:float, cos_theta0: float, lamb: float) -> tuple:
    return omath.calc_fresnel_coefficients(n0, n1, cos_theta0)

@singledispatch
def deflect_ray(self:Deflector, ray: Ray, x0:np.ndarray, x1:np.ndarray, dielectric0, normal: np.ndarray, dielectric1) -> tuple:
    raise NotImplementedError()

@deflect_ray.register
def deflect_ray(self: SimpleDeflector, ray: Ray, x0:np.ndarray, x1:np.ndarray, dielectric0, normal: np.ndarray, dielectric1) -> tuple:
    assert is_isotropic(dielectric0)
    assert is_isotropic(dielectric1)
    n0 = dielectric0[0, 0]**0.5
    n1 = dielectric1[0, 0]**0.5
    n_ratio = n1/n0

    # Calculate normalized refracted and reflected vectors.
    refracted_vector = geo3.refract_vector(ray.line.vector, normal, n_ratio)/n_ratio
    reflected_vector = geo3.reflect_vector(ray.line.vector, normal)

    # Generate unit vector perpendicular to normal and incident.
    s_pol_vector = geo3.make_perpendicular(normal, ray.line.vector)

    incident_p_pol_vector = v4.cross(ray.line.vector, s_pol_vector)
    refracted_p_pol_vector = v4.cross(refracted_vector, s_pol_vector)
    reflected_p_pol_vector = v4.cross(reflected_vector, s_pol_vector)

    cos_theta0 = abs(v4.dot(normal, ray.line.vector))
    amplitudes = calc_amplitudes(self.interface, n0, n1, cos_theta0, ray.lamb)
    incident_components = [v4.dot(ray.polarization, i) for i in (incident_p_pol_vector, s_pol_vector)]

    def calc_ray(origin, amplitudes, axis_vectors, ray_vector, n):
        pol_unnorm = sum(c*a*v for c, a, v in zip(incident_components, amplitudes, axis_vectors))
        pol_factor = v4.norm_squared(pol_unnorm)
        pol = pol_unnorm/pol_factor**0.5
        line = Line(origin, ray_vector)
        deflected_ray = Ray(line, n*2*np.pi/ray.lamb*ray_vector, pol, ray.flux*pol_factor*n/n0, ray.phase_origin,
            ray.lamb)
        return deflected_ray

    rays = []
    if self.reflects:
        rays.append(
            calc_ray(x0, amplitudes[0], (reflected_p_pol_vector, s_pol_vector), reflected_vector, n0))
    if self.refracts:
        rays.append(calc_ray(x1, amplitudes[1], (refracted_p_pol_vector, s_pol_vector), refracted_vector, n1))

    return tuple(rays)


def is_isotropic(tensor):
    n = tensor.shape[0]
    assert tensor.shape == (n, n)
    return np.all(tensor == np.diag((tensor[0, 0], )*n))


@get_dielectric_tensor.register
def _(self, lamb: float, x: Sequence[float]) -> np.ndarray:
    xp = np.dot(x, self.root_to_local)
    return get_dielectric_tensor(self.region, lamb, xp)

@dataclass
class Segment:
    ray: Ray
    length: Union[float, None]

def get_points(segments: Sequence[Segment], default_length: float = None, max_sep: float=None) -> np.ndarray:
    last_end = None
    points = [segments[0].ray.line.origin]
    for seg in segments:
        if last_end is not None and max_sep is not None:
            sep = v4.norm(seg.ray.line.origin - last_end)
            if sep > max_sep:
                raise ValueError(f'Separation between start and last end point is {sep}, greater than {max_sep}.')
        if seg.length is None:
            if default_length is None:
               raise ValueError(f'Unbounded ray - specify default_length.')
            else:
                length = default_length
        else:
            length = seg.length
        last_end = seg.ray.line.eval(length)
        points.append(last_end)
    return np.asarray(points)

@dataclass
class Branch:
    ray: Ray
    length: Union[float, None]
    children: List['Branch']

    def get_num_deflections(self) -> int:
        if self.length is None:
            return 0
        else:
            return max(child.get_num_deflections() for child in self.children) + 1

    def flatten(self) -> List[Segment]:
        me = [Segment(self.ray, self.length)]
        if len(self.children) == 0:
            return me
        elif len(self.children) == 1:
            return me + self.children[0].flatten()
        else:
            raise ValueError(f'{len(self.children)} - cannot flatten.')

@dataclass
class TransformedElement:
    root_to_local: np.ndarray
    element: Element

    def __post_init__(self):
        self.local_to_root = np.linalg.inv(self.root_to_local)

    def get_dielectric_tensor(self, lamb: float, x: Sequence[float]) -> np.ndarray:
        xp = np.dot(x, self.root_to_local)
        return get_dielectric_tensor(self.element.medium, lamb, xp)

    def getnormal(self, x:Sequence[float]) -> np.ndarray:
        xp = np.dot(x, self.root_to_local)
        return np.dot(npscalar.getnormal(self.element.surface, xp), self.local_to_root)

    def get_deflector(self, x:Sequence[float]) -> Deflector:
        xp = np.dot(x, self.root_to_local)
        return get_deflector(self.element, xp).transform(self.local_to_root)

@get_dielectric_tensor.register
def _(self: Assembly, lamb: float, x: Sequence[float]) -> np.ndarray:
    element = _get_transformed_element(self, x)
    if element is None:
        return get_dielectric_tensor(self.exterior, lamb, x)
    else:
        return element.get_dielectric_tensor(lamb, x)

def _get_transformed_element(self: Assembly, x) -> TransformedElement:
    insides = []
    for surface, d in npscalar.traverse(self.surface, x):
        if d <= 0 and surface in self.elements:
            insides.append(surface)
    if len(insides) == 0:
        return None
    elif len(insides) == 1:
        element = self.elements[insides[0]]
        return TransformedElement(sdb.get_root_to_local(element.surface, x), element)
    else:
        raise ValueError(f'Point {x} is inside {len(insides)} elements.')

def _process_ray(self: Assembly, ray:Ray, sphere_trace_kwargs:dict, spheretrace = npscalar.spheretrace) -> tuple:
    """Intersect ray with geometry, the deflect it."""
    # Ray travels from element/medium 0 to 1.
    element0 = _get_transformed_element(self, ray.line.origin)
    if element0 is None:
        trace = spheretrace(self.surface, ray.line.origin, ray.line.vector, sign=1., through=True,
            **sphere_trace_kwargs)
        if trace.d >= 0:
            if trace.steps >= sphere_trace_kwargs['max_steps']:
                warn(f"Spheretrace reached max_steps ({sphere_trace_kwargs['max_steps']}.")
            return None, ()
        else:
            x0 = trace.last_x
            x1 = trace.x
            tensor0 = get_dielectric_tensor(self.exterior, ray.lamb, trace.last_x)
            element1 = _get_transformed_element(self, trace.x)
            tensor1 = element1.get_dielectric_tensor(ray.lamb, trace.x)
            normal = -element1.getnormal(trace.xm) # Normal points from 0 to 1.
            deflector = element1.get_deflector(trace.x)
    else:
        line0 = ray.line.transform(element0.root_to_local)
        trace = spheretrace(element0.element.surface, line0.origin, line0.vector, sign=-1., through=True,
            **sphere_trace_kwargs)
        if trace.d <= 0:
            if trace.steps >= sphere_trace_kwargs['max_steps']:
                warn(f"Spheretrace reached max_steps ({sphere_trace_kwargs['max_steps']}.")
            return None, ()
        else:
            # TODO some recomputation of transforms here
            x0 = np.dot(trace.last_x, element0.local_to_root)
            xm = np.dot(trace.xm, element0.local_to_root)
            x1 = np.dot(trace.x, element0.local_to_root)

            tensor0 = element0.get_dielectric_tensor(ray.lamb, x0)
            # normal should point from 0 to 1, so no minus sign.
            normal = element0.getnormal(xm)
            deflector = element0.get_deflector(x0)

            element1 = _get_transformed_element(self, x1)
            if element1 is None:
                tensor1 = get_dielectric_tensor(self.exterior, ray.lamb, x1)
            else:
                tensor1 = element1.get_dielectric_tensor(ray.lamb, x1)
                normal1 = -element1.getnormal(xm)
                assert np.allclose(normal, normal1)
                deflector1 = element1.get_deflector(x1)
                deflector = deflector1 # TODO HACK
                # TODO assert deflectors are equal

    ray = ray.advance(trace.tm)
    deflected_rays = deflect_ray(deflector, ray, x0, x1, tensor0, normal, tensor1)
    return trace.tm, deflected_rays

def nonseq_trace(self: Assembly, start_ray:Ray, sphere_trace_kwargs_:dict=None, min_flux:float=None, num_deflections:int=None, spheretrace = npscalar.spheretrace) -> Branch:
    if sphere_trace_kwargs_ is None:
        sphere_trace_kwargs_ = {}
    sphere_trace_kwargs = dict(epsilon=start_ray.lamb*1e-3, t_max=1e9, max_steps=100)
    sphere_trace_kwargs.update(sphere_trace_kwargs_)

    if num_deflections == 0:
        return Branch(start_ray, None, [])
    length, deflected_rays = _process_ray(self, start_ray, sphere_trace_kwargs, spheretrace)
    segments = []
    if num_deflections is not None:
        num_deflections -= 1
    for ray in deflected_rays:
        if min_flux is not None and ray.flux < min_flux:
            continue
        segments.append(nonseq_trace(self, ray, sphere_trace_kwargs, min_flux, num_deflections, spheretrace))
    return Branch(start_ray, length, segments)

@make_ray.register
def _(self: Assembly, ox, oy, oz, vx, vy, vz, px, py, pz, lamb, flux=1, phase_origin=0):
    x = v4.to_vector((ox, oy, oz))
    dielectric = get_dielectric_tensor(self, lamb, x)
    assert is_isotropic(dielectric)
    n = dielectric[0, 0]**0.5
    return make_ray(ox, oy, oz, vx, vy, vz, px, py, pz, n, flux, phase_origin, lamb)




