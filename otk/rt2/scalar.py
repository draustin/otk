"""
TODOs:
#      * More concise notation.
#      * Anitropic media
#      * Absorptive media
#      * diffraction grating
"""
import logging
from warnings import warn
import numpy as np
from functools import singledispatch
from dataclasses import dataclass
from typing import Sequence, Union, List

from ..types import Sequence4, Matrix4, Vector4
from ..functions import normalize, dot, norm_squared, reflect_vector, make_perpendicular
from .. import functions
from .. import sdb
from .. import v4h
from .. sdb import npscalar, Surface
from . import *

logger = logging.getLogger(__name__)

__all__ = ['Line', 'Ray', 'make_line', 'make_ray', 'perfect_refractor', 'Branch', 'get_points', 'get_deflector',
           'intersect', 'Segment', 'nonseq_trace', 'set_backend', 'form_ray']


backend = npscalar


def set_backend(module):
    global backend
    logging.info(f'Set rt2 scalar ray math backend to {module.__name__}.')
    backend = module


@dataclass
class Line:
    origin: np.ndarray
    vector: np.ndarray

    def __post_init__(self):
        assert v4h.is_point(self.origin)
        assert v4h.is_vector(self.vector)

    def eval(self, t: float) -> np.ndarray:
        return self.origin + t*self.vector

    def advance(self, t: float) -> 'Line':
        return Line(self.eval(t), self.vector)

    def pass_point(self, x:np.ndarray):
        t = dot(self.vector, x - self.origin)/norm_squared(self.vector)
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
    d = dot(l.vector[:3], s.n)
    try:
        t = -(s.c + dot(l.origin[:3], s.n))/d
    except ZeroDivisionError:
        t = float('inf')
    return t


def make_line(ox, oy, oz,  vx, vy, vz):
    return Line(np.array((ox, oy, oz, 1.)), np.array((vx, vy, vz, 0.)))


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
        return np.dot(backend.getnormal(self.element.surface, xp), self.local_to_root)

    def get_deflector(self, x:Sequence[float]) -> Deflector:
        xp = np.dot(x, self.root_to_local)
        return get_deflector(self.element, xp).transform(self.local_to_root)


@dataclass
class Ray:
    line: Line
    k: np.ndarray
    polarization: np.ndarray
    flux: float
    phase_origin: float
    lamb: float
    element: TransformedElement

    def __post_init__(self):
        assert v4h.is_vector(self.k)
        assert v4h.is_vector(self.polarization)
        assert np.isscalar(self.flux)

    def advance(self, t: float):
        phi = self.phase_origin + np.dot(self.k, self.line.vector)*t
        return Ray(self.line.advance(t), self.k, self.polarization, self.flux, phi, self.lamb, self.element)

@singledispatch
def make_ray(*args, **kwargs) -> Ray:
    raise NotImplementedError(args[0])

@make_ray.register
def _(ox: float, oy, oz, vx, vy, vz, px, py, pz, n, flux, phase_origin, lamb, element=None):
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
    vector = normalize(v4h.to_vector((vx, vy, vz)))
    line = Line(np.asarray((ox, oy, oz, 1.)), vector)
    # Make polarization perpendicular to vector.
    y = v4h.cross(vector, v4h.to_vector((px, py, pz)))
    pol = normalize(v4h.cross(y, vector))

    k = line.vector*n*2*np.pi/lamb
    return Ray(line, k, pol, flux, phase_origin, lamb, element)

@singledispatch
def get_deflector(e: Element, x: Sequence[float]) -> Deflector:
    raise NotImplementedError()

@get_deflector.register
def _(r:SimpleElement, x:Sequence[float]) -> Deflector:
    return r.deflector

@singledispatch
def get_dielectric_tensor(obj, lamb: float, x: Sequence[float]) -> np.ndarray:
    raise NotImplementedError(obj)

@get_dielectric_tensor.register
def _(self:UniformIsotropic, lamb: float, x: Sequence[float]) -> np.ndarray:
    eps = self.n(lamb)**2
    return np.diag((eps, eps, eps))

@calc_amplitudes.register
def _(self:ConstantInterface, n0: float, n1:float, cos_theta0: float, lamb: float) -> tuple:
    return (self.rp, self.rs), (self.tp, self.ts)

@calc_amplitudes.register
def _(self:FresnelInterface, n0: float, n1:float, cos_theta0: float, lamb: float) -> tuple:
    return functions.calc_fresnel_coefficients(n0, n1, cos_theta0)

@singledispatch
def deflect_ray(self:Deflector, ray: Ray, x0:np.ndarray, x1:np.ndarray, dielectric0: np.ndarray, normal: np.ndarray,
    dielectric1: np.ndarray, element1: TransformedElement) -> tuple:
    raise NotImplementedError()

def refract_vector(incident: Vector4, normal: Vector4, n_ratio: float) -> Union[Vector4, None]:
    """Calculate refracted wave vector given incident, normal and ratio of refractive indices.

    Surface normal must be normalized.

    Result has length of incident vector.
    """
    incident_normal_component = dot(incident, normal)
    projected = incident - normal*incident_normal_component
    epsilon = n_ratio**2*norm_squared(incident) - norm_squared(projected)
    if epsilon < 0:
        # Total internal reflection.
        return None
    refracted_normal_component = np.sign(incident_normal_component)*epsilon**0.5
    refracted = (refracted_normal_component*normal + projected)/n_ratio
    return refracted

@deflect_ray.register
def deflect_ray(self: SimpleDeflector, ray: Ray, x0:np.ndarray, x1:np.ndarray, dielectric0: np.ndarray, normal: np.ndarray,
    dielectric1: np.ndarray, element1: TransformedElement) -> tuple:
    assert is_isotropic(dielectric0)
    assert is_isotropic(dielectric1)
    n0 = dielectric0[0, 0]**0.5
    n1 = dielectric1[0, 0]**0.5
    n_ratio = n1/n0

    # Calculate normalized refracted and reflected vectors.
    refracted_vector = refract_vector(ray.line.vector, normal, n_ratio)
    reflected_vector = reflect_vector(ray.line.vector, normal)

    # Generate unit vector perpendicular to normal and incident.
    s_pol_vector = make_perpendicular(normal, ray.line.vector)
    incident_p_pol_vector = v4h.cross(ray.line.vector, s_pol_vector)


    cos_theta0 = abs(dot(normal, ray.line.vector))
    amplitudes = calc_amplitudes(self.interface, n0, n1, cos_theta0, ray.lamb)
    incident_components = [dot(ray.polarization, i) for i in (incident_p_pol_vector, s_pol_vector)]

    def calc_ray(origin, amplitudes, axis_vectors, ray_vector, n, element):
        pol_unnorm = sum(c*a*v for c, a, v in zip(incident_components, amplitudes, axis_vectors))
        pol_factor = norm_squared(pol_unnorm)
        pol = pol_unnorm/pol_factor**0.5
        line = Line(origin, ray_vector)
        deflected_ray = Ray(line, n*2*np.pi/ray.lamb*ray_vector, pol, ray.flux*pol_factor*n/n0, ray.phase_origin,
            ray.lamb, element)
        return deflected_ray

    rays = []
    if self.reflects:
        reflected_p_pol_vector = v4h.cross(reflected_vector, s_pol_vector)
        rays.append(
            calc_ray(x0, amplitudes[0], (reflected_p_pol_vector, s_pol_vector), reflected_vector, n0, ray.element))
    if self.refracts and refracted_vector is not None:
        refracted_p_pol_vector = v4h.cross(refracted_vector, s_pol_vector)
        rays.append(calc_ray(x1, amplitudes[1], (refracted_p_pol_vector, s_pol_vector), refracted_vector, n1, element1))

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
            sep = v4h.norm(seg.ray.line.origin - last_end)
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



@get_dielectric_tensor.register
def _(self: Assembly, lamb: float, x: Sequence[float]) -> np.ndarray:
    element = _get_transformed_element(self, x)
    if element is None:
        return get_dielectric_tensor(self.exterior, lamb, x)
    else:
        return element.get_dielectric_tensor(lamb, x)

def get_root_to_local(self, x: Sequence4, ancestry: List[Surface] = None) -> Matrix4:
    if ancestry is None:
        ancestry = self.get_ancestry_at(x)[0]
    m = np.eye(4)
    for surface0 in ancestry[::-1]:
        m0 = surface0.get_parent_to_child(x)
        m = np.dot(m, m0)
        x = np.dot(x, m0)
    return m

def get_root_to_local(x: Sequence4, ancestry: List[Surface]) -> Matrix4:
    m = np.eye(4)
    for surface0 in ancestry[::-1]:
        m0 = surface0.get_parent_to_child(x)
        m = np.dot(m, m0)
        x = np.dot(x, m0)
    return m

def _get_transformed_element(self: Assembly, x: Sequence4) -> TransformedElement:
    # insides = []
    # for surface, d in npscalar.traverse(self.surface, x):
    #     if d <= 0 and surface in self.elements:
    #         insides.append(surface)
    # if len(insides) == 0:
    #     return None
    # elif len(insides) == 1:
    #     element = self.elements[insides[0]]
    #     return TransformedElement(sdb.get_root_to_local(element.surface, x), element)
    # else:
    #     raise ValueError(f'Point {x} is inside {len(insides)} elements.')

    #insides = []
    # d0 = np.inf
    # surface0 = None
    # for surface, d in npscalar.traverse(self.surface, x):
    #     if d <= d0:
    #         d0 = d
    #         surface0 = surface
    # if d0 <= 0:
    #     for surface, element in self.elements.items():
    #         if surface0 in surface.descendants():
    #             return TransformedElement(sdb.get_root_to_local(surface, x), element)

    ancestry, d = self.surface.get_ancestry_at(x)
    if d > 0:
        return None

    for num, surface in enumerate(ancestry):
        try:
            element = self.elements[surface]
        except KeyError:
            pass
        else:
            break
    else:
        raise ValueError('No element found.')

    root_to_local = get_root_to_local(x, ancestry[num:])

    return TransformedElement(root_to_local, element)

    # if len(insides) == 0:
    #     return None
    # elif len(insides) == 1:
    #     element = self.elements[insides[0]]
    #     return TransformedElement(sdb.get_root_to_local(element.surface, x), element)
    # else:
    #     raise ValueError(f'Point {x} is inside {len(insides)} elements.')


def _process_ray(self: Assembly, ray:Ray, sphere_trace_kwargs:dict) -> tuple:
    """Intersect ray with geometry, the deflect it."""
    # Ray travels from element/medium 0 to 1.
    element0 = ray.element
    if element0 is None:
        trace = backend.spheretrace(self.surface, ray.line.origin, ray.line.vector, sign=1., through=True,
            **sphere_trace_kwargs)
        if trace.d >= 0:
            if trace.steps >= sphere_trace_kwargs['max_steps']:
                warn(f"Spheretrace reached max_steps ({sphere_trace_kwargs['max_steps']}.")
            return None, ()

        x0 = trace.last_x
        x1 = trace.x
        tensor0 = get_dielectric_tensor(self.exterior, ray.lamb, trace.last_x)
        element1 = _get_transformed_element(self, trace.x)
        tensor1 = element1.get_dielectric_tensor(ray.lamb, trace.x)
        normal = -element1.getnormal(trace.xm) # Normal points from 0 to 1.
        deflector = element1.get_deflector(trace.x)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{trace} - {element1.element.surface}')
    else:
        line0 = ray.line.transform(element0.root_to_local)
        trace = backend.spheretrace(element0.element.surface, line0.origin, line0.vector, sign=-1., through=True,
            **sphere_trace_kwargs)
        if trace.d <= 0:
            if trace.steps >= sphere_trace_kwargs['max_steps']:
                warn(f"Spheretrace reached max_steps={sphere_trace_kwargs['max_steps']}.")
            return None, ()

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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{element0.element.surface} - {trace}')
        else:
            tensor1 = element1.get_dielectric_tensor(ray.lamb, x1)
            normal1 = -element1.getnormal(xm)
            assert np.allclose(normal, normal1)
            deflector1 = element1.get_deflector(x1)
            deflector = deflector1 # TODO HACK
            # TODO assert deflectors are equal
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{element0.element.surface} - {trace} - {element1.element.surface}')

    ray = ray.advance(trace.tm)
    deflected_rays = deflect_ray(deflector, ray, x0, x1, tensor0, normal, tensor1, element1)
    return trace.tm, deflected_rays


def nonseq_trace(self: Assembly, start_ray:Ray, sphere_trace_kwargs_:dict=None, min_flux:float=None, num_deflections:int=None) -> Branch:
    if sphere_trace_kwargs_ is None:
        sphere_trace_kwargs_ = {}
    sphere_trace_kwargs = dict(epsilon=start_ray.lamb*1e-3, t_max=1e9, max_steps=100)
    sphere_trace_kwargs.update(sphere_trace_kwargs_)

    if num_deflections == 0:
        return Branch(start_ray, None, [])
    length, deflected_rays = _process_ray(self, start_ray, sphere_trace_kwargs)
    segments = []
    if num_deflections is not None:
        num_deflections -= 1
    for ray in deflected_rays:
        if min_flux is not None and ray.flux < min_flux:
            continue
        segments.append(nonseq_trace(self, ray, sphere_trace_kwargs, min_flux, num_deflections))
    return Branch(start_ray, length, segments)


@make_ray.register
def _(self: Assembly, ox, oy, oz, vx, vy, vz, px, py, pz, lamb, flux=1, phase_origin=0):
    x = v4h.to_vector((ox, oy, oz))
    # TODO tidy
    dielectric = get_dielectric_tensor(self, lamb, x)
    assert is_isotropic(dielectric)
    n = dielectric[0, 0]**0.5
    element = _get_transformed_element(self, x)
    return make_ray(ox, oy, oz, vx, vy, vz, px, py, pz, n, flux, phase_origin, lamb, element)


def form_ray(assembly: Assembly, line: Line, pol: np.ndarray, lamb: float, flux: float=1., phase_origin: float=0.):
    dielectric = get_dielectric_tensor(assembly, lamb, line.origin)
    assert is_isotropic(dielectric)
    n = dielectric[0, 0]**0.5
    k = line.vector*n*2*np.pi/lamb
    element = _get_transformed_element(assembly, line.origin)
    return Ray(line, k, pol, flux, phase_origin, lamb, element)





