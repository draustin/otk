from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Mapping, Tuple, Callable, Union, List
from .. import ri
import numpy as np
from .. import math as omath
from ..sdb import *
#from ..vector3 import *
from ..geo3 import *
from ..sdb.scalar import *

__all__ = ['Medium', 'UniformIsotropic', 'SimpleInterface', 'make_constant_interface', 'make_fresnel_interface', 'SimpleDeflector',
    'Assembly', 'Element', 'SimpleElement', 'Line', 'Ray', 'make_line', 'make_ray', 'perfect_refractor', 'Branch', 'get_points']

def is_point(x:np.ndarray):
    return (x.shape == (4,)) and (x[3] == 1.)

def is_vector(v:np.ndarray):
    return (v.shape == (4,)) and (v[3] == 0.)

@dataclass
class Line:
    origin: np.ndarray
    vector: np.ndarray

    def __post_init__(self):
        assert is_point(self.origin)
        assert is_vector(self.vector)

    def eval(self, t: float) -> np.ndarray:
        return self.origin + t*self.vector

    def advance(self, t: float) -> 'Line':
        return Line(self.eval(t), self.vector)

    def pass_point(self, x:np.ndarray):
        t = dot(self.vector, x - self.origin)/norm_squared(self.vector)
        return t, self.advance(t)

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
        assert is_vector(self.k)
        assert is_vector(self.polarization)
        assert np.isscalar(self.flux)

    def advance(self, t: float):
        phi = self.phase_origin + np.dot(self.k, self.line.vector)*t
        return Ray(self.line.advance(t), self.k, self.polarization, self.flux, phi, self.lamb)

def make_ray(ox, oy, oz, vx, vy, vz, px, py, pz, n, flux, phase_origin, lamb):
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
    vector = normalize((vx, vy, vz, 0))
    line = Line(np.asarray((ox, oy, oz, 1.)), vector)
    # Make polarization perpendicular to vector.
    y = cross(vector, (px, py, pz, 0))
    pol = normalize(cross(y, vector))

    k = line.vector*n*2*np.pi/lamb
    return Ray(line, k, pol, flux, phase_origin, lamb)

class Medium(ABC):
    @abstractmethod
    def get_dielectric_tensor(self, lamb, x):
        pass

    @property
    @abstractmethod
    def uniform(self):
        pass

@dataclass
class UniformIsotropic(Medium):
    n: ri.Index

    def get_dielectric_tensor(self, lamb, x):
        eps = self.n(lamb)**2
        return np.diag((eps, eps, eps))

    @property
    def uniform(self):
        return True

class Deflector(ABC):
    @abstractmethod
    def deflect_ray(self, ray0:Ray, x1:np.ndarray, dielectric0, normal: np.ndarray, dielectric1) -> tuple:
        pass

@dataclass
class SimpleInterface:
    calc_amplitudes: Callable
    reflects: bool
    refracts: bool

def make_constant_interface(rp:float, rs:float, tp:float, ts:float, reflects:bool = True, refracts:bool = True):
    calc_amplitudes = lambda n1, n2, cos_theta1, lamb: ((rp, rs), (tp, ts))
    return SimpleInterface(calc_amplitudes, reflects, refracts)

def make_fresnel_interface(reflects:bool = True, refracts:bool = True):
    calc_amplitudes = lambda n0, n1, cos_theta0, lamb: omath.calc_fresnel_coefficients(n0, n1, cos_theta0)
    return SimpleInterface(calc_amplitudes, reflects, refracts)

def deflect_ray_simple(ray0: Ray, x1:np.ndarray, dielectric0, normal: np.ndarray, dielectric1, interface):
    """
    TODOs:
     * More concise notation.
     * Anitropic media
     * Absorptive media
     * diffraction grating

    Args:
        ray0:
        medium0:
        normal:
        interface:
        medium1:

    Returns:

    """

    assert is_isotropic(dielectric0)
    assert is_isotropic(dielectric1)
    n0 = dielectric0[0, 0]**0.5
    n1 = dielectric1[0, 0]**0.5
    n_ratio = n1/n0

    # Calculate normalized refracted and reflected vectors.
    refracted_vector = refract_vector(ray0.line.vector, normal, n_ratio)/n_ratio
    reflected_vector = reflect_vector(ray0.line.vector, normal)

    # Generate unit vector perpendicular to normal and incident.
    s_pol_vector = make_perpendicular(normal, ray0.line.vector)

    incident_p_pol_vector = cross(ray0.line.vector, s_pol_vector)
    refracted_p_pol_vector = cross(refracted_vector, s_pol_vector)
    reflected_p_pol_vector = cross(reflected_vector, s_pol_vector)

    cos_theta1 = abs(dot(normal, ray0.line.vector))
    amplitudes = interface.calc_amplitudes(n0, n1, cos_theta1, ray0.lamb)
    incident_components = [dot(ray0.polarization, i) for i in (incident_p_pol_vector, s_pol_vector)]

    def calc_ray(origin, amplitudes, axis_vectors, ray_vector, n):
        pol_unnorm = sum(c*a*v for c, a, v in zip(incident_components, amplitudes, axis_vectors))
        pol_factor = norm_squared(pol_unnorm)
        pol = pol_unnorm/pol_factor**0.5
        line = Line(origin, ray_vector)
        deflected_ray = Ray(line, n*2*np.pi/ray0.lamb*ray_vector, pol, ray0.flux*pol_factor*n/n0, ray0.phase_origin, ray0.lamb)
        return deflected_ray

    rays = []
    if interface.reflects:
        rays.append(calc_ray(ray0.line.origin, amplitudes[0], (reflected_p_pol_vector, s_pol_vector),
            reflected_vector, n0))
    if interface.refracts:
        rays.append(calc_ray(x1, amplitudes[1], (refracted_p_pol_vector, s_pol_vector), refracted_vector, n1))

    return tuple(rays)

class SimpleDeflector(Deflector):
    """Isotropic and non-scattering."""
    def __init__(self, interface:SimpleInterface):
        self.interface = interface

    def deflect_ray(self, ray0: Ray, x1:np.ndarray, dielectric0, normal: np.ndarray, dielectric1):
        return deflect_ray_simple(ray0, x1, dielectric0, normal, dielectric1, self.interface)

perfect_refractor = SimpleDeflector(make_constant_interface(0, 0, 1, 1, False, True))

def is_isotropic(tensor):
    n = tensor.shape[0]
    assert tensor.shape == (n, n)
    return np.all(tensor == np.diag((tensor[0, 0], )*n))

class Element(ABC):
    @property
    @abstractmethod
    def surface(self) -> Surface:
        pass

    @property
    @abstractmethod
    def interior(self) -> Medium:
        pass

    @abstractmethod
    def get_deflector(self, x) -> Deflector:
        pass

class SimpleElement(Element):
    def __init__(self, surface: Surface, n:ri.Index, get_deflector: Callable[[np.ndarray], Deflector]):
        self._surface = surface
        self._interior = UniformIsotropic(n)
        self._get_deflector = get_deflector

    @property
    def surface(self):
        return self._surface

    @property
    def interior(self):
        return self._interior

    def get_deflector(self, x) -> Deflector:
        #isdb = identify(self._surface, x)
        return self._get_deflector(x)

# def make_simple_element(surface: Surface, n: Union[ri.Index, float], deflector: Deflector=None,
#     deflectors: Mapping[Tuple[Surface, int], Deflector]=None) -> SimpleElement:
#     """Convenience SimpleElement creator.
#
#     If n is float then ri.FixedIndex is created.
#
#     One of deflector or deflectors must be given."""
#     if not isinstance(n, ri.Index):
#         n = ri.FixedIndex(n)
#
#     if deflectors is None:
#         deflectors = {()}

@dataclass
class Segment:
    ray: Ray
    length: Union[float, None]

def get_points(segments: Sequence[Segment], default_length: float = None, max_sep: float=None) -> np.ndarray:
    last_end = None
    points = [segments[0].ray.line.origin]
    for seg in segments:
        if last_end is not None and max_sep is not None:
            sep = norm(seg.ray.line.origin - last_end)
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

class Assembly:
    def __init__(self, elements:Sequence[Element], exterior:Medium):
        self.elements = elements
        self.exterior = exterior
        self.surface = UnionOp([e.surface for e in elements])

    def get_element(self, x) -> Element:
        insides = [e for e in self.elements if getsdb(e.surface, x) < 0]
        if len(insides) == 0:
            return None
        elif len(insides) == 1:
            return insides[0]
        else:
            raise ValueError(f'Point {x} is inside {len(insides)} elements.')

    def process_ray(self, ray:Ray, sphere_trace_kwargs:dict) -> tuple:
        """Intersect ray with geometry, the deflect it."""
        # Ray travels from element/medium 0 to 1.
        element0 = self.get_element(ray.line.origin)
        if element0 is None:
            medium0 = self.exterior
            trace = spheretrace(self.surface, ray.line.origin, ray.line.vector, sign=1., through=True,
                **sphere_trace_kwargs)
            if trace.d >= 0:
                return None, ()
            else:
                element1 = self.get_element(trace.x)
                medium1 = element1.interior
                normal = -getnormal(element1.surface, trace.xm) # Normal points from 0 to 1.
                deflector = element1.get_deflector(trace.xm)
        else:
            medium0 = element0.interior
            trace = spheretrace(element0.surface, ray.line.origin, ray.line.vector, sign=-1., through=True,
                **sphere_trace_kwargs)
            if trace.d <= 0:
                return None, ()
            else:
                element1 = self.get_element(trace.x)
                # normal should point from 0 to 1, so no minus sign.
                normal = getnormal(element0.surface, trace.xm)
                if element1 is None:
                    medium1 = self.exterior
                else:
                    medium1 = element1.interior
            deflector = element0.get_deflector(trace.xm)

        ray0 = ray.advance(trace.last_t)
        x1 = ray.line.origin + ray.line.vector*trace.t
        tensor0 = medium0.get_dielectric_tensor(ray.lamb, ray.line.origin)
        tensor1 = medium1.get_dielectric_tensor(ray.lamb, ray.line.origin)
        deflected_rays = deflector.deflect_ray(ray0, x1, tensor0, normal, tensor1)
        return trace.tm, deflected_rays

    def nonseq_trace(self, start_ray:Ray, sphere_trace_kwargs_:dict=None, min_flux:float=None, num_deflections:int=None) -> Branch:
        if sphere_trace_kwargs_ is None:
            sphere_trace_kwargs_ = {}
        sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)
        sphere_trace_kwargs.update(sphere_trace_kwargs_)

        if num_deflections == 0:
            return Branch(start_ray, None, [])
        length, deflected_rays = self.process_ray(start_ray, sphere_trace_kwargs)
        segments = []
        if num_deflections is not None:
            num_deflections -= 1
        for ray in deflected_rays:
            if min_flux is not None and ray.flux < min_flux:
                continue
            segments.append(self.nonseq_trace(ray, sphere_trace_kwargs, min_flux, num_deflections))
        return Branch(start_ray, length, segments)


