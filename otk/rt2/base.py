import numpy as np
from typing import Sequence
from functools import singledispatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from otk import ri
from ..sdb import Surface, UnionOp

__all__ = ['Medium', 'UniformIsotropic', 'Element', 'SimpleElement', 'Deflector', 'SimpleInterface', 'ConstantInterface',
    'calc_amplitudes', 'perfect_reflector',
    'FresnelInterface', 'SimpleDeflector', 'make_constant_deflector', 'make_fresnel_deflector', 'perfect_refractor',
    'Assembly']

class Medium(ABC):
    @property
    @abstractmethod
    def uniform(self):
        pass

    @classmethod
    def make(cls, x):
        if isinstance(x, ri.Index):
            return UniformIsotropic(x)
        else:
            n = float(x)
            return UniformIsotropic(ri.FixedIndex(n))

@dataclass
class UniformIsotropic(Medium):
    n: ri.Index

    @property
    def uniform(self):
        return True

class Deflector(ABC):
    @abstractmethod
    def transform(self, m: np.ndarray) -> 'Deflector':
        pass

@dataclass
class Element:
    surface: Surface
    medium: Medium

@dataclass
class SimpleElement(Element):
    deflector: Deflector

class SimpleInterface(ABC):
    @abstractmethod
    def transform(self, m: np.ndarray) -> 'SimpleInterface':
        pass

class FresnelInterface(SimpleInterface):
    def transform(self, m: np.ndarray):
        return self

@dataclass
class ConstantInterface(SimpleInterface):
    rp: float
    rs: float
    tp: float
    ts: float

    def transform(self, m: np.ndarray):
        return self

@singledispatch
def calc_amplitudes(self:SimpleInterface, n0: float, n1:float, cos_theta0: float, lamb: float) -> tuple:
    raise NotImplementedError()

@dataclass
class SimpleDeflector(Deflector):
    interface: SimpleInterface
    reflects: bool
    refracts: bool

    def transform(self, m: np.ndarray):
        return SimpleDeflector(self.interface.transform(m), self.reflects, self.refracts)

def make_constant_deflector(rp:float, rs:float, tp:float, ts:float, reflects:bool = True, refracts:bool = True):
    return SimpleDeflector(ConstantInterface(rp, rs, tp, ts), reflects, refracts)

def make_fresnel_deflector(reflects:bool = True, refracts:bool = True):
    #calc_amplitudes = lambda n0, n1, cos_theta0, lamb: omath.calc_fresnel_coefficients(n0, n1, cos_theta0)
    return SimpleDeflector(FresnelInterface(), reflects, refracts)

perfect_refractor = make_constant_deflector(0, 0, 1, 1, False, True)
perfect_reflector = make_constant_deflector(1, 1, 0, 0, True, False)

class Assembly:
    def __init__(self, surface: Surface, elements:Sequence[Element], exterior: Medium):
        self.surface = surface
        self.elements = {e.surface:e for e in elements}
        self.exterior = exterior

    @classmethod
    def make(cls, elements: Sequence[Element], exterior):
        surface = UnionOp([e.surface for e in elements])
        exterior = Medium.make(exterior)
        return Assembly(surface, elements, exterior)


