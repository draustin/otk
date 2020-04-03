import numpy as np
from typing import Sequence
from functools import singledispatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from otk import ri
from ..sdb import Surface

__all__ = ['Medium', 'UniformIsotropic', 'Element', 'SimpleElement', 'Deflector', 'SimpleInterface', 'ConstantInterface',
    'calc_amplitudes',
    'FresnelInterface', 'SimpleDeflector', 'make_constant_deflector', 'make_fresnel_deflector', 'perfect_refractor']

class Medium(ABC):
    @property
    @abstractmethod
    def uniform(self):
        pass

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


