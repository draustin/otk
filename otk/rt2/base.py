import numpy as np
from typing import Sequence
from functools import singledispatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from otk import ri
from otk.types import Matrix4
from ..sdb import Surface, UnionOp, AffineOp


class Medium(ABC):
    @property
    @abstractmethod
    def uniform(self) -> bool:
        pass

    @classmethod
    def make(cls, x):
        if isinstance(x, ri.Index):
            return UniformIsotropic(x)
        else:
            n = float(x)
            return UniformIsotropic(ri.FixedIndex(n))

    @abstractmethod
    def transform(self, to_local: Matrix4) -> 'Medium':
        """Return copy of self with transformed coordinate system.

        Transform rule: if
            new = old.transform()
        and f is a scalar property then
            new.f(x) = old.f(x@to_local)
        """
        pass


@dataclass
class UniformIsotropic(Medium):
    n: ri.Index

    @property
    def uniform(self):
        return True

    def transform(self, to_local: Matrix4) -> 'Medium':
        return self

class Interface(ABC):
    @abstractmethod
    def transform(self, to_local: Matrix4) -> 'Interface':
        """Return copy of self with transformed coordinate system.

        Transform rule: if
            new = old.transform()
        and f is a scalar property then
            new.f(x) = old.f(x@to_local)
        """
        pass


@dataclass
class Element:
    surface: Surface
    medium: Medium
    interface: Interface

    def transform(self, to_local: Matrix4) -> 'Element':
        """Return copy of self with transformed coordinate system.

        Transform rule: if
            new = old.transform()
        and f is a scalar property then
            new.f(x) = old.f(x@to_local)
        """
        surface = AffineOp(self.surface, np.linalg.inv(to_local)) # TODO check direction
        medium = self.medium.transform(to_local)
        interface = self.interface.transform(to_local)
        return Element(surface, medium, interface)


class FresnelInterface(Interface):
    def transform(self, m: np.ndarray):
        return self


@dataclass
class ConstantInterface(Interface):
    rp: float
    rs: float
    tp: float
    ts: float

    def transform(self, to_local: Matrix4):
        return self


perfect_refractor = ConstantInterface(0, 0, 1, 1)
perfect_reflector = ConstantInterface(1, 1, 0, 0)


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


