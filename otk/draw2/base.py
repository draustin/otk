import typing
from dataclasses import dataclass

__all__ = ['Drawing', 'Translation', 'Scaling', 'Arc', 'Line', 'Sequence']

class Drawing:
    pass

@dataclass
class Translation(Drawing):
    x: float
    y: float
    child: Drawing

@dataclass
class Scaling(Drawing):
    x: float
    y: float
    child: Drawing

@dataclass
class Arc(Drawing):
    xc: float
    yc: float
    radius: float
    theta0: float
    theta1: float

@dataclass
class Line(Drawing):
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class Sequence(Drawing):
    children: typing.Sequence[Drawing]