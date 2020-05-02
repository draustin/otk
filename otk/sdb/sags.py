from typing import Sequence, Union
from dataclasses import dataclass
import numpy as np
from . import *
from ..vectors import norm

__all__ = ['ZemaxConicSagFunction', 'RectangularArraySagFunction', 'SinusoidSagFunction', "RectangularArrayLevel"]

class ZemaxConicSagFunction(SagFunction):
    """
    roc: Radius of curvature.
    kappa: Conic parameter. Special values:
        kappa < 0: Hyperboloid.
        kappa = 0: Paraboloid.
        0 < kappa < 1: Elipsoid of revolution about major axis.
        kappa = 1: Sphere
        kappa > 1: Elipsoid of revolution about minor axis.
    alphas: Second and higher order coefficients.
    """
    def __init__(self, roc: float, radius: float, kappa: float = 1., alphas: Sequence[float] = None):
        assert radius > 0
        if kappa*radius**2 >= roc**2:
            raise ValueError(f'Surface is vertical with radius {radius}, roc {roc} and kappa {kappa}.')
        if alphas is None:
            alphas = []
        ns = np.arange(2, len(alphas) + 2)
        # For now use loose Lipschitz bound equal to sum of bounds.
        lipschitz = radius/(roc**2 - kappa*radius**2)**0.5 + sum(abs(alpha)*n*radius**(n - 1) for n, alpha in zip(ns, alphas))

        self.roc = roc
        self.radius = radius
        self.kappa = kappa
        self.alphas = alphas
        self._lipschitz = lipschitz

    @property
    def lipschitz(self) -> float:
        return self._lipschitz

@dataclass
class RectangularArrayLevel:
    pitch: np.ndarray
    size: Union[np.ndarray, None]
    clamp: bool

    @classmethod
    def make(cls, pitch, size = None, clamp: bool = False):
        try:
            pitch = (float(pitch),)*2
        except TypeError:
            pass
        pitch = np.array(pitch, float)
        assert pitch.shape == (2,)
        assert all(pitch > 0)

        if size is not None:
            try:
                size = (int(size),)*2
            except TypeError:
                pass
            size = np.array(size, int)
            assert size.shape == (2,)
            assert all(size > 0)

        return cls(pitch, size, clamp)

class RectangularArraySagFunction(SagFunction):
    """Infinite rectangular array of sag functions.

    The unit function is evaluated for x in [0, pitch[0]/2) and y in [0, pitch[1]/2) i.e. it is implicitly symmetric
    about the x and y axis. This ensures continuity at the boundaries.

    The number of copies of the unit function along each dimension is given by size.

    If size is None the array is infinite with one unit centered on the origin. Otherwise, the offset is chosen so that
    the center of the entire array is at the origin. The coordinates passed to the unit function are clipped.

    TODO otpions for smoothing at boundaries. Simple possibility is quadratic function in transition zone matching slope.
    In corner transition zones, quadratic (in distance from corner) matching slope at corners of active region.
    """
    def __init__(self, unit: SagFunction, pitch: Sequence[float], size: Sequence[int] = None, clamp: bool = False):
        pitch = np.array(pitch, float)
        assert pitch.shape == (2,)
        assert all(pitch > 0)

        if size is not None:
            size = np.array(size, int)
            assert size.shape == (2,)
            assert all(size > 0)

        self.unit = unit
        self.pitch = pitch
        self.size = size
        self.clamp = clamp

    @property
    def lipschitz(self) -> float:
        return self.unit.lipschitz

    @classmethod
    def make_multi(cls, unit: SagFunction, levels: Sequence[RectangularArrayLevel]):
        """Make multi-level rectangular array.

        Args:
            unit: Function to be repeated.
            levels: From smallest to largest scale.
        """
        sag = unit
        for l in levels:
            sag = RectangularArraySagFunction(sag, l.pitch, l.size, l.clamp)
        return sag

class SinusoidSagFunction(SagFunction):
    """Fixed wave-vector sinusoidal sag function.

    sag(x, y) = amplitude*cos(x*vector[0] + y*vector[1])
    """
    def __init__(self, amplitude: float, vector: Sequence[float]):
        vector = np.array(vector, float)
        assert vector.shape == (2,)
        self.amplitude = amplitude
        self.vector = vector
        self._lipschitz = amplitude*norm(vector)

    @property
    def lipschitz(self) -> float:
        return self._lipschitz
