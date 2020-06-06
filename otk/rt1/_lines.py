import numpy as np
from ..v4hb import *

class Line:
    """A bundle of rays.

    A rt is invalid if any of its components are NaN.
    """

    def __init__(self, origin, vector):
        origin = np.asarray(origin)
        vector = np.asarray(vector)
        assert origin.shape[-1] == 4
        invalid = np.isnan(origin).any(axis=-1, keepdims=True) | np.isnan(vector).any(axis=-1, keepdims=True)
        assert (np.isclose(origin[..., [3]], 1) | invalid).all()
        assert (np.isclose(vector[..., [3]], 0) | invalid).all()
        assert (np.isclose(dot(vector), 1) | invalid).all()
        origin, vector = np.broadcast_arrays(origin, vector)
        self.origin = origin
        self.vector = vector
        self.shape = self.origin.shape[:-1]

    def __eq__(self, other):
        return isinstance(other, Line) and np.array_equal(self.origin, other.origin) and np.array_equal(self.vector,
            other.vector)

    def __repr__(self):
        return 'Line(%r, %r)'%(self.origin, self.vector)

    def __str__(self):
        origin = self.origin.reshape((-1, 4))[0, :3]
        vector = self.vector.reshape((-1, 4))[0, :3]
        string = 'origin[0]=(%.3f, %.3f, %.3f) mm, vector[0]=(%.3f, %.3f, %.3f), shape=%s'
        return string%(*(origin*1e3), *vector, self.shape)

    def allclose(self, other):
        return np.allclose(self.origin, other.origin) and np.allclose(self.vector, other.vector)

    def project_point(self, point):
        """
        We solve dot((point - (origin + d*vector)), vector) = 0

        Args:
            point:

        Returns:

        """
        delta = point - self.origin
        d = dot(delta, self.vector)
        r = delta - d*self.vector
        # assert np.allclose(dot(r, self.vector), 0) # Just for early debugging.
        return d, r

    def project_line(self, line):
        """Project a line onto self.

        Expresses line as r = h + u*d where d is distance along self from self.origin.

        Args:
            line (Line): Line to project.

        Returns:
            h (...x4 array): The displacement vector, perpendicular to self.vector, from self.origin to a point on the line.
            u (...x4 array): The vector, perpendicular to self.vector, that when multiplied by distance along self describes
                line.
        """
        vector_dot = dot(line.vector, self.vector)
        # Intersect ray with plane perpendicular to self.vector containing self.origin. We solve
        # dot((bundle.origin + d*bundle.vector) - self.origin, self.vector) = 0.
        d = dot(self.origin - line.origin, self.vector)/vector_dot
        h = line.origin + line.vector*d - self.origin
        u = line.vector/vector_dot - self.vector
        return h, u

    def __getitem__(self, item):
        return Line(self.origin[item, :], self.vector[item, :])

    def reshape(self, shape):
        return Line(self.origin.reshape(shape + (4,)), self.vector.reshape(shape + (4,)))

    def transform(self, matrix):
        return Line(transform(self.origin, matrix), transform(self.vector, matrix))

    def advance(self, d):
        return Line(self.origin + self.vector*d, self.vector)

    def flip(self):
        """Invert line through its origin."""
        return Line(self.origin, -self.vector)


class ComplexLine:
    def __init__(self, real: Line, imag: Line):
        assert real.shape == imag.shape
        self.real = real
        self.imag = imag
        self.shape = real.shape

    def __getitem__(self, item):
        return ComplexLine(self.real[item], self.imag[item])

    def __str__(self):
        return 'complex line real %s, imag %s'%(self.real, self.imag)

    def __repr__(self):
        return 'ComplexLine(real=%r, imag=%r)'%(self.real, self.imag)

    def reshape(self, shape):
        return ComplexLine(self.real.reshape(shape), self.imag.reshape(shape))

    def transform(self, matrix):
        return ComplexLine(self.real.transform(matrix), self.imag.transform(matrix))

    def advance(self, d):
        return ComplexLine(self.real.advance(d), self.imag.advance(d))


