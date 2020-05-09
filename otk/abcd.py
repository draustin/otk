"""Tools for dealing with 1D ray transfer ("ABCD") matrices.

Rays are column vectors (x, theta).

TODO consistent capitalization.
"""
from typing import Tuple
import dataclasses
from dataclasses import dataclass
import numpy as np

# def propagate_Gaussian(d, q1, E1):
#     q2 = apply_to_Gaussian(propagation(d), q)
#     return q2, q1/q2*E1

def propagation(d):
    return np.asarray([[1, d], [0, 1]])


def thin_lens(f):
    return np.asarray([[1, 0], [-1/f, 1]])


def curved_interface(n1, n2, R):
    """
    Args:
        n1: incident refractive index
        n2: final refractive index
        R: radius of curvature, >0 for convex
    """
    return np.asarray([[1, 0], [(n1 - n2)/(R*n2), n1/n2]])

def interface(n1, n2):
    """
    Args:
        n1: incident refractive index
        n2: final refractive index
        R: radius of curvature, >0 for convex
    """
    return np.asarray([[1, 0], [0, n1/n2]])


def thick_lens(n2, r1, r2, t, n1=1):
    return curved_interface(n2, n1, -r2) @ propagation(t) @ curved_interface(n1, n2, r1)


def transform_Gaussian(m, q):
    return (m[0][0]*q + m[0][1])/(m[1][0]*q + m[1][1])


def Gaussian_wR_to_q(w, R, lamb, m=1):
    return 1/(1/R - 1j*lamb/(np.pi*(w/m)**2))

def fourier_transform(f):
    return np.asarray([[0, f], [-1/f, 0]])

def fractional_fourier_transform(f, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, f*s], [-s/f, c]])

def Gaussian_q_to_wR(q, lamb, m=1):
    iq = 1/q
    w = (-lamb/(np.pi*iq.imag))**0.5*m
    try:
        R = 1/iq.real
    except ZeroDivisionError:
        R = float('inf')
    return w, R

def solve_qs(q1, q2, unity='D'):
    """Find positive unimodular matrix mapping Gaussian q1 to q2 with given element equal to unity."""
    if unity == 'D':
        # See Dane's logbook 3 p 32.
        Bs = np.roots([-q1.conjugate().imag, (q1.conjugate()*(q2.conjugate() - (q1 - q2))).imag, ((q1-q2)*(q1*q2).conjugate()).imag])
        print(Bs)
        Cs = (q1 - q2 + Bs)/(q1*q2 - Bs*q1)
        assert np.allclose(Cs.imag, 0)
        Cs = Cs.real
        Ms = tuple((np.asarray([[1 + B*C, B], [C, 1]]) for B, C in zip(Bs, Cs)))
        return Ms
    else:
        raise ValueError('Not implemented')

def image_qs(q1:complex, q2:complex, f:float) -> Tuple:
    """Compute object and image distances to map one Gaussian beam to another with given focal length lens.

    Args:
        q1: Input beam.
        q2: Output beam.
        f: Focal length.

    Returns:
        us: Distances from q1 plane to lens.
        vs: Distances from lens to q2 plane.

    There are two solutions, returned sorted by increasing us.
    """
    # See Dane's logbook 3 p 34.
    A = q2*f - q1*q2 - f*q1
    B = q2+f
    C = f-q1

    coeffs = np.asarray((B, -(A+B*C.conjugate()), A*C.conjugate())).imag
    us = np.sort(np.roots(coeffs))
    assert np.isrealobj(us)
    vs = (A - B*us)/(C - us)
    assert np.allclose(vs.imag, 0)
    vs = vs.real

    return us, vs

def reverse(m):
    """Derivation p111 Dane's logbook 3."""
    #assert np.isclose(np.linalg.det(m), 1)
    return np.linalg.inv(m)*[[1, -1], [-1, 1]]
    #return np.asarray(((m[1, 1], m[0, 1]), (m[1, 0], m[0, 0])))

@dataclass
class FundamentalGaussian:
    x: float
    theta: float
    w: float
    roc: float
    lamb: float
    m: float = 1

    @property
    def q(self):
        return Gaussian_wR_to_q(self.w, self.roc, self.lamb, self.m)

    @property
    def ray(self):
        return np.array((self.x, self.theta))

    def transform(self, matrix: np.ndarray) -> 'FundamentalGaussian':
        assert matrix.shape == (2, 2)
        ray = matrix@self.ray
        q = transform_Gaussian(matrix, self.q)
        w, roc = Gaussian_q_to_wR(q, self.lamb, self.m)
        return FundamentalGaussian(ray[0], ray[1], w, roc, self.lamb, self.m)

    def translate(self, dx) -> 'FundamentalGaussian':
        return dataclasses.replace(self, x=(self.x + dx))