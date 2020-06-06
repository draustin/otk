from typing import Dict, Sequence, Tuple
from enum import Enum
import numpy as np
import otk.functions
import scipy.interpolate
from otk.functions import make_perpendicular

from .. import v4hb
from .. import functions
from .. import ri

class Directions(Enum):
    REFLECTED = 0
    TRANSMITTED = 1

class InterfaceMode:
    def __init__(self, direction: int, matrix: np.ndarray, vector: np.ndarray, n: np.ndarray):
        """

        Args:
            direction: Either REFLECTED or TRANSMITTED.
            matrix: Projection matrix.
            vector: Outgoing k vector.
            n: Outgoing refractive index.
        """
        self.direction = Directions(direction)
        self.matrix = np.asarray(matrix)
        assert self.matrix.shape[-2:] == (4, 4)
        self.vector = np.asarray(vector)
        assert self.vector.shape[-1] == 4
        # Row space of matrix should be orthogonal to outgoing k vector.
        assert np.allclose(v4hb.dot(self.matrix, self.vector[..., None, :]), 0, atol=1e-7)
        # This checks that the shapes are consistent.
        self.shape = np.broadcast(self.matrix, self.vector[..., None]).shape
        self.n = n

    def __repr__(self):
        return 'InterfaceMode(matrix=%r, vector=%r, n=%r)'%(self.matrix, self.vector, self.n)


class Interface:
    def calc_modes(self, point: np.ndarray, normal: np.ndarray, lamb: float, vector: np.ndarray, n: np.ndarray) -> Dict[
        str, InterfaceMode]:
        """

        Args:
            point: ...x4 array in surface local coordinates.
            normal: ...x4 array in surface local coordinates.
            lamb: wavelength
            vector: ...x4 array of normalized incident k vectors in local coordinates.

        """
        raise NotImplementedError()


def calc_outer_product(vector1, vector2, amplitude):
    """Output[...,i,j] = vector1[...,i]*amplitude*vector2[...,j]."""
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    amplitude = np.atleast_1d(amplitude)
    assert amplitude.shape[-1] == 1
    return vector1[..., :, None]*amplitude[..., None]*vector2[..., None, :]


def calc_matrix(incident_vectors, deflected_vectors, amplitudes):
    return sum(calc_outer_product(incident_vector, deflected_vector, amplitude) for incident_vector, deflected_vector, amplitude in
               zip(incident_vectors, deflected_vectors, amplitudes))


class Mirror(Interface):
    def calc_modes(self, point: np.ndarray, normal: np.ndarray, lamb: float, incident_vector: np.ndarray,
            n: np.ndarray) -> Dict:
        reflected_vector = otk.functions.reflect_vector(incident_vector, normal)
        s_pol_vector = v4hb.cross(normal, incident_vector)
        incident_p_pol_vector = v4hb.cross(incident_vector, s_pol_vector)
        reflected_p_pol_vector = v4hb.cross(reflected_vector, s_pol_vector)

        matrix = calc_matrix((incident_p_pol_vector, s_pol_vector), (reflected_p_pol_vector, s_pol_vector),
            np.asarray((1, 1)))
        mode = InterfaceMode(Directions.REFLECTED, matrix, reflected_vector, n)
        modes = dict(reflected=mode)
        return modes


class IsotropicMediaInterface(Interface):
    def __init__(self, n1, n2, reflects: bool = True, transmits: bool = True):
        self.n1 = n1
        self.n2 = n2
        self.reflects = reflects
        self.transmits = transmits

    def calc_amplitudes(self, n1, n2, cos_theta1, lamb) -> Tuple[Tuple]:
        """Returns amplitudes ((rp, rs), (tp, ts))."""
        raise NotImplementedError()

    def calc_modes(self, point: np.ndarray, normal: np.ndarray, lamb: float, incident_vector: np.ndarray,
            n: np.ndarray) -> Dict:
        """

        Args:
            point:
            normal:
            lamb:
            incident_vector:

        Returns:
            Mapping of (Outgoing, Polarization) pairs to InterfaceMode objects.
        """
        n1 = self.n1(lamb)
        n2 = self.n2(lamb)
        cos_theta1 = v4hb.dot(normal, incident_vector)

        if 0:
            na = np.choose(cos_theta1 < 0, (n1, n2))
            nb = np.choose(cos_theta1 < 0, (n2, n1))
        else:
            assert np.all(cos_theta1>=0) or np.all(cos_theta1<=0)
            cos_theta1 = cos_theta1.ravel()[0]
            if cos_theta1>0:
                na, nb = n1, n2
            else:
                na, nb = n2, n1

        cos_theta1 = abs(cos_theta1)

        refracted_vector = otk.functions.refract_vector(incident_vector, normal, nb/na)*na/nb
        reflected_vector = otk.functions.reflect_vector(incident_vector, normal)

        # Generate unit vector perpendicular to normal and incident.
        s_pol_vector = make_perpendicular(normal, incident_vector)

        incident_p_pol_vector = v4hb.cross(incident_vector, s_pol_vector)
        refracted_p_pol_vector = v4hb.cross(refracted_vector, s_pol_vector)
        reflected_p_pol_vector = v4hb.cross(reflected_vector, s_pol_vector)

        amplitudes = self.calc_amplitudes(na, nb, cos_theta1, lamb)

        modes = {}
        if self.reflects:
            matrix = calc_matrix((incident_p_pol_vector, s_pol_vector), (reflected_p_pol_vector, s_pol_vector),
                amplitudes[0])
            modes['reflected'] = InterfaceMode(Directions.REFLECTED, matrix, reflected_vector, na)

        if self.transmits:
            matrix = calc_matrix((incident_p_pol_vector, s_pol_vector), (refracted_p_pol_vector, s_pol_vector),
                amplitudes[1])
            modes['transmitted'] = InterfaceMode(Directions.TRANSMITTED, matrix, refracted_vector, nb)

        return modes

class PerfectRefractor(IsotropicMediaInterface):
    def __init__(self, n1, n2):
        IsotropicMediaInterface.__init__(self, n1, n2, False, True)

    def calc_amplitudes(self, n1, n2, cos_theta1, lamb):
        return ((0, 0), (1, 1))

class FresnelInterface(IsotropicMediaInterface):
    def calc_amplitudes(self, n1, nb, cos_theta1, lamb):
        return functions.calc_fresnel_coefficients(n1, nb, cos_theta1)

    def __repr__(self):
        return 'FresnelInterface(n1=%r, n2=%r)'%(self.n1, self.n2)

    def flip(self):
        return FresnelInterface(self.n2, self.n1)


class SampledCoating(IsotropicMediaInterface):
    """Symmetric - amplitudes are the same from both sides."""

    def __init__(self, n1: ri.Index, n2: ri.Index, lambs: Sequence, thetas: Sequence, amplitudes: np.ndarray):
        """

        Args:
            lambs: Sampled wavelengths.
            thetas: Sampled angles.
            amplitudes: Array with dimensions (Outgoing, Polarization, wavelength, angle).
        """
        IsotropicMediaInterface.__init__(self, n1, n2)
        self.lambs = np.asarray(lambs)
        assert self.lambs.ndim == 1
        self.thetas = np.asarray(thetas)
        assert self.thetas.ndim == 1
        self.amplitudes = amplitudes
        assert self.amplitudes.shape == (2, 2, len(self.lambs), len(self.thetas))

    def __repr__(self):
        return 'SampledCoating(n1=%r, n2=%r, lambs=%r, thetas=%r, amplitudes=%r)'%(
            self.n1, self.n2, self.lambs, self.thetas, self.amplitudes)

    def calc_amplitudes(self, n1, n2, cos_theta1, lamb):
        results = []
        theta1 = np.arccos(cos_theta1)
        # Loop over reflected, transmitted.
        for amplitudes in self.amplitudes:
            results.append([])
            # Loop over p, s.
            for amplitude in zip(amplitudes):
                # TODO switch to complex interpolation.
                amplitude_lamb = scipy.interpolate.interp1d(self.lambs, amplitude, axis=0, copy=False)(lamb)
                amplitude_lamb_theta = scipy.interpolate.interp1d(self.thetas, amplitude_lamb, axis=0, copy=False)(
                    theta1)
                results[-1].append(amplitude_lamb_theta)
        return results
