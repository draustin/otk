import numpy as np
from numpy import testing
import mathx
from otk import geo3

def test_refract_vector():
    assert np.allclose(geo3.refract_vector((0, 0, 1), (0, 0, 1), 2), (0, 0, 2))
    assert np.allclose(geo3.refract_vector((0, 0, -1), (0, 0, 1), 2), (0, 0, -2))
    assert np.allclose(geo3.refract_vector((0, 0, -1), (0, 0, -1), 2), (0, 0, -2))
    assert np.allclose(geo3.refract_vector((0, 2**-0.5, 2**-0.5), (0, 0, 1), 1), (0, 2**-0.5, 2**-0.5))
    vr = geo3.refract_vector((0, 2**-0.5, 2**-0.5), (0, 0, 1), 2)
    assert np.allclose(vr[1], 2**-0.5)
    assert np.isclose(np.dot(vr, vr), 4)

def test_intersect_planes():
    testing.assert_allclose(geo3.intersect_planes([1, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1]), [1, 1, 0, 1])

def test_reflect_vector():
    assert np.allclose(geo3.reflect_vector(np.array((0, 0, 1, 0)), np.array((0, 0, 1, 0))), (0, 0, -1, 0))
    assert np.allclose(geo3.reflect_vector(np.array((0, 0, 1, 0)), np.array((0, 2**-0.5, 2**-0.5, 0))), (0, -1, 0, 0))

