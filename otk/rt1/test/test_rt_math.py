import numpy as np
import otk.functions
from numpy import testing
import mathx

def test_refract_vector():
    assert np.allclose(otk.functions.refract_vector((0, 0, 1), (0, 0, 1), 2), (0, 0, 2))
    assert np.allclose(otk.functions.refract_vector((0, 0, -1), (0, 0, 1), 2), (0, 0, -2))
    assert np.allclose(otk.functions.refract_vector((0, 0, -1), (0, 0, -1), 2), (0, 0, -2))
    assert np.allclose(otk.functions.refract_vector((0, 2**-0.5, 2**-0.5), (0, 0, 1), 1), (0, 2**-0.5, 2**-0.5))
    vr = otk.functions.refract_vector((0, 2**-0.5, 2**-0.5), (0, 0, 1), 2)
    assert np.allclose(vr[1], 2**-0.5)
    assert np.isclose(np.dot(vr, vr), 4)

# TODO move to test_functions.
def test_intersect_planes():
    testing.assert_allclose(otk.functions.intersect_planes([1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]), [1, 1, 0])

def test_reflect_vector():
    assert np.allclose(otk.functions.reflect_vector(np.array((0, 0, 1, 0)), np.array((0, 0, 1, 0))), (0, 0, -1, 0))
    assert np.allclose(otk.functions.reflect_vector(np.array((0, 0, 1, 0)), np.array((0, 2**-0.5, 2**-0.5, 0))), (0, -1, 0, 0))

