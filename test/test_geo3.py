import otk.geo3
import numpy as np
from otk.rt import interfaces
from otk import vector3


def test_make_perpendicular():
    assert np.array_equal(otk.geo3.make_perpendicular(vector3.xhat, vector3.yhat), vector3.zhat)

    # Single degenerate vector.
    v = otk.geo3.make_perpendicular(vector3.xhat, vector3.xhat)
    assert np.isclose(vector3.dot(vector3.xhat, v), 0)
    assert np.isclose(vector3.dot(v), 1)

    # Single degenerate vector.
    m = np.c_[vector3.xhat, vector3.yhat].T
    v = otk.geo3.make_perpendicular(m, m)
    assert np.allclose(vector3.dot(m, v), 0)
    assert np.allclose(vector3.dot(v), 1)