import otk.geo3
import numpy as np
from otk.rt import interfaces
from otk import v4b, v4


def test_make_perpendicular():
    assert np.array_equal(otk.geo3.make_perpendicular(v4.xhat, v4.yhat), v4.zhat)

    # Single degenerate vector.
    v = otk.geo3.make_perpendicular(v4.xhat, v4.xhat)
    assert np.isclose(v4b.dot(v4.xhat, v), 0)
    assert np.isclose(v4b.dot(v), 1)

    # Single degenerate vector.
    m = np.c_[v4.xhat, v4.yhat].T
    v = otk.geo3.make_perpendicular(m, m)
    assert np.allclose(v4b.dot(m, v), 0)
    assert np.allclose(v4b.dot(v), 1)