import numpy as np
from otk.sdb import *

def test_transforms():
    m = orthographic(-2, 3, -4, 5, 6, 7)
    assert np.allclose(np.dot([-2,-4,-6,1], m), [-1.0, -1.0, -1.0, 1.0])
    assert np.allclose(np.dot([3,5,-7,1], m), [1.0, 1.0, 1.0, 1.0])

    assert np.allclose(lookat([1.0, 3.0, -1.0], [1.0, -4.0, -1.0], [0.0, 0.0, 2.0]),
        np.asarray((
            (-1.0,   0.0,  0.0,   1.0),
            (0.0,  -0.0,  1.0,  3.0),
            (0.0,   1.0,  0.0,  -1.0),
            (0.0,   0.0,  0.0,   1.0))).T)

def test_misc():
    assert all(pix2norm(np.asarray((0,3)), 4) == (-0.75, 0.75))

    invP = np.linalg.inv(orthographic(-2.0, 3.0, -4.0, 5.0, 6.0, 7.0))
    x0, v, d_max = ndc2ray(-1, -1, invP)
    assert np.allclose(x0, (-2.0, -4.0, -6.0, 1.0))
    assert np.allclose(v, (0.0, 0.0, -1.0, 0.0))
    assert np.isclose(d_max, 1.0)