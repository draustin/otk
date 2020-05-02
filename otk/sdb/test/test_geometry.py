import numpy as np
import pytest
from otk.h4t import make_translation
from otk import sdb
from otk.v4hb import *

# def assert_isaabb(b):
#     assert len(b) == 2
#     for c in b:
#         c = np.asarray(c, float)
#         assert c.shape == (4,)
#         assert np.isclose(c[3], 1)
#
# def assert_aabb_close(b1, b2):
#     assert_isaabb(b1)
#     assert_isaabb(b2)
#     return all(np.allclose(c1, c2) for c1, c2 in zip(b1, b2))

def test_Sphere():
    s = sdb.Sphere(0.5, (0.5, 1, 1.5))
    assert sdb.isclose(s.get_aabb(make_translation(1, 2, 3)), sdb.AABB.make((1, 2.5, 4), (2, 3.5, 5)))

def test_Plane():
    n = 1, 2, 3
    c = 2
    s = sdb.Plane(n, c)
    assert np.array_equal(s.n, normalize(n))
    with pytest.raises(NotImplementedError):
        s.get_aabb(np.eye(4))


def test_AffineOp():
    m = make_translation(1, 2, 3)
    s0 = sdb.Sphere(1.0)
    s1 = sdb.AffineOp(s0, m)
    assert np.allclose(s1.get_parent_to_child((0, 0, 0, 1)), np.linalg.inv(m))
    assert sdb.isclose(s1.get_aabb(np.eye(4)), sdb.AABB.make((0, 1, 2), (2, 3, 4)))

def test_Surface():
    s0 = sdb.Sphere(1.0)
    s1 = sdb.Sphere(1.0)
    s2 = sdb.UnionOp((s0, s1))
    s3 = sdb.Sphere(1.0)
    s4 = sdb.IntersectionOp((s2, s3))
    assert s0.get_ancestors() == [s0, s2, s4]
