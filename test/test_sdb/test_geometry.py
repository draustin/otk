import numpy as np
from otk.h4t import make_translation
from otk.sdb import *
from otk.v4b import *

def test_Plane():
    n = 1, 2, 3
    c = 2
    s = Plane(n, c)
    assert np.array_equal(s.n, normalize(n))

def test_AffineOp():
    m = make_translation(1, 2, 3)
    s0 = Sphere(1.0)
    s1 = AffineOp(s0, m)
    assert np.allclose(s1.get_parent_to_child((0, 0, 0, 1)), np.linalg.inv(m))

def test_Surface():
    s0 = Sphere(1.0)
    s1 = Sphere(1.0)
    s2 = UnionOp((s0, s1))
    s3 = Sphere(1.0)
    s4 = IntersectionOp((s2, s3))
    assert s0.get_ancestors() == [s0, s2, s4]
