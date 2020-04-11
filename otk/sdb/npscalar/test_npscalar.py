import numpy as np
from otk.sdb import *
from otk.sdb.npscalar import *

def is_traverse_equal(t1, t2):
    return all(s1 is s2 and np.isclose(d1, d2) for (s1, d1), (s2, d2) in zip(t1, t2))

def test_Sphere():
    s = Sphere(1.0)
    d = getsdb(s, np.asarray((1.0, 1.0, 1.0, 1.0)))
    assert np.isscalar(d)
    assert d == 3**0.5 - 1

    s = Sphere(1.0, (1.0, 1.0, 1.0))
    assert getsdb(s, np.asarray((2.0, 2.0, 2.0))) == 3**0.5 - 1

    assert list(traverse(s, (1, 1, 3))) == [(s, 1.)]
    assert list(traverse(s, (1, 1, 1))) == [(s, -1.)]

def test_InfiniteCylinder():
    s = InfiniteCylinder(1.0)
    d = getsdb(s, np.asarray((1.0, 1.0, 1.0, 1.0)))
    assert np.isscalar(d)
    assert d == 2**0.5 - 1

    s = InfiniteCylinder(1.0, (1.0, 1.0))
    assert getsdb(s, np.asarray((2.0, 2.0, 1.0))) == 2**0.5 - 1

def test_UnionOp():
    s0 = Sphere(1.0)
    s1 = Sphere(1.0, (0.5, 0, 0))
    s = UnionOp((s0, s1))

    assert getsdb(s, np.asarray((2.5, 0.0, 0.0, 1.0))) == 1.0
    assert getsdb(s, np.asarray((-2.0, 0.0, 0.0, 1.0))) == 1.0
    assert is_traverse_equal(list(traverse(s, (-0.6, 0, 0))), [(s0, -0.4), (s1, 0.1), (s, -0.4)])

def test_IntersectionOp():
    s0 = Sphere(1.0)
    s1 = InfiniteCylinder(0.5)
    s = IntersectionOp((s0, s1))
    assert getsdb(s, np.asarray((0.0, 0.0, 3.0))) == 2.0
    assert getsdb(s, np.asarray((1, 0.0, 0.0))) == 0.5

    assert is_traverse_equal(list(traverse(s, (0, 0, 0))), [(s0, -1.), (s1, -0.5), (s, -0.5)])


def test_DifferenceOp():
    s = DifferenceOp(Sphere(1.0), Sphere(0.5))
    assert getsdb(s, np.asarray((0.0, 0.0, 0.25, 1.0))) == 0.25
    assert getsdb(s, np.asarray((0.0, 0.0, 1.25, 1.0))) == 0.25

#def test

def test_SphereTrace():
    surface = Sphere(1.0)
    epsilon = 1e-4
    max_steps = 1000
    # Pointing right at sphere - takes one step.
    trace = spheretrace(surface, [0.0, 0.0, -2.0, 1.0], [0.0, 0.0, 1.0, 0.0], 100, epsilon, max_steps)
    assert trace.t == 1.0
    assert trace.d == 0
    assert trace.steps == 1
    assert np.array_equal(trace.x, [0, 0, -1, 1])

    trace = spheretrace(surface, [0.5, 0.0, -1.0, 1.0], [0.0, 0.0, 1.0, 0.0], 100, epsilon, max_steps)
    assert abs(trace.t - (1.0 - 3**0.5/2)) < epsilon