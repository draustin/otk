import numpy as np
from otk.sdb import *
from otk.sdb.scalar import *

def test_Sphere():
    s = Sphere(1.0)
    d = getsdb(s, np.asarray((1.0, 1.0, 1.0, 1.0)))
    assert np.isscalar(d)
    assert d == 3**0.5 - 1

    s = Sphere(1.0, (1.0, 1.0, 1.0))
    assert getsdb(s, np.asarray((2.0, 2.0, 2.0))) == 3**0.5 - 1

def test_InfiniteCylinder():
    s = InfiniteCylinder(1.0)
    d = getsdb(s, np.asarray((1.0, 1.0, 1.0, 1.0)))
    assert np.isscalar(d)
    assert d == 2**0.5 - 1

    s = InfiniteCylinder(1.0, (1.0, 1.0))
    assert getsdb(s, np.asarray((2.0, 2.0, 1.0))) == 2**0.5 - 1

def test_UnionOp():
    s = UnionOp((Sphere(1.0), Sphere(1.0, (1.0, 0, 0))))

    assert getsdb(s, np.asarray((3.0, 0.0, 0.0, 1.0))) == 1.0
    assert getsdb(s, np.asarray((-2.0, 0.0, 0.0, 1.0))) == 1.0

def test_IntersectionOp():
    s = IntersectionOp((Sphere(1.0), InfiniteCylinder(0.5)))
    assert getsdb(s, np.asarray((0.0, 0.0, 3.0))) == 2.0
    assert getsdb(s, np.asarray((1, 0.0, 0.0))) == 0.5

def test_DifferenceOp():
    s = DifferenceOp(Sphere(1.0), Sphere(0.5))
    assert getsdb(s, np.asarray((0.0, 0.0, 0.25, 1.0))) == 0.25
    assert getsdb(s, np.asarray((0.0, 0.0, 1.25, 1.0))) == 0.25

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