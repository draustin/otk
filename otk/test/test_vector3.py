# TODO rename
import otk.h4t
from numpy import testing
import numpy as np
from otk import v4hb

def test_make_frame():
    assert np.allclose(otk.h4t.make_frame((0, 0, 0.1), (0, 2, 0)), np.eye(4))
    assert np.allclose(otk.h4t.make_frame((0.1, 0, 0), (0, 0, 0.2)),
        [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

def test_make_rotation():
    for vector in ([1, 0, 0], [0, 1, 0], [0, 0, 1], v4hb.normalize([1, -1, 2])):
        testing.assert_allclose(otk.h4t.make_rotation(vector, 0), np.eye(4))

    for theta in (np.pi/4, 5*np.pi/4):
        testing.assert_allclose(otk.h4t.make_rotation([0, 1, 0], theta), otk.h4t.make_y_rotation(theta))
        testing.assert_allclose(otk.h4t.make_rotation([0, 0, 1], theta), otk.h4t.make_z_rotation(theta))

def test_dot():
    assert np.isreal(v4hb.dot([1, 2, 3], [1, 2, 3]))
    testing.assert_array_equal(v4hb.dot([1, 2, 3], [1, 2, 3]), 14)
    testing.assert_array_equal(v4hb.dot([1+1j, 2, 3], [1-1j, 2, 3]), 13+2j)
    testing.assert_array_equal(v4hb.dot([1, 2, 3]), 14)

    a = np.arange(3) + np.arange(4)[:, None]
    b = np.arange(3) + np.arange(4)[:, None] + np.arange(5)[:, None, None]*1j
    testing.assert_array_equal(v4hb.dot(a, b), (a*b.conj()).sum(-1, keepdims=True))

def test_cross():
    # Single complex cross product.
    a = [1, 2 - 1j, -1 + 1j, 0]
    b = [1, 1 - 1j, 2 + 2j, 0]
    cross = v4hb.cross(a, b)
    testing.assert_array_equal(cross[:3], np.cross(a[:3], b[:3]).conj())
    assert cross[3] == 0

    # Single real cross product.
    a = [1, 2, -1, 0]
    b = [1, 1, 2, 0]
    cross = v4hb.cross(a, b)
    testing.assert_array_equal(cross[:3], np.cross(a[:3], b[:3]))
    assert np.isrealobj(cross)

def test_triple():
    # Single real.
    a = [2, 3, 4]
    b = [1, 2, -1]
    c = [1, 1, 2]
    triple = v4hb.triple(a, b, c)
    assert triple == np.dot(a, np.cross(b, c))
    assert np.isreal(triple)

    # Single complex.
    a = [2 + 1j, 3 + 3j, 4 + 4j]
    b = [1 + 1j, 2 - 2j, -1 + 1j]
    c = [1 - 1j, 1 + 1j, 2 - 2j]
    triple = v4hb.triple(a, b, c)
    # Our cross product is conjugated, then conjugated again in the dot product.
    assert triple == np.dot(a, np.cross(b, c))

