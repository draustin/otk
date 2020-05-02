"""Test the ...rt.profiles module"""
import numpy as np
from numpy import testing
from mathx import RegularFiniteSquareLattice
from otk.rt1 import profiles
from otk import v4hb


def test_SphericalProfile():
    profile = profiles.SphericalProfile(10)
    assert np.allclose(profile.calc_normal(np.array((0, 0, 0, 1))), (0, 0, 1, 0))
    assert np.allclose(profile.calc_normal(np.array((10, 0, 10, 1))), (-1, 0, 0, 0))
    assert np.allclose(profile.calc_normal(np.array((0, -10, 10, 1))), (0, 1, 0, 0))

    profile = profiles.SphericalProfile(-10)
    assert np.allclose(profile.calc_normal(np.array((0, 0, 0, 1))), (0, 0, 1, 0))
    assert np.allclose(profile.calc_normal(np.array((10, 0, -10, 1))), (1, 0, 0, 0))
    assert np.allclose(profile.calc_normal(np.array((0, -10, -10, 1))), (0, -1, 0, 0))

    assert np.isclose(profile.calc_z(5, 6), profiles.Profile.calc_z(profile, 5, 6))


def test_SphericalSquareArrayProfile():
    profile = profiles.SphericalSquareArrayProfile(10, 20)
    assert np.allclose(profile.calc_normal(np.array((10, 30, 0, 1))), (0, 0, 1, 0))
    assert np.allclose(profile.calc_normal(np.array((20, 30, 10, 1))), (-1, 0, 0, 0))
    assert np.allclose(profile.calc_normal(np.array((10, 20.000001, 10, 1))), (0, 1, 0, 0))

    testing.assert_allclose(profile.intersect(np.asarray((30, 30, -10, 1)), np.asarray((0, 0, 1, 0))), 10)



def test_LatticeProfile():
    lattice = RegularFiniteSquareLattice(20, 4)
    spherical_profile = profiles.SphericalProfile(10)
    profile = profiles.LatticeProfile(lattice, spherical_profile)

    x = np.arange(4)[:, None]
    y = np.arange(5)
    profile.calc_z(x, y) # Method has some internal checks.

    assert np.allclose(profile.calc_normal(np.array((10, 30, 0, 1))), (0, 0, 1, 0))
    assert np.allclose(profile.calc_normal(np.array((20, 30, 10, 1))), (-1, 0, 0, 0))
    assert np.allclose(profile.calc_normal(np.array((10, 20.000001, 10, 1))), (0, 1, 0, 0))

    testing.assert_allclose(profile.intersect(np.asarray((30, 30, -10, 1)), np.asarray((0, 0, 1, 0))), 10)

def test_ConicProfile():
    p = profiles.ConicProfile(10, 0.5, [0.05])
    origin = v4hb.stack_xyzw([1, -1.1, 0, 0], [0, 0, -1.2, 1.3], [-1, -1.1, -1.2, -1.3], 1)
    vector = v4hb.normalize(v4hb.stack_xyzw([0.1, 0.2, -0.1, 0.1], [-0.1, -0.1, 0.2, 0.2], [1, 1, 1, 1], 0))
    pd = p.intersect(origin, vector)
    point = origin + pd*vector
    rho = v4hb.dot(point[..., :2])**0.5
    assert np.allclose(p.calc_z(rho), point[..., [2]])

    p.calc_normal(point)




