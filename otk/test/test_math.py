import numpy as np
from otk import functions as math

def test_calc_sphere_sag_normal_xy():
    assert math.calc_sphere_sag_normal_xy(10, 0, 0)==(0, (0, 0, 1))
    assert math.calc_sphere_sag_normal_xy(10, 10, 0)==(10.0, (-1.0, 0.0, 0.0))
    assert math.calc_sphere_sag_normal_xy(10, -10, 0)==(10.0, (1.0, 0.0, 0.0))
    assert math.calc_sphere_sag_normal_xy(10, 0, 10)==(10.0, (0, -1, 0))
    ##
    assert np.allclose((np.asarray(math.calc_sphere_sag_normal_xy(2, *np.random.rand(2, 10))[1])**2).sum(axis=0), 1)

def test_calc_sphere_sag():
    assert math.calc_sphere_sag(10, 0) == 0
    assert math.calc_sphere_sag(10, 0, True) == 0
    assert math.calc_sphere_sag(10, 10, True) == np.inf
    assert np.isclose(math.calc_sphere_sag(10, 10/2**0.5, True), 1)

def test_unwrap_quadratic_phase():
    def eval_quad(a, b, c, z):
        return a*z**2 + b*z + c

    a = 0.5 - 0.5j
    b = -2
    c = 1 + 1j
    # z_end=4 means start in top-right quadrant and finish in bottom right, via other two.
    z_end = np.array([4.0, 1.0, 2])
    v_end = eval_quad(a, b, c, z_end)
    phi_end = math.unwrap_quadratic_phase(a, b, c, z_end)
    assert np.allclose(phi_end, np.angle(v_end) + np.array([2, 0, 2])*np.pi)