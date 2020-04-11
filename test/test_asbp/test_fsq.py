import numpy as np
import mathx
from otk import asbp, rt1

def test_calc_gradxyE_spherical():
    rs_support = 200e-6
    num_points = 64
    qs_center = (100e3, -50e3)
    rs_center = (1e-3, 0.5e-3)
    k = 2*np.pi/860e-9
    x, y = asbp.calc_xy(rs_support, num_points, rs_center)
    Er = asbp.calc_gaussian(k, x, y, 20e-6, 0.5e-3, rs_center, qs_center)
    gradxyE_flat = asbp.calc_gradxyE(rs_support, Er, qs_center)
    gradxyE_spherical = asbp.calc_gradxyE_spherical(k, rs_support, Er, 1e-3, rs_center, qs_center)
    assert all(mathx.allclose(g1, g2, atol_frac=1e-6) for g1, g2 in zip(gradxyE_flat, gradxyE_spherical))
