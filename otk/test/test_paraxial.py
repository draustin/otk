import numpy as np
from numpy import testing
from otk import paraxial
from otk import functions as math

def test_design_spherical_inverse_lens():
    roc, d = paraxial.design_spherical_inverse_lens(1.45, 1e-6, 4e-3, 200e-6)
    sag = math.calc_sphere_sag(roc, 32*10e-6*2**0.5)
    f, h1, h2 = paraxial.calc_thick_spherical_lens(1/1.45, -roc, roc, d)
    assert np.isclose(f, 4e-3)

def test_calc_multi_element_lens_abcd():
    roc1 = 0.5
    roc2 = -0.3
    d = 0.1
    n = 1.5
    f, h1, h2 = paraxial.calc_thick_spherical_lens(n, roc1, roc2, d)

    ds = [f - h1, d, f + h2]
    ns = [1, n, 1]
    rocs = [roc1, roc2]
    m = paraxial.calc_multi_element_lens_abcd(ns, rocs, ds)
    assert np.allclose(m, [[0, f], [-1/f, 0]])

def test_design_singlet_transform():
    ws = 1, 2
    f = 10
    n = 1.5
    rocs, d = paraxial.design_singlet_transform(n, ws, f)

    f_, h1, h2 = paraxial.calc_thick_spherical_lens(n, *rocs, d)

    assert np.isclose(f_, f)
    assert np.isclose(f_ - h1, ws[0])
    assert np.isclose(f_ + h2, ws[1])

    ne = 1.2
    rocs, d = paraxial.design_singlet_transform(n, ws, f, ne)
    f_, h1, h2 = paraxial.calc_thick_spherical_lens(n, *rocs, d, ne)
    assert np.isclose(f_, f)
    assert np.isclose(f_ - h1, ws[0])
    assert np.isclose(f_ + h2, ws[1])

def test_design_singlet():
    n = 1.5
    f = 10
    ss = -1, 0, 1
    d = 0.1
    for s in ss:
        r1, r2 = paraxial.design_singlet(n, f, s, d)
        assert np.isclose(paraxial.calc_thick_spherical_lens(n, r1, r2, d)[0], f)

