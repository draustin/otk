import numpy as np
import mathx
from otk.asbp import source

def test_calc_bessel():
    x = np.linspace(-5, 5, 512)
    y = np.linspace(-5.5, 5.5, 513)[:, None]
    radius = 2
    E, (gradxE, gradyE) = source.calc_bessel(x, y, radius, True)

    # Test normalization.
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    power = (abs(E)**2).sum()*dx*dy
    assert np.isclose(power, 1)

    # Test gradients. There is a discontinuity at the edge, so multiply by E.
    gradxE_num = mathx.usv.differentiate(x, E)
    assert np.allclose(gradxE*E, gradxE_num*E, atol=1e-3)
    gradyE_num = mathx.usv.differentiate(y, E)
    assert np.allclose(gradyE*E, gradyE_num*E, atol=1e-3)