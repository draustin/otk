import os
import numpy as np
from otk import zemax

def test_read_train_conic_aspheric_telecentric_lens():
    train_full = zemax.read_train(os.path.join(os.path.dirname(__file__), 'conic_aspheric_telecentric_lens.zmx'), encoding='ascii')
    assert len(train_full.interfaces) == 9
    train=train_full.subset(2, -1)
    assert np.isclose(train.get_effective_focal_length(550e-9)*1e3, 199.1243, atol=0.5)

    i0 = train.interfaces[0]
    assert np.isclose(i0.kappa, 1 + 0.285773)
    assert np.allclose(i0.alphas[1:5], [0, -2.866109e2, 0, -8.075138e4], rtol=1e-3)

    i2 = train.interfaces[2]
    assert np.allclose(i2.segments[0].kappa, 1 + -0.922024)
