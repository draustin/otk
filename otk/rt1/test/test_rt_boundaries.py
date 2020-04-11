import numpy as np
from otk.rt1 import boundaries

def test_SquareBoundary():
    sb = boundaries.SquareBoundary(0.2)
    str(sb)
    assert np.array_equal(sb.is_inside([0, 0.1, 0.1, 0.11, 0.11], [0, 0, 0.1, 0.1, 0.11]), [True, True, True, False, False])