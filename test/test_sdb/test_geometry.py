import numpy as np
from otk.sdb import *
from otk.vector3 import *

def test_Plane():
    n = 1, 2, 3
    c = 2
    s = Plane(n, c)
    assert np.array_equal(s.n, normalize(n))