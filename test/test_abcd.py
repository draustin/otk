import numpy as np
from otk import abcd

def test_solve_qs():
    q1s = 1/(1-2j), 1/(-1-2j)
    q2s = 1/(3-0.4j), 1/(3-0.4j)
    for q1, q2 in zip(q1s, q2s):
        Ms = abcd.solve_qs(q1, q2)
        assert all(np.isclose(np.linalg.det(M), 1) for M in Ms)
        assert all(np.isclose(abcd.transform_Gaussian(M, q1), q2) for M in Ms)

def test_image_qs():
    q1 = 1/(1-2j)
    q2 = 1/(3-0.4j)
    f = 0.5
    us, vs = abcd.image_qs(q1, q2, f)
    for u, v in zip(us, vs):
        q2p = abcd.transform_Gaussian(abcd.propagation(v).dot(abcd.thin_lens(f).dot(abcd.propagation(u))), q1)
        assert np.isclose(q2, q2p)
