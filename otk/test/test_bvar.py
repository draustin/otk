import numpy as np
from otk import bvar

def test_calc_sziklas_siegman_roc():
    k = 860e-9
    var_r0 = 1e-3**2
    phi_c0 = 1.2
    var_q = 1e3**2
    z = 1e-3
    var_rz = bvar.calc_propagated_variance_1d(k, var_r0, phi_c0, var_q, z)[0]
    m = (var_rz/var_r0)**0.5
    rocz_c = z/(m-1)
    zw, var_rw, Msqd, z_R = bvar.calc_waist(k, var_r0, phi_c0, var_q)

    rocz = bvar.calc_sziklas_siegman_roc(k, var_r0, phi_c0, var_q, z)
    assert np.isclose(rocz, rocz_c)

    roc0_c = -(z_R**2 + zw**2)/zw

    roczs = bvar.calc_sziklas_siegman_roc(k, var_r0, phi_c0, var_q, [z, 0])
    assert np.allclose(roczs, (rocz_c, roc0_c))