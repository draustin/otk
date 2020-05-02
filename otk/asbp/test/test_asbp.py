import mathx
import numpy as np
from otk import asbp, bvar


def test_misc_variance_funs():
    lamb = 860e-9
    k = 2*np.pi/lamb
    r_support = 1e-3
    num_points = 256
    waist0 = 0.1e-3
    r = asbp.calc_r(r_support, num_points)
    q = asbp.calc_q(r_support, num_points)

    def doit(order):
        Er_0f = np.exp(-(r/waist0)**order)
        phi_c = 2
        var_r0 = mathx.mean_and_variance(r, abs(Er_0f)**2)[1]
        Er_0 = Er_0f*np.exp(1j*r**2*phi_c/var_r0)
        f = -k*var_r0/(2*phi_c)  # Lens applied to _0f field to get _0.

        Eq_0f = asbp.fft(Er_0f)
        Eq_0 = asbp.fft(Er_0)
        var_q0 = mathx.mean_and_variance(q, abs(Eq_0)**2)[1]
        var_q0f = mathx.mean_and_variance(q, abs(Eq_0f)**2)[1]

        assert np.isclose(bvar.transform_angular_variance_lens_1d(k, var_r0, 0, var_q0f, -k*var_r0/(2*phi_c)), var_q0)

        power_range = bvar.calc_correction_lens_range_1d(k, var_r0, phi_c, var_q0, var_q0f*1.1)
        assert power_range[0] <= -1/f <= power_range[1]

        assert np.isclose(bvar.infer_angular_variance_spherical(var_r0, phi_c, var_q0f), var_q0)

        assert np.isclose(bvar.calc_minimum_angular_variance_1d(var_r0, phi_c, var_q0), var_q0f)

        zw, var_rw, Msqd, z_R = bvar.calc_waist(k, var_r0, phi_c, var_q0)

        Er_w = asbp.propagate_plane_to_plane_flat_1d(k, r_support, Er_0, zw, paraxial=True)
        var_rw_ = mathx.mean_and_variance(r, abs(Er_w)**2)[1]
        assert np.isclose(var_rw**0.5, var_rw_**0.5)
        if order == 2:
            # Gaussian with quadratic phase in real space is Gaussian with quadratic phase in angular space,  which
            # is a perfect Gaussian after propagation.
            assert np.isclose(Msqd, 1)

        z = 1e-3
        Er_1 = asbp.propagate_plane_to_plane_flat_1d(k, r_support, Er_0, z, paraxial=True)
        var_r1 = mathx.mean_and_variance(r, abs(Er_1)**2)[1]
        assert np.isclose(bvar.calc_propagated_variance_1d(k, var_r0, phi_c, var_q0, z)[0]**0.5, var_r1**0.5)

        m = asbp.calc_propagation_m_1d(k, r_support, var_r0, phi_c, var_q0, z, num_points)

        Er_1m, _ = asbp.propagate_plane_to_plane_spherical_paraxial_1d(k, r_support, Er_0.copy(), z, m)
        var_r1m = mathx.mean_and_variance(r*m, abs(Er_1m)**2)[1]
        assert np.isclose(var_r1m**0.5, var_r1**0.5)

    doit(2)
    doit(4)
