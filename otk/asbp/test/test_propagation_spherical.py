import pytest
import numpy as np
import mathx
import otk.h4t
from mathx import matseq
from otk import asbp, rt1, bvar, functions


def test_propagate_curved_paraxial_1d():
    k = 2*np.pi/860e-9
    num_points = 2**6
    waist0 = 20e-6
    z_R = waist0**2*k/2
    z = 10e-3
    m_gaussian = (1 + (z/z_R)**2)**0.5
    x0_support = (np.pi*num_points)**0.5*waist0
    x0_center = 20e-6
    q_center = 10e3
    x_offset = 30e-6
    q_offset = 15e3
    m = asbp.calc_curved_propagation_m(k, x0_support, num_points, np.inf, z)
    assert np.isclose(m, m_gaussian)
    Er0 = asbp.calc_gaussian_1d(k, asbp.calc_r(x0_support, num_points, x0_center), waist0, 0, x_offset, q_offset)
    Er1, x1_center = asbp.propagate_plane_to_plane_spherical_paraxial_1d(k, x0_support, Er0.copy(), z, m, x0_center,
        q_center)
    x1_support = x0_support*m
    Er1_theory = asbp.calc_gaussian_1d(k, asbp.calc_r(x1_support, num_points, x1_center), waist0, z, x_offset, q_offset)
    assert mathx.allclose(Er1, Er1_theory, atol=1e-6)
    ##
    # ru, Er1u=asbp.unroll_r_1d(x1_support, Er1, x1_center)
    # plot=pg.plot(ru*1e6, abs(Er1u), pen=pg.mkPen('b', width=2))
    # Er1u_theory=asbp.unroll_r_1d(x1_support, Er1_theory, x1_center)[1]
    # plot.plot(ru*1e6, abs(Er1u_theory), pen='r')


def test_propagate_1d():
    """Compare (i) propagate_1d with initial roc (curved), (ii) propagate_1d without, and (iii) theory."""
    k = 2*np.pi/860e-9
    num_points = 2**9
    waist0 = 200e-6
    z_R = waist0**2*k/2

    def calc_roc(z):
        if z == 0:
            return np.inf
        else:
            return z + z_R**2/z

    z1 = -300e-3
    roc1 = calc_roc(z1)
    waist1 = waist0*(1 + (z1/z_R)**2)**0.5
    x1_support = (np.pi*num_points)**0.5*waist1
    x_offset = 1e-3
    q_offset = 5e3
    x1_center = 0.9e-3
    q_center = -3e3

    x1 = asbp.calc_r(x1_support, num_points, x1_center)
    Er1 = asbp.calc_gaussian_1d(k, x1, waist0, z1, x_offset, q_offset)
    z2 = 400e-3

    Er2, m12, x2_center = asbp.propagate_plane_to_plane_spherical_1d(k, x1_support, Er1.copy(), z2 - z1, x1_center,
        q_center, roc1)

    x2 = asbp.calc_r(x1_support*m12, num_points, x2_center)

    Er2_theory = asbp.calc_gaussian_1d(k, x2, waist0, z2, x_offset, q_offset)

    assert mathx.allclose(Er2, Er2_theory, 1e-5)

    if 0:
        ## Plotting code used during development - keep here.
        plot = pg.plot(x2*1e6, abs(Er2), pen=pg.mkPen('b', width=3))
        plot.plot(x2*1e6, abs(Er2_theory), pen='g')
        plot.plot(x1*1e6, abs(Er1), pen='k')
        ##
        plot = pg.plot(x2*1e6, np.angle(Er2/Er2_theory), pen='b')
        plot.plot(x2_curved*1e6, np.angle(Er2_curved/Er2_theory_curved), pen='r')

        plot = pg.plot(x2*1e6, abs(Cr), pen='b')
        plot.plot(x2*1e6, abs(Cr_curved), pen='r')


def test_propagate_plane_to_plane_spherical():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0 = 20e-6
    z_R = waist0**2*k/2
    z = 10e-3
    m_gaussian = (1 + (z/z_R)**2)**0.5

    r_centerss = (0, 0), (10e-6, -15e-6)
    q_centerss = (0, 0), (10e3, 15e3)
    r_offsetss = (0, 0), (-10e-6, 20e-6)
    q_offsetss = (0, 0), (10e3, -15e3)

    for rs_center, qs_center, r_offsets, q_offsets in zip(r_centerss, q_centerss, r_offsetss, q_offsetss):
        r0_supports = (np.pi*num_pointss)**0.5*waist0
        m = asbp.calc_curved_propagation_m(k, r0_supports, num_pointss, np.inf, z)
        assert np.allclose(m, m_gaussian)
        x0, y0 = asbp.calc_xy(r0_supports, num_pointss, rs_center)
        Er0 = asbp.calc_gaussian(k, x0, y0, waist0, 0, r_offsets, q_offsets)
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'
        r1_centers = asbp.adjust_r(k, rs_center, z, qs_center, kz_mode)
        Er1 = asbp.propagate_plane_to_plane_spherical(k, r0_supports, Er0.copy(), z, m, rs_center, qs_center, r1_centers,
            kz_mode)
        r1_supports = r0_supports*m
        x1, y1 = asbp.calc_xy(r1_supports, num_pointss, r1_centers)
        Er1_theory = asbp.calc_gaussian(k, x1, y1, waist0, z, r_offsets, q_offsets)
        assert mathx.allclose(Er1, Er1_theory, atol=1e-6)
    ##
    # Er1_fig = asbp.plot_abs_waves_r_q(r1_supports, Er1, r1_centers, qs_center)
    # Er1_theory_fig = asbp.plot_abs_waves_r_q(r1_supports, Er1_theory, r1_centers, qs_center)


# def test_propagate():
#     k=2*np.pi/860e-9
#     num_pointss=np.asarray((2**8, 2**8))
#     waist0s=np.asarray((200e-6, 300e-6))
#     z_Rs=waist0s**2*k/2
#
#     def calc_roc(z, z_R):
#         if z==0:
#             return np.inf
#         else:
#             return z+z_R**2/z
#
#     z1=-500e-3
#     roc1s=calc_roc(z1, z_Rs)
#     waist1s=waist0s*(1+(z1/z_Rs)**2)**0.5
#
#     r1_supports=(np.pi*num_pointss)**0.5*waist1s
#     r_offsets=1e-3, -1e-3
#     q_offsets=1e3, 2e3
#     r1_centers=1.1e-3, -0.8e-3
#     qs_center=10e3, 10e3
#
#     x1, y1=asbp.calc_xy(r1_supports, num_pointss, r1_centers)
#     Er1=asbp.calc_gaussian(k, x1, y1, waist0s, z1, r_offsets, q_offsets)
#     z2=1000e-3
#
#     Er2, m12s, r2_centers=asbp.propagate_plane_to_plane_spherical(k,  r1_supports,  Er1.copy(),  z2-z1,  r1_centers,  qs_center,  roc1s)
#
#     x2, y2=asbp.calc_xy(r1_supports*m12s, num_pointss, r2_centers)
#
#     Er2_theory=asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets)
#
#     assert mathx.allclose(Er2, Er2_theory, 1e-5)
#     # plot=pg.image(abs(Er2-Er2_theory))

def test_propagate_curved_paraxial_surface_1d():
    """Check propagation of Gaussian waist to an arbitrary surface."""
    k = 2*np.pi/860e-9
    num_points = 2**8
    waist0 = 20e-6
    r_support = (np.pi*num_points)**0.5*waist0
    r1 = asbp.calc_r(r_support, num_points)
    z_R = waist0**2*k/2
    num_rayleighs_mean = 5
    m = (1 + num_rayleighs_mean**2)**0.5
    num_rayleighs = num_rayleighs_mean + np.random.randn(num_points) # Get rid of random - want deterministic.
    z2 = z_R*num_rayleighs
    Er1 = asbp.calc_gaussian_1d(k, r1, waist0, 0)
    Er2 = asbp.propagate_plane_to_plane_spherical_paraxial_1dE(k, r_support, Er1, z2, m)
    r2 = r1*m
    Er2_theory = asbp.calc_gaussian_1d(k, r2, waist0, z2)
    assert mathx.allclose(Er2, Er2_theory, 1e-6)
    ##
    # plot=pg.plot(r1*1e6, abs(Er1), pen='b')
    # plot.plot(r2*1e6, abs(Er2), pen=pg.mkPen('r', width=2))
    # plot.plot(r2*1e6, abs(Er2_theory), pen='g')

@pytest.mark.slow
def test_propagate_plane_to_curved_spherical():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0s = np.asarray((20e-6, 25e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    for r_offsets, q_offsets, rs_center, qs_center in (((0, 0), (0, 0), (0, 0), (0, 0)),
                                                       ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (30e3, 100e3)),
                                                       ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (0, 0))):
        x1, y1 = asbp.calc_xy(rs_support, num_pointss, rs_center)
        z_Rs = waist0s**2*k/2
        num_rayleighs_mean = 5
        m = (1 + num_rayleighs_mean**2)**0.5
        z_R = np.prod(z_Rs)**0.5
        num_rayleighs = num_rayleighs_mean + np.random.randn(*num_pointss)/2
        z2 = z_R*num_rayleighs
        z2_mean = num_rayleighs_mean*z_R
        r2_supports = rs_support*m
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'
        r2_centers = asbp.adjust_r(k, rs_center, z2_mean, qs_center, kz_mode)
        Er1 = asbp.calc_gaussian(k, x1, y1, waist0s, 0, r_offsets, q_offsets)
        Er2, gradxyEr2 = asbp.propagate_plane_to_curved_spherical(k, rs_support, Er1, z2, m, rs_center, qs_center,
            r2_centers, kz_mode)
        x2, y2 = asbp.calc_xy(r2_supports, num_pointss, r2_centers)
        Er2_theory, gradxyEr2_theory = asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets, gradr=True)
        assert mathx.allclose(Er2, Er2_theory, 1e-7)
        assert mathx.allclose(gradxyEr2, gradxyEr2_theory, 1e-6)
        propagator = asbp.prepare_plane_to_curved_spherical(k, rs_support, Er1.shape, z2, m, rs_center, qs_center,
            r2_centers, kz_mode)
        Er2p = propagator.apply(Er1)
        assert mathx.allclose(Er2, Er2p, 1e-15)
    ##
    # Er2_fig = asbp.plot_abs_waves_r_q(r2_supports, Er2, r2_centers, qs_center)
    # Er2_theory_fig = asbp.plot_abs_waves_r_q(r2_supports, Er2_theory, r2_centers, qs_center)

@pytest.mark.slow
def test_propagate_plane_to_curved_spherical_arbitrary():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0s = np.asarray((20e-6, 25e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    num_pointss2 = 100, 101
    for r_offsets, q_offsets, rs_center, qs_center in (((0, 0), (0, 0), (0, 0), (0, 0)),
                                                       ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (30e3, 100e3)),
                                                       ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (0, 0))):
        x1, y1 = asbp.calc_xy(rs_support, num_pointss, rs_center)
        z_Rs = waist0s**2*k/2
        num_rayleighs_mean = 5
        m = (1 + num_rayleighs_mean**2)**0.5
        z_R = np.prod(z_Rs)**0.5
        num_rayleighs = num_rayleighs_mean + np.random.uniform(-0.5, 0.5, num_pointss2)/2
        z2 = z_R*num_rayleighs
        z2_mean = num_rayleighs_mean*z_R
        r2_supports = rs_support*m
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'
        r2_centers = asbp.adjust_r(k, rs_center, z2_mean, qs_center, kz_mode)

        xo, yo = asbp.calc_xy(r2_supports, num_pointss2, r2_centers)
        mx = m*(1 + 0.1*np.random.uniform(-1, 1, num_pointss2))
        my = m*(1 + 0.1*np.random.uniform(-1, 1, num_pointss2))
        roc_x = z2/(mx - 1)
        roc_y = z2/(my - 1)

        Er1 = asbp.calc_gaussian(k, x1, y1, waist0s, 0, r_offsets, q_offsets)
        Er2, gradxyEr2 = asbp.propagate_plane_to_curved_spherical_arbitrary(k, rs_support, Er1, z2, xo, yo, roc_x, roc_y,
            rs_center, qs_center, r2_centers, kz_mode)
        x2, y2 = asbp.calc_xy(r2_supports, num_pointss2, r2_centers)
        Er2_theory, gradxyEr2_theory = asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets, gradr=True)
        assert mathx.allclose(Er2, Er2_theory, 1e-7)
        assert mathx.allclose(gradxyEr2, gradxyEr2_theory, 1e-6)
        propagator = asbp.prepare_plane_to_curved_spherical_arbitrary(k, rs_support, Er1.shape, z2, xo, yo, roc_x, roc_y,
            rs_center, qs_center, r2_centers, kz_mode)
        Er2p = propagator.apply(Er1)
        assert mathx.allclose(Er2, Er2p, 1e-15)
    ##
    # Er2_fig = asbp.plot_abs_waves_r_q(r2_supports, Er2, r2_centers, qs_center)
    # Er2_theory_fig = asbp.plot_abs_waves_r_q(r2_supports, Er2_theory, r2_centers, qs_center)

@pytest.mark.slow
def test_propagate_plane_to_curved_spherical_gradxy_localxy():
    """Use propagate_plane_to_curved_spherical with plane surface to allow numerical calculation of the gradient."""
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0s = np.asarray((20e-6, 25e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    z_Rs = waist0s**2*k/2
    z_R = np.prod(z_Rs)**0.5
    num_rayleighs = 5
    m = (1 + num_rayleighs**2)**0.5
    z2_mean = num_rayleighs*z_R
    r2_supports = rs_support*m
    rocs = z2_mean + z_Rs**2/z2_mean
    for r_offsets, q_offsets, rs_center, qs_center in (((0, 0), (0, 0), (0, 0), (0, 0)),
                                                       ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (30e3, 100e3))):
        x1, y1 = asbp.calc_xy(rs_support, num_pointss, rs_center)
        z2 = z_R*num_rayleighs*np.ones(num_pointss)
        r2_centers = asbp.adjust_r(k, rs_center, z2_mean, qs_center, 'local_xy')
        Er1 = asbp.calc_gaussian(k, x1, y1, waist0s, 0, r_offsets, q_offsets)
        Er2, gradxyEr2 = asbp.propagate_plane_to_curved_spherical(k, rs_support, Er1, z2, m, rs_center, qs_center,
            r2_centers, 'local_xy')
        x2, y2 = asbp.calc_xy(r2_supports, num_pointss, r2_centers)
        gradxyEr2_num = asbp.calc_gradxyE_spherical(k, r2_supports, Er2, rocs, r2_centers, qs_center)
        assert mathx.allclose(gradxyEr2, gradxyEr2_num, 1e-6)

def test_invert_plane_to_curved_spherical():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**7, 2**7))
    waist0s = np.asarray((20e-6, 25e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    r_offsets = 30e-6, -10e-6
    q_offsets = 30e3, 100e3
    rs_center = 30e-6, -15e-6
    qs_center = 20e3, 80e3
    x1, y1 = asbp.calc_xy(rs_support, num_pointss, rs_center)
    z_Rs = waist0s**2*k/2
    num_rayleighs_mean = 5
    m = (1 + num_rayleighs_mean**2)**0.5
    xf = (x1 - rs_center[0])/rs_support[0]*2
    yf = (y1 - rs_center[1])/rs_support[0]*2
    num_rayleighs = num_rayleighs_mean + xf*yf - xf + yf**2  # np.random.randn(*num_pointss)/2
    z_R = np.prod(z_Rs)**0.5
    z2 = z_R*num_rayleighs
    z2_mean = z_R*num_rayleighs_mean

    kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'

    r2_supports = rs_support*m
    r2_centers = asbp.adjust_r(k, rs_center, z2_mean, qs_center, kz_mode)
    x2, y2 = asbp.calc_xy(r2_supports, num_pointss, r2_centers)
    Er1_theory = asbp.calc_gaussian(k, x1, y1, waist0s, 0, r_offsets, q_offsets)
    Er2 = asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets)

    Er1, propagator = asbp.invert_plane_to_curved_spherical(k, rs_support, Er2, z2, m, rs_center, qs_center, r2_centers,
        kz_mode=kz_mode, max_iterations=10, tol=1e-8)
    assert mathx.allclose(Er2, propagator.apply(Er1), 1e-6)
    assert mathx.allclose(Er1, Er1_theory, 1e-7)
    ##
    # Er1_fig = asbp.plot_abs_waves_r_q(rs_support, Er1, rs_center, qs_center)
    # Er1_theory_fig = asbp.plot_abs_waves_r_q(rs_support, Er1_theory, rs_center, qs_center)

@pytest.mark.slow
def test_invert_plane_to_curved_spherical_arbitrary():
    # Number of points to invert to.
    num_pointss1 = 48, 64
    # Starting number of points.
    num_pointss2 = 52, 68
    roc_x2 = 50e-3
    roc_y2 = 75e-3
    k = 2*np.pi/860e-9
    waist0s = np.asarray((20e-6, 20e-6))
    r0_supports = waist0s*8
    z_Rs = waist0s**2*k/2
    trials = ((0, (0, 0), (0, 0), (0, 0), (0, 0), 5e-6, (0, 0), 1),  # 0
              (0, (0, 0), (0, 0), (0, 0), (0, 0), 50e-3, (0, 0), 1),  # 1
              (20e-3, (0, 0), (0, 0), (0, 0), (0, 0), 50e-3, (0, 0), 1),  # 2
              (50e-3, (0, 0), (0, 0), (0, 0), (0, 0), 50e-3, (0, 0), 1),  # 3
              (0, (30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (20e3, 80e3), 0, (30e-6, -10e-6), 1),  # 4
              (20e-3, (30e-6, -10e-6), (30e3, 25e3), (0, 0), (20e3, 35e3), 20e-3, (30e-6, -10e-6), 1),  # 5
              ((0, (0, 0), (0, 0), (0, 0), (0, 0), 0e-6, (0, 0), 1))) # 6

    invert_kwargs = dict(max_iterations=50, tol=1e-11)
    for trial_num, trial in tuple(enumerate(trials)):
        z1, r_offsets, q_offsets, r1_centers, qs_center, z2_mean, r2_centers, r2_supports_factor = trial

        m1s = (1 + (z1/z_Rs)**2)**0.5
        r1_supports = r0_supports*m1s
        x1, y1 = asbp.calc_xy(r1_supports, num_pointss1, r1_centers)
        Er1_theory = asbp.calc_gaussian(k, x1, y1, waist0s, z1, r_offsets, q_offsets)

        m2s_mean = (1 + (z2_mean/z_Rs)**2)**0.5
        r2_supports = r0_supports*m2s_mean*r2_supports_factor

        x2, y2 = asbp.calc_xy(r2_supports, num_pointss2, r2_centers)
        z2 = x2**2/(2*roc_x2) + y2**2/(2*roc_y2) + z2_mean
        Er2 = asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets)



        roc12_x = bvar.calc_sziklas_siegman_roc_from_waist(z_Rs[0], -z1, z2 - z1)
        roc12_y = bvar.calc_sziklas_siegman_roc_from_waist(z_Rs[1], -z1, z2 - z1)

        # calc_gaussian uses paraxial propagation, but the angles are small enough so local_xy agrees too. For
        # thoroughness, test both.
        for kz_mode in ('local_xy', 'paraxial'):
            Er1, propagator = asbp.invert_plane_to_curved_spherical_arbitrary(k, r1_supports, num_pointss1, Er2, z2 - z1,
                x2, y2, roc12_x, roc12_y, r1_centers, qs_center, r2_centers, kz_mode, invert_kwargs)
            assert Er1.shape == num_pointss1
            Er2p = propagator.apply(Er1)
            print((mathx.sum_abs_sqd(Er2 - Er2p)/mathx.sum_abs_sqd(Er2))**0.5)
            assert mathx.allclose(Er2, Er2p, 1e-4), (kz_mode, trial_num)
            assert mathx.allclose(Er1, Er1_theory, 1e-4)

            if trial == trials[-1]:
                roc21_x = bvar.calc_sziklas_siegman_roc_from_waist(z_Rs[0], -z2, z1 - z2)
                roc21_y = bvar.calc_sziklas_siegman_roc_from_waist(z_Rs[1], -z2, z1 - z2)
                Er1 = asbp.propagate_arbitrary_curved_to_plane_spherical(k, x2, y2, Er2, roc21_x, roc21_y, z1 - z2, r1_supports,
                    num_pointss1, r2_centers, qs_center, r1_centers, kz_mode, invert_kwargs)

                assert mathx.allclose(Er2, propagator.apply(Er1), 1e-4), trial_num
                assert mathx.allclose(Er1, Er1_theory, 1e-4)
    if 0:
        ##
        Er1_fig = asbp.plot_r_q_polar(r1_supports, Er1, r1_centers, qs_center)
        Er1_fig[0].setWindowTitle('Er1')
        Er1_theory_fig = asbp.plot_r_q_polar(r1_supports, Er1_theory, r1_centers, qs_center)
        Er1_theory_fig[0].setWindowTitle('Er1_theory')
        Er2_fig = asbp.plot_r_q_polar(r2_supports, Er2, r2_centers, qs_center)
        Er2_fig[0].setWindowTitle('Er2')
        Er1_diff_fig = asbp.plot_r_q_polar(r1_supports, Er1 - Er1_theory, r1_centers, qs_center)
        Er1_diff_fig[0].setWindowTitle('Er1 diff')
        # Er2p =\
        # asbp.propagate_plane_to_curved_spherical_arbitrary(k, r1_supports, Er1_theory, z2 - z1, x2, y2, m12x, m12y)[0]
        # Er2p_fig = asbp.plot_r_q_polar(r2_supports, Er2p, r2_centers, qs_center)
        # Er2p_fig[0].setWindowTitle('Er2p')
        Er2_diff_fig = asbp.plot_r_q_polar(r1_supports, Er2p - Er2, r1_centers, qs_center)
        Er2_diff_fig[0].setWindowTitle('Er2 diff')
        ##
        absEr1_diff_fig = asbp.plot_r_q_polar(r1_supports, abs(Er1) - abs(Er1_theory), r1_centers, qs_center)
        absEr1_diff_fig[0].setWindowTitle('abs Er1 diff')

@pytest.mark.slow
def test_curved_interface_collimate():
    lamb = 587.6e-9
    waist0 = 150e-6
    num_points = 2**7
    k = 2*np.pi/lamb
    r0_support = (np.pi*num_points)**0.5*waist0
    x0, y0 = asbp.calc_xy(r0_support, num_points)
    Er0 = asbp.calc_gaussian(k, x0, y0, waist0)
    f = 100e-3
    n = 1.5
    roc = f*(n - 1)
    z_R = np.pi*waist0**2/lamb
    m = (1 + (f/z_R)**2)**0.5
    r1_support = m*r0_support
    x1, y1 = asbp.calc_xy(r1_support, num_points)
    sag = functions.calc_sphere_sag_xy(roc, x1, y1)
    Er1, _ = asbp.propagate_plane_to_curved_spherical(k, r0_support, Er0, f + sag, m)
    Er2, propagator = asbp.invert_plane_to_curved_spherical(k*n, r1_support, Er1, -f*n + sag, 1)
    ##
    waist2 = f*lamb/(np.pi*waist0)
    Er2_theory = asbp.calc_gaussian(k, x1, y1, waist2)
    Er2_theory *= mathx.expj(np.angle(Er2[0, 0]/Er2_theory[0, 0]))
    assert mathx.allclose(Er2, Er2_theory, 1e-3)

@pytest.mark.slow
def test_curved_interface_collimate_offset():
    lamb = 587.6e-9
    waist0 = 30e-6
    num_points = 2**7
    k = 2*np.pi/lamb
    r0_support = (np.pi*num_points)**0.5*waist0
    # Define x lateral offset of beam.
    r_offsets = np.asarray((4e-3, -2e-3))
    # We center the numerical window on the beam.
    rs_center = r_offsets
    x0, y0 = asbp.calc_xy(r0_support, num_points, rs_center)
    # Create input beam.
    Er0 = asbp.calc_gaussian(k, x0, y0, waist0, r0s=r_offsets)
    # We will propagate it a distance f.
    f = 100e-3
    z_R = np.pi*waist0**2/lamb
    m = (1 + (f/z_R)**2)**0.5
    # We collimate it with a spherical interface.
    n = 1.5
    roc = f*(n - 1)
    # At the interface,  support is expanded by the curved wavefront propagation.
    r1_support = m*r0_support
    x1, y1 = asbp.calc_xy(r1_support, num_points, rs_center)
    # Calculate and plot interface sag.
    sag = functions.calc_sphere_sag_xy(roc, x1, y1)
    xu, yu, sagu = asbp.unroll_r(r1_support, sag)
    # Propagate to curved surface.
    Er1, _ = asbp.propagate_plane_to_curved_spherical(k, r0_support, Er0, f + sag, m, rs_center)
    xu, yu, Er1u = asbp.unroll_r(r1_support, Er1)
    #
    # x2, y2=asbp.calc_xy(r1_support, num_points)
    qs_centers2 = -r_offsets/f*k
    Er2, propagator = asbp.invert_plane_to_curved_spherical(k*n, r1_support, Er1, -f*n + sag, 1, (0, 0), qs_centers2,
        max_iterations=10)  # -x_offset/f*k
    xu, yu, Er2u = asbp.unroll_r(r1_support, Er2)
    tilt_factor = mathx.expj(-(xu*r_offsets[0] + yu*r_offsets[1])/f*k)
    ##
    waist2 = f*lamb/(np.pi*waist0)
    Er2_theory = asbp.calc_gaussian(k, x1, y1, waist2)*tilt_factor
    Er2_theory *= mathx.expj(np.angle(Er2[0, 0]/Er2_theory[0, 0]))
    assert mathx.allclose(abs(Er2), abs(Er2_theory), 2e-2)
    if 0:
        ##
        Er2_fig = asbp.plot_r_q_polar(r1_support, Er2, qs_center=qs_centers2)
        Er2_theory_fig = asbp.plot_r_q_polar(r1_support, Er2_theory, qs_center=qs_centers2)


def test_propagate_plane_to_curved_spherical_inclined():
    lamb = 860e-9
    theta = np.pi/6
    waist = 100e-6
    num_points = 64
    k = 2*np.pi/lamb
    z_R = np.pi*waist**2/lamb
    r0_support = waist*12
    z0 = -z_R*0.5
    z1m = z_R*2
    m = 2
    for phi, apod_frac in ((0, 1e-2), (np.pi/2, 1e-2), (np.pi/4, 2e-1)):
        vector = np.asarray(mathx.polar_to_cart(1, theta, phi))
        r0_centers = vector[:2]/vector[2]*z0
        qs_center = vector[:2]*k
        x0, y0 = asbp.calc_xy(r0_support, num_points, r0_centers)
        matrix = otk.h4t.make_frame(vector, (0, 1, 0))
        x0l, y0l, z0l, _ = matseq.mult_mat_vec(np.linalg.inv(matrix), (x0, y0, z0, 1))
        Er0 = asbp.calc_gaussian(k, x0l, y0l, waist, z0l)
        r1_centers = vector[:2]/vector[2]*z1m
        r1_support = r0_support*m
        x1, y1 = asbp.calc_xy(r1_support, num_points, r1_centers)
        z1 = z1m + 1e0*(x1 - r1_centers[0]) + 1e2*(y1 - r1_centers[1])**2
        x1l, y1l, z1l, _ = matseq.mult_mat_vec(np.linalg.inv(matrix), (x1, y1, z1, 1))
        Er1_theory = asbp.calc_gaussian(k, x1l, y1l, waist, z1l)
        Er1, gradxyEr1 = asbp.propagate_plane_to_curved_spherical(k, r0_support, Er0, z1 - z0, m, r0_centers, qs_center,
            r1_centers, 'local_xy')
        assert mathx.allclose(Er1, Er1_theory, apod_frac)
    if 0:
        ##
        plot0 = asbp.plot_r_q_polar(r0_support, Er0, r0_centers, qs_center)
        plot0[0].setWindowTitle('Er0')
        plot1 = asbp.plot_r_q_polar(r1_support, Er1, r1_centers, qs_center)
        plot1[0].setWindowTitle('Er1')
        plot1_theory = asbp.plot_r_q_polar(r1_support, Er1_theory, r1_centers, qs_center)
        plot1_theory[0].setWindowTitle('Er1_theory')
        plot_diff = asbp.plot_r_q_polar(r1_support, Er1 - Er1_theory, r1_centers, qs_center)
        plot_diff[0].setWindowTitle('diff')

        plot_diff_abs = asbp.plot_r_q_polar(r1_support, abs(Er1) - abs(Er1_theory), r1_centers, qs_center)
        plot_diff_abs[0].setWindowTitle('diff_abs')