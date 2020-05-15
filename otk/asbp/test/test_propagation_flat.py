import pytest
import mathx
import otk.h4t
from mathx import matseq
import numpy as np
from otk import asbp, bvar, rt1

def test_propagate_flat_1d():
    k = 2*np.pi/860e-9
    num_points = 2**6
    waist0 = 20e-6
    z_R = waist0**2*k/2
    z1 = -0.5e-3
    x_center = 10e-6
    q_center = 7e3
    x_offset = 25e-6
    q_offset = 50e3
    x_support = (np.pi*num_points)**0.5*waist0
    r = asbp.calc_r(x_support, num_points, x_center)
    Er1 = asbp.calc_gaussian_1d(k, r, waist0, z1, x_offset, q_offset)
    z2 = 1e-3
    Er2 = asbp.propagate_plane_to_plane_flat_1d(k, x_support, Er1, z2 - z1, q_center, paraxial=True)
    Er2_theory = asbp.calc_gaussian_1d(k, r, waist0, z2, x_offset, q_offset)
    assert mathx.allclose(Er2, Er2_theory, atol=1e-6)
    ##
    # plot=pg.plot(r*1e6, abs(Er1), pen=pg.mkPen('b', width=2))
    # ru, Er1u=asbp.unroll_r_1d(x_support, Er1, x_center)
    # plot.plot(ru*1e6, abs(Er1u), pen='r')
    # plot=pg.plot(ru*1e6, abs(asbp.unroll_r_1d(x_support, Er2, x_center)[1]), pen=pg.mkPen('r', width=2))
    # plot.plot(ru*1e6, abs(asbp.unroll_r_1d(x_support, Er2_theory, x_center)[1]), pen='b')

def test_propagate_plane_to_plane_flat():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0 = 20e-6
    z_R = waist0**2*k/2
    z1 = -0.5e-3

    r_centerss = (0, 0), (10e-6, -15e-6)
    q_centerss = (0, 0), (10e3, 15e3)
    r_offsetss = (0, 0), (-10e-6, 20e-6)
    q_offsetss = (0, 0), (10e3, -15e3)

    for rs_center, qs_center, r_offsets, q_offsets in zip(r_centerss, q_centerss, r_offsetss, q_offsetss):
        rs_support = (np.pi*num_pointss)**0.5*waist0
        x, y = asbp.calc_xy(rs_support, num_pointss, rs_center)
        Er1 = asbp.calc_gaussian(k, x, y, waist0, z1, r_offsets, q_offsets)
        z2 = 1e-3
        if qs_center == (0, 0):
            kz_mode = 'local_xy'
        else:
            kz_mode = 'paraxial'
        Er2 = asbp.propagate_plane_to_plane_flat(k, rs_support, Er1, z2 - z1, qs_center, kz_mode=kz_mode)
        Er2_theory = asbp.calc_gaussian(k, x, y, waist0, z2, r_offsets, q_offsets)
        assert mathx.allclose(Er2, Er2_theory, atol=1e-6)
    ##
    # Er2_fig = asbp.plot_abs_waves_r_q(rs_support, Er2, rs_center, qs_center)
    # Er2_theory_fig = asbp.plot_abs_waves_r_q(rs_support, Er2_theory, rs_center, qs_center)

@pytest.mark.slow
def test_propagate_curved_to_plane_flat():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**7, 2**7))
    waist0s = np.asarray((200e-6, 300e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    rqs = ((0, 0), (0, 0), (0, 0), (0, 0)), ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (20e3, 80e3))
    for r_offsets, q_offsets, rs_center, qs_center in rqs:
        x1, y1 = asbp.calc_xy(rs_support, num_pointss, rs_center)
        z_Rs = waist0s**2*k/2
        num_rayleighs_mean = 0.5
        xf = (x1 - rs_center[0])/rs_support[0]*0.1
        yf = (y1 - rs_center[1])/rs_support[0]*0.1
        num_rayleighs = num_rayleighs_mean + 0.1*(xf*yf - xf + yf**2)  # np.random.randn(*num_pointss)/2
        z_R = np.prod(z_Rs)**0.5
        z2 = z_R*num_rayleighs
        z2_mean = z_R*num_rayleighs_mean

        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'

        r2_centers = asbp.adjust_r(k, rs_center, z2_mean, qs_center, kz_mode)
        x2, y2 = asbp.calc_xy(rs_support, num_pointss, r2_centers)
        Er1_theory = asbp.calc_gaussian(k, x1, y1, waist0s, 0, r_offsets, q_offsets)
        Er2 = asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets)

        Er1, propagator = asbp.invert_plane_to_curved_flat(k, rs_support, Er2, z2, qs_center, kz_mode=kz_mode,
            max_iterations=10, tol=1e-20)
        assert abs(Er2 - propagator.apply(Er1)).max() < 1e-8
        assert mathx.allclose(Er1, Er1_theory, 1e-7)
    ##
    # Er1_fig = asbp.plot_abs_waves_r_q(rs_support, Er1, rs_center, qs_center)
    # # Er2_fig = asbp.plot_abs_waves_r_q(r2_supports, Er2, r2_centers, qs_center)
    # Er1_theory_fig = asbp.plot_abs_waves_r_q(rs_support, Er1_theory, rs_center, qs_center)
    # diff_fig = asbp.plot_abs_waves_r_q(rs_support, Er1 - Er1_theory, rs_center, qs_center)


def test_propagate_plane_to_curved_flat():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0s = np.asarray((20e-6, 25e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    rqs = ((0, 0), (0, 0), (0, 0), (0, 0)), ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (20e3, 80e3))
    for r_offsets, q_offsets, rs_center, qs_center in rqs:
        x, y = asbp.calc_xy(rs_support, num_pointss, rs_center)
        z_Rs = waist0s**2*k/2
        z_R = np.prod(z_Rs)**0.5
        num_rayleighs = np.random.randn(*num_pointss)/5
        z2 = z_R*num_rayleighs
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'
        Er1 = asbp.calc_gaussian(k, x, y, waist0s, 0, r_offsets, q_offsets)
        Er2, gradxyEr2 = asbp.propagate_plane_to_curved_flat(k, rs_support, Er1, z2, qs_center, kz_mode)
        Er2_theory, gradxyEr2_theory = asbp.calc_gaussian(k, x, y, waist0s, z2, r_offsets, q_offsets, True)
        assert mathx.allclose(Er2, Er2_theory, 1e-7)
        assert mathx.allclose(gradxyEr2, gradxyEr2_theory, 1e-7)
        propagator = asbp.prepare_plane_to_curved_flat(k, rs_support, Er1.shape, z2, qs_center, kz_mode)
        Er2p = propagator.apply(Er1)
        assert mathx.allclose(Er2, Er2p, 1e-15)
    ##
    #Er2_fig = asbp.plot_abs_waves_r_q(rs_support, Er2, rs_center, qs_center)
    #Er2_theory_fig = asbp.plot_abs_waves_r_q(rs_support, Er2_theory, rs_center, qs_center)

@pytest.mark.slow
def test_propagate_plane_to_curved_flat_arbitrary():
    k = 2*np.pi/860e-9
    num_pointss = np.asarray((2**6, 2**7))
    waist0s = np.asarray((20e-6, 25e-6))
    rs_support = (np.pi*num_pointss)**0.5*waist0s
    num_pointss2 = 100, 101
    z_Rs = waist0s**2*k/2
    z_R = np.prod(z_Rs)**0.5
    num_rayleighs_mean = 0
    z2_mean = num_rayleighs_mean*z_R
    rqs = ((0, 0), (0, 0), (0, 0), (0, 0)), ((30e-6, -10e-6), (30e3, 100e3), (30e-6, -15e-6), (20e3, 80e3))
    for r_offsets, q_offsets, rs_center, qs_center in rqs:
        x, y = asbp.calc_xy(rs_support, num_pointss, rs_center)
        num_rayleighs = num_rayleighs_mean + np.random.uniform(-0.5, 0.5, num_pointss2)/2
        z2 = z_R*num_rayleighs
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'
        Er1 = asbp.calc_gaussian(k, x, y, waist0s, 0, r_offsets, q_offsets)
        r2_centers = asbp.adjust_r(k, rs_center, z2_mean, qs_center, kz_mode)
        x2, y2 = asbp.calc_xy(rs_support, num_pointss2, r2_centers)
        Er2, gradxyEr2 = asbp.propagate_plane_to_curved_flat_arbitrary(k, rs_support, Er1, z2, x2, y2, qs_center, kz_mode)
        Er2_theory, gradxyEr2_theory = asbp.calc_gaussian(k, x2, y2, waist0s, z2, r_offsets, q_offsets, True)
        assert mathx.allclose(Er2, Er2_theory, 1e-7)
        assert mathx.allclose(gradxyEr2, gradxyEr2_theory, 1e-7)
        propagator = asbp.prepare_plane_to_curved_flat_arbitrary(k, rs_support, Er1.shape, z2, x2, y2, qs_center, kz_mode)
        Er2p = propagator.apply(Er1)
        assert mathx.allclose(Er2, Er2p, 1e-7)
    ##
    #Er2_fig = asbp.plot_abs_waves_r_q(rs_support, Er2, rs_center, qs_center)
    #Er2_theory_fig = asbp.plot_abs_waves_r_q(rs_support, Er2_theory, rs_center, qs_center)


def test_propagate_plane_to_plane_flat_inclined():
    """Propagate an inclined Gaussian beam between two planes."""
    lamb = 860e-9
    theta = np.pi/6
    phi = np.pi/4
    waist = 100e-6
    num_points = 64
    k = 2*np.pi/lamb
    z_R = np.pi*waist**2/lamb
    r_support = waist*12
    vector = np.asarray(mathx.polar_to_cart(1, theta, phi))
    z0 = -z_R*0.5
    z1 = z_R*0.5
    r0_centers = vector[:2]/vector[2]*z0
    qs_center = vector[:2]*k
    x0, y0 = asbp.calc_xy(r_support, num_points, r0_centers)
    matrix = otk.h4t.make_frame(vector, (0, 1, 0))
    x0l, y0l, z0l, _ = matseq.mult_mat_vec(np.linalg.inv(matrix), (x0, y0, z0, 1))
    Er0 = asbp.calc_gaussian(k, x0l, y0l, waist, z0l)
    r1_centers = vector[:2]/vector[2]*z1
    x1, y1 = asbp.calc_xy(r_support, num_points, r1_centers)
    x1l, y1l, z1l, _ = matseq.mult_mat_vec(np.linalg.inv(matrix), (x1, y1, z1, 1))
    Er1_theory = asbp.calc_gaussian(k, x1l, y1l, waist, z1l)
    for kz_mode, atol_frac in (('local_xy', 1e-1), ('local', 1e-2), ('exact', 1e-5)):
        ##
        Er1 = asbp.propagate_plane_to_plane_flat(k, r_support, Er0, z1 - z0, qs_center, kz_mode)
        assert mathx.allclose(Er1, Er1_theory, atol_frac)
    if 0:
        ##
        plot0 = asbp.plot_r_q_polar(r_support, Er0, r0_centers, qs_center)
        plot0[0].setWindowTitle('Er0')
        plot1 = asbp.plot_r_q_polar(r_support, Er1, r1_centers, qs_center)
        plot1[0].setWindowTitle('Er1')
        plot1_theory = asbp.plot_r_q_polar(r_support, Er1_theory, r1_centers, qs_center)
        plot1_theory[0].setWindowTitle('Er1_theory')
        plot_diff = asbp.plot_r_q_polar(r_support, Er1 - Er1_theory, r1_centers, qs_center)
        plot_diff[0].setWindowTitle('diff')

        plot_diff_abs = asbp.plot_r_q_polar(r_support, abs(Er1) - abs(Er1_theory), r1_centers, qs_center)
        plot_diff_abs[0].setWindowTitle('diff_abs')

def test_propagate_plane_to_curved_flat_inclined():
    lamb = 860e-9
    theta = np.pi/6
    waist = 100e-6
    num_points = 64
    k = 2*np.pi/lamb
    z_R = np.pi*waist**2/lamb
    r_support = waist*12
    z0 = -z_R*0.5
    z1m = z_R*0.5

    # When the central k vector is in the xz or yz planes then in local_xy mode, the kz expansion is exact to second
    # order. Otherwise it is only approximate to second order since it ignores the crossed second derivatives.
    for phi, atol_frac in ((0, 2e-3), (np.pi/2, 2e-3), (np.pi/4, 1e-1)):
        vector = np.asarray(mathx.polar_to_cart(1, theta, phi))
        r0_centers = vector[:2]/vector[2]*z0
        qs_center = vector[:2]*k
        x0, y0 = asbp.calc_xy(r_support, num_points, r0_centers)
        matrix = otk.h4t.make_frame(vector, (0, 1, 0))
        x0l, y0l, z0l, _ = matseq.mult_mat_vec(np.linalg.inv(matrix), (x0, y0, z0, 1))
        Er0 = asbp.calc_gaussian(k, x0l, y0l, waist, z0l)
        r1_centers = vector[:2]/vector[2]*z1m
        x1, y1 = asbp.calc_xy(r_support, num_points, r1_centers)
        z1 = z1m + 0*(x1 - r1_centers[0]) + 0*(y1 - r1_centers[1])
        x1l, y1l, z1l, _ = matseq.mult_mat_vec(np.linalg.inv(matrix), (x1, y1, z1, 1))
        Er1_theory, gradxyEr1_theory = asbp.calc_gaussian(k, x1l, y1l, waist, z1l, gradr=True)
        Er1, gradxyEr1 = asbp.propagate_plane_to_curved_flat(k, r_support, Er0, z1 - z0, qs_center, 'local_xy')
        assert mathx.allclose(Er1, Er1_theory, atol_frac)
        # We don't expect the derivatives to agree because they are along the inclined surface.

    if 0:
        ##
        plot0 = asbp.plot_r_q_polar(r_support, Er0, r0_centers, qs_center)
        plot0[0].setWindowTitle('Er0')
        plot1 = asbp.plot_r_q_polar(r_support, Er1, r1_centers, qs_center)
        plot1[0].setWindowTitle('Er1')
        plot1_theory = asbp.plot_r_q_polar(r_support, Er1_theory, r1_centers, qs_center)
        plot1_theory[0].setWindowTitle('Er1_theory')
        plot_diff = asbp.plot_r_q_polar(r_support, Er1 - Er1_theory, r1_centers, qs_center)
        plot_diff[0].setWindowTitle('diff')

        plot_diff_abs = asbp.plot_r_q_polar(r_support, abs(Er1) - abs(Er1_theory), r1_centers, qs_center)
        plot_diff_abs[0].setWindowTitle('diff_abs')

