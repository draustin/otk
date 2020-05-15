import pytest
import numpy as np
from numpy import testing
import mathx
from otk import asbp, beams, bvar, trains, ri

@pytest.mark.slow
def test_profile_propagation():
    # Propagate Gaussian beam onto curved surface. Plane 0 is the waist, plane 1 is the start surface and plane 2 is the
    # final surface.
    lamb = 860e-9
    n = 1
    waist0s = np.asarray((20e-6, 20e-6))
    r0_supports = waist0s*8

    num_pointss1 = 48, 64

    num_pointss2 = 63, 64
    roc_x2 = 50e-3
    roc_y2 = 75e-3
    k = 2*np.pi*n/lamb
    z_Rs = waist0s**2*k/2

    trials = (
        # 0 - propagate from waist a short distance.
        (0, (0, 0), (0, 0), (0, 0), (0, 0), 5e-6, (0, 0)),
        # 1 - propagate from waist many Rayleighs.
        (0, (0, 0), (0, 0), (0, 0), (0, 0), 50e-3, (0, 0)),
        # 2 - propagate between two places both with significant curvature.
        (20e-3, (0, 0), (0, 0), (0, 0), (0, 0), 50e-3, (0, 0)),
        # 3 - propagate from plane to curved surface through plane.
        (50e-3, (30e-6, -10e-6), (30e3, 10e3), (50e-6, -40e-6), (30e3, 10e3), 50e-3, (-100e-6, 50e-6)),
        # 4
        (0, (30e-6, -10e-6), (30e3, 50e3), (0, 0), (30e3, 50e3), 0, (0, 0)),
        # 5
        (20e-3, (30e-6, -10e-6), (30e3, 25e3), (0, 0), (30e3, 25e3), 20e-3, (0, 0)))

    for trial_num, trial in enumerate(trials):
        z1, rs_waist, qs_waist, delta_rs_center1, qs_center, z2_mean, delta_rs_center2 = trial
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'

        # Setup plane 1 (initial).
        rs_center1 = asbp.math.adjust_r(k, rs_waist, z1, qs_waist, kz_mode) + delta_rs_center1
        m1s = (1 + (z1/z_Rs)**2)**0.5
        r1_supports = r0_supports*m1s
        x1, y1 = asbp.calc_xy(r1_supports, num_pointss1, rs_center1)
        profile1 = asbp.PlaneProfile.make_gaussian(lamb, n, waist0s, r1_supports, num_pointss1, rs_waist, qs_waist, 0,
            rs_center1, qs_center, z1)

        # Setup plane 2 (destination).
        m2s_mean = (1 + (z2_mean/z_Rs)**2)**0.5
        rs_support2 = r0_supports*m2s_mean
        rs_center2 = asbp.math.adjust_r(k, rs_waist, z2_mean, qs_waist, kz_mode) + delta_rs_center2

        # Test propagate to plane surface 2.
        profile2_plane = profile1.propagate_to_plane(z2_mean, rs_center2, m2s_mean/m1s, kz_mode)
        profile2_theory_plane = asbp.PlaneProfile.make_gaussian(lamb, n, waist0s, rs_support2, num_pointss1, rs_waist,
            qs_waist, 0, rs_center2, qs_center, z2_mean)
        assert mathx.allclose(profile2_theory_plane.Er, profile2_plane.Er, 1e-4)

        # Setup test at curved curved surface 2
        x2, y2 = asbp.calc_xy(rs_support2, num_pointss2, rs_center2)
        z2fun = lambda x, y: x**2/(2*roc_x2) + y**2/(2*roc_y2) + z2_mean
        z2 = z2fun(x2, y2)
        profile2_theory = asbp.CurvedProfile.make_gaussian(lamb, n, waist0s, rs_support2, num_pointss2, rs_waist,
            qs_waist, 0, rs_center2, qs_center, z2fun)

        # Test with ROCs specified.
        roc_x = bvar.calc_sziklas_siegman_roc_from_waist(z_Rs[0], -z1, z2 - z1)
        roc_y = bvar.calc_sziklas_siegman_roc_from_waist(z_Rs[1], -z1, z2 - z1)
        profile2 = profile1.propagate_to_curved(rs_support2, num_pointss2, rs_center2, z2fun, roc_x, roc_y, kz_mode)
        assert mathx.allclose(profile2_theory.Er, profile2.Er, 1e-4)

        # Test automatic determination of correct ROC.
        profile2 = profile1.propagate_to_curved(rs_support2, num_pointss2, rs_center2, z2fun, None, None, kz_mode)
        assert mathx.allclose(profile2_theory.Er, profile2.Er, 1e-3)

    ##
    if 0:
        profile1_fig = profile1.plot_r_q_polar(True)
        profile2_plane_fig = profile2_plane.plot_r_q_polar(True)
        profile2_theory_plane_fig = profile2_theory_plane.plot_r_q_polar(True)
        ##
        profile2_fig = profile2.plot_r_q_polar(True)
        profile2_theory_fig = profile2_theory.plot_r_q_polar(True)

def test_profile_plotting(qtbot):
    profile0 = asbp.PlaneProfile.make_gaussian(860e-9, 1.5, 20e-6, 200e-6, 2**7)
    qtbot.addWidget(profile0.plot_r_q_polar()[0])

    widget = asbp.PlaneProfileWidget()
    qtbot.addWidget(widget)
    widget.set_profile(profile0)

    profile0 = asbp.CurvedProfile.make_gaussian(860e-9, 1.5, 20e-6, 200e-6, 2**7)
    qtbot.addWidget(profile0.plot_r_q_polar()[0])
    widget = asbp.CurvedProfileWidget()
    qtbot.addWidget(widget)
    widget.set_profile(profile0)


def test_PlaneProfile_interpolate():
    lamb = 587.6e-9
    waist0 = 30e-6

    r_offsets = (1e-3, -1e-3)
    q_offsets = (10e3, 30e3)
    z_offset = 40e-3
    r_support0 = beams.FundamentalGaussian(lamb, w_0=waist0).waist(z_offset)*5
    r_centers0 = (0.99e-3, -1.01e-3)
    qs_center = (11e3, 32e3)
    profile0 = asbp.PlaneProfile.make_gaussian(lamb, 1, waist0, r_support0, 2**7, r_offsets, q_offsets, z_offset, r_centers0,
        qs_center)

    r_support1 = 50e-6
    r_centers1 = (0.95e-3, -1.05e-3)
    num_points1 = 2**7
    profile1 = profile0.interpolate(r_support1, num_points1, r_centers1)
    profile1_theory = asbp.PlaneProfile.make_gaussian(lamb, 1, waist0, r_support1, num_points1, r_offsets, q_offsets,
        z_offset, r_centers1, qs_center)
    ##
    assert mathx.allclose(profile1.Er, profile1_theory.Er, 1e-3)
    assert mathx.allclose(profile1.gradxyE[0], profile1_theory.gradxyE[0], 1e-3)
    assert mathx.allclose(profile1.gradxyE[1], profile1_theory.gradxyE[1], 1e-3)


def test_CurvedProfile():
    # Propagate Gaussian beam onto curved surface. Plane 0 is the waist, plane 1 is the start surface and plane 2 is the
    # final surface.
    lamb = 860e-9
    n = 1.5
    waist0s = np.asarray((20e-6, 22e-6))
    rs_support0 = waist0s*8

    num_pointss1 = 48, 64

    num_pointss2 = 63, 64
    roc_x2 = 50e-3
    roc_y2 = 50e-3
    k = 2*np.pi*n/lamb
    z_Rs = waist0s**2*k/2

    trials = (
        (0, (0, 0), (0, 0), (0, 0), (0, 0)),  # 0 - propagate from waist a short distance.
        (50e-3, (0, 0), (0, 0), (0, 0), (0, 0)), # 1 - propagate between two places both with significant curvature.
        (50e-3, (30e-6, -10e-6), (5e3, 10e3), (30e-6, -15e-6), (4e3, 11e3)),  # 2 - with offsets.
        (-5e-3, (1e-3, 0.5e-3), (-10e3, 5e3), (1e-3, 0.5e-3), (-10e3, 5e3)))

    for trial_num, trial in enumerate(trials):
        z1, rs_waist, qs_waist, rs_center1, qs_center = trial
        kz_mode = 'local_xy' if qs_center == (0, 0) else 'paraxial'

        m1s = (1 + (z1/z_Rs)**2)**0.5
        rs_support1 = rs_support0*m1s

        profile1_theory = asbp.PlaneProfile.make_gaussian(lamb, n, waist0s, rs_support1, num_pointss1, rs_waist,
            qs_waist, 0,
            rs_center1, qs_center, z1)

        def calc_z2(x2, y2):
            return z1 + x2**2/(2*roc_x2) + y2**2/(2*roc_y2)

        x2, y2 = asbp.calc_xy(rs_support1, num_pointss2, rs_center1)
        z2 = calc_z2(x2, y2)

        profile2 = asbp.CurvedProfile.make_gaussian(lamb, n, waist0s, rs_support1, num_pointss2, rs_waist, qs_waist, 0,
            rs_center1, qs_center, calc_z2)
        assert np.isclose(profile2.z_center, calc_z2(*rs_center1))

        profile1 = profile2.planarize(z1, rs_support1, num_pointss1, kz_mode)
        assert profile1.n == profile2.n
        assert mathx.allclose(profile1_theory.Er, profile1.Er, 1e-4)
    ##
    # profile1_theory_fig = profile1_theory.plot_r_q_polar(True)
    # profile1_fig = profile1.plot_r_q_polar(True)
    # profile2_fig = profile2.plot_r_q_polar(True)


def test_PlaneProfile_fourier_transform():
    lamb = 860e-9
    waist0s = 30e-6
    r0_supports = waist0s*6
    num_points = 64
    rs_waist = np.asarray((1e-3, 1.2e-3))
    qs_waist = np.asarray((10e3, 20e3))
    rs_center0 = np.asarray((1.01e-3, 1.21e-3))
    qs_center0 = np.asarray((15e3, 25e3))
    profile0 = asbp.PlaneProfile.make_gaussian(lamb, 1, waist0s, r0_supports, num_points, rs_waist, qs_waist, 0,
        rs_center0, qs_center0)
    f = 100e-3
    rs0 = np.asarray((0.9e-3, 1.1e-3))
    profile1 = profile0.fourier_transform(f, rs0)
    testing.assert_allclose(profile1.rs_center, profile0.qs_center*f/profile0.k)
    testing.assert_allclose(profile1.qs_center, (profile0.rs_center - rs0)*profile0.k/f)

def test_PlaneProfile_interface():
    roc = 1
    lamb = 800e-9
    n1 = ri.vacuum
    n2 = ri.fused_silica
    interface = trains.Interface(n1, n2, roc, 2e-3)
    rs_center = 0.5e-3, -0.2e-3

    p0 = asbp.PlaneProfile.make_gaussian(lamb, n1(lamb), 1e-3, 5e-3, 64)

    dx = p0.x - rs_center[0]
    dy = p0.y - rs_center[1]
    rho = (dx**2 + dy**2)**0.5

    p1 = p0.apply_interface_thin(interface, rs_center, None)
    assert np.allclose(p1.Er, p0.Er*interface.calc_mask(p0.lamb, rho))
    assert p1.n == n2(lamb)

    for shape in ('circle', 'square'):
        p1 = p0.apply_interface_thin(interface, rs_center, shape)
        assert np.allclose(p1.Er, p0.Er*interface.calc_mask(p0.lamb, rho)*interface.calc_aperture(dx, dy, shape))
