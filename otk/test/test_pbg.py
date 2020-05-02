import otk.h4t
import otk.rt1.lines
import numpy as np
from numpy import testing
from otk import beams
import mathx
from otk import rt1, pbg, ri
from otk import functions as omath

def test_pbg_calc_field():
    origin = rt1.stack_xyzw(0, 0, 0, 1)
    vector = rt1.stack_xyzw(0, 0, 1, 0)
    axis1 = rt1.normalize(rt1.stack_xyzw(1, 1j, 0.5j, 0))
    lamb = 860e-9
    k = 2*np.pi/lamb
    waist = 100e-6
    z_R = np.pi*waist**2/lamb
    z_waist = 1*z_R
    mode = pbg.Mode.make(otk.rt1.lines.Line(origin, vector), axis1, lamb, waist, z_waist)[0]

    z = np.linspace(-z_R*16, z_R*16, 200)[:, None]
    x = np.arange(3)[:, None, None]*waist
    #zv, xv = [array.ravel() for array in np.broadcast_arrays(zs, xs)]
    points = rt1.concatenate_xyzw(x, 0, z, 1)
    field, phi_axis, grad_phi = mode.calc_field(k, points, calc_grad_phi=True)
    psis = field*mathx.expj(-z*k)
    psis_true = beams.FundamentalGaussian(lamb, w_0=waist, flux=1).E(z - z_waist, x)*np.exp(
        -1j*np.arctan(z_waist/z_R))

    assert np.allclose(psis, psis_true)



    if 0: # Useful for testing in IPython.
        glw = pg.GraphicsLayoutWidget()
        absz_plot = glw.addAlignedPlot()
        glw.nextRows()
        phase_plot = glw.addAlignedPlot()
        for x, color, psi, psi_true in zip(xs, pg.tableau10, psis.T, psis_true.T):
            absz_plot.plot(zs[:, 0]*1e3, abs(psi), pen=color)
            absz_plot.plot(zs[:, 0]*1e3, abs(psi_true), pen=pg.mkPen(color, style=pg.DashLine))
            phase_plot.plot(zs[:, 0]*1e3, np.angle(psi), pen=color)
            phase_plot.plot(zs[:, 0]*1e3, np.angle(psi_true), pen=pg.mkPen(color, style=pg.DashLine))
        glw.show()

def test_pbg_simple():
    waist = 100e-6
    n = 1
    lamb = 800e-9
    k = 2*np.pi*n/lamb
    # Totally arbitrary origin and axes.
    origin = np.asarray([1, 2, 3, 1])
    vector = rt1.normalize(np.asarray([1, -2, -1, 0]))
    axis_x = rt1.normalize(np.asarray([1, 1, 0, 0]))

    mode, (axis_x, axis_y) = pbg.Mode.make(otk.rt1.lines.Line(origin, vector), axis_x, lamb/n, waist)

    z_R = waist**2*k/2
    z = np.asarray([-10, 1, 0, 1, 10])[:, None]*z_R*0
    x = np.arange(3)[:, None, None]*waist
    y = np.arange(3)[:, None, None, None]*waist
    #z = np.asarray([[z_R]])
    #x = np.asarray([[waist]])
    #y = np.asarray([[waist]])
    rho = (x**2 + y**2)**0.5
    xhat = np.asarray([1, 0, 0, 0])
    yhat = np.asarray([0, 1, 0, 0])
    rhohat = mathx.divide0(xhat*x + yhat*y, rho)
    mode_matrix = np.stack([axis_x, axis_y, vector, origin], 0)
    points = rt1.concatenate_xyzw(x, y, z, 1).dot(mode_matrix)

    field, phi_axis, grad_phi = mode.calc_field(k, points, calc_grad_phi=True)

    beam = beams.FundamentalGaussian(lamb/n, w_0=waist, flux=1)
    field_true = beam.E(z, rho)*mathx.expj(z*k)
    field_true_axis = beam.E(z, 0)*mathx.expj(z*k)
    grad_phi_true = (beam.dphidz(z, rho) + k)*vector + beam.dphidrho(z, rho)*rhohat.dot(mode_matrix)

    testing.assert_allclose(abs(field), abs(field_true))
    testing.assert_allclose(field, field_true)

    testing.assert_allclose(abs(mathx.wrap_to_pm(phi_axis - np.angle(field_true_axis), np.pi)), 0, atol=1e-6)

    testing.assert_allclose(grad_phi, grad_phi_true, atol=1e-6)

def test_pbg_transform_advance():
    # Check that the field of a totally non-degenerate PBG is identical under all sorts of transforms.
    lamb = 860e-9
    k = 2*np.pi/lamb
    waist1 = 100e-6  # + 0e-6j
    waist2 = 50e-6
    z_R = np.pi*waist1**2/lamb
    x0 = 1
    y0 = 2
    z0 = 3
    origin = rt1.stack_xyzw(x0, y0, z0, 1)
    vector = rt1.normalize(rt1.stack_xyzw(0, 0, 1, 0))
    axis1 = rt1.normalize(rt1.stack_xyzw(1 + 1j, 1, 0, 0))

    z_waist1 = (1 + 1j)*z_R
    z_waist2 = (2 - 1j)*z_R
    mode = pbg.Mode.make(otk.rt1.lines.Line(origin, vector), axis1, lamb, waist1, z_waist1, waist2, z_waist2)[0]
    z = np.linspace(-z_R*16, z_R*16, 200)[:, None] + z0
    rho = np.arange(3)[:, None, None]*waist1
    theta = np.pi/4
    x = np.cos(theta)*rho + x0
    y = np.sin(theta)*rho + y0
    points = rt1.concatenate_xyzw(x, y, z, 1)
    field, phi_axis, grad_phi = mode.calc_field(k, points, calc_grad_phi=True)
    #print(abs(field).max()) # Check we don't have all zeros.

    matrices = [otk.h4t.make_translation(-1, 2, 3), otk.h4t.make_scaling(-1, -1, -1),
                np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

    advances = [0, 0.2]

    for advance in advances:
        modep, phi_axis_origin = mode.advance(k, advance)
        for matrix in matrices:
            modepp = modep.transform(matrix)
            pointsp = rt1.transform(points, matrix)

            fieldp, phi_axisp, grad_phipp = modepp.calc_field(k, pointsp, 1, phi_axis_origin, calc_grad_phi=True)
            grad_phip = grad_phipp.dot(np.linalg.inv(matrix))

            assert np.allclose(fieldp, field)
            assert np.allclose(phi_axisp, phi_axis)
            assert np.allclose(grad_phip, grad_phi)

def test_pbg_project_field():
    # Define a fundamental Gaussian beam point along z axis.
    lamb = 860e-9
    k = 2*np.pi/lamb
    waist = 10e-6
    flux = 2
    beam = beams.FundamentalGaussian(lamb=lamb, w_0=waist, flux=flux)
    z_waist = beam.z_R # Put waist of Gaussian at one Rayleigh range.
    z = 2*beam.z_R # Sample at two Rayleigh ranges.
    y = np.linspace(-0.5, 0.5, 16)[:, None]*6*waist
    x = np.linspace(-0.5, 0.5, 16)[:, None, None]*6*waist
    Er = beam.E(z - z_waist, (x**2 + y**2)**0.5)*np.exp(1j*k*z)
    Er *= np.exp(1j*beam.Gouy(-z_waist)) # Want phase at origin to be zero for comparison below.

    # Define a set of PBGs of different angles with same waist plane as Er.
    origin = rt1.stack_xyzw(0, 0, 0, 1)
    thetay = np.linspace(-0.5, 0.5, 3)[:, None]*6*lamb/(np.pi*waist)
    thetax = np.linspace(-0.5, 0.5, 3)[:, None, None]*6*lamb/(np.pi*waist)
    vector = rt1.normalize(rt1.concatenate_xyzw(thetax, thetay, 1, 0))
    axis1 = rt1.normalize(rt1.stack_xyzw(1, 0, 0, 0))
    mode_bundle = pbg.Mode.make(otk.rt1.lines.Line(origin, vector), axis1, lamb, waist, z_waist)[0]

    flux_, phi_axis_origin_, projected_field = mode_bundle.project_field(k, rt1.concatenate_xyzw(x, y, z, 1), Er)
    coefficients = flux_**0.5 * np.exp(1j*phi_axis_origin_)
    testing.assert_allclose(coefficients, np.asarray([[0, 0, 0], [0, flux**0.5, 0], [0, 0, 0]])[:, :, None], atol=1e-10)

# TODO restore me
# def test_pbg_collimate(qtbot):
#     lamb = 860e-9
#     waist = 10e-6
#     n1 = 1.5
#     n2 = 3
#     f = 0.1
#     surface = rt.Surface(rt.SphericalProfile(f*(n2 - n1)), interface=rt.FresnelInterface(ri.FixedIndex(n1), ri.FixedIndex(n2)))
#     line_base = otk.rt.lines.Line([0, 0, -f*n1, 1], [0, 0, 1, 0])
#     beam0 = pbg.Beam.make(line_base, [1, 0, 0, 0], lamb, waist, [1,0,0,0], 1, n=n1)
#     surfaces = (surface, )
#     segments = pbg.trace_surfaces(surfaces, ['transmitted'], beam0)
#     assert len(segments) == 2
#     beam1 = segments[1].beam
#     assert beam1.n == n2
#     assert np.isclose(beam1.flux, abs(omath.calc_fresnel_coefficients(n1, n2, 1)[1][0])**2*n2/n1)
#     for surface_index in range(2): # TODO fix this
#         widget = segments[-1].make_profile_widget(0)
#         qtbot.addWidget(widget)