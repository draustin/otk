import otk.h4t
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph_extended as pg
import numpy as np
import mathx
from otk import rt1
from otk import asbp, ri, paraxial


class SimpleLensPropagator(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        def make_roc_spin_box():
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setRange(1e-3, 1e6)
            spin_box.setDecimals(3)
            spin_box.setValue(20)
            spin_box.valueChanged.connect(self._update_lens)
            return spin_box

        self.roc1_widget = make_roc_spin_box()
        self.roc2_widget = make_roc_spin_box()

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setValue(40)
        spin_box.valueChanged.connect(self._update_lens)
        self.thickness_widget = spin_box

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setRange(1e-3, 1e6)
        spin_box.setDecimals(3)
        spin_box.setValue(1.4523)
        spin_box.valueChanged.connect(self._update_lens)
        self.n_widget = spin_box

        self.f_widget = QtWidgets.QLabel()
        self.working1_widget = QtWidgets.QLabel()
        self.working2_widget = QtWidgets.QLabel()

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setRange(200, 2000)
        spin_box.setValue(860)
        self.wavelength_widget = spin_box

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setRange(1e-3, 10)
        spin_box.setDecimals(3)
        spin_box.setSingleStep(1e-3)
        spin_box.setValue(0.1)
        self.source_diameter_widget = spin_box

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setValue(0)
        spin_box.setRange(-100, 100)
        self.source_x_widget = spin_box

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setValue(0)
        spin_box.setRange(-100, 100)
        self.source_y_widget = spin_box

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setValue(0)
        spin_box.setRange(-100, 100)
        spin_box.setDecimals(3)
        spin_box.setSingleStep(1e-3)
        self.source_z_widget = spin_box

        button = QtWidgets.QPushButton('Propagate')
        button.clicked.connect(self.propagate)
        self.propagate_widget = button

        combo_box = QtWidgets.QComboBox()
        combo_box.addItem('Angular centroid')
        combo_box.addItem('Peak intensity')
        self.tilt_mode_widget = combo_box
        combo_box.currentIndexChanged.connect(self.update_plots)

        def make_tilt_nudge_spin_box():
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setDecimals(3)
            spin_box.setSingleStep(0.1)
            spin_box.setRange(-100, 100)
            spin_box.valueChanged.connect(self.update_plots)
            return spin_box

        self.tilt_nudge_x_widget = make_tilt_nudge_spin_box()
        self.tilt_nudge_y_widget = make_tilt_nudge_spin_box()

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        def make_glw(title, amplitude_label, domain='r'):
            label_str = {'r': ('x (mm)', 'y (mm)'), 'q': ('kx (rad/mm)', 'ky (rad/mm)')}
            bottom_label, left_label = label_str[domain]
            glw = pg.GraphicsLayoutWidget()
            glw.addLabel(title)
            glw.nextRows()
            gl = glw.addLayout()
            abs_plot = gl.addAlignedPlot(labels={'bottom': bottom_label, 'left': left_label}, title='Amplitude')
            abs_image = abs_plot.image(lut=pg.get_colormap_lut())
            gl.addHorizontalSpacer(10)
            gl.addColorBar(image=abs_image, label=amplitude_label, rel_row=2)
            gl.addHorizontalSpacer(20)
            # gl.nextRows()
            phase_plot = gl.addAlignedPlot(labels={'bottom': bottom_label, 'left': left_label}, title='Phase')
            phase_plot.setXYLink(abs_plot)
            phase_image = phase_plot.image(lut=pg.get_colormap_lut('bipolar'))
            gl.addHorizontalSpacer(10)
            gl.addColorBar(image=phase_image, label='Waves', rel_row=2)
            return glw, (abs_image, phase_image)

        before_glw, self.before_images = make_glw('Before lens', 'Amplitude')
        inside_r_glw, self.inside_r_images = make_glw('Middle of lens', 'Amplitude')
        inside_q_glw, self.inside_q_images = make_glw('Middle of lens - angular spectrum', 'Amplitude', 'q')
        after_glw, self.after_images = make_glw('After lens', 'Amplitude')
        output_r_glw, self.output_r_images = make_glw('Output focal plane', 'Amplitude (normalized to ideal)')
        output_q_glw, self.output_q_images = make_glw('Output focal plane - angular spectrum', 'Amplitude', 'q')

        self.invert_kwargs = dict(max_iterations=20, tol=1e-8)
        self.num_points = 64
        self.m01_factor = 1
        self.m12_factor = 1
        self.m23_factor = 1
        self.kz_mode = False
        self.qfd1 = 1
        self.qfd2 = 1
        self.waist0_factor = 6
        self.center_fraction = 1e-4
        self.kz_mode = 'local_xy'

        self._update_lens()

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(QtWidgets.QLabel('ROC 1 (mm)'))
        hbox.addWidget(self.roc1_widget)
        hbox.addWidget(QtWidgets.QLabel('ROC 2 (mm)'))
        hbox.addWidget(self.roc2_widget)
        hbox.addWidget(QtWidgets.QLabel('Center thickness'))
        hbox.addWidget(self.thickness_widget)
        hbox.addWidget(QtWidgets.QLabel('Refractive index'))
        hbox.addWidget(self.n_widget)
        hbox.addWidget(QtWidgets.QLabel('Focal length (mm)'))
        hbox.addWidget(self.f_widget)
        hbox.addWidget(QtWidgets.QLabel('Working distance 1 (mm)'))
        hbox.addWidget(self.working1_widget)
        hbox.addWidget(QtWidgets.QLabel('Working distance 2(mm)'))
        hbox.addWidget(self.working2_widget)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(QtWidgets.QLabel('Wavelength (nm)'))
        hbox.addWidget(self.wavelength_widget)
        hbox.addWidget(QtWidgets.QLabel('e<sup>-2</sup> diameter (mm)'))
        hbox.addWidget(self.source_diameter_widget)
        hbox.addWidget(QtWidgets.QLabel('X offset (mm)'))
        hbox.addWidget(self.source_x_widget)
        hbox.addWidget(QtWidgets.QLabel('Y offset (mm)'))
        hbox.addWidget(self.source_y_widget)
        hbox.addWidget(QtWidgets.QLabel('Z offset (mm)'))
        hbox.addWidget(self.source_z_widget)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(self.propagate_widget)
        hbox.addWidget(QtWidgets.QLabel('Tilt mode:'))
        hbox.addWidget(self.tilt_mode_widget)
        hbox.addWidget(QtWidgets.QLabel('Tilt nudge x:'))
        hbox.addWidget(self.tilt_nudge_x_widget)
        hbox.addWidget(QtWidgets.QLabel('y:'))
        hbox.addWidget(self.tilt_nudge_y_widget)
        hbox.setAlignment(Qt.AlignLeft)

        tab = QtWidgets.QTabWidget()
        vbox.addWidget(tab)
        tab.addTab(before_glw, 'Before lens')
        tab.addTab(inside_r_glw, 'Middle')
        tab.addTab(inside_q_glw, 'Middle (Angular)')
        tab.addTab(after_glw, 'After lens')
        tab.addTab(output_r_glw, 'Output focal plane')
        tab.addTab(output_q_glw, 'Output focal plane (Angular)')
        tab.setCurrentIndex(4)

    def _update_lens(self):
        n, roc1, roc2, d = [widget.value() for widget in
                            (self.n_widget, self.roc1_widget, self.roc2_widget, self.thickness_widget)]
        f, h1, h2 = paraxial.calc_thick_spherical_lens(n, roc1, -roc2, d)
        w1 = f - h1
        w2 = f + h2
        for widget, value in ((self.f_widget, f), (self.working1_widget, w1), (self.working2_widget, w2)):
            widget.setText('%.3f'%value)
        self.f = f*1e-3
        self.w1 = w1*1e-3
        self.w2 = w2*1e-3

    def propagate(self):
        lamb = self.wavelength_widget.value()*1e-9
        waist0 = self.source_diameter_widget.value()/2*1e-3
        num_points = self.num_points
        rs_waist = np.asarray((self.source_x_widget.value()*1e-3, self.source_y_widget.value()*1e-3))
        source_z = self.source_z_widget.value()*1e-3
        roc1 = self.roc1_widget.value()*1e-3
        roc2 = self.roc2_widget.value()*1e-3
        d = self.thickness_widget.value()*1e-3
        n = ri.FixedIndex(self.n_widget.value())
        # These are calculated by lens update.
        f = self.f
        w1 = self.w1
        w2 = self.w2

        r0_centers = rs_waist
        q0_centers = (0, 0)
        k = 2*np.pi/lamb
        r0_support = waist0*self.waist0_factor  # *#(np.pi*num_points)**0.5*waist0
        x0, y0 = asbp.calc_xy(r0_support, num_points, r0_centers)

        # Create input beam and prepare for propagation to first surface.
        Er0 = asbp.calc_gaussian(k, x0, y0, waist0, r0s=rs_waist)

        profile0 = asbp.PlaneProfile(lamb, 1, source_z, r0_support, Er0, asbp.calc_gradxyE(r0_support, Er0, q0_centers),
            r0_centers, q0_centers)
        b0 = asbp.Beam(profile0)
        s1 = rt1.Surface(rt1.SphericalProfile(roc1), otk.h4t.make_translation(0, 0, w1),
                        interface=rt1.PerfectRefractor(ri.air, n))
        s2 = rt1.Surface(rt1.SphericalProfile(-roc2), otk.h4t.make_translation(0, 0, w1 + d),
                        interface=rt1.PerfectRefractor(n, ri.air))
        s3 = rt1.Surface(rt1.PlanarProfile(), otk.h4t.make_translation(0, 0, w1 + d + w2))



        segments = asbp.trace_surfaces(b0, (s1, s2, s3), ('transmitted', 'transmitted', None))[0]
        b1_incident = segments[0].beams[1]
        b1_refracted = segments[1].beams[0]
        b1_plane = segments[1].planarized_beam
        b2_incident = segments[1].beams[1]
        b2_refracted = segments[2].beams[0]
        b2_plane = segments[2].planarized_beam
        b3 = segments[2].beams[1]
        # b1_incident = b0.propagate(s1, self.kz_mode)
        # b1_refracted = b1_incident.refract(s1, self.kz_mode)
        # b1_plane = b1_refracted.planarize(kz_mode=self.kz_mode, invert_kwargs=self.invert_kwargs)
        # b2_incident = b1_plane.propagate(s2, self.kz_mode)
        # b2_refracted = b2_incident.refract(s2, self.kz_mode)
        # b2_plane = b2_refracted.planarize(kz_mode=self.kz_mode, invert_kwargs=self.invert_kwargs)
        # b3 = b2_plane.propagate(s3, kz_mode=self.kz_mode)

        self.b0 = b0
        self.b1_incident = b1_incident
        self.b1_refracted = b1_refracted
        self.b1_plane = b1_plane
        self.b2_incident = b2_incident
        self.b2_plane = b2_plane
        self.b3 = b3
        waist3 = f*lamb/(np.pi*waist0)
        self.b3_scale = waist3/waist0

        self.update_plots()

    def update_plots(self):
        def plot_r(images, beam, scale=1):
            profile = beam.profile
            xu, yu, Eru = asbp.unroll_r(profile.rs_support, profile.Er*scale, profile.rs_center)
            tilt_mode = self.tilt_mode_widget.currentIndex()
            if not isinstance(beam.profile, asbp.PlaneProfile):
                q_profile = profile.app
            else:
                q_profile = profile
            if tilt_mode == 0:
                q0s = q_profile.centroid_qs_flat
            else:
                q0s = q_profile.peak_qs
            q0s += np.asarray((self.tilt_nudge_x_widget.value(), self.tilt_nudge_y_widget.value()))/1e3*profile.k
            Eru = Eru*mathx.expj(
                -(q0s[0]*(xu - profile.peak_rs[0]) + q0s[1]*(yu - profile.peak_rs[1]) + np.angle(profile.peak_Er)))
            images[0].setImage(abs(Eru))
            images[0].setRect(pg.axes_to_rect(xu*1e3, yu*1e3))
            images[1].setImage(np.angle(Eru)/(2*np.pi))
            images[1].setRect(pg.axes_to_rect(xu*1e3, yu*1e3))

        # def plot_r_waist(images, b):
        #    plot_r_(images, b.r_supports_waist, b.Er_waist, b.r_centers_waist, b.qs_center)

        def plot_q_(images, r_support, Eq, qs_center, rs_center):
            kxu, kyu, Equ = asbp.unroll_q(r_support, Eq, qs_center)
            Equ = Equ*mathx.expj((rs_center[0]*kxu + rs_center[1]*kyu))
            images[0].setImage(abs(Equ))
            images[0].setRect(pg.axes_to_rect(kxu/1e3, kyu/1e3))
            images[1].setImage(np.angle(Equ)/(2*np.pi))
            images[1].setRect(pg.axes_to_rect(kxu/1e3, kyu/1e3))

        def plot_q(images, beam, flat=False):
            profile = beam.profile
            if flat:
                Eq = asbp.fft2(beam.Er_flat)
            else:
                Eq = asbp.fft2(profile.Er)
            plot_q_(images, profile.rs_support, Eq, profile.qs_center, profile.rs_center)

        # def plot_q_waist(images, b):
        #    plot_q_(images, b.r_supports_waist, b.Eq_waist, b.qs_center, b.r_centers_waist)

        plot_r(self.before_images, self.b1_incident)

        # plot_r_waist(self.inside_r_images, b1_after)
        # plot_q_waist(self.inside_q_images, b1_after)
        plot_r(self.after_images, self.b2_plane)
        plot_r(self.output_r_images, self.b3, self.b3_scale)
        plot_q(self.output_q_images,
               self.b3)  # waist3 = f*lamb/(np.pi*waist0)  # Eq3 = asbp.fft2(b3.Er)  # # Subtract numerical center tilt.  # #Er3p = b3.Er*mathx.expj(-x3*b3.qs_center[0] - y3*b3.qs_center[1])*waist3/waist0  # kxm3, kym3 = asbp.calc_kxky_moment(r3_support, mathx.abs_sqd(Eq3), b3.qs_center)  # Er3p = b3.Er*mathx.expj(-x3*kxm3 - y3*kym3)*waist3/waist0  # plot_r(self.output_r_images, r3_support, Er3p, b3.rs_center)  # plot_q(self.output_q_images, r3_support, Eq3, b3.qs_center)
