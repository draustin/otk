from collections import namedtuple
import numpy as np
import pyqtgraph_extended as pg
from PyQt5 import QtWidgets, Qt
import mathx
from . import sa, profiles, plotting
from .. import rt1 as rt

scatter_kwargs = dict(pen=None, brush='g', size=4)


class ProfileWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.profile = None
        self.line = None
        self.items = []

    @classmethod
    def make_absEr(cls, gl):
        plot = gl.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Real space amplitude')
        image = pg.ImageItem(lut=pg.get_colormap_lut())
        plot.addItem(image)
        scatter = pg.ScatterPlotItem(**scatter_kwargs)
        plot.addItem(scatter)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Amplitude')
        return plot, image, scatter

    @classmethod
    def make_absEq(cls, gl):
        plot = gl.addAlignedPlot(labels={'left': 'ky (rad/mm)', 'bottom': 'kx (rad/mm)'},
            title='Angular space amplitude')
        image = pg.ImageItem(lut=pg.get_colormap_lut())
        plot.addItem(image)
        scatter = pg.ScatterPlotItem(**scatter_kwargs)
        plot.addItem(scatter)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Amplitude')
        return plot, image, scatter

    @classmethod
    def make_wavesr(cls, gl, absEr_plot):
        plot = gl.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Real space wavefront')
        image = pg.ImageItem(lut=pg.get_colormap_lut('bipolar'))
        plot.addItem(image)
        plot.setXYLink(absEr_plot)
        scatter = pg.ScatterPlotItem(**scatter_kwargs)
        plot.addItem(scatter)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Waves')
        return plot, image, scatter

    @classmethod
    def make_wavesq(cls, gl, absEq_plot):
        plot = gl.addAlignedPlot(labels={'left': 'ky (rad/mm)', 'bottom': 'kx (rad/mm)'},
            title='Angular space wavefront')
        image = pg.ImageItem(lut=pg.get_colormap_lut('bipolar'))
        plot.addItem(image)
        plot.setXYLink(absEq_plot)
        scatter = pg.ScatterPlotItem(**scatter_kwargs)
        plot.addItem(scatter)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Waves')
        return plot, image, scatter

    def set_profile(self, profile):
        self.profile = profile
        self._update()

    def set_line(self, line):
        self.line = line
        self._update()

    def set_items(self, items):
        for item in self.items:
            item.hide()
        for item in items:
            self.plots.r.abs.addItem(item)
            item.show()
        self.items = items


class PlaneProfileWidget(ProfileWidget):
    def __init__(self, parent=None):
        ProfileWidget.__init__(self, parent)

        combo_box = QtWidgets.QComboBox()
        self.field_combo_box = combo_box
        combo_box.addItem('true')
        combo_box.addItem('flattened')
        combo_box.currentIndexChanged.connect(self._update)

        check_box = QtWidgets.QCheckBox('Remove tilt')
        self.remove_tilt_check_box = check_box
        check_box.setChecked(True)
        check_box.stateChanged.connect(self._update)

        check_box = QtWidgets.QCheckBox('Remove constant')
        self.remove_constant_check_box = check_box
        check_box.setChecked(True)
        check_box.stateChanged.connect(self._update)

        glw = pg.GraphicsLayoutWidget()
        self.label = glw.addLabel()
        glw.nextRow()
        gl = glw.addLayout()
        absEr_plot, absEr_image, absEr_scatter = self.make_absEr(gl)
        gl.addHorizontalSpacer(10)
        absEq_plot, absEq_image, absEq_scatter = self.make_absEq(gl)
        gl.nextRows()
        wavesr_plot, wavesr_image, wavesr_scatter = self.make_wavesr(gl, absEr_plot)
        gl.addHorizontalSpacer(10)
        wavesq_plot, wavesq_image, wavesq_scatter = self.make_wavesr(gl, absEq_plot)

        self.plots = sa.RQ(sa.AbsPhase(absEr_plot, wavesr_plot), sa.AbsPhase(absEq_plot, wavesq_plot))
        self.images = sa.RQ(sa.AbsPhase(absEr_image, wavesr_image), sa.AbsPhase(absEq_image, wavesq_image))
        self.scatters = sa.RQ(sa.AbsPhase(absEr_scatter, wavesr_scatter), sa.AbsPhase(absEq_scatter, wavesq_scatter))

        # Place widgets.
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(QtWidgets.QLabel('Field:'))
        hbox.addWidget(self.field_combo_box)
        hbox.addWidget(QtWidgets.QLabel('Phase:'))
        hbox.addWidget(self.remove_tilt_check_box)
        hbox.addWidget(self.remove_constant_check_box)
        hbox.setAlignment(Qt.Qt.AlignLeft)
        vbox.addWidget(glw)

    def _update(self):
        p = self.profile
        field_index = self.field_combo_box.currentIndex()
        if p is not None:
            if field_index == 0:
                Er = p.Er
                Eq = p.Eq
            elif field_index == 1:
                Er = p.Er_flat
                Eq = p.Eq_flat
            else:
                raise ValueError('Unknown mode %d.', field_index)

            if self.remove_tilt_check_box.isChecked():
                Er = Er*mathx.expj(-(p.qs_center[0]*(p.x - p.rs_center[0]) + p.qs_center[1]*(p.y - p.rs_center[1])))
                Eq = Eq*mathx.expj(p.rs_center[0]*(p.kx - p.qs_center[0]) + p.rs_center[1]*(p.ky - p.qs_center[1]))

            if self.remove_constant_check_box.isChecked():
                Er = Er*mathx.expj(-np.angle(Er[p.r_center_indices]))
                Eq = Eq*mathx.expj(-np.angle(Eq[p.q_center_indices]))

            plotting.set_Er_image_item(self.images.r.abs, p.rs_support, Er, p.rs_center, 'amplitude')
            plotting.set_Er_image_item(self.images.r.phase, p.rs_support, Er, p.rs_center, 'waves')
            plotting.set_Eq_image_item(self.images.q.abs, p.rs_support, Eq, p.qs_center, 'amplitude')
            plotting.set_Eq_image_item(self.images.q.phase, p.rs_support, Eq, p.qs_center, 'waves')

            title = p.title_str
            idx = title.find('qs_center')
            self.label.setText('<br>'.join((title[:idx], title[idx:])))
        else:
            for images_ in self.images:
                for image in images_:
                    image.setImage(None)
            self.label.setText('')

        if self.line is None or p is None:
            x = np.asarray([])
            y = np.asarray([])
            kx = np.asarray([])
            ky = np.asarray([])
        else:
            point = self.line.origin
            vector = self.line.vector
            x = point[..., 0].ravel()
            y = point[..., 1].ravel()
            kx = vector[..., 0].ravel()*p.k
            ky = vector[..., 1].ravel()*p.k

            if field_index in (1, 2):
                # Remove spherical wavefront from transverse wavenumbers.
                kx -= (x - p.rs_center[0])*p.k/p.rocs[0]
                ky -= (y - p.rs_center[1])*p.k/p.rocs[1]

        self.scatters.r.abs.setData(x*1e3, y*1e3)
        self.scatters.r.phase.setData(x*1e3, y*1e3)
        self.scatters.q.abs.setData(kx/1e3, ky/1e3)
        self.scatters.q.phase.setData(kx/1e3, ky/1e3)


class CurvedProfileWidget(ProfileWidget):
    def __init__(self, parent=None):
        ProfileWidget.__init__(self, parent)

        combo_box = QtWidgets.QComboBox()
        self.field_combo_box = combo_box
        combo_box.addItem('true')
        combo_box.addItem('approximate planar')
        combo_box.addItem('approximate planar flattened')
        combo_box.currentIndexChanged.connect(self._update)

        check_box = QtWidgets.QCheckBox('Remove tilt')
        self.remove_tilt_check_box = check_box
        check_box.setChecked(True)
        check_box.stateChanged.connect(self._update)

        check_box = QtWidgets.QCheckBox('Remove constant')
        self.remove_constant_check_box = check_box
        check_box.setChecked(True)
        check_box.stateChanged.connect(self._update)

        glw = pg.GraphicsLayoutWidget()
        self.label = glw.addLabel()
        glw.nextRow()
        gl = glw.addLayout()
        absEr_plot, absEr_image, absEr_scatter = self.make_absEr(gl)
        gl.addHorizontalSpacer(10)
        wavesr_plot, wavesr_image, wavesr_scatter = self.make_wavesr(gl, absEr_plot)
        gl.addHorizontalSpacer(10)
        self.dz_plot = gl.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'})
        self.dz_image = pg.ImageItem(lut=pg.get_colormap_lut())
        self.dz_plot.addItem(self.dz_image)
        self.dz_plot.setXYLink(wavesr_plot)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=self.dz_image, rel_row=2, label='Relative z (mm)')

        glw.nextRows()
        gl = glw.addLayout()
        absEq_plot, absEq_image, absEq_scatter = self.make_absEq(gl)
        gl.addHorizontalSpacer(10)
        wavesq_plot, wavesq_image, wavesq_scatter = self.make_wavesr(gl, absEq_plot)

        self.plots = sa.RQ(sa.AbsPhase(absEr_plot, wavesr_plot), sa.AbsPhase(absEq_plot, wavesq_plot))
        self.images = sa.RQ(sa.AbsPhase(absEr_image, wavesr_image), sa.AbsPhase(absEq_image, wavesq_image))
        self.scatters = sa.RQ(sa.AbsPhase(absEr_scatter, wavesr_scatter), sa.AbsPhase(absEq_scatter, wavesq_scatter))

        # Place widgets.
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        hbox.addWidget(QtWidgets.QLabel('Mode:'))
        hbox.addWidget(self.field_combo_box)
        hbox.addWidget(QtWidgets.QLabel('Phase:'))
        hbox.addWidget(self.remove_tilt_check_box)
        hbox.addWidget(self.remove_constant_check_box)
        hbox.setAlignment(Qt.Qt.AlignLeft)
        vbox.addWidget(glw)

    def _update(self):
        p = self.profile
        if p is not None:
            # Set relative z plot.
            x, y, z = sa.unroll_r(p.rs_support, p.z, p.rs_center)
            self.dz_image.setImage((z - p.mean_z)*1e3)
            self.dz_image.setRect(pg.axes_to_rect(x, y, 1e3))

            field_index = self.field_combo_box.currentIndex()
            if field_index == 0:
                Er = p.Er
                Eq = p.Eq
            elif field_index == 1:
                Er = p.app.Er
                Eq = p.app.Eq
            elif field_index == 2:
                Er = p.app.Er_flat
                Eq = p.app.Eq_flat
            else:
                raise ValueError('Unknown mode %d.', field_index)

            if self.remove_tilt_check_box.isChecked():
                Er = Er*mathx.expj(-(p.qs_center[0]*(p.x - p.rs_center[0]) + p.qs_center[1]*(p.y - p.rs_center[1])))
                Eq = Eq*mathx.expj(
                    p.rs_center[0]*(p.app.kx - p.app.qs_center[0]) + p.app.rs_center[1]*(p.app.ky - p.app.qs_center[1]))

            if self.remove_constant_check_box.isChecked():
                Er = Er*mathx.expj(-np.angle(Er[p.r_center_indices]))
                Eq = Eq*mathx.expj(-np.angle(Eq[p.app.q_center_indices]))

            plotting.set_Er_image_item(self.images.r.abs, p.rs_support, Er, p.rs_center, 'amplitude')
            plotting.set_Er_image_item(self.images.r.phase, p.rs_support, Er, p.rs_center, 'waves')
            plotting.set_Eq_image_item(self.images.q.abs, p.rs_support, Eq, p.qs_center, 'amplitude')
            plotting.set_Eq_image_item(self.images.q.phase, p.rs_support, Eq, p.qs_center, 'waves')

            title = p.title_str
            idx = title.find('qs_center')
            self.label.setText('<br>'.join((title[:idx], title[idx:])))
        else:
            for images_ in self.images:
                for image in images_:
                    image.setImage(None)
            self.label.setText('')

        if self.line is None or p is None:
            x = np.asarray([])
            y = np.asarray([])
            kx = np.asarray([])
            ky = np.asarray([])
        else:
            point = self.line.origin
            vector = self.line.vector
            # Scatter plot item wants 1D arrays.
            x, y = rt1.to_xy(point.reshape((-1, 4)))
            kx, ky = rt1.to_xy(vector.reshape((-1, 4)))*p.k

            if field_index in (1, 2):
                # Remove spherical wavefront from transverse wavenumbers.
                kx -= (x - p.rs_center[0])*p.k/p.rocs[0]
                ky -= (y - p.rs_center[1])*p.k/p.rocs[1]

        self.scatters.r.abs.setData(x*1e3, y*1e3)
        self.scatters.r.phase.setData(x*1e3, y*1e3)
        self.scatters.q.abs.setData(kx/1e3, ky/1e3)
        self.scatters.q.phase.setData(kx/1e3, ky/1e3)


def make_profile_widget(profile):
    if isinstance(profile, profiles.PlaneProfile):
        widget = PlaneProfileWidget()
    else:
        widget = CurvedProfileWidget()
    widget.set_profile(profile)
    widget.show()
    return widget


class PlaneCurvedProfileWidget(QtWidgets.QStackedWidget):
    def __init__(self, parent=None):
        QtWidgets.QStackedWidget.__init__(self, parent)
        self.widgets = PlaneProfileWidget(), CurvedProfileWidget()
        self.addWidget(self.widgets[0])
        self.addWidget(self.widgets[1])

    def set_profile(self, p):
        if isinstance(p, profiles.PlaneProfile):
            index = 0
        elif isinstance(p, profiles.CurvedProfile):
            index = 1
        else:
            raise ValueError('Unknown profile type %s.', type(p))

        self.setCurrentIndex(index)
        self.widgets[index].set_profile(p)

    def set_line(self, line):
        for widget in self.widgets:
            widget.set_line(line)

    def set_items(self, items):
        self.widgets[self.currentIndex()].set_items(items)


class MultiProfileWidget(QtWidgets.QWidget):
    """Widget for showing one of a list of profiles, which the user selects from a list."""

    """Defines an entry, one of which is selected by the user and displayed.
    
    Attributes:
        string (str): Shown in the list widget for selection.
        profile (Profile): To be displayed. Its type (plane or curved) determines the widget type.
        line (rt.Line): Optional ray lines to be added to plot.
        items (sequence of pyqtgraph.PlotItem): Annotations, added to real space amplitude plot.    
    """
    Entry = namedtuple('Entry', ('string', 'profile', 'line', 'items'))

    def __init__(self, entries, parent=None):
        self.entries = entries

        QtWidgets.QWidget.__init__(self, parent)
        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setMaximumWidth((max(len(e.string) for e in entries) + 4)*6)
        hbox.addWidget(self.list_widget)
        for e in entries:
            QtWidgets.QListWidgetItem(e.string, self.list_widget)
        self.list_widget.currentRowChanged.connect(self._set_index)
        self.profile_widget = PlaneCurvedProfileWidget()
        hbox.addWidget(self.profile_widget)
        self._set_index(0)

        self.resize(1000, 550)

    def _set_index(self, index):
        entry = self.entries[index]
        self.profile_widget.set_profile(entry.profile)
        self.profile_widget.set_line(entry.line)
        self.profile_widget.set_items(entry.items)

    def set_index(self, index):
        self.list_widget.setCurrentRow(index)

    @classmethod
    def plot_segments(cls, segments):
        """Make widget allowing user to inspect profiles of all beams in a sequence of segments.

        Args:
            segments (sequence of asbp.BeamSegment): Segments to plot. Each beam

        Returns:
            MultiProfileWidget: One profile entry for every beam in all the segments.
        """
        sbrs = [sbr for segment in segments for sbr in segment.flatten()]

        def make_entry(surface, beam, ray):
            name = beam.segment_string
            if surface is not None:
                name = surface.name + ' ' + name

            if ray is not None:
                ray_local = ray.line.transform(beam.inverse_matrix)
            else:
                ray_local = None
            try:
                make_pg2d_items = surface.make_pg2d_items
            except AttributeError:
                items = ()
            else:
                items = make_pg2d_items(beam.inverse_matrix)
            return cls.Entry(name, beam.profile, ray_local, items)

        entries = [make_entry(*sbr) for sbr in sbrs]
        widget = cls(entries)
        widget.show()
        return widget

# class BeamTraceWidget(QtWidgets.QWidget):
#     def __init__(self, beam_trace, parent=None):
#         self.beam_trace = beam_trace
#
#         QtWidgets.QWidget.__init__(self, parent)
#         hbox = QtWidgets.QHBoxLayout()
#         self.setLayout(hbox)
#         self.list_widget = QtWidgets.QListWidget()
#         self.list_widget.setMaximumWidth((max(len(beam_trace.lookup_beam_name(i)) for i in range(beam_trace.num_beams)) + 4)*6)
#         hbox.addWidget(self.list_widget)
#         for index in range(self.beam_trace.num_beams):
#             name = '%d. %s'%(index, beam_trace.lookup_beam_name(index))
#             QtWidgets.QListWidgetItem(name, self.list_widget)
#         self.list_widget.currentRowChanged.connect(self._set_index)
#         self.profile_widget = ProfileWidget()
#         hbox.addWidget(self.profile_widget)
#         self._set_index(0)
#
#         self.resize(1000, 550)
#
#     def _set_index(self, index):
#         self.current_beam = self.beam_trace.lookup_beam_info(index)[0].beam
#         self.profile_widget.set_profile(self.current_beam.profile)

# class BeamRayTraceWidget(BeamTraceWidget):
#     def __init__(self, beam_ray_trace, parent=None):
#         self.beam_ray_trace = beam_ray_trace
#         BeamTraceWidget.__init__(self, beam_ray_trace.beam_trace, parent)
#         self.ray_trace = beam_ray_trace.ray_trace
#
#     def _set_index(self, index):
#         BeamTraceWidget._set_index(self, index)
#         line, beam = self.beam_ray_trace.lookup_beam_ray(index)
#         if line is None:
#             self.profile_widget.set_ray_bundle(None)
#         else:
#             ray_bundle_local = line.transform(beam.inverse_matrix)
#             self.profile_widget.set_ray_bundle(ray_bundle_local)
# # Convert rt point to beam local space and set spots and convert rt vector to transverse wavenumber.
# x, y = beam.to_local(point)[:2, :]
# kx, ky = beam.to_local(vector)[:2, :]*beam.profile.k

# # Remove spherical wavefront from transverse wavenumbers.
# kx -= (x - beam.profile.rs_center[0])*beam.profile.k/beam.profile.rocs[0]
# ky -= (y - beam.profile.rs_center[1])*beam.profile.k/beam.profile.rocs[1]

# Update plot.
