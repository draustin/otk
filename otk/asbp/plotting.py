import numpy as np
from PyQt5 import QtCore
import pyqtgraph_extended as pg
from . import sa, math

def make_Er_image_item(r_support, Er, rs_center=(0, 0), quantity='waves'):
    x, y, Eru = sa.unroll_r(r_support, Er, rs_center)
    if quantity == 'amplitude':
        data = abs(Eru)
        lut = pg.get_colormap_lut()
        levels = 0, data.max()
    elif quantity == 'waves':
        data = np.angle(Eru)/(2*np.pi)
        lut = pg.get_colormap_lut('bipolar')
        levels = -0.5, 0.5
    item = pg.ImageItem(data, lut=lut)
    item.setRect(pg.axes_to_rect(x*1e3, y*1e3))
    item.setLevels(levels)
    return item

def set_Er_image_item(item, r_support, Er, rs_center=(0, 0), quantity='waves'):
    x, y, Eru = sa.unroll_r(r_support, Er, rs_center)
    if quantity == 'amplitude':
        data = abs(Eru)
        levels = 0, data.max()
    elif quantity == 'waves':
        data = np.angle(Eru)/(2*np.pi)
        levels = -0.5, 0.5
    else:
        raise ValueError('Unknown quantity %s.', quantity)
    item.setImage(data)
    item.setRect(pg.axes_to_rect(x*1e3, y*1e3))
    item.setLevels(levels)
    return item

def make_Eq_image_item(r_support, Eq, qs_center=(0, 0), quantity='waves'):
    kx, ky, Equ = sa.unroll_q(r_support, Eq, qs_center)
    if quantity == 'amplitude':
        data = abs(Equ)
        lut = pg.get_colormap_lut()
        levels = 0, data.max()
    elif quantity == 'waves':
        data = np.angle(Equ)/(2*np.pi)
        lut = pg.get_colormap_lut('bipolar')
        levels = -0.5, 0.5
    item = pg.ImageItem(data, lut=lut)
    item.setRect(pg.axes_to_rect(kx/1e3, ky/1e3))
    item.setLevels(levels)
    return item

def set_Eq_image_item(item, r_support, Eq, qs_center=(0, 0), quantity='waves'):
    kx, ky, Equ = sa.unroll_q(r_support, Eq, qs_center)
    if quantity == 'amplitude':
        data = abs(Equ)
        levels = 0, data.max()
    elif quantity == 'waves':
        data = np.angle(Equ)/(2*np.pi)
        levels = -0.5, 0.5
    else:
        raise ValueError('Unknown quantity %s.', quantity)
    item.setImage(data)
    item.setRect(pg.axes_to_rect(kx/1e3, ky/1e3))
    item.setLevels(levels)
    return item


def plot_r_q_polar(rs_support, Er, rs_center=(0, 0), qs_center=(0, 0), gl=None, Eq=None):
    if gl is None:
        glw = pg.GraphicsLayoutWidget()
        plots = plot_r_q_polar(rs_support, Er, rs_center, qs_center, glw.ci, Eq)
        glw.resize(830, 675)
        glw.show()
        return glw, plots
    else:
        if Eq is None:
            Eq = math.fft2(Er)
        absEr_plot = gl.addAlignedPlot(labels={'left':'y (mm)', 'bottom':'x (mm)'}, title='Real space amplitude')
        image = make_Er_image_item(rs_support, Er, rs_center, 'amplitude')
        absEr_plot.addItem(image)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Amplitude')
        gl.addHorizontalSpacer(10)
        wavesr_plot = gl.addAlignedPlot(labels={'left':'y (mm)', 'bottom':'x (mm)'}, title='Real space wavefront')
        image = make_Er_image_item(rs_support, Er, rs_center, 'waves')
        wavesr_plot.addItem(image)
        wavesr_plot.setXYLink(absEr_plot)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Waves')
        gl.nextRows()
        absEq_plot = gl.addAlignedPlot(labels={'left':'ky (rad/mm)', 'bottom':'kx (rad/mm)'}, title='Angular space amplitude')
        image = make_Eq_image_item(rs_support, Eq, qs_center, 'amplitude')
        absEq_plot.addItem(image)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Amplitude')
        gl.addHorizontalSpacer(10)
        wavesq_plot = gl.addAlignedPlot(labels={'left':'ky (rad/mm)', 'bottom':'kx (rad/mm)'},
                                       title='Angular space wavefront')
        image = make_Eq_image_item(rs_support, Eq, qs_center, 'waves')
        wavesq_plot.addItem(image)
        wavesq_plot.setXYLink(absEq_plot)
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Waves')
        return sa.RQ(sa.AbsPhase(absEr_plot, wavesr_plot), sa.AbsPhase(absEq_plot, wavesq_plot))

def add_r_q_polar_scatter(plots, rs, qs, **kwargs):
    for plot in plots.r:
        item = pg.ScatterPlotItem(rs[0]*1e3, rs[1]*1e3, **kwargs)
        plot.addItem(item)

    for plot in plots.q:
        item = pg.ScatterPlotItem(qs[0]/1e3, qs[1]/1e3, **kwargs)
        plot.addItem(item)


def plot_projection(profile, projected, residual):
    glw = pg.GraphicsLayoutWidget()
    field_plot = glw.addAlignedPlot(labels={'left':'y (mm)', 'bottom':'x (mm)'}, title='Field')
    field_image = make_Er_image_item(profile.rs_support, profile.Er, profile.rs_center, 'amplitude')
    field_plot.addItem(field_image)
    glw.addHorizontalSpacer()

    projected_plot = glw.addAlignedPlot(labels={'left':'y (mm)', 'bottom':'x (mm)'}, title='Projected')
    projected_plot.setXYLink(field_plot)
    projected_image = make_Er_image_item(profile.rs_support, projected, profile.rs_center, 'amplitude')
    projected_plot.addItem(projected_image)
    glw.addHorizontalSpacer()
    glw.addColorBar(images=(field_image, projected_image), rel_row=2)
    glw.addHorizontalSpacer()

    residual_plot = glw.addAlignedPlot(labels={'left':'y (mm)', 'bottom':'x (mm)'}, title='Residual')
    residual_image = make_Er_image_item(profile.rs_support, residual, profile.rs_center, 'amplitude')
    residual_plot.addItem(residual_image)
    residual_plot.setXYLink(projected_plot)
    glw.addColorBar(image=residual_image, rel_row=2)

    glw.resize(1200, 360)
    glw.show()
    return glw
