import pyqtgraph_extended as pg

from .. import asbp


def plot_projection(profile, projected):
    residual = profile.Er - projected

    glw = pg.GraphicsLayoutWidget()
    field_plot = glw.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Field')
    field_image = asbp.make_Er_image_item(profile.rs_support, profile.Er, profile.rs_center, 'amplitude')
    field_plot.addItem(field_image)
    glw.addHorizontalSpacer()

    projected_plot = glw.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Projected')
    projected_plot.setXYLink(field_plot)
    projected_image = asbp.make_Er_image_item(profile.rs_support, projected, profile.rs_center, 'amplitude')
    projected_plot.addItem(projected_image)
    glw.addHorizontalSpacer()
    glw.addColorBar(images=(field_image, projected_image), rel_row=2)
    glw.addHorizontalSpacer()

    residual_plot = glw.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'}, title='Residual')
    residual_image = asbp.make_Er_image_item(profile.rs_support, residual, profile.rs_center, 'amplitude')
    residual_plot.addItem(residual_image)
    residual_plot.setXYLink(projected_plot)
    glw.addColorBar(image=residual_image, rel_row=2)

    glw.resize(1200, 360)
    glw.show()
    return glw
