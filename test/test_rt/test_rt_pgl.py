import otk.rt.lines
from pyqtgraph_extended.opengl import pgl

import otk.rt.raytrace
from otk import rt
from otk.rt import pgl as rtpgl

def test_rt_pgl(qtbot):
    surface = rt.Surface(rt.PlanarProfile())
    line = rt.Line((0, 0, -1, 1), (0, 0, 1, 0))
    segments = otk.rt.raytrace.Ray(line, [1,0,0,0], 0, 860e-9, 1).trace_surfaces((surface,), ['incident'])

    widget = rtpgl.plot_surfaces((surface,))
    qtbot.addWidget(widget)

    item = rtpgl.ParentItem()
    item.add_surface(surface)
    rtpgl.SegmentsItem(segments, item)
    widget = pgl.GLViewWidget()
    widget.addItem(item)
    qtbot.addWidget(widget)
