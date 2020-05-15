import pytest
import otk.rt1.lines
from pyqtgraph_extended.opengl import pgl

import otk.rt1.raytrace
from otk import rt1
from otk.rt1 import pgl as rtpgl

@pytest.mark.skip
def test_rt_pgl(qtbot):
    surface = rt1.Surface(rt1.PlanarProfile())
    line = rt1.Line((0, 0, -1, 1), (0, 0, 1, 0))
    segments = otk.rt1.raytrace.Ray(line, [1,0,0,0], 0, 860e-9, 1).trace_surfaces((surface,), ['incident'])

    widget = rtpgl.plot_surfaces((surface,))
    qtbot.addWidget(widget)

    item = rtpgl.ParentItem()
    item.add_surface(surface)
    rtpgl.SegmentsItem(segments, item)
    widget = pgl.GLViewWidget()
    widget.addItem(item)
    qtbot.addWidget(widget)
