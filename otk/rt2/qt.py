import numpy as np
from typing import Mapping, Sequence, Iterable
from PyQt5 import QtWidgets, QtCore
from ..sdb import Surface
from ..sdb.glsl import gen_get_all_recursive
from .scalar import Assembly
from ..sdb.qt import SphereTraceRender, SphereTraceViewer

def view_assembly(a: Assembly, set_properties:Mapping[Surface, Mapping]=None, parent_properties:Mapping=None):
    if set_properties is None:
        set_properties = {}
    if parent_properties is None:
        parent_properties = {}
    parent_properties.setdefault('surface_color', (0, 0, 1))
    parent_properties.setdefault('edge_color', (0, 0, 0))
    parent_properties.setdefault('edge_width', 0.1e-3)
    sdb_glsl = gen_get_all_recursive(a.surface, set_properties, parent_properties)
    return AssemblyViewer(sdb_glsl)

class AssemblyViewer(QtWidgets.QWidget):
    def __init__(self, sdb_glsl: str, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        display_widget = SphereTraceRender([sdb_glsl])

        max_steps = QtWidgets.QSpinBox()
        max_steps.setRange(1, 1000)
        max_steps.setValue(display_widget.max_steps)
        max_steps.valueChanged.connect(self.maxStepsChanged)

        sb = QtWidgets.QDoubleSpinBox()
        self.log10epsilon = sb
        sb.setRange(-20, 2)
        sb.setValue(np.log10(display_widget.epsilon))
        sb.valueChanged.connect(self.log10EpsilonChanged)

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        vbox = QtWidgets.QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addWidget(max_steps)
        vbox.addWidget(self.log10epsilon)
        vbox.addStretch(1)
        hbox.addWidget(display_widget)

        timer = QtCore.QTimer()
        timer.timeout.connect(display_widget.update)
        #timer.start(0)
        self.timer = timer

        self.display_widget = display_widget

    def maxStepsChanged(self, value):
        self.display_widget.max_steps = value
        self.display_widget.update()

    def log10EpsilonChanged(self, value):
        self.display_widget.epsilon = 10**value
        self.display_widget.update()

    def set_rays(self, rays: Sequence[np.ndarray], colors: Iterable = None):
        self.display_widget.set_rays(rays, colors)
