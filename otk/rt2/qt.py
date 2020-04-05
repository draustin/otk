import numpy as np
from typing import Mapping, Sequence, Iterable
from PyQt5 import QtWidgets, QtCore
from ..sdb import Surface
from ..sdb.glsl import gen_get_all_recursive
from .scalar import Assembly
from ..sdb.qt import SphereTraceRender, SphereTraceViewer
from ..delegate import Delegate

def view_assembly(a: Assembly, set_properties:Mapping[Surface, Mapping]=None, parent_properties:Mapping=None):
    if set_properties is None:
        set_properties = {}
    if parent_properties is None:
        parent_properties = {}
    parent_properties.setdefault('surface_color', (0, 0, 1))
    parent_properties.setdefault('edge_color', (0, 0, 0))
    parent_properties.setdefault('edge_width', 0.1e-3)
    sdb_glsl = gen_get_all_recursive(a.surface, set_properties, parent_properties)
    viewer = AssemblyViewer(sdb_glsl)
    return viewer

class AssemblyViewer(QtWidgets.QWidget):
    def __init__(self, sdb_glsl: str, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        display_widget = SphereTraceRender([sdb_glsl])

        max_steps = QtWidgets.QSpinBox()
        max_steps.setRange(1, 1000)
        max_steps.setValue(display_widget.max_steps)
        max_steps.valueChanged.connect(self.maxStepsChanged)

        sb = QtWidgets.QDoubleSpinBox()
        self._log10epsilon = sb
        sb.setRange(-20, 2)
        sb.setValue(np.log10(display_widget.epsilon))
        sb.valueChanged.connect(self.log10EpsilonChanged)

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        vbox = QtWidgets.QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addWidget(max_steps)
        vbox.addWidget(self._log10epsilon)
        vbox.addStretch(1)
        hbox.addWidget(display_widget)

        timer = QtCore.QTimer()
        timer.timeout.connect(display_widget.update)
        #timer.start(0)
        self.timer = timer

        self.display_widget = display_widget

    def maxStepsChanged(self, value):
        self.display_widget.max_steps = value

    def log10EpsilonChanged(self, value):
        self.display_widget.epsilon = 10**value

    def sizeHint(self):
        return QtCore.QSize(640, 480)

    def set_rays(self, rays: Sequence[np.ndarray], colors: Iterable = None):
        self.display_widget.set_rays(rays, colors)

    projection = Delegate('display_widget', 'projection')
    eye_to_world = Delegate('display_widget', 'eye_to_world')

    @property
    def max_steps(self) -> int:
        return self.display_widget.max_steps

    @max_steps.setter
    def max_steps(self, v: int):
        self._max_steps.setValue(v)

    @property
    def epsilon(self) -> float:
        return self.display_widget.epsilon

    @epsilon.setter
    def epsilon(self, v: float):
        self._log10epsilon.setValue(np.log10(v))
