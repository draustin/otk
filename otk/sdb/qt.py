import time
from typing import Sequence, Iterable, Tuple
import numpy as np
from OpenGL import GL
from PyQt5.QtCore import QPoint
from PyQt5 import QtWidgets, QtCore, QtGui
from otk.types import Matrix4
from otk.h4t import make_x_rotation, make_y_rotation, make_translation
from .._utility import Delegate
from . import *
from . import opengl
from ..qt import application

__all__ = ['SphereTraceRender', 'SphereTraceViewer', 'Scene', 'ScenesViewer', 'application']

class SphereTraceRender(QtWidgets.QOpenGLWidget):
    def __init__(self, sdb_glsls:Sequence[str], parent=None):
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        self.start_time = time.time()
        self._max_steps = 100
        self._epsilon = 1e-2
        self.sdb_glsls = sdb_glsls
        self.rays = []
        self.ray_colors = []
        self.ray_program = None
        self.wireframe_program = None
        self.wireframe_models = []
        surface_format = QtGui.QSurfaceFormat()
        surface_format.setDepthBufferSize(24)
        self.setFormat(surface_format)
        self._projection = Orthographic(0.5, 100)
        self.program_num = 0
        self._eye_to_world = np.eye(4)
        self.background_color = 1., 1., 1., 1.

        self.ndc_orbit = None
        self.mouse_drag_mode = None

    @property
    def eye_to_world(self):
        return self._eye_to_world

    @eye_to_world.setter
    def eye_to_world(self, m):
        m = np.array(m, float)
        assert m.shape == (4, 4)
        self._eye_to_world = m
        self.update()

    @property
    def projection(self) -> Projection:
        return self._projection

    @projection.setter
    def projection(self, v: Projection):
        self._projection = v
        self.update()

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @max_steps.setter
    def max_steps(self, v:int):
        self._max_steps = v
        self.update()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, v: float):
        self._epsilon = v
        self.update()

    def set_rays(self, rays: Sequence[np.ndarray], colors: Iterable = None):
        self.rays = rays
        self.ray_colors = colors
        if self.ray_program is not None:
            self.ray_program.set_rays(rays, colors)
        self.update()

    def set_wireframe_models(self, models:Sequence[render.WireframeModel]):
        self.wireframe_models = models
        if self.wireframe_program is not None:
            self.wireframe_program.set_models(models)
        self.update()

    def initializeGL(self) -> None:
        self.trace_programs = [opengl.make_sphere_trace_program(s) for s in self.sdb_glsls]
        self.ray_program = opengl.make_ray_program(10000)
        self.ray_program.set_rays(self.rays, self.ray_colors)
        self.wireframe_program = opengl.WireframeProgram.make()
        self.wireframe_program.set_models(self.wireframe_models)

        #print(self.trace_program.program, self.ray_program.program)

    def resizeGL(self, width, height):
        # Decided to put everything in paintGL unless performance dictates otherwise.
        pass

    def paintGL(self):
        x0, y0, width, height = GL.glGetFloatv(GL.GL_VIEWPORT)
        # https://stackoverflow.com/questions/39455504/qt-mainwindow-with-qopenglwidget-in-retina-display-displays-wrong-size
        #ratio = self.devicePixelRatio()
        #width *= ratio
        #height *= ratio

        #half_height = self.half_width*height/width

        eye_to_clip = self._projection.eye_to_clip(height/width)
        #-self.half_width, self.half_width, -half_height, half_height, self.z_near, self.z_far)

        self.clip_to_eye = np.linalg.inv(eye_to_clip)

        elapsed_time = time.time() - self.start_time

        clip_to_world = self.clip_to_eye.dot(self._eye_to_world)
        world_to_clip = np.linalg.inv(clip_to_world)

        # Took some experimentation to get right.
        # Must enable depth testing even though we don't want it for first pass. See
        # https://community.khronos.org/t/does-depth-buffer-allow-writing-into-when-disabling-depth-test/14100
        # Don't need to clear depth buffer since first pass writes every fragment.
        GL.glDepthMask(GL.GL_TRUE) # Defensive - appears to be default.
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_ALWAYS)
        self.trace_programs[self.program_num].draw(self.eye_to_world, eye_to_clip, self._max_steps,
            self._epsilon, self.background_color)
        GL.glDepthFunc(GL.GL_LESS)
        self.ray_program.draw(world_to_clip)
        self.wireframe_program.draw(world_to_clip)

    def pos_to_ndc(self, pos: QPoint) -> np.ndarray:
        ratio = self.devicePixelRatio()
        x_window = pos.x()*ratio
        y_window = (self.height() - pos.y())*ratio
        self.makeCurrent()
        viewport_x0, viewport_y0, viewport_width, viewport_height = GL.glGetFloatv(GL.GL_VIEWPORT)
        depth_range_near, depth_range_far = GL.glGetFloatv(GL.GL_DEPTH_RANGE)
        z_window = GL.glReadPixels(x_window, y_window, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT).item()
        # https://www.khronos.org/opengl/wiki/Compute_eye_space_from_window_space
        ndc = np.asarray((2*(x_window - viewport_x0)/viewport_width - 1, 2*(y_window - viewport_y0)/viewport_height - 1,
        (2*z_window - (depth_range_far + depth_range_near))/(depth_range_far - depth_range_near), 1))
        return ndc

    def ndc_to_eye(self, ndc: np.ndarray) -> np.ndarray:
        eyep = ndc.dot(self.clip_to_eye)
        eye = eyep/eyep[3]
        return eye

    def pos_to_eye(self, pos: QPoint):
        ndc = self.pos_to_ndc(pos)
        eye = self.ndc_to_eye(ndc)
        return eye

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mouse_drag_mode is None:
            self.mouse_press_pos = event.pos()
            self.mouse_press_eye_to_world = self.eye_to_world.copy()
            self.mouse_press_ndc = self.pos_to_ndc(event.pos())
            self.mouse_press_eye = self.ndc_to_eye(self.mouse_press_ndc)
            if event.button() == QtCore.Qt.RightButton:
                self.mouse_drag_mode = 'orbit'
            elif event.button() == QtCore.Qt.LeftButton:
                self.mouse_drag_mode = 'pan'

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        rel_pos = (event.pos() - self.mouse_press_pos)
        if self.mouse_drag_mode == 'orbit':
            phi = -rel_pos.x()/self.width()*4*np.pi
            theta = -rel_pos.y()/self.height()*4*np.pi
            transform = make_translation(*(-self.mouse_press_eye[:3])).dot(make_y_rotation(phi)).dot(make_x_rotation(theta)).dot(make_translation(*(self.mouse_press_eye[:3])))
            self.eye_to_world = transform.dot(self.mouse_press_eye_to_world)
            self.update()
        elif self.mouse_drag_mode == 'pan':
            ndc = self.pos_to_ndc(event.pos())
            # If the user moves the mouse too fast, the panning doesn't keep up. The mouse can fall over a point with very
            # different z, which can result is unpredictable movement.
            ndc[2] = self.mouse_press_ndc[2]
            eye = self.ndc_to_eye(ndc)
            delta = eye - self.mouse_press_eye
            transform = make_translation(-delta[0], -delta[1], 0)
            self.eye_to_world = transform.dot(self.mouse_press_eye_to_world)
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if ((event.button() == QtCore.Qt.RightButton and self.mouse_drag_mode == 'orbit') or
            (event.button() == QtCore.Qt.LeftButton and self.mouse_drag_mode == 'pan')):
            self.mouse_drag_mode = None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        by = 1.5
        factor = by if event.angleDelta().y() > 0 else 1/by
        self.projection = self.projection.zoom(factor)

class SphereTraceViewer(QtWidgets.QWidget):
    def __init__(self, sdb_glsl: str, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        display_widget = SphereTraceRender([sdb_glsl])

        max_steps = QtWidgets.QSpinBox()
        max_steps.setRange(1, 1000)
        max_steps.setValue(display_widget.max_steps)
        max_steps.valueChanged.connect(self.maxStepsChanged)
        self._max_steps = max_steps

        sb = QtWidgets.QDoubleSpinBox()
        self._log10epsilon = sb
        sb.setRange(-20, 2)
        sb.setValue(np.log10(display_widget.epsilon))
        sb.valueChanged.connect(self.log10EpsilonChanged)

        b = QtWidgets.QPushButton('Home')
        self._home_button = b
        b.pressed.connect(self.go_home)

        # Place widgets in layout.
        def label(widget, string):
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(QtWidgets.QLabel(string))
            hbox.addWidget(widget)
            return hbox

        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        vbox = QtWidgets.QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addLayout(label(self._max_steps, 'Max. steps:'))
        vbox.addLayout(label(self._log10epsilon, 'log10(epsilon):'))
        vbox.addWidget(self._home_button)
        vbox.addStretch(1)
        hbox.addWidget(display_widget)

        # Timer not used at the moment. But we might want an animate feature so leaving this here.
        # timer = QtCore.QTimer()
        # timer.timeout.connect(display_widget.update)
        # #timer.start(0)
        # self.timer = timer

        self.display_widget = display_widget
        self.set_home(None)

    def maxStepsChanged(self, value):
        self.display_widget.max_steps = value

    def log10EpsilonChanged(self, value):
        self.display_widget.epsilon = 10**value

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)

    def set_home(self, home: Tuple[Matrix4, Projection]):
        self.home = home
        self._home_button.setEnabled(home is not None)

    def go_home(self):
        self.eye_to_world, self.projection = self.home

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

class ScenesViewer(QtWidgets.QWidget):
    def __init__(self, scenes: Sequence[Scene], parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self._scenes = scenes

        scene_list = QtWidgets.QListWidget()
        for scene in scenes:
            scene_list.addItem(QtWidgets.QListWidgetItem(scene.name))
        scene_list.currentRowChanged.connect(self.sceneChanged)
        scene_list.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred))
        self.scene_list = scene_list

        display_widget = SphereTraceRender([s.sdb_glsl for s in scenes])

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
        vbox.addWidget(scene_list)
        vbox.addWidget(max_steps)
        vbox.addWidget(self.log10epsilon)
        vbox.addStretch(1)
        hbox.addWidget(display_widget)

        self.display_widget = display_widget
        self.set_scene(0)

    def maxStepsChanged(self, value):
        self.display_widget.max_steps = value

    def log10EpsilonChanged(self, value):
        self.display_widget.epsilon = 10**value

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def sceneChanged(self, num: int):
        self._set_scene(num)
        self.display_widget.update()

    def _set_scene(self, num: int):
        self.display_widget.program_num = num
        scene = self._scenes[num]
        self.display_widget.set_wireframe_models(scene.wireframe_models)
        self.display_widget.eye_to_world = lookat(scene.eye, scene.center)
        self.display_widget.projection = Perspective(np.pi/3, scene.z_near, scene.z_far)

    def set_scene(self, num: int):
        self.scene_list.setCurrentRow(num)