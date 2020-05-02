"""Pyqtgraph OpenGL extensions."""
import itertools
from typing import Sequence, Tuple
import numpy as np
import pyqtgraph_extended as pg
import pyqtgraph_extended.opengl as pgl
from . import surfaces
from OpenGL import GL
from .. import v4hb

# class FlatSurfaceItem(pgl.GLGraphicsItem.GLGraphicsItem):
#     def __init__(self, surface, parent=None):
#         pgl.GLGraphicsItem.GLGraphicsItem.__init__(self, parentItem=None)
#         self.setTransform(surface.matrix)
#         self.surface = surface
#
#     def paint(self):
#         xs, ys = self.surface.boundary.make_perimeter()
#         GL.glBegin(GL.GL_LINES)
#         for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
#             GL.glVertex3f(x0, y0, 0)
#             GL.glVertex3f(x1, y1, 0)
#         GL.glEnd()

class ParentItem(pgl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self):
        pgl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        # We will make all items children of brick_item to allow for rotation of the global coordinate system. This
        # is purely because pyqtgraph's mouse interaction rotates and pans along inconvenient axes.
        self.rotate(90, 1, 0, 0)

    def add_surface(self, surface):
        surface_item = SurfaceItem(surface)
        surface_item.setParentItem(self)


def plot_surfaces(surfaces):
    item = ParentItem()
    for surface in surfaces:
        item.add_surface(surface)
    widget = pgl.GLViewWidget()
    widget.addItem(item)
    widget.show()
    return widget


class SurfaceItem(pgl.GLSurfacePlotItem):
    def __init__(self, surface: surfaces.Surface, color: Tuple[float] = (0, 0, 1, 0.2), smooth: bool = True,
            shader: str = 'balloon', glOptions: str = 'additive', **kwargs):
        """Sample surface and create item.

        The default drawing style is translucent blue, which is good for a black background. Another good set of choices
        is shader='edgeHilight', glOptions='opaque', which looks nice on white.

        Args:
            surface:
            **kwargs: Passed on to  pgl.GLSurfacePlotItem.__init__.
        """
        x, y = surface.boundary.sample()
        xv, yv = np.broadcast_arrays(x, y)
        xv = xv.ravel()
        yv = yv.ravel()
        zv = surface.profile.intersect(v4hb.stack_xyzw(xv, yv, 0, 1), v4hb.stack_xyzw(0, 0, 1, 0))
        z = zv.reshape((len(x), len(y)))
        pgl.GLSurfacePlotItem.__init__(self, x, y, z, color=color, smooth=smooth, shader=shader, glOptions=glOptions,
            **kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.setTransform(surface.matrix.T)
        self.surface = surface

    def paint(self):
        pgl.GLSurfacePlotItem.paint(self)
        # xs, ys = self.surface.boundary.make_perimeter()
        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)

        def draw(xs, ys, zs):
            xs, ys, zs = np.broadcast_arrays(xs, ys, zs)
            for x0, y0, z0, x1, y1, z1 in zip(xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]):
                GL.glVertex3f(x0, y0, z0)
                GL.glVertex3f(x1, y1, z1)

        draw(self.x[:, 0], self.y[0], self.z[:, 0])
        draw(self.x[:, 0], self.y[-1], self.z[:, -1])
        draw(self.x[0, 0], self.y, self.z[0, :])
        draw(self.x[-1, 0], self.y, self.z[-1, :])
        GL.glEnd()


class SegmentsItem(pgl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, segments, parent=None, colors: Sequence = None):
        """

        Args:
            segments:
            parent:
            colors:
        """
        pgl.GLGraphicsItem.GLGraphicsItem.__init__(self, parentItem=parent)
        self.segments = segments
        self.setGLOptions('opaque')
        self.colors = colors  # self.points = [array.reshape((-1, 4)) for array in np.broadcast_arrays(self.segments.initial.origin,  #    *[i.point for i in self.segments.intersections])]

    def paint(self):
        self.setupGLState()

        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)
        for segment in self.segments:
            if segment.length is None:
                continue
            point0 = segment.ray.line.origin
            point1 = segment.ray.line.advance(segment.length).origin
            if self.colors is None:
                colors = itertools.cycle(pg.tableau20)
            else:
                colors = itertools.cycle(self.colors)
            for color, p0, p1 in zip(colors, point0.reshape((-1, 4)), point1.reshape((-1, 4))):
                GL.glColor3ub(*color)
                GL.glVertex3f(*p0[:3])
                GL.glVertex3f(*p1[:3])
        GL.glEnd()
