import otk.rt1.lines
import numpy as np
from typing import Tuple, Sequence

from otk.rt1 import Ray
import mathx
from .. import ri
from .. import v4hb
from . import raytrace
from .. import trains
from .surfaces import make_analysis_surfaces, Surface
from .profiles import PlanarProfile

import pyqtgraph_extended as pg


def make_square_field_vectors(side_length: float, num_spots: int, x_axis: int = -2, y_axis: int = -1):
    # Generate square grid of plane wave vectors along -1 and -2 axes.
    thetax = mathx.reshape_vec(np.linspace(-0.5, 0.5, num_spots), x_axis)*side_length
    thetay = mathx.reshape_vec(np.linspace(-0.5, 0.5, num_spots), y_axis)*side_length
    vx = np.sin(thetax)
    vy = np.sin(thetay)
    vz = (1 - vx**2 - vy**2)**0.5
    return (thetax, thetay), (vx, vy, vz)

def make_field_vectors(size: float, num_spots: int, shape:str='circle', axes:Tuple[int,int]=(-2,-1)):
    if shape=='circle':
        raise NotImplementedError()
    elif shape=='square':
        return make_square_field_vectors(size, num_spots, *axes)
    else:
        raise ValueError()

def make_stop_origins(size:float, num_rays:int, shape:str='circle', axes:Tuple[int,int]=(-2,-1)):
    if shape=='circle':
        raise NotImplementedError()
    else:
        return make_square_stop_origins(size, num_rays, *axes)

def make_square_stop_origins(side_length:float, num_rays:int, x_axis:int=-2, y_axis:int=-1):
    ox = mathx.reshape_vec(np.linspace(-0.5, 0.5, num_rays), x_axis)*side_length
    oy = mathx.reshape_vec(np.linspace(-0.5, 0.5, num_rays), y_axis)*side_length
    return ox, oy

class SpotArray:
    def __init__(self, thetax, thetay, ox, oy, ix, iy, iu, iv):
        self.thetax = thetax
        self.thetay = thetay
        self.ox = ox
        self.oy = oy
        self.ix = ix
        self.iy = iy
        self.iu = iu
        self.iv = iv

    def advance(self, z):
        return SpotArray(self.thetax, self.thetay, self.ox, self.oy, self.ix+z*self.iu, self.iy+z*self.iv, self.iu, self.iv)

    def show(self):
        rms = (self.ix.var(axis=(-1, -2)) + self.iy.var(axis=(-1, -2)))**0.5
        ixmu = self.ix.mean(axis=(-1, -2))
        iymu = self.iy.mean(axis=(-1, -2))

        glw = pg.GraphicsLayoutWidget()
        # title = 'Input & output aperture side length %.1f mm. %dx%d plane waves, each with %dx%d rays.'%(
        #     self.field_side_length*1e3, num_spots, num_spots, num_rays, num_rays)
        # glw.addLabel(title, colspan=2)
        # glw.nextRow()

        gl = glw.addLayout()
        spot_plot = gl.addAlignedPlot(labels={'left': 'field y (mm)', 'bottom': 'field x (mm)'}, title='Spots')
        index = 0
        for ix_, iy_ in zip(self.ix, self.iy):
            for ix, iy in zip(ix_, iy_):
                item = pg.ScatterPlotItem(ix.ravel()*1e3, iy.ravel()*1e3, pen=None, brush=pg.tableau20[index%20],
                    size=2)
                spot_plot.addItem(item)
                index += 1

        gl = glw.addLayout()
        rms_plot = gl.addAlignedPlot(labels={'left': 'theta y (mrad)', 'bottom': 'theta x (mrad)'},
            title='RMS spot size')
        rms_image = rms_plot.image(rms*1e6, rect=pg.axes_to_rect(self.thetax*1e3, self.thetay*1e3),
            lut=pg.get_colormap_lut())
        gl.addHorizontalSpacer()
        gl.addColorBar(image=rms_image, rel_row=2, label='Size (micron)')

        glw.resize(920, 400)
        glw.show()
        return glw

    @classmethod
    def trace(cls, stop_surface, image_surface, trace_fun, lamb: float, stop_size: float, num_rays: int,
            field_size: float, num_spots: int, stop_shape: str = 'circle', field_shape: str = 'circle',
            n0:ri.Index = ri.air):
        """Trace square angular lattice of square spatial lattices of rays.

        Args:
            surfaces (tuple): Zeroth element is starting surface. Rays are traced through subsequent focal_surfaces. Final element
                is detector surface.
            f (scalar): Nominal focal length.
            side_length (scalar): Side length of square input and output apertures. Rays fill input aperture uniformly
                    and output spots fill output aperture uniformly if lens has no distortion.
                num_spots (int): Number of spots to a side.
                num_rays (int): Number of rays to cast to side.

        Returns:
            tuple: (ox, oy), (vx, vy, vz), (ix, iy), defined as follows:
                ox ((num_rays,1) array): Incident x coordinates.
                oy (num_rays) array): Incident y coordinates.
                ixi ((num_spots,1,1,1) array): Incident vector x components.
                ixi ((num_spots,1,1) array): Incident vector y components.
                ix ((num_spots, num_spots, num_rays, num_rays) array): Output focal plane x positions.
                iy ((num_spots, num_spots, num_rays, num_rays) array): Output focal plane y positions.
        """
        # Generate grid of plane wave vectors along -3 and -4 axes.
        (thetax, thetay), (vx, vy, vz) = make_field_vectors(field_size, num_spots, field_shape, (-4, -3))

        # Generate square grid of incident rays along -1 and -2 axes.
        ox, oy = make_stop_origins(stop_size, num_rays, stop_shape)

        oxv, oyv, vxv, vyv, vzv = [array.ravel() for array in np.broadcast_arrays(ox, oy, vx, vy, vz)]

        origin_local = v4hb.stack_xyzw(oxv, oyv, 0, 1)
        vector_local = v4hb.stack_xyzw(vxv, vyv, vzv, 0)
        origin = stop_surface.to_global(origin_local)
        vector = stop_surface.to_global(vector_local)
        pol = v4hb.normalize(v4hb.cross(vector, [0, 1, 0, 0]))
        segments = trace_fun(raytrace.Ray(raytrace.Line(origin, vector), pol, 0, lamb, n0(lamb)))
        shape = num_spots, num_spots, num_rays, num_rays, 4
        line = segments[-1].ray.line
        ix, iy = v4hb.to_xy(image_surface.to_local(line.origin).reshape(shape))
        ivx, ivy, ivz = v4hb.to_xyz(image_surface.to_local(line.vector).reshape(shape))
        iu = ivx/ivz
        iv = ivy/ivz

        #ix = ixv.reshape(shape)
        #iy = iyv.reshape(shape)

        return cls(thetax, thetay, ox, oy, ix, iy, iu, iv)


def trace_distortion(stop_surface, image_surface, trace_fun, lamb: float, stop_size: float, num_rays: int, thetas: Sequence, stop_shape:str='circle'):
    """

    Args:
        stop_surface:
        image_surface:
        trace_fun:
        lamb:
        stop_size: For square stop shape, side length.
        num_rays: For square stop shape, number of rays to a side.
        thetas: Angles to use.
        stop_shape: 'circle' or 'square'.

    Returns:

    """
    ox, oy = make_stop_origins(stop_size, num_rays, stop_shape)
    oxv, oyv = [array.ravel() for array in np.broadcast_arrays(ox, oy)]

    origin_local = v4hb.stack_xyzw(oxv, oyv, 0, 1)
    origin = stop_surface.to_global(origin_local)

    ixs = []
    for theta in thetas:
        vx = np.sin(theta)
        vz = (1 - vx**2)**0.5
        vector_local = v4hb.stack_xyzw(vx, 0, vz, 0)
        vector = stop_surface.to_global(vector_local)
        pol = stop_surface.matrix[1, :]
        segments = trace_fun(raytrace.Ray(otk.rt1.lines.Line(origin, vector), pol, 1, 0, lamb))
        ix, iy = v4hb.to_xy(image_surface.to_local(segments[-1].ray.line.origin))
        ixs.append(ix.mean())

    return np.asarray(ixs)


def connect_mapped_points(ray0, point1, map_to, map_from, max_error_distance=1e-9, max_num_iterations=100):
    point0 = ray0.line.origin
    num_iterations = 0
    while num_iterations <= max_num_iterations:
        # Trace from 0 to 1.
        ray1i = map_to(ray0)
        error1 = v4hb.dot(ray1i.line.origin - point1)

        # Make ray from point1 in opposite direction to ray1i.
        ray1 = Ray(otk.rt1.lines.Line(point1, -ray1i.line.vector), ray1i.pol, ray1i.flux, 0, ray1i.lamb, ray1i.n)

        # Trace from 1 to 0.
        ray0i = map_from(ray1)
        error0 = v4hb.dot(ray0i.line.origin - point0)

        # Make ray from point0 in opposite direction to ray0i.
        ray0 = Ray(otk.rt1.lines.Line(point0, -ray0i.line.vector), ray0i.pol, ray0i.flux, 0, ray0i.lamb, ray0i.n)

        error_distance = max(error0.max(), error1.max())
        if error_distance <= max_error_distance:
            break

        num_iterations += 1

    if num_iterations > max_num_iterations:
        raise ValueError(f'error_distance, {error_distance}, did not reach max_error_distance, {max_error_distance}, '
                         f'in {num_iterations} iterations.')

    return ray0, ray1


def make_phase_space_cross(x_w, y_w, vx_w, vy_w, num=10, colors=('b', 'r'), symbols=('x', 'o')):
    zeros = np.zeros(num)
    origin = np.c_[np.r_[np.linspace(-x_w/2, x_w/2, num), zeros, zeros, zeros], np.r_[
        zeros, np.linspace(-y_w/2, y_w/2, num), zeros, zeros], np.zeros(4*num), np.zeros(4*num)]
    vector = v4hb.normalize(np.c_[np.r_[zeros, zeros, np.linspace(-vx_w/2, vx_w/2, num), zeros], np.r_[
        zeros, zeros, zeros, np.linspace(-vy_w/2, vy_w/2, num)], np.ones(4*num), np.zeros(4*num)])
    return origin, vector

def trace_train_spot_array(train: trains.Train, lamb: float, stop_size: float, num_rays: int, field_size: float, num_spots: int,
        stop_shape: str = 'circle', field_shape: str = 'circle') -> SpotArray:
    """Trace square angular lattice of 'plane waves'."""
    surfaces, keys = make_analysis_surfaces(train)
    stop_surface = Surface(PlanarProfile())
    trace_fun = lambda ray: ray.trace_surfaces(surfaces, keys)[0]
    spot_array = SpotArray.trace(stop_surface, surfaces[-1], trace_fun, lamb, stop_size, num_rays, field_size,
                                    num_spots, stop_shape, field_shape)
    return spot_array