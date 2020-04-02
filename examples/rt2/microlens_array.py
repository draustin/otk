import numpy as np
import itertools
from PyQt5 import QtWidgets
from otk.sdb import qt
from otk import paraxial, ri
#from otk.sdb import *
#from otk.sdb.lens import *
from otk import sdb
from otk.rt2 import *
import otk.rt2.scalar as rt2

# TODO chrome mask
def make_MLA150(small:bool=False):
    # https://www.thorlabs.com/drawings/2387c2b71216f558-2D5B16EB-0174-2DDB-0867F5C4641A7CDC/MLA150-5C-SpecSheet.pdf
    roc = 2.54e-3
    d = 140e-6
    p = 150e-6
    t = 1.2e-3
    w = 10e-3
    h = 10e-3
    unitfn = sdb.ZemaxConicSagFunction(roc, d/2)
    fn = sdb.RectangularArraySagFunction(unitfn, (p, p))
    front = sdb.Sag(fn, 1)
    sides = sdb.InfiniteRectangularPrism(w, h)
    back = sdb.Plane((0, 0, 1), -t)
    surface = sdb.IntersectionOp((front, sides, back))
    element = SimpleElement(surface, UniformIsotropic(ri.fused_silica), rt2.perfect_refractor)
    return element

element = make_MLA150()
assembly = rt2.Assembly(element.surface, [element], UniformIsotropic(ri.air))
sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)

num_rays = 300
lamb = 532e-9
rays_list = []
height = 10e-3
for y in np.linspace(-height/2, height/2, num_rays):
    ray0 = rt2.make_ray(0, y, -5e-3, 0, 0, 1, 1, 0, 0, ri.air(lamb), 1, 0, lamb)
    segments = assembly.nonseq_trace(ray0, sphere_trace_kwargs).flatten()
    rays = rt2.get_points(segments, 10e-3)[:, :3]
    rays_list.append(rays)

properties = dict(edge_width = 0.5e-6, edge_color = (0, 0, 0), surface_color = (0.2, 0.4, 1))
sdb_glsl = sdb.gen_get_all_recursive(assembly.surface, {}, properties)

app = QtWidgets.QApplication([])
w = qt.SphereTraceViewer(sdb_glsl)
w.display_widget.half_width = 8e-3
w.display_widget.eye_to_world = sdb.lookat((-20e-3, 0, 2e-3), (0, 0, 2e-3))
w.display_widget.z_near = 1e-6
w.display_widget.z_far = 1
w.log10epsilon.setValue(-7) # mysterious artefacts for smaller values
w.display_widget.set_rays(rays_list)
w.resize(800, 600)
w.show()
app.exec()
