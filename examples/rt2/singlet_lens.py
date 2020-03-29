import numpy as np
import itertools
from PyQt5 import QtWidgets
from otk.sdb import qt
from otk import paraxial, ri
from otk.sdb import *
from otk.sdb.lens import *
from otk.rt2 import *

ne = ri.air
n = ri.fused_silica
f = 0.1
lamb = 800e-9
thickness = 5e-3
vertex0 = np.asarray((0, 0, 0, 1))
vertex1 = vertex0 + (0, 0, thickness, 0)
roc0, roc1 = paraxial.design_singlet(n(lamb), f, 0, thickness, ne(lamb))
f_, h0, h1 = paraxial.calc_thick_spherical_lens(n(lamb), roc0, roc1, thickness, ne(lamb))
assert np.isclose(f_, f)

r = 20e-3
surface = make_spherical_singlet(roc0, roc1, thickness, vertex0[:3], 'circle', r)
element = SimpleElement(surface, n, lambda x: perfect_refractor)
assembly = Assembly([element], UniformIsotropic(ne))
sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)

# Compute object and image points for 2f-2f imaging. zp is relative to vertex.
zp_object = -(2*f - h0)
zp_image = 2*f + h1 + thickness
object = vertex0 + (0, 0, zp_object, 0)
image = vertex0 + (0, 0, zp_image, 0)

def get_rays(theta, phi) -> np.ndarray:
    vx = np.sin(theta)*np.cos(phi)
    vy = np.sin(theta)*np.sin(phi)
    vz = np.cos(theta)
    incident_ray0 = make_ray(*object[:3], vx, vy, vz, 0, 1, 0, ne(lamb), 1, 0, lamb)
    rays = get_points(assembly.nonseq_trace(incident_ray0, sphere_trace_kwargs).flatten(), 2*f)[:, :3]
    return rays

num_theta = 6
num_phi = 16
theta_max = np.arctan(r/zp_object*0.9)
thetas = np.arange(1, num_theta)/(num_theta - 1)*theta_max
phis = np.arange(num_phi)/num_phi*2*np.pi
rays_list = [get_rays(theta, phi) for theta, phi in itertools.product(thetas, phis)]

ids = add_ids(assembly.surface)
properties = dict(edge_width = 0.51e-6, edge_color = (0, 0, 0), surface_color = (0.2, 0.4, 1))
sdb_glsl = gen_getSDB_recursive(assembly.surface, ids, set()) + gen_getColor_recursive(assembly.surface, ids, {}, properties, set())

app = QtWidgets.QApplication([])
w = qt.SphereTraceViewer(sdb_glsl)
w.display_widget.half_width = 4*r
w.display_widget.eye_to_world = lookat((0, 0, 5*f), (0, 0, 0))
w.display_widget.z_near = 0.01*f
w.display_widget.z_far = 10*f
w.log10epsilon.setValue(-7) # mysterious artefacts for smaller values
w.display_widget.set_rays(rays_list)
w.resize(800, 600)
w.show()
app.exec()
