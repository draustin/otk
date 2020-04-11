import numpy as np
from otk.rt2 import rt2_scalar_qt as rt2
from otk import ri
from otk import sdb
from otk.rt2 import stock

assembly = rt2.Assembly.make([stock.make_MLA150()], ri.air)

num_rays = 300
lamb = 532e-9
rays_list = []
height = 10e-3
for y in np.linspace(-height/2, height/2, num_rays):
    ray0 = rt2.make_ray(0., y, -5e-3, 0, 0, 1, 1, 0, 0, ri.air(lamb), 1, 0, lamb)
    segments = rt2.nonseq_trace(assembly, ray0).flatten()
    rays = rt2.get_points(segments, 10e-3)[:, :3]
    rays_list.append(rays)

with rt2.application():
    w = rt2.view_assembly(assembly)
    w.projection = sdb.Orthographic(8e-3, 0.1)
    w.eye_to_world = sdb.lookat((-20e-3, 0, 2e-3), (0, 0, 2e-3))
    w.epsilon = 1e-7 # mysterious artefacts for smaller values
    w.display_widget.set_rays(rays_list)
    w.show()
