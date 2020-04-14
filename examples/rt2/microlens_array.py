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
    ray0 = rt2.make_ray(assembly, 0., y, -5e-3, 0, 0, 1, 1, 0, 0, lamb)
    segments = rt2.nonseq_trace(assembly, ray0).flatten()
    rays = rt2.get_points(segments, 10e-3)[:, :3]
    rays_list.append(rays)

# TODO after better (per-element?) rendering control is supported, make lenslets more visible by exaggerating lighting.
with rt2.application():
    w = rt2.view_assembly(assembly, projection_type='perspective')
    w.display_widget.set_rays(rays_list)
    w.show()
