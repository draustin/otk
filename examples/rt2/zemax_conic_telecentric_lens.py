import os
import itertools
import numpy as np
from matplotlib import colors as mcolors
from PyQt5 import QtWidgets
from otk.sdb import lookat, projection
from otk import zemax, trains
from otk import ri
from otk import rt2, sdb
from otk.rt2 import qt
from otk.rt2.scalar import Assembly, make_ray, get_points

train_full = zemax.read_train(os.path.join(os.path.dirname(__file__), 'conic_telecentric_lens.zmx'), encoding='ascii')
train = train_full.subset(2, -1)
singlet_sequence = trains.SingletSequence.from_train(train)
# For fun make it a square aperture.
elements = rt2.make_elements(singlet_sequence, 'square')
assembly = Assembly(sdb.UnionOp([e.surface for e in elements]), elements, rt2.UniformIsotropic(ri.air))

lamb = 850e-9
# Get paraxial focal length.
f = train.get_effective_focal_length(lamb)
stop_half_width = train_full.interfaces[1].radius/2**0.5
field_half_width = train_full.interfaces[-1].radius/2**0.5
traced_rays = []
colors = []
num_field_points = 3
num_rays_side = 3
for xy, color in zip(np.linspace(0, field_half_width, num_field_points), mcolors.TABLEAU_COLORS):
    # Loop over entrance pupil.
    for epx, epy in itertools.product(np.linspace(-stop_half_width, stop_half_width, num_rays_side), repeat=2):
        start_ray = make_ray(epx, epy, 0, xy, xy, f, 1, 0, 0, ri.air(lamb), 1, 0, lamb)
        # Get segmented ray.
        traced_rays.append(get_points(assembly.nonseq_trace(start_ray).flatten(), f)[:, :3])
        colors.append(mcolors.to_rgb(color))

app = QtWidgets.QApplication([])
viewer = qt.view_assembly(assembly)
size = singlet_sequence.center_length
viewer.display_widget.z_near = size*0.1
# TODO when decent view controls added, remove this
if False:
    viewer.display_widget.half_width = viewer.display_widget.z_near/2
    viewer.display_widget.make_eye_to_clip = projection
else:
    viewer.display_widget.half_width = size*0.6
viewer.display_widget.eye_to_world = lookat((size*1.5, 0, size/2), (0, 0, size/2))
viewer.log10epsilon.setValue(-4)
viewer.set_rays(traced_rays, colors)
viewer.show()
app.exec()





