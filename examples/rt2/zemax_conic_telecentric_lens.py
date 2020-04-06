import os
import itertools
import numpy as np
from matplotlib import colors as mcolors
from PyQt5 import QtWidgets
from otk.sdb import lookat, projection
from otk import zemax, trains
from otk import ri
from otk import sdb
from otk import rt2_scalar_qt as rt2

train_full = zemax.read_train(os.path.join(os.path.dirname(__file__), 'conic_telecentric_lens.zmx'), encoding='ascii')
train = train_full.subset(2, -1)
singlet_sequence = trains.SingletSequence.from_train(train)
# For fun make it a square aperture.
elements = rt2.make_elements(singlet_sequence, 'square')
assembly = rt2.Assembly.make(elements, ri.air)

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
        start_ray = rt2.make_ray(epx, epy, 0, xy, xy, f, 1, 0, 0, ri.air(lamb), 1, 0, lamb)
        # Get segmented ray.
        traced_rays.append(rt2.get_points(assembly.nonseq_trace(start_ray).flatten(), f)[:, :3])
        colors.append(mcolors.to_rgb(color))

with rt2.application():
    viewer = rt2.view_assembly(assembly)
    size = singlet_sequence.center_length
    viewer.projection = sdb.Orthographic(size*0.6, 1)
    viewer.eye_to_world = lookat((size*1.5, 0, size/2), (0, 0, size/2))
    viewer.epsilon = 1e-4
    viewer.set_rays(traced_rays, colors)






