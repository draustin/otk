import os
import time
import itertools
import numpy as np
from matplotlib import colors as mcolors
from PyQt5 import QtWidgets
from otk.sdb import lookat, projection
from otk import zemax, trains
from otk import ri
from otk import sdb
from otk import rt2_scalar_qt as rt2

# Load Zemax file.
train_full = zemax.read_train(os.path.join(os.path.dirname(__file__), 'conic_telecentric_lens.zmx'), encoding='ascii')
# Remove object, stop and image surfaces.
train = train_full.subset(2, -1)
# Convert to a sequence of axisymemtric singlet lenses.
singlet_sequence = trains.SingletSequence.from_train(train)
# Convert to rt2 Elements. For fun make the lenses square.
elements = rt2.make_elements(singlet_sequence, 'square')
# Create assembly object for ray tracing.
assembly = rt2.Assembly.make(elements, ri.air)

lamb = 850e-9
# Get paraxial focal length.
f = train.get_effective_focal_length(lamb)
stop_half_width = train_full.interfaces[1].radius/2**0.5
field_half_width = train_full.interfaces[-1].radius/2**0.5

traced_rays = []
num_field_points = 3
num_rays_side = 3

t0 = time.time()
# Loop over field positions.
for xy, color in zip(np.linspace(0, field_half_width, num_field_points), mcolors.TABLEAU_COLORS):
    # Loop over entrance pupil.
    for epx, epy in itertools.product(np.linspace(-stop_half_width, stop_half_width, num_rays_side), repeat=2):
        start_ray = rt2.make_ray(epx, epy, 0, xy, xy, f, 1, 0, 0, ri.air(lamb), 1, 0, lamb)
        # Trace ray and convert to sequence of points for plotting.
        traced_rays.append(rt2.get_points(assembly.nonseq_trace(start_ray, dict(epsilon=1e-10)).flatten(), 10e-3)[:, :3])
print(time.time() - t0)







