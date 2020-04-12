import os
import time
import itertools
import numpy as np
from matplotlib import colors as mcolors
from PyQt5 import QtWidgets
from otk.sdb import lookat, projection
from otk import zemax, trains
from otk import ri
from otk.sdb import npscalar
from otk.sdb import numba as sdb_numba
from otk.rt2 import rt2_scalar_qt as rt2

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

times = []
# Loop over field positions.
for xy, color in zip(np.linspace(0, field_half_width, num_field_points), mcolors.TABLEAU_COLORS):
    # Loop over entrance pupil.
    for epx, epy in itertools.product(np.linspace(-stop_half_width, stop_half_width, num_rays_side), repeat=2):
        start_ray = rt2.make_ray(epx, epy, 0, xy, xy, f, 1, 0, 0, ri.air(lamb), 1, 0, lamb)
        # Trace ray and convert to sequence of points for plotting.
        times.append([])
        for spheretrace in (npscalar.spheretrace, sdb_numba.spheretrace):
            t0 = time.time()
            rt2.nonseq_trace(assembly, start_ray, dict(epsilon=1e-10), spheretrace=spheretrace)
            times[-1].append(time.time() - t0)

print(times)







