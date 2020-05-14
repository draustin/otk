import os
import itertools
import numpy as np
from matplotlib import colors as mcolors
import mathx
from PyQt5 import QtWidgets
from otk.sdb import lookat, projection
from otk import zemax, trains
from otk import ri, sdb, ROOT_DIR
from otk.rt2 import rt2_scalar_qt as rt2

# Load Zemax file.
train_full = zemax.read_train(os.path.join(ROOT_DIR, '..', 'designs',  'aspheric_telecentric_lens.zmx'), encoding='ascii')
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
field_half_angle = np.radians(10)

traced_rays = []
colors = []
num_field_points = 3
num_rays_side = 3
# Loop over field positions.
for theta, color in zip(np.linspace(0, field_half_angle, num_field_points), mcolors.TABLEAU_COLORS):
    # Loop over entrance pupil.
    for epx, epy in itertools.product(np.linspace(-stop_half_width, stop_half_width, num_rays_side), repeat=2):
        start_ray = rt2.make_ray(assembly, epx, epy, 0, *mathx.polar_to_cart(1, theta, np.pi/4), 1, 0, 0, lamb)
        # Trace ray and convert to sequence of points for plotting.
        traced_rays.append(rt2.get_points(rt2.nonseq_trace(assembly, start_ray, {'max_steps':101}).flatten(), 80e-3)[:, :3])
        colors.append(mcolors.to_rgb(color))

# For fun we'll render a cutaway view.
cutout = sdb.IntersectionOp((sdb.Plane((1, 0, 0), 0), sdb.Plane((0, 1, 0), 0)))
view_surface = sdb.DifferenceOp(assembly.surface, cutout, assembly.surface)

# Create HTML of scene.
rt2.gen_scene_html('aspheric_telecentric_lens.html', elements, surface=view_surface, rays=traced_rays, colors=colors)

with rt2.application():
    viewer = rt2.view_elements(elements, surface=view_surface)
    viewer.set_rays(traced_rays, colors)






