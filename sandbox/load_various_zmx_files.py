import logging
from otk import zemax, trains, sdb
from otk.rt2 import rt2_scalar_qt as rt2
logging.getLogger('otk.rt2.qt').setLevel(logging.DEBUG)
logging.basicConfig()
train0 = zemax.read_train('US08934179-4.zmx')
train1 = train0.crop_to_finite()
sequence0 = trains.SingletSequence.from_train2(train1, 'max')
sequence1 = sequence0#.split(4)[1].split(1)[0]
# Convert to rt2 Elements.
elements = rt2.make_elements(sequence1, 'circle')
# Create assembly object for ray tracing.
assembly = rt2.Assembly.make(elements, sequence1.n_external)

view_surface = sdb.IntersectionOp((assembly.surface, sdb.Plane((-1, 0, 0), 0)), assembly.surface).scale(1e3)

with rt2.application():
    viewer = rt2.view_assembly(assembly, surface=view_surface)
    viewer.max_steps = 1000
