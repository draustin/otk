import sys
import numpy as np
from .. import zemax, trains, sdb
from otk.rt2 import rt2_scalar_qt as rt2

def view_zmx():
    filename = sys.argv[1]
    train0 = zemax.read_train(filename)
    train1 = train0.crop_to_finite()

    # Convert to a sequence of axisymemtric singlet lenses.
    singlet_sequence = trains.SingletSequence.from_train2(train1, 'max')
    # Convert to rt2 Elements.
    elements = rt2.make_elements(singlet_sequence, 'circle')
    # Create assembly object for ray tracing.
    assembly = rt2.Assembly.make(elements, singlet_sequence.n_external)

    scale_factor = abs(assembly.surface.get_aabb(np.eye(4)).size[:3]).prod()**(-1/3)
    view_surface = sdb.IntersectionOp((assembly.surface, sdb.Plane((-1, 0, 0), 0)), assembly.surface).scale(scale_factor)

    with rt2.application():
        viewer = rt2.view_assembly(assembly, surface=view_surface)


