"""Functional test of conversion between data formats / representations."""
import os
from otk import zemax, trains, rt2, DESIGNS_DIR

def test_zemax_to_elements():
    train0 = zemax.read_train(os.path.join(DESIGNS_DIR, 'aspheric_telecentric_lens.zmx'), encoding='ascii',
        glass_catalog_paths=zemax.SUPPLIED_GLASS_CATALOG_PATHS)
    train1 = train0.crop_to_finite()

    # Convert to a sequence of axisymemtric singlet lenses.
    sequence = trains.SingletSequence.from_train2(train1, 'max')
    # Convert to rt2 Elements.
    elements = rt2.make_elements(sequence, 'circle')

    assert len(elements) == 3