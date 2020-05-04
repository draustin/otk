from otk import rt2, trains, ri, sdb

def test_make_square_element_train():
    index = ri.FixedIndex(1.5)
    train = trains.Singlet.from_focal_length(100e-3, index, 5e-3, 12.5e-3, 0)
    element = rt2.to_rectangular_array_element(train, [sdb.RectangularArrayLevel.make(12.5e-3, 3)], (1, 2, 3))

    assert element.medium == rt2.UniformIsotropic(index)
    front = element.surface.surfaces[0]
    assert front.sagfun.unit.roc == train.surfaces[0].roc





