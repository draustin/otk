from otk import trains, ri

def test_SpotArray():
    f = 50e-3
    lamb = 850e-9
    train = trains.Train.design_singlet(ri.N_BK7, f, 0.5, 10e-3, 25e-3, lamb).pad_to_transform(lamb)
    spot_array = train.trace_spot_array(lamb, 15e-3, 8, 100e-3, 8, 'square', 'square')
