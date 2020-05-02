import numpy as np
from numpy import testing
from otk import trains, paraxial, ri, functions


def test_Train_make_singlet():
    n = 3
    f = 10
    shapes = np.asarray((-1, 0, 1))
    d = 0.1
    for ne in (1, 1.5):
        for shape in shapes:
            train = trains.Train.design_singlet(ri.FixedIndex(n), f, shape, d, 0.01, ne=ri.FixedIndex(ne))
            assert np.allclose(train.get_focal_lengths()/ne, f)


def test_Train_make_singlet_transform1():
    n = 1.5
    ws = 1, 2
    f = 10
    l = trains.Train.make_singlet_transform1(ri.FixedIndex(n), ws, f, 0.01)
    testing.assert_allclose(l.get_focal_lengths(), (f, f))
    assert len(l.spaces) == 3
    assert np.isclose(l.spaces[0], ws[0])
    assert np.isclose(l.spaces[2], ws[1])
    testing.assert_allclose(l.get_focal_lengths(), (f, f))

    l2 = l.pad_to_transform()
    testing.assert_allclose(l2.get_working_distances(), (0, 0), atol=2e-15)


def test_pad_to_half_transform():
    l = trains.Train.make_singlet_transform2(ri.FixedIndex(1.5), 0.5, 0.1, 0.01, 0.01)
    fp = 1
    lp = l.pad_to_half_transform(f=fp)
    assert np.allclose((lp + lp.reverse()).get_focal_lengths(), fp)


def test_Train_principal_planes():
    n = 1.5
    f = 0.5
    shape = 0.1
    thickness = 0.01
    l = trains.Train.design_singlet(ri.FixedIndex(n), f, shape, thickness, 0.01)
    ppb, ppf = l.get_principal_planes()
    _, ppb_, ppf_ = paraxial.calc_thick_spherical_lens(n, l.interfaces[0].roc, l.interfaces[1].roc, thickness)
    assert np.isclose(ppb, ppb_)
    assert np.isclose(ppf, ppf_)


def test_Train_make_singlet_transform2():
    n = 1.5
    f = 10
    shapes = np.asarray((-1, 0, 1))
    d = 0.1
    for shape in shapes:
        train = trains.Train.make_singlet_transform2(ri.FixedIndex(n), f, shape, d, 0.01)
        assert np.allclose(train.get_focal_lengths(), f)


def test_Singlet():
    f = 100e-3
    lamb=532e-9
    singlet = trains.Singlet.from_focal_length(f, ri.fused_silica, 5e-3, 12.5e-3, 0.5, lamb=lamb)
    train = singlet.to_train(ri.vacuum)
    assert np.isclose(train.get_effective_focal_length(lamb), f)

    reversed = singlet.reverse()
    assert reversed.surfaces == tuple(s.reverse() for s in singlet.surfaces[::-1])


def test_SingletSequence():
    spaces = 0.1, 0.2
    f = 100e-3
    lamb = 532e-9
    singlet = trains.Singlet.from_focal_length(f, ri.fused_silica, 5e-3, 12.5e-3, 0.5, lamb=lamb)
    sequence = trains.SingletSequence((singlet,), spaces)
    train = sequence.to_train()
    assert np.isclose(train.get_effective_focal_length(lamb), f)

    reversed = sequence.reverse()
    assert reversed.spaces == sequence.spaces[::-1]


def test_design_symmetric_singlet_transform():
    f_transform = 100e-3
    working_distance = 10e-3
    edge_thickness = 5e-3
    field_radius = 10e-3
    lamb=532e-9
    sequence = trains.SingletSequence.design_symmetric_singlet_transform(ri.fused_silica, f_transform, working_distance,
        edge_thickness, field_radius, lamb=lamb)
    train = (sequence + sequence.reverse()).to_train()
    assert np.isclose(train.get_effective_focal_length(lamb), f_transform)
    testing.assert_allclose(train.get_working_distances(lamb), 0, atol=1e-15)

def test_Interface_calc_mask():
    n1 = ri.vacuum
    n2 = ri.FixedIndex(1.5)
    roc = 0.1
    lamb = 530e-9
    rho = 0.05
    interface = trains.Interface(n1, n2, roc, 10e-3)
    f, gradf = interface.calc_mask(lamb, rho, True)

    sag = functions.calc_sphere_sag(roc, rho)
    k = 2*np.pi/lamb
    deltan = n1(lamb) - n2(lamb)
    assert np.isclose(f, np.exp(1j*sag*k*deltan))

    grad_sag = functions.calc_sphere_sag(roc, rho, True)
    assert np.isclose(gradf, 1j*k*deltan*grad_sag*f)
