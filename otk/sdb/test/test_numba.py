import numpy as np
from itertools import product
from otk import sdb
from otk.sdb import npscalar as sdbs
from otk.sdb import numba as sdbn


def check(s: sdb.Surface):
    getsdb_numba = sdbn.gen_getsdb(s)
    for x, y, z in product(np.linspace(-5, 5, 10), repeat=3):
        r = np.array((x, y, z, 1))
        assert np.isclose(sdbs.getsdb(s, r), getsdb_numba(r))

def test_primitives():
    check(sdb.Box((1, 2, 3), (4, 5, 6), 0.5))
    check(sdb.InfiniteCylinder(2, (2, 3)))
    check(sdb.InfiniteRectangularPrism(2, 3, (1, 2)))
    check(sdb.Plane((1, 2, 3), 4))
    check(sdb.ZemaxConic(1., 0.3, 1., 0.5, (0, 0.1, 0.2), (1., 2, 3)))
    check(sdb.SphericalSag(1., -1., (1., 2, 3)))

    check(sdb.SegmentedRadial((sdb.Plane((0, 0, 1), 2), sdb.SphericalSag(0.1, 1.)), (0.1,)))

def test_sags():
    def test(sagfun):
        check(sdb.Sag(sagfun, 1, (1, 2, 3)))
        check(sdb.Sag(sagfun, -1, (1, 2, 3)))

    test(sdb.ZemaxConicSagFunction(1, 0.2, 0.5, [0.1, 0.2, 0.3]))

    unit_sagfun = sdb.ZemaxConicSagFunction(1, 0.2, 0.5, [0.1, 0.2, 0.3])
    test(sdb.RectangularArraySagFunction(unit_sagfun, (0.2, 0.3)))
    test(sdb.RectangularArraySagFunction(unit_sagfun, (0.2, 0.3), (2, 3), False))
    test(sdb.RectangularArraySagFunction(unit_sagfun, (0.2, 0.3), (2, 3), True))

def test_csgs():
    def test(op_cls):
        p0 = sdb.Box((1, 2, 3), (0, 1, 2), 0.5)
        p1 = sdb.InfiniteCylinder(1, (1, 0.5))
        p2 = sdb.Plane((1, 1, 1), 1)
        surface = op_cls((p0, p1, p2))
        check(surface)

    test(sdb.UnionOp)
    test(sdb.IntersectionOp)

    p0 = sdb.Box((1, 2, 3), (0, 1, 2), 0.5)
    p1 = sdb.InfiniteCylinder(1, (1, 0.5))
    surface = sdb.DifferenceOp(p0, p1)
    check(surface)




