import numpy as np
from itertools import product
from otk import sdb
from otk.sdb import scalar as sdbs
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

def test_csgs():
    def test(op_cls):
        p0 = sdb.Box((1, 2, 3), (0, 1, 2), 0.5)
        p1 = sdb.InfiniteCylinder(1, (1, 0.5))
        # p2 = sdb.Plane((1, 1, 1), 1)
        surface = op_cls((p0, p1))
        check(surface)

    test(sdb.UnionOp)
    test(sdb.IntersectionOp)

    p0 = sdb.Box((1, 2, 3), (0, 1, 2), 0.5)
    p1 = sdb.InfiniteCylinder(1, (1, 0.5))
    surface = sdb.DifferenceOp(p0, p1)
    check(surface)

def


