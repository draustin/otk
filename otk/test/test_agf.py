import os
from otk import agf
import numpy as np
from numpy.testing import assert_allclose

def get_record(name: str):
    if name == 'BASF5':
        return agf.Record('BASF5', 5, 1.603233, 42.504788, False, agf.Status.STANDARD, 0,
            'source:  Laikin, Lens Design', 0., 1., 0., False,
            [1.574103270E+000, 1.397861490E-002, 8.300451100E-004, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000,
                0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000], 3.34000000E-001, 2.32500000E+000,
            0., 0., 0., 0., 0., 0., 20.)
    elif name == 'CAF2':
        return agf.Record('CAF2', 2, 1.433849, 94.995854, False, agf.Status.STANDARD, 0, 'source:  Handbook of Optics Vol. II',
            1.890000000E+001, 3.181000000E+000, 0., False,
            [5.675888000E-001, 2.526430000E-003, 4.710914000E-001, 1.007833300E-002, 3.848472300E+000, 1.200556000E+003,
                0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000], 2.30000000E-001,
            9.70000000E+000, -2.662000000E-005, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000,
            0.000000000E+000, 2.000000000E+001)
    elif name == 'COC':
        return agf.Record('COC', 1, 1.533732, 56.227932, False, agf.Status.STANDARD, 0, 'source:  Hoechst Celanese Spec sheet',
            60.0, 1.02, 0.0, False, [2.28448546, 0.0102952211, 0.0373493703, -0.00928409653, 0.00173289808, -0.000115203047, 0.0, 0.0, 0.0, 0.0],
            0.334, 2.325, -0.0002278, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0)

def test_load_catalog():
    with open(os.path.join(agf.AGFS_DIR, 'misc.agf'), 'rt') as file:
        comments, records = agf.parse_catalog(file)
    assert comments == ['From https://github.com/nzhagen/zemaxglass']
    assert records[1] == get_record('BASF5')

    assert records[3] == get_record('CAF2')

    assert records[-1].name == 'WATER'

def test_calc_index():
    # Check without temperature adjustment.

    # Formula num. 1
    assert_allclose(get_record('COC').calc_index([0.4e-6, 0.6e-6, 0.8e-6], None), [1.5505454, 1.5331623, 1.5273148], atol=1e-6)

    # Formula num. 2
    assert_allclose(get_record('CAF2').calc_index([0.4e-6, 0.6e-6, 0.8e-6], None), [1.4418537, 1.4335639, 1.4305293],
        atol=1e-6)

    # Formula num. 5
    assert_allclose(get_record('BASF5').calc_index([0.4e-6, 0.6e-6, 0.8e-6], None),  [1.6295563, 1.6023620, 1.5933891], atol=1e-6)


