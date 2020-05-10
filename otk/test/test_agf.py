import os
from otk import agf

def test_load_catalog():
    with open(os.path.join(agf.AGFS_DIR, 'misc.agf'), 'rt') as file:
        comments, records = agf.parse_catalog(file)
    assert comments == ['From https://github.com/nzhagen/zemaxglass']
    assert records[1] == agf.Record('BASF5', 5, 1.603233, 42.504788, False, agf.Status.STANDARD, 0,
        'source:  Laikin, Lens Design', 0., 1., 0., False,
        [1.574103270E+000, 1.397861490E-002, 8.300451100E-004, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000,
            0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000], 3.34000000E-001, 2.32500000E+000,
        0., 0., 0., 0., 0., 0., 20.)

    assert records[3] == agf.Record('CAF2', 2, 1.433849, 94.995854, False, agf.Status.STANDARD, 0, 'source:  Handbook of Optics Vol. II',
        1.890000000E+001, 3.181000000E+000, 0., False,
        [5.675888000E-001, 2.526430000E-003, 4.710914000E-001, 1.007833300E-002, 3.848472300E+000,1.200556000E+003,
            0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000], 2.30000000E-001, 9.70000000E+000,
        -2.662000000E-005, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 0.000000000E+000, 2.000000000E+001)

    assert records[-1].name == 'WATER'

#def test_
