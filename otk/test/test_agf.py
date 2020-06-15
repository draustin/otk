import os
from otk import agf, zemax
import numpy as np
from numpy.testing import assert_allclose


def get_record(name: str):
    if name == 'BASF5':
        return agf.Record(name='BASF5', dispersion_formula=5, nd=1.603233, vd=42.504788, exclude_substitution=False,
                          status=agf.Status.STANDARD, melt_freq=0,
                          comments='source:  Laikin, Lens Design', tce=0., density=1., dPgF=0.,
                          ignore_thermal_expansion=False,
                          dispersion_coefficients=[1.574103270E+000, 1.397861490E-002, 8.300451100E-004,
                                                   0.000000000E+000, 0.000000000E+000, 0.000000000E+000,
                                                   0.000000000E+000, 0.000000000E+000, 0.000000000E+000,
                                                   0.000000000E+000], min_lamb=3.34000000E-001,
                          max_lamb=2.32500000E+000,
                          d0=0., d1=0., d2=0., e0=0., e1=0., lamb_tk=0., reference_temperature=20.)
    elif name == 'CAF2':
        return agf.Record(name='CAF2', dispersion_formula=2, nd=1.433849, vd=94.995854, exclude_substitution=False,
                          status=agf.Status.STANDARD, melt_freq=0,
                          comments=['source:  Handbook of Optics Vol. II'],
                          tce=1.890000000E+001, density=3.181000000E+000, dPgF=0., ignore_thermal_expansion=False,
                          dispersion_coefficients=[5.675888000E-001, 2.526430000E-003, 4.710914000E-001,
                                                   1.007833300E-002, 3.848472300E+000,
                                                   1.200556000E+003,
                                                   0.000000000E+000, 0.000000000E+000, 0.000000000E+000,
                                                   0.000000000E+000], min_lamb=2.30000000E-001,
                          max_lamb=.70000000E+000, d0=-2.662000000E-005, d1=0.000000000E+000, d2=0.000000000E+000,
                          e0=0.000000000E+000,
                          e1=0.000000000E+000,
                          lamb_tk=0.000000000E+000, reference_temperature=2.000000000E+001)
    elif name == 'COC':
        return agf.Record(name='COC', dispersion_formula=1, nd=1.533732, vd=56.227932, exclude_substitution=False,
                          status=agf.Status.STANDARD, melt_freq=0,
                          comments=['source:  Hoechst Celanese Spec sheet'],
                          tce=60.0, density=1.02, dPgF=0.0, ignore_thermal_expansion=False,
                          dispersion_coefficients=[2.28448546, 0.0102952211, 0.0373493703, -0.00928409653,
                                                   0.00173289808, -0.000115203047, 0.0,
                                                   0.0, 0.0, 0.0],
                          min_lamb=0.334, max_lamb=2.325, d0=-0.0002278, d1=0.0, d2=0.0, e0=0.0, e1=0.0, lamb_tk=0.0,
                          reference_temperature=20.0)
    elif name == 'F_SILICA':
        return agf.Record(name='F_SILICA', dispersion_formula=2, nd=1.458464, vd=67.821433, exclude_substitution=False,
                          status=agf.Status.STANDARD, melt_freq=-1,
                          comments=['source:  The Infrared & Electro-Optical Systems Handbook V. III'],
                          tce=0.51, density=2.2, dPgF=0.0, ignore_thermal_expansion=False,
                          dispersion_coefficients=[6.961663000E-01, 4.679148000E-03, 4.079426000E-01, 1.351206300E-02,
                                                   8.974794000E-01,
                                                   9.793400250E+01, 0.000000000E+00, 0.000000000E+00, 0.000000000E+00,
                                                   0.000000000E+00],
                          min_lamb=0.21, max_lamb=3.71, d0=2.237000000E-05, d1=0.000000000E+00, d2=0.000000000E+00,
                          e0=0.000000000E+00,
                          e1=0.000000000E+00, lamb_tk=0.000000000E+00, reference_temperature=2.000000000E+01)


def test_load_catalog():
    catalog = agf.load_catalog(os.path.join(zemax.SUPPLIED_AGFS_DIR, 'Edmund-Optics-2018.agf'))

    assert catalog['F_SILICA'] == get_record('F_SILICA')

    catalog = agf.load_catalog(os.path.join(zemax.SUPPLIED_AGFS_DIR, 'schottzemax-20180601.agf'))
    assert catalog['P-SF67'].max_lamb == 2.5

def test_calc_index():
    # Check without temperature adjustment.

    # Formula num. 1
    assert_allclose(get_record('COC').calc_index([0.4e-6, 0.6e-6, 0.8e-6], None), [1.5505454, 1.5331623, 1.5273148],
                    atol=1e-6)

    # Formula num. 2
    assert_allclose(get_record('CAF2').calc_index([0.4e-6, 0.6e-6, 0.8e-6], None), [1.4418537, 1.4335639, 1.4305293],
                    atol=1e-6)

    # Formula num. 5
    assert_allclose(get_record('BASF5').calc_index([0.4e-6, 0.6e-6, 0.8e-6], None), [1.6295563, 1.6023620, 1.5933891],
                    atol=1e-6)
