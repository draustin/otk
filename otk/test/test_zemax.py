import os
import numpy as np
from numpy.testing import  assert_allclose
from otk import zemax, trains, ri, agf, DESIGNS_DIR

def test_read_train_conic_aspheric_telecentric_lens():
    # Ensure we use the supplied glass catalogs.
    train = zemax.read_train(os.path.join(DESIGNS_DIR, 'aspheric_telecentric_lens.zmx'), encoding='ascii',
        glass_catalog_paths=zemax.SUPPLIED_GLASS_CATALOG_PATHS)
    assert len(train.interfaces) == 9
    assert_allclose(train.spaces, (0., np.inf, 10e-3, 20e-3, 6.757645743585563e-2, 20e-3, 2.868233931997107e-2, 20e-3, 7.045493144939738e-2, 0))

    bk7 = agf.load_catalog(zemax.SUPPLIED_GLASS_CATALOG_PATHS['SCHOTT'])['N-BK7'].fix_temperature()
    radius0 = 2.859595844931864e-2
    radius1 = 3.398390496818528e-2 - radius0
    surface0 = trains.ConicSurface(-3.399643783726705e-2, radius0, 1 - 2.667544379512378E+000, (0, 0, -1.899747134198353e3, 0, 2.093291560636944e5))
    surface1 = trains.SphericalSurface(np.inf, radius1)
    interface = train.interfaces[4]
    assert isinstance(interface, trains.SegmentedInterface)
    assert interface.n1 == ri.air
    assert interface.n2 == bk7
    assert interface.segments[0].isclose(surface0)
    assert interface.segments[1].isclose(surface1)
    assert_allclose(interface.sags, (0, surface0.calc_sag(surface0.radius)))