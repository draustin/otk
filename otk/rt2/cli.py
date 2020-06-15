import sys
import argparse
import numpy as np
from .. import zemax, trains, sdb, _utility, agf
from otk.rt2 import rt2_scalar_qt as rt2


def view_zmx():
    parser = argparse.ArgumentParser(
        description='Load and view a Zemax lens. Glass catalogs are obtained the directory zemax_glass_catalog_dir'
                    'defined in the otk configuration file.')
    parser.add_argument('filename', help='file to view')
    parser.add_argument('-n', help='accept N- prefix as substitute if named glass not found', action='store_true')
    args = parser.parse_args()

    config = _utility.load_config()
    dir = config.get('zemax_glass_catalog_dir')
    if dir is not None:
        glass_catalog_paths = zemax.read_glass_catalog_dir(dir)
    else:
        glass_catalog_paths = zemax.SUPPLIED_GLASS_CATALOG_PATHS

    try:
        train0 = zemax.read_train(args.filename, glass_catalog_paths=glass_catalog_paths, try_n_prefix=args.n)
    except (agf.ParseError, zemax.GlassNotFoundError, zemax.NoCatalogError) as e:
        print(e.args[0])
        sys.exit(1)
    train1 = train0.crop_to_finite()

    # Convert to a sequence of axisymmetric singlet lenses.
    singlet_sequence = trains.SingletSequence.from_train2(train1, 'max')
    # Convert to rt2 Elements.
    elements = rt2.make_elements(singlet_sequence, 'circle')
    # Create assembly object for ray tracing.
    assembly = rt2.Assembly.make(elements, singlet_sequence.n_external)

    scale_factor = abs(assembly.surface.get_aabb(np.eye(4)).size[:3]).prod()**(-1/3)
    view_surface = sdb.IntersectionOp((assembly.surface, sdb.Plane((-1, 0, 0), 0)), assembly.surface).scale(scale_factor)

    with rt2.application():
        viewer = rt2.view_assembly(assembly, surface=view_surface)


