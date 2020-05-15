import os
import numpy as np
from typing import TextIO, Tuple, Mapping, Dict
import chardet
from . import ri, trains, agf, PROPERTIES_DIR


# Folder containing supplied (public domain) AGF files.
SUPPLIED_AGFS_DIR = os.path.join(PROPERTIES_DIR, 'agfs')

# Translate from Zemax glass database to ri module.
#glasses = {'PMMA':ri.PMMA_Zemax, 'F_SILICA':ri.fused_silica, 'BK7':ri.N_BK7, 'K-VC89':ri.KVC89}

SUPPLIED_GLASS_CATALOG_PATHS = {
    'HOYA': os.path.join(SUPPLIED_AGFS_DIR, 'HOYA20200314_include_obsolete.agf'),
    'SCHOTT': os.path.join(SUPPLIED_AGFS_DIR, 'schottzemax-20180601.agf'),
    'OHARA': os.path.join(SUPPLIED_AGFS_DIR, 'OHARA_200306_CATALOG.agf'),
    'SUMITA': os.path.join(SUPPLIED_AGFS_DIR, 'sumita-opt-dl-data-20200511235805.agf'),
    'NIKON': os.path.join(SUPPLIED_AGFS_DIR, 'NIKON-HIKARI_201911.agf')
    }

default_glass_catalog_paths = SUPPLIED_GLASS_CATALOG_PATHS.copy()


def read_glass_catalog_dir(dir: str) -> Dict[str, str]:
    paths = {}
    dir = os.path.expanduser(dir)
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if not os.path.isfile(path): continue
        root, ext = os.path.splitext(name)
        if ext.lower() != '.agf': continue
        paths[root.upper()] = path
    return paths


def read_interface(file:TextIO, n1, catalog: Dict[str, agf.Record], temperature: float) -> Tuple[trains.Interface, float, float]:
    commands = {}
    parms = {}
    while True:
        pos = file.tell()
        line = file.readline()
        words = line.split()
        if len(words) > 0:
            if line[:2] != '  ':
                file.seek(pos)
                break
            if words[0] == 'PARM':
                parm_index = int(words[1])
                parm_value = float(words[2])
                assert len(words) == 3
                parms[parm_index] = parm_value
            else:
                commands[words[0]]=words[1:]

    if 'GLAS' in commands:
        record = catalog[commands['GLAS'][0]]
        n2 = record.fix_temperature(temperature)
    else:
        n2 = ri.air
    thickness = float(commands['DISZ'][0])*1e-3
    clear_semi_dia = float(commands['DIAM'][0])*1e-3
    chip_zone = float(commands.get('OEMA', ['0'])[0])*1e-3
    clear_radius = clear_semi_dia + chip_zone

    with np.errstate(divide='ignore'):
        roc = np.divide(1e-3,float(commands['CURV'][0]))
    kappa = float(commands.get('CONI', [0])[0]) + 1
    surface_type = commands['TYPE'][0]
    if surface_type == 'STANDARD' and np.isclose(kappa,1):
        inner_surface = trains.SphericalSurface(roc, clear_radius)
    elif surface_type in ('STANDARD','EVENASPH'):
        alphas = []
        for parm_index, parm_value in parms.items():
            # Term is (2*parm_index)th-order coefficient, so the (2*parm_index-2)th element of alphas.
            order = 2*parm_index
            index = order-2
            if parm_value != 0:
                assert index >= 0
                alphas += [0]*(index - len(alphas) + 1)
                alphas[index] = parm_value*(1e3)**(order - 1)
        inner_surface = trains.ConicSurface(roc, clear_radius, kappa, alphas)
    else:
        raise ValueError('Unknown surface type %s.'%surface_type)

    if 'MEMA' in commands:
        mech_semi_dia = float(commands['MEMA'][0])*1e-3
    else:
        mech_semi_dia = clear_radius

    if mech_semi_dia - clear_radius > 1e-6:
        # TODO tidy this up - get rid of rt first?
        # Somehow need to offset this surface (in z). No way of describing this at present. Could add an offset
        # attribute to Surface. Then would need an offset in Profile.
        outer_surface = trains.SphericalSurface(np.inf, mech_semi_dia - clear_radius)
        outer_sag = inner_surface.calc_sag(clear_radius)
        surface = trains.SegmentedSurface((inner_surface, outer_surface), (0, outer_sag))
    else:
        surface = inner_surface

    interface = surface.to_interface(n1, n2)

    return interface, n2, thickness


def read_train(filename:str, n: ri.Index = ri.air, encoding: str = None, temperature: float = None, glass_catalog_paths: Dict[str, str] = None) -> trains.Train:
    """Read optical train from Zemax file.

    The given refractive index defines the surrounding medium.

    If encoding is not given it is detected automatically."""
    if encoding is None:
        with open(filename, 'rb') as file:
            raw = file.read()
        encoding = chardet.detect(raw)['encoding']
    if glass_catalog_paths is None:
        glass_catalog_paths = default_glass_catalog_paths
    surface_num = 0
    spaces = [0]
    interfaces = []
    full_catalog = {}
    with open(filename, 'rt', encoding=encoding) as file:
        while True:
            line = file.readline()
            if not line:
                break
            words = line.split()
            if words[0] == 'SURF':
                assert int(words[1]) == surface_num
                try:
                    interface, n, space = read_interface(file, n, full_catalog, temperature)
                except Exception as e:
                    raise ValueError(f'Exception reading SURF {surface_num}.') from e

                interfaces.append(interface)
                spaces.append(space)
                surface_num += 1
            elif words[0] == 'GCAT':
                for name in line.split()[1:]:
                    full_catalog.update(agf.load_catalog(glass_catalog_paths[name]))
    return trains.Train(interfaces, spaces)