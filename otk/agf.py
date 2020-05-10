"""Read ANSI glass format (.agf) files used by Zemax.

Inspired by https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass.py.
"""
import os
from enum import Enum
from typing import Sequence, TextIO, Tuple, Dict
from dataclasses import dataclass
from . import ROOT_DIR

AGFS_DIR = os.path.join(ROOT_DIR, '..', 'glasses')

class ParseError(Exception):
    pass

class Status(Enum):
    STANDARD = 0
    PREFERRED = 1
    OBSOLETE = 2
    SPECIAL = 3
    MELT = 4

@dataclass
class Record:
    name: str
    dispersion_formula: int
    nd: float
    vd: float
    exclude_substitution: bool
    status: Status
    melt_freq: int
    comment: str
    tce: float
    density: float
    dPgF: float
    ignore_thermal_expansion: bool
    dispersion_coefficients: Sequence[float]
    min_lamb: float # in micron
    max_lamb: float # in micron
    d0: float
    d1: float
    d2: float
    e0: float
    e1: float
    ltk: float
    reference_temperature: float

def parse_catalog(file: TextIO) -> Tuple[Sequence[str], Sequence[Record]]:
    catalog_comments = []
    records = []
    data = None
    for line in file:
        if line.startswith('CC'):
            catalog_comments.append(line[2:].strip())

        elif line.startswith('NM'):
            if data is not None:
                records.append(Record(**data))
            data = {}
            terms = line.split()

            data['name'] = terms[1]
            data['dispersion_formula'] = int(terms[2])
            # terms[3] is MIL#, not used.
            data['nd'] = float(terms[4])
            data['vd'] = float(terms[5])
            data['exclude_substitution'] = bool(int(terms[6])) if len(terms) > 6 else False
            data['status'] = Status(int(terms[7])) if len(terms) > 7 else Status.STANDARD
            data['melt_freq'] = int(terms[8]) if len(terms) > 8 else 0

        elif line.startswith('GC'):
            data['comment'] = line[2:].strip()

        elif line.startswith('ED'): # Extra Data
            terms = line.split()
            data['tce'] = float(terms[1])
            # terms[2] is TCE in 100 to 300 deg. range - not used
            data['density'] = float(terms[3])
            data['dPgF'] = float(terms[4])
            data['ignore_thermal_expansion'] = bool(int(terms[5])) if len(terms) > 5 else False

        elif line.startswith('CD'): # Coefficient Data
            terms = line.split()
            data['dispersion_coefficients'] = [float(term) for term in terms[1:]]

        elif line.startswith('TD'): # Thermal Data
            terms = line.split()
            data['d0'] = float(terms[1])
            data['d1'] = float(terms[2])
            data['d2'] = float(terms[3])
            data['e0'] = float(terms[4])
            data['e1'] = float(terms[5])
            data['ltk'] = float(terms[6])
            data['reference_temperature'] = float(terms[7])

        elif line.startswith('LD'): # Lambda Data
            terms = line.split()
            data['min_lamb'] = float(terms[1])
            data['max_lamb'] = float(terms[2])

    if data is not None:
        records.append(Record(**data))

    return catalog_comments, records

def load_catalog(path: str) -> Dict[str, Record]:
    with open(path, 'rt') as file:
        return parse_catalog(file)





