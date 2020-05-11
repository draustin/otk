"""Read ANSI glass format (.agf) files used by Zemax.

Inspired by https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass.py.
"""
import os
from enum import Enum
from typing import Sequence, TextIO, Tuple, Dict
from dataclasses import dataclass
import numpy as np
from . import ROOT_DIR
from .types import Numeric

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
    lamb_tk: float
    reference_temperature: float

    def calc_index(self, lamb: Numeric, temperature: float = None):
        """Calculate refractive index.

        TODO braodcasting? return value type?

        Args:
            temperature: In deg C.
            pressure: in Pa.
        """
        cd = self.dispersion_coefficients
        w = np.asarray(lamb)*1e6 # TODO rename to mum

        # Calculate n at self.reference_temperature.
        if self.dispersion_formula == 1: ## Schott
            n = np.sqrt(cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8))
        elif self.dispersion_formula == 2: ## Sellmeier1
            n = np.sqrt((cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + (cd[4] * w**2 / (w**2 - cd[5])) + 1.)
        elif self.dispersion_formula == 5: ## Conrady
            n = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
        else:
            raise ValueError(f'Unknown dispersion formula {self.dispersion_formula}.')

        if temperature is not None:
            # Schott Glass Technologies Inc formula.
            dT = temperature - self.reference_temperature
            dn = ((n**2 - 1.0) / (2.0 * n)) * (self.d0 * dT + self.d1 * dT**2 + self.d2 * dT**3 + ((self.e0 * dT + self.e1 * dT**2) / (w**2 - np.sign(self.lamb_tk)*self.lamb_tk**2)))
            n += dn

        return n

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
            data['lamb_tk'] = float(terms[6])
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
        comments, records = parse_catalog(file)

    catalog = {r.name:r for r in records}
    return catalog





