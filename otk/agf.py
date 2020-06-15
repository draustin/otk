"""Read ANSI glass format (.agf) files used by Zemax.

Inspired by https://github.com/nzhagen/zemaxglass/blob/master/ZemaxGlass.py.
"""
import os
from enum import Enum
from typing import Sequence, TextIO, Tuple, Dict, Callable, Iterable, List
from dataclasses import dataclass
import numpy as np
import chardet
from .types import Numeric
from . import ri

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
    comments: Sequence[str]
    tce: float
    density: float
    dPgF: float
    dispersion_coefficients: Sequence[float]
    min_lamb: float # in micron
    max_lamb: float # in micron
    ignore_thermal_expansion: bool = False
    exclude_substitution: bool = False
    status: Status = Status.STANDARD
    melt_freq: int = 0
    d0: float = 0.
    d1: float = 0.
    d2: float = 0.
    e0: float = 0.
    e1: float = 0.
    lamb_tk: float = 0.
    reference_temperature: float = 0.

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

    def fix_temperature(self, temperature: float = None) -> 'Index':
        return Index(self, temperature)

    def __str__(self):
        return self.name

@dataclass
class Index(ri.Index):
    record: Record
    temperature: float

    def __call__(self, lamb: Numeric) -> Numeric:
        return self.record.calc_index(lamb, self.temperature)

    def __str__(self):
        return self.record.name


Catalog =  Dict[str, Record]


def is_command(token: str):
    return len(token) == 2 and token.isalnum()

def parse_tokens(data: dict, records: List[Record]):
    token = yield
    while True:
        command = token
        if command == 'NM':
            if len(data) > 0:
                records.append(Record(**data))
                data.clear()
            data['comments'] = []

            data['name'] = yield
            # Schott catalog sometimes uses exponential notation here and below.
            token = yield
            data['dispersion_formula'] = int(float(token))
            # terms[3] is MIL#, not used.
            yield
            token = yield
            data['nd'] = float(token)
            token = yield
            data['vd'] = float(token)
            token = yield
            if is_command(token):
                continue
            data['exclude_substitution'] = bool(int(float(token)))
            token = yield
            if is_command(token):
                continue
            data['status'] = Status(int(float(token)))
            token = yield
            if is_command(token):
                continue
            if token != '-':
                data['melt_freq'] = int(token)
            token = yield
        elif command == 'ED': # Extra Data
            token = yield
            data['tce'] = float(token)
            # terms[2] is TCE in 100 to 300 deg. range - not used
            yield
            token = yield
            data['density'] = float(token)
            token = yield
            data['dPgF'] = float(token)
            token = yield
            data['ignore_thermal_expansion'] = bool(int(token))
            token = yield
        elif command == 'CD': # Coefficient Data
            data['dispersion_coefficients'] = []
            while True:
                token = yield
                if is_command(token):
                    break
                data['dispersion_coefficients'].append(float(token))
        elif command == 'TD': # Thermal Data
            for key in ('d0', 'd1', 'd2', 'e0', 'e1', 'lamb_tk', 'reference_temperature'):
                token = yield
                if is_command(token):
                    # Some catalogs (e.g. SCHOTT) have missing TD entries.
                    break
                data[key] = float(token)
            token = yield
        elif command == 'LD': # Lambda Data
            for key in ('min_lamb', 'max_lamb'):
                token = yield
                data[key] = float(token)
            token = yield
        elif command == 'OD':
            # Don't record this
            # OD <rel cost> <CR> <FR> <SR> <AR> <PR>
            token = yield
        else:
            token = (yield token)


def parse_catalog(lines: Iterable[str]) -> Tuple[Sequence[str], Sequence[Record]]:
    catalog_comments = []
    records = []
    data = {}
    token_parser = parse_tokens(data, records)
    ignored_tokens = []
    next(token_parser)
    for line_num, line in enumerate(lines):
        try:
            if line.startswith('CC'):
                catalog_comments.append(line[2:].strip())
            elif line.startswith('GC'):
                data['comments'].append(line[2:].strip())
            else:
                for token in line.split():
                    token = token_parser.send(token)
                    if token is not None:
                        ignored_tokens.append((line_num, token))
        except (ParseError, ValueError) as e:
            raise ParseError(f'Line {line_num + 1}: ' + e.args[0]) from e

    if len(data) > 0:
        records.append(Record(**data))

    return catalog_comments, records


def read_lines(path: str) -> Iterable[str]:
    """Detect file encoding and read lines."""
    with open(path, 'rb') as file:
        raw = file.read()
    encoding = chardet.detect(raw)['encoding']
    text = raw.decode(encoding)
    lines = text.splitlines()
    return lines


def load_catalog(path: str) -> Catalog:
    lines = read_lines(path)
    try:
        comments, records = parse_catalog(lines)
    except ParseError as e:
        raise ParseError(f'{path}: ' + e.args[0]) from e
    catalog = {r.name:r for r in records}
    return catalog







