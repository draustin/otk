"""Functions for accessing the refractiveindex.info database."""
import os
import yaml
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Tuple, Callable
from otk.types import Numeric, Array1D
from otk import PROPERTIES_DIR

DB_DIR = os.path.join(PROPERTIES_DIR, 'rii')


def get_library(cached=[None]) -> List[Dict]:
    # print('DB_DIR = ', DB_DIR)
    if cached[0] is None:
        db_path = os.path.join(DB_DIR, 'library.yml')
        cached[0] = yaml.load(open(db_path, 'r'), Loader=yaml.FullLoader)
    return cached[0]

def in_lim(x, a, b):
    """Greater than or equal to and less than or equal to."""
    return (x >= a) & (x <= b)

def print_lib():
    """Print all pages in the library in hierachical format."""
    for shelf in get_library():
        print(shelf['name'])
        for book in shelf['content']:
            if not book.has_key('BOOK'):
                continue
            print('  ' + book['BOOK'])
            for page in book['content']:
                if not page.has_key('PAGE'):
                    continue
                print('    ' + page['PAGE'])


def load_page(page: dict) -> Dict:
    file = open(os.path.join(DB_DIR, page['path']), 'r')
    string = ''
    for line in file:
        string = string + line.replace('\t', '')
    entry = yaml.load(string, Loader=yaml.FullLoader)
    return entry

@dataclass
class TabNK:
    """Tabulated (n,k) entryractiveindex.info database entry."""
    _lamb: Array1D
    _n: Array1D
    _k: Array1D
    range: Tuple[float, float] # TODO why is this in micron? Can get rid of it and use first and last entries of lamb.
    check_range: bool

    def __call__(self, lamb: Numeric, check_range: bool = None):
        check_range = (check_range if check_range is not None else self.check_range)
        lamb = np.array(lamb)
        mum = lamb * 1e6
        if check_range and in_lim(mum, *self.range).all():
            raise ValueError('Out of range ({0}--{1} micron). Pass check_range=False to ignore.'.format(*self.range))
        # interp gives error if passed complex valued yp, so must split up
        # real and imaginary parts
        # TODO lamb sorted? Speed?
        return np.interp(lamb, self._lamb, self._n) + 1j*np.interp(lamb, self._lamb, self._k)

    @classmethod
    def from_entry(cls, entry: dict, check_range: bool = True):
        data = []
        for line in entry['data'].split('\n'):
            try:
                data.append([float(x) for x in line.split(' ')])
            except ValueError as e:
                pass
        data = np.array(data)
        _lamb = data[:, 0] * 1e-6
        _n = data[:, 1]
        _k = data[:, 2]
        range = _lamb.min() * 1e6, _lamb.max() * 1e6
        return cls(_lamb, _n, _k, range, check_range)

@dataclass
class Formula:
    """Dispersion formulae - see database/doc/Dispersion formulas.pdf"""
    form_num: int
    c: Array1D
    range: Tuple[float, float]
    check_range: bool

    @classmethod
    def from_entry(cls, entry: dict, check_range: bool = True):
        form_num = int(entry['type'][7:])
        c = np.array([float(x) for x in entry['coefficients'].split(' ')])
        range = tuple(float(x) for x in entry['range'].split(' '))
        check_range = check_range
        return Formula(form_num, c, range, check_range)

    def __call__(self, lamb, check_range=None):
        check_range = (check_range if check_range is not None else self.check_range)
        lamb = np.array(lamb)
        mum = lamb * 1e6
        if check_range and not in_lim(mum, *self.range).all():
            raise ValueError('Out of range ({0}--{1} micron). Pass check_range=False to ignore.'.format(*self.range))
        if self.form_num == 1:
            # Sellmeier
            ns = 1 + self.c[0]
            mum2 = mum ** 2
            for a, b in zip(self.c[1::2], self.c[2::2]):
                ns += a * mum2 / (mum2 - b ** 2)
            n = ns ** 0.5
        elif self.form_num == 2:
            ns = 1 + self.c[0]
            mum2 = mum ** 2
            for a, b in zip(self.c[1::2], self.c[2::2]):
                ns += a * mum2 / (mum2 - b)
            n = ns ** 0.5
        elif self.form_num == 4:
            mum2 = mum ** 2
            ns = self.c[0]
            for a, b, c, d in self.c[1:9].reshape((2, 4)):
                ns += a * mum ** b / (mum2 - c ** d)
            for a, b in self.c[9:].reshape((-1, 2)):
                ns += a * mum ** b
            n = ns ** 0.5
        elif self.form_num == 6:
            # gases
            n = 1 + self.c[0]
            for a, b in zip(self.c[1::2], self.c[2::2]):
                n += a / (b - mum ** -2)
        else:
            raise ValueError('Unknown formula number %d' % self.form_num)
        return n


def parse_entry(entry: dict, check_range: bool = True) -> Callable:
    """Parse a yaml dictionary representing an entry, returning object that returns index given wavelength (in m)."""
    entry = entry['DATA'][0]
    type = entry['type']
    if type == 'tabulated nk':
        return TabNK.from_entry(entry, check_range)
    elif type[0:7] == 'formula':
        return Formula.from_entry(entry, check_range)
    else:
        raise ValueError('Unknown type ' + type)


def search(page_str: str = None, book_str: str = None, shelf_str: str = None) -> List[Dict]:
    """Return list of matching pages.

    None means wildcard.
    """
    pages = []
    for shelf in get_library():
        if shelf_str not in (shelf['name'], shelf['SHELF'], None):
            continue
        for book in shelf['content']:
            if book_str not in (book.get('BOOK'), None):
                continue
            for page in book['content']:
                if page_str not in (None, page.get('PAGE')):
                    continue
                pages.append(page)
    return pages


def lookup_index(page_str: str = None, book_str: str = None, shelf_str: str = None, check_range: bool = True) -> Callable:
    pages = search(page_str, book_str, shelf_str)
    if len(pages) != 1:
        raise ValueError(f'Found {len(pages)} matching pages.')
    return parse_entry(load_page(pages[0]), check_range)