"""Abstractions for the refractive index of materials.

TODO merge wth refindinf.
"""
from typing import Callable
import scipy.interpolate
import numpy as np

# TODO Having color attribute introduces ugly code coupling - get rid of.
# TODO consider changing Index -> Callable.
class Index:
    """Defines a refractive index as a function of wavelength.

    Attributes:
        section_color (tuple): RGB(A) color for 2D section plots, or None for no color.
    """

    def __init__(self, section_color=None):
        assert section_color is None or len(section_color) in (3, 4)
        self.section_color = section_color

    def __call__(self, lamb):
        raise NotImplementedError()


class FixedIndex(Index):
    """A material with a refractive index that is independent of wavelength.
    """

    def __init__(self, index, name=None, section_color=None):
        Index.__init__(self, section_color)
        self.index = index
        if name is None:
            name = 'n=%.4f'%index
        self.name = name

    def __call__(self, lamb):
        return self.index

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'FixedIndex(%.6f, %s, %r)'%(self.index, self.name, self.section_color)


class MicronFormulaIndex(Index):
    def __init__(self, n_fun: Callable, name: str, section_color: tuple = None):
        Index.__init__(self, section_color)
        self.n_fun = n_fun
        self.name = name
        self.offset = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'MicronFormulaIndex(%r, %s, %r)'%(self.n_fun, self.name, self.section_color)

    def __call__(self, lamb):
        return self.n_fun(lamb*1e6)+self.offset

class InterpolatedIndex(Index):
    def __init__(self, lambs, ns, section_color: tuple=None):
        Index.__init__(self, section_color)
        self.lambs = lambs
        self.ns = ns

    def __call__(self, lamb):
        return scipy.interpolate.interp1d(self.lambs, self.ns, 'cubic', -1, False, False, 'extrapolate')(lamb)


# Define some specific materials. Data from refractiveindex.info.
vacuum = FixedIndex(1, 'vacuum')
air = MicronFormulaIndex(lambda x: 1+0.05792105/(238.0185-x**-2)+0.00167917/(57.362-x**-2), 'air')
fused_silica = MicronFormulaIndex(lambda x: (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5, 'fused_silica', (0.1, 0.4, 0.8))
#fused_silica = FixedIndex(1.4523, 'SiO2', )  # At 860 nm. TODO include wavelength dependence.
N_LASF41 = FixedIndex(1.8194, 'N-LASF41')  # at 850 nm.
N_LASF31 = FixedIndex(1.8637, 'N-LASF31')  # 850 nm.
D263T = FixedIndex(1.51514523, 'D263T')
N_BK7 = MicronFormulaIndex(lambda x: (1 + 1.03961212/(1 - 0.00600069867/x ** 2) + 0.231792344/(
            1 - 0.0200179144/x ** 2) + 1.01046945/(1 - 103.560653/x ** 2)) ** .5, 'N-BK7')
AlGaAs = FixedIndex(3.4, 'AlGaAs', (0.5, 0, 0.5))
#PMMA = MicronFormulaIndex(lambda x:(1+0.99654/(1-0.00787/x**2)+0.18964/(1-0.02191/x**2)+0.00411/(1-3.85727/x**2))**.5, 'PMMA')
PMMA = MicronFormulaIndex(lambda x:(2.1778+6.1209e-3*x**2-1.5004e-3*x**4+2.3678e-2*x**-2-4.2137e-3*x**-4+7.3417e-4*x**-6-4.5042e-5*x**-8)**.5, 'PMMA')

# Used by Ingeneric microlens arrays: https://ingeneric.com/wp-content/uploads/2018/11/INGENERIC_MLA.pdf.
KVC89 = MicronFormulaIndex(lambda x:(3.1860388-0.013756822*x**2+0.029614017*x**-2+0.0012383727*x**-4-8.0134175e-05*x**-6+7.2330635e-06*x**-8)**.5, 'K-VC89')

# Schott Borofloat 33 - substrate for Moxtek wire grid plate polarizers
Borofloat33 = InterpolatedIndex(np.arange(400, 1100, 100)*1e-9, [1.484, 1.476, 1.471, 1.468, 1.466, 1.464, 1.463])

class ShiftedIndex(Index):
    def __init__(self, index:Index, shift:float):
        Index.__init__(self)
        self.index = index
        self.shift = shift

    def __call__(self, lamb):
        return self.index(lamb) + self.shift

# Hack - there's a slight difference.
PMMA_Zemax = ShiftedIndex(PMMA, -0.0002)