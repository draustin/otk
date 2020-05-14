"""Defining and analysing axisymmetric optical systems."""
import itertools
from functools import singledispatch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Mapping
import numpy as np
import scipy.optimize
from . import abcd, paraxial, functions, ri
from .functions import calc_sphere_sag

# TODO Make Interface composition of Surface and refractive indeices.

@dataclass
class Interface:
    """An interface between two media in an axisymmetric optical system.

    Attributes:
        n1: Refractive index on first side.
        n2: Index on second side.
        roc (scalar): Radius of curvature.
        radius (scalar): Radius/half-diagonal of interface.
    """
    n1: ri.Index
    n2: ri.Index
    roc: float
    radius: float

    def __post_init__(self):
        self.sag = functions.calc_sphere_sag(self.roc, self.radius)

    def __str__(self):
        return '%s - ROC %.3g mm, radius %.3g mm - %s'%(self.n1, self.roc*1e3, self.radius*1e3, self.n2)
    #
    # def __repr__(self):
    #     return 'Interface(n1=%r, n2=%r, roc=%g, radius=%g)'%(self.n1, self.n2, self.roc, self.radius)

    def get_abcd(self, lamb):
        return abcd.curved_interface(self.n1(lamb), self.n2(lamb), self.roc)

    def make_parameter_string(self):
        if np.isfinite(self.roc):
            return 'ROC %.3f mm'%(self.roc*1e3)
        else:
            return 'plane'

    def reverse(self):
        return Interface(self.n2, self.n1, -self.roc, self.radius)

    def get_points(self, num_points=32):
        """Returns (num_points, 2) array."""
        x = np.linspace(-self.radius, self.radius, num_points)
        # TODO use calc_sag.
        h = calc_sphere_sag(self.roc, x)
        xys = np.c_[x, h]
        return xys

    def calc_sag(self, rho, derivative: bool = False):
        """Calculate sag of surface.

        Positive ROC means positive sag.

        Args:
            rho: Distance from center.
            derivative: If True, derivative is returned as well.

        """
        sag = functions.calc_sphere_sag(self.roc, rho)
        if derivative:
            grad_sag = functions.calc_sphere_sag(self.roc, rho, True)
            return sag, grad_sag
        else:
            return sag

    def calc_aperture(self, x, y, shape: str):
        """Returns binary aperture mask at (x, y)."""
        if shape == 'circle':
            return x**2 + y**2 <= self.radius
        elif shape == 'square':
            half_side = self.radius/2**0.5
            return (abs(x) <= half_side) & (abs(y) <= half_side)
        else:
            raise ValueError(f'Unknown shape {shape}.')

    def calc_mask(self, lamb, rho, derivative: bool = False):
        n1 = self.n1(lamb)
        n2 = self.n2(lamb)
        jdeltak = 2.j*np.pi/lamb*(n1 - n2)

        sag, grad_sag = self.calc_sag(rho, True)

        f = np.exp(jdeltak*sag)
        if derivative:
            gradf = jdeltak*grad_sag*f
            return f, gradf
        else:
            return f

@dataclass
class ConicInterface(Interface):
    """An Interface with conic and aspheric terms.

    We take kappa as defined in Spencer and Murty, JOSA 52(6) p 672, 1962. Note that in some other contexts,
    k = kappa - 1 is used as the conic constant. This is the case in Zemax i.e. kappa here equals Zemax conic constant
    plus 1.

    Useful links:
    https://www.iap.uni-jena.de/iapmedia/de/Lecture/Advanced+Lens+Design1393542000/ALD13_Advanced+Lens+Design+7+_+Aspheres+and+freeforms.pdf

    Args:
        n1: Index on first side.
        n2: Index on second side.
        roc (scalar): Radius of curvature.
        kappa (scalar): Conic parameter. Special values:
            kappa < 0: Hyperboloid.
            kappa = 0: Paraboloid.
            0 < kappa < 1: Elipsoid of revolution about major axis.
            kappa = 1: Sphere
            kappa > 1: Elipsoid of revolution about minor axis.
        alphas (sequence): Second and higher order coefficients.
    """
    kappa: float
    alphas: Sequence

    def __post_init__(self):
        self.alphas = np.asarray(self.alphas)

    def __str__(self):
        return '%s - ROC %g mm, radius %g mm, kappa %.3f, alphas %s - %s)'%(
        self.n1, self.roc*1e3, self.radius*1e3, self.kappa, ', '.join('%g'%v for v in self.alphas), self.n2)
    #
    # def __repr__(self):
    #     return 'ConicInterface(%r, %r, %r, %r, %r, %r)'%(
    #     self.n1, self.n2, self.roc, self.radius, self.kappa, self.alphas)

    def reverse(self):
        return ConicInterface(self.n2, self.n1, -self.roc, self.radius, self.kappa, -self.alphas)

    def calc_sag(self, rho, derivative: bool = False):
        """Calculate sag of surface.

        Positive ROC means positive sag.

        Args:
            rho: Distance from center.
            derivative: If True, tuple of sag and its derivative is returned.
        """
        sag = functions.calc_conic_sag(self.roc, self.kappa, self.alphas, rho, False)
        if derivative:
            grad_sag = functions.calc_conic_sag(self.roc, self.kappa, self.alphas, rho, True)
            return sag, grad_sag
        else:
            return sag


@dataclass
class Train:
    """Immutable class representing a sequence of Interfaces with defined spaces in between and at the ends.

    Systems that begins and/or ends on an interface is handled as special cases by setting the first and/or last element
    of spaces to zero.

    TODO If needed, many methods are candidates for memoization.
    TODO make frozen
    TODO add make method instead of correcting in post_init (cleaner type checking).
    """
    interfaces: Tuple[Interface]
    spaces: Tuple[float]

    def __post_init__(self):
        assert len(self.spaces) == len(self.interfaces) + 1
        # assert interfaces[0].n1 is None
        # assert interfaces[-1].n2 is None
        for i0, i1 in zip(self.interfaces[:-1], self.interfaces[1:]):
            assert i0.n2 == i1.n1
        self.interfaces = tuple(self.interfaces)
        self.spaces = tuple(float(space) for space in self.spaces)
        self.length = sum(self.spaces)

        # self.mf = self._calc_abcd()  # self.mb = np.linalg.inv(self.mf)

    def __str__(self):
        def format_space(s):
            return f'{s*1e3:.3g} mm'
        return format_space(self.spaces[0]) + ' / ' + ' / '.join(f'{i} / {format_space(s)}' for i, s in zip(self.interfaces, self.spaces[1:]))

    def make_parameter_strings(self):
        strs = ['total length %.3f mm'%(self.length*1e3)]
        interface = self.interfaces[0]
        strs.append('%s, thickness %.3f mm'%(interface.n1, self.spaces[0]*1e3))
        for interface, space in zip(self.interfaces, self.spaces[1:]):
            strs.append(interface.make_parameter_string())
            strs.append('%s, thickness %.3f mm'%(interface.n2, space*1e3))
        return tuple(strs)

    def pad(self, space0, space1=None):
        if space1 is None:
            space1 = space0
        spaces = (self.spaces[0] + space0,) + self.spaces[1:-1] + (self.spaces[-1] + space1,)
        return Train(self.interfaces, spaces)

    def pad_to_transform(self, lamb=None):
        """Pad first and last space so train performs Fourier transform.

        Returns:
            train: Same as original except that first and last space are modified.
        """
        return self.pad(*self.get_working_distances(
            lamb))  # ws = self.working_distances  # spaces = (self.spaces[0] + ws[0],) + self.spaces[1:-1] + (self.spaces[-1] + ws[1],)  # return Train(self.interfaces, spaces)

    def calc_abcd(self, lamb=None):
        m = abcd.propagation(self.spaces[0])
        for interface, space in zip(self.interfaces, self.spaces[1:]):
            # Apply interface.
            m = np.matmul(interface.get_abcd(lamb), m)
            # Propagate to next surface (or exit surface).
            m = np.matmul(abcd.propagation(space), m)
        return m

    def calc_abcd_bi(self, lamb=None):
        mf = self.calc_abcd(lamb)
        mb = np.linalg.inv(mf)
        return mf, mb

    def get_focal_lengths(self, lamb=None):
        """Rear and front focal lengths."""
        mf, mb = self.calc_abcd_bi(lamb)
        return np.asarray((1/mb[1, 0], -1/mf[1, 0]))

    def get_effective_focal_length(self, lamb=None):
        focal_lengths = self.get_focal_lengths(lamb)
        efl = focal_lengths[0]/self.interfaces[0].n1(lamb)
        assert np.isclose(focal_lengths[1]/self.interfaces[-1].n2(lamb), efl)
        return efl

    def get_working_distances(self, lamb=None):
        # Move this into abcd module?
        mf, mb = self.calc_abcd_bi(lamb)
        wf = -mf[0, 0]/mf[1, 0]
        wb = mb[0, 0]/mb[1, 0]
        return wb, wf

    def get_principal_planes(self, lamb=None):
        """Calculate principal planes.

        The distances are defined from the ends of the system, including the first and last spaces.  They are positive to
        the right.

        Returns:
            ppb, ppf: Distances to principal planes before and after the system.
        """
        fb, ff = self.get_focal_lengths(lamb)
        wb, wf = self.get_working_distances(lamb)
        ppb = fb - wb
        ppf = wf - ff
        return ppb, ppf

    def get_petzval_sum(self, lamb=None):
        """Calculate Petzval sum of train.

        Args:
            lamb (scalar): Wavelength.

        Returns:
            scalar: Petzval sum.
        """
        ns = itertools.chain((i.n1 for i in self.interfaces), [self.interfaces[-1].n2])
        n0s = [n(lamb) for n in ns]
        rocs = [i.roc for i in self.interfaces]
        return paraxial.calc_multi_element_lens_petzval_sum(n0s, rocs)

    @classmethod
    def make_singlet_transform1(cls, n: ri.Index, ws: Tuple[float, float], f: float, radius: float, lamb: float = None,
            interface_class=Interface, interface_args=None, interface_kwargs=None, n_ext: ri.Index = ri.vacuum) -> 'Train':
        """Make a singlet Fourier transform given working distances and focal length.

        Args:
            n (ri.Index): Defines refractive index.
            w (scalar of pair of scalars): Working distance(s).
            f (scalar): Transform focal length.
            lamb (scalar): Design wavelength.

        Returns:
            roc: radius of curvature
            d: center thickness
        """
        try:
            ws = ws[0], ws[1]
        except TypeError:
            ws = ws, ws
        if interface_args is None:
            interface_args = (), ()
        if interface_kwargs is None:
            interface_kwargs = {}, {}
        rocs, d = paraxial.design_singlet_transform(n(lamb), ws, f, n_ext(lamb))
        interfaces = (interface_class(n_ext, n, rocs[0], radius, *interface_args[0], **interface_kwargs[0]),
                      interface_class(n, n_ext, rocs[1], radius, *interface_args[1], **interface_kwargs[1]))
        train = cls(interfaces, (ws[0], d, ws[1]))
        return train

    @classmethod
    def make_singlet_transform2(cls, n, f, shape, thickness, radius, lamb=None):
        """Make singlet transform given shape and thickness.

        Args:
            n: Refractive index.
            f: Focal length.
            shape: Coddington shape factor. 0 = symmetric. +1 is convex-plano, -1 is plano-convex.
            thickness: Center thickness.
            lamb (scalar): Design wavelength.

        Returns:
            Train object, including working distances.
        """
        n0 = n(lamb)
        rocs = paraxial.design_singlet(n0, f, shape, thickness)
        f, h1, h2 = paraxial.calc_thick_spherical_lens(n0, *rocs, thickness)
        interfaces = Interface(ri.vacuum, n, rocs[0], radius), Interface(n, ri.vacuum, rocs[1], radius)
        train = Train(interfaces, (f - h1, thickness, f + h2))
        assert np.allclose(train.get_focal_lengths(lamb), (f, f))
        assert np.allclose(train.get_working_distances(lamb), (0, 0))
        return train

    @classmethod
    def design_singlet(cls, n:ri.Index, f:float, shape:float, thickness:float, radius:float, lamb=None, ne=ri.vacuum):
        """Make train representing a singlet lens.

        Args:
            n: Lens refractive index.
            f: Focal length, inverse of lens power. Sometimes called effective focal length. Distance to
                focus of parallel rays divided by external refractive index.
            shape: Coddington shape factor. 0 = symmetric. +1 is convex-plano, -1 is plano-convex.
            thickness: Center thickness.
            radius: Radius/half-diagonal.
            lamb: Design wavelength.
            ne: External refractive index.

        Returns:
            Train: First and last space is zero.
        """
        n0 = n(lamb)
        ne0 = ne(lamb)
        rocs = paraxial.design_singlet(n0, f, shape, thickness, ne0)
        f_, h1, h2 = paraxial.calc_thick_spherical_lens(n0, *rocs, thickness, ne0)
        assert np.isclose(f_, f)
        interfaces = Interface(ne, n, rocs[0], radius), Interface(n, ne, rocs[1], radius)
        train = Train(interfaces, (0, thickness, 0))
        assert np.allclose(train.get_focal_lengths(lamb)/ne0, f)
        return train

    def make_singlet_transform_conic(cls, n, ws, f, radius, lamb=None, kappas=(1, 1), alphass=((), ())):
        return cls.make_singlet_transform1(n, ws, f, lamb, radius, ConicInterface, list(zip(kappas, alphass)))

    def reverse(self):
        interfaces = tuple(i.reverse() for i in self.interfaces[::-1])
        return Train(interfaces, self.spaces[::-1])

    def __add__(self, other):
        interfaces = self.interfaces + other.interfaces
        if len(self.interfaces) > 0 and len(other.interfaces) > 0:
            assert self.interfaces[-1].n2 == other.interfaces[0].n1
        spaces = self.spaces[:-1] + (self.spaces[-1] + other.spaces[0],) + other.spaces[1:]
        return Train(interfaces, spaces)

    def pad_to_half_transform(self, lamb: float = None, inner_space: float = None, f: float = None):
        """Adjust input and output space to perform half of a lens.

        Only one of inner_space and f should be given.

        Args:
            lamb: Design wavelength.
            inner_space: Space added between self and reversed self.
            f: Focal length of resulting transform.

        Returns:
            train
        """
        if inner_space is None:
            # Need to add the inner principal plane distances.
            inner_space = paraxial.infer_combined_lens_separation(*self.get_focal_lengths(lamb), f) + 2* \
                          self.get_principal_planes(lamb)[1]
        else:
            assert f is None
        pair = (self + Train([], [inner_space]) + self.reverse()).pad_to_transform(lamb)
        train = Train(self.interfaces, (pair.spaces[0],) + self.spaces[1:-1] + (self.spaces[-1] + inner_space/2,))
        return train

    def make_html_table(self, doc, lamb=None):
        """Make HTML table list spaces and interfaces.

        Args:
            doc (yattag.Doc object): Table is added to this.
        """
        with doc.tag('table', cellspacing=10):
            doc.line('caption', 'List of material, its thickness, and profile of the following interface')
            with doc.tag('tr'):
                doc.line('th', 'Color')
                doc.line('th', 'Material')
                doc.line('th', 'Index')
                doc.line('th', 'Thickness (mm)')
                doc.line('th', 'ROC (mm)')
            n = self.interfaces[0].n1
            for space, interface in zip(self.spaces, self.interfaces + (None,)):
                with doc.tag('tr'):
                    color = n.section_color
                    if color is None:
                        color = (1, 1, 1)
                    # doc.line('td', ' ', bgcolor='rgb(%d,%d,%d)'%tuple(int(round(c*255)) for c in color), width=100)
                    doc.line('td', ' ', style='background-color:rgb(%d,%d,%d)'%tuple(int(round(c*255)) for c in color),
                             width=100)
                    doc.line('td', n.name)
                    doc.line('td', n(lamb))
                    doc.line('td', '%.3f'%(space*1e3))
                    if interface is not None:
                        doc.line('td', '%.3f mm'%(interface.roc*1e3))
                        n = interface.n2

    def subset(self, start: int = None, stop: int = None):
        """Make train consisting of interval subset of interfaces."""
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.interfaces)
        if stop < 0:
            spaces_stop = stop + len(self.spaces)
        else:
            spaces_stop = stop + 1
        return Train(self.interfaces[start:stop], self.spaces[start:spaces_stop])

    def crop_to_finite(self) -> 'Train':
        # Crop to exclude surfaces of infinite thickness.
        infs = [n for n, space in enumerate(self.spaces) if np.isinf(space)]
        if len(infs) == 0:
            cropped = self
        else:
            if len(infs) == 1:
                inf = infs[0]
                if inf < len(self.spaces)/2:
                    spaces = (0.,) + self.spaces[inf + 1:]
                    interfaces = self.interfaces[inf:]
                else:
                    spaces = self.spaces[:inf] + (0.,)
                    interfaces = self.interfaces[:inf]
            else:
                raise ValueError(f"Don't know how to handle {len(infs)} surfaces with infinite thickness.")
            cropped = Train(interfaces, spaces)
        return cropped

    def consolidate(self) -> 'Train':
        """Remove interfaces with same material on either side."""
        # Remove same material interfaces
        space = self.spaces[0]
        n = self.interfaces[0].n1
        spaces = []
        interfaces = []
        for next_space, interface in zip(self.spaces[1:], self.interfaces):
            if interface.n2 == n:
                space += next_space
            else:
                spaces.append(space)
                interfaces.append(interface)
                space = next_space
                n = interface.n2
        spaces.append(space)
        return Train(interfaces, spaces)

class Surface(ABC):
    """An axisymmetric surface between two media.

    TODO define radius. Is sag constant outside this i.e. is sag(rho) = sag(radius) for rho > radius?"""
    roc: float
    radius: float

    def __init__(self, roc: float, radius: float):
        self.roc = float(roc)
        self.radius = float(radius)

    def __eq__(self, other):
        return type(self) is type(other) and self.roc == other.roc and self.radius == other.radius

    def isclose(self, other):
        return np.isclose(self.roc, other.roc) and np.isclose(self.radius, other.radius)

    @property
    @abstractmethod
    def sag_range(self) -> np.ndarray:
        """Returns minimum and maximum sag.

        TODO must be tight bound? Convert to method so can specify rho interval?"""
        pass

    @abstractmethod
    def to_interface(self, n1: ri.Index, n2: ri.Index):
        pass

    @abstractmethod
    def reverse(self) -> 'Surface':
        pass

    @abstractmethod
    def calc_sag(self, rho, derivative: bool = False):
        """Calculate sag of surface.

        Positive ROC means positive sag.

        Args:
            rho: Distance from center.
            derivative: If True, derivative is returned as well.

        """
        pass

class SphericalSurface(Surface):
    @property
    def sag_range(self) -> np.ndarray:
        # TODO this assumes monotonicity
        sag = self.calc_sag(self.radius)
        return np.asarray((min(sag, 0), max(sag, 0)))

    def to_interface(self, n1, n2):
        return Interface(n1, n2, self.roc, self.radius)

    def reverse(self) -> 'SphericalSurface':
        return SphericalSurface(-self.roc, self.radius)

    def __repr__(self):
        return f'SphericalSurface(roc={self.roc}, radius={self.radius})'

    # TODO move to rt1
    # def make_profile(self):
    #     if np.isfinite(self.roc):
    #         return rt1.SphericalProfile(self.roc)
    #     else:
    #         return rt1.PlanarProfile()

    def calc_sag(self, rho, derivative: bool = False):
        """Calculate sag of surface.

        Positive ROC means positive sag.

        Args:
            rho: Distance from center.
            derivative: If True, derivative is returned as well.

        """
        sag = functions.calc_sphere_sag(self.roc, rho)
        if derivative:
            grad_sag = functions.calc_sphere_sag(self.roc, rho, True)
            return sag, grad_sag
        else:
            return sag

@singledispatch
def to_surface(obj, *args, **kwargs) -> Surface:
    raise NotImplementedError

@to_surface.register
def _(obj: Interface, radius: float=None) -> Surface:
    if radius is None:
        radius = obj.radius
    return SphericalSurface(obj.roc, radius)

class ConicSurface(Surface):
    def __init__(self, roc: float, radius: float, kappa: float, alphas: Sequence[float] = None):
        Surface.__init__(self, roc, radius)
        self.kappa = kappa
        if alphas is None:
            alphas = []
        self.alphas = np.asarray(alphas)

    def to_interface(self, n1, n2):
        return ConicInterface(n1, n2, self.roc, self.radius, self.kappa, self.alphas)

    def reverse(self) -> 'ConicSurface':
        return ConicSurface(-self.roc, self.radius, self.kappa, -self.alphas)

    # TODO move to rt1
    # def make_profile(self):
    #     return rt1.ConicProfile(self.roc, self.kappa, self.alphas)

    @property
    def sag_range(self) -> np.ndarray:
        s = self.calc_sag(self.radius)
        return np.array((min(s, 0), max(s, 0)))

    def __repr__(self):
        return f'ConicSurface(roc={self.roc}, radius={self.radius}, kappa={self.kappa}, alphas={self.alphas})'

    def __eq__(self, other):
        return Surface.__eq__(self, other) and self.kappa == other.kappa and np.array_equal(self.alphas, other.alphas)

    def isclose(self, other):
        return Surface.isclose(self, other) and np.isclose(self.kappa, other.kappa) and np.allclose(self.alphas, other.alphas)

    def calc_sag(self, rho, derivative: bool = False):
        """Calculate sag of surface.

        Positive ROC means positive sag.

        Args:
            rho: Distance from center.
            derivative: If True, tuple of sag and its derivative is returned.
        """
        sag = functions.calc_conic_sag(self.roc, self.kappa, self.alphas, rho, False)
        if derivative:
            grad_sag = functions.calc_conic_sag(self.roc, self.kappa, self.alphas, rho, True)
            return sag, grad_sag
        else:
            return sag

@to_surface.register
def _(obj: ConicInterface, radius: float=None) -> ConicSurface:
    if radius is None:
        radius = obj.radius
    return ConicSurface(obj.roc, radius, obj.kappa, obj.alphas)

class SegmentedInterface(Interface):
    def __init__(self, n1, n2, segments: Sequence[Surface], sags: Sequence[float]):
        radius = sum(s.radius for s in segments)
        Interface.__init__(self, n1, n2, segments[0].roc, radius)
        if sags is None:
            sags = np.zeros((len(segments),))
        sags = np.asarray(sags)
        assert len(sags) == len(segments)
        self.segments = segments
        self.sags = sags

    def __repr__(self):
        return 'SegmentedInterface(%r, %r, %r, %r)'%(self.n1, self.n2, self.segments, self.sags)

    def __str__(self):
        # TODO improve
        return '%s - (%s) - %s'%(self.n1, ', '.join(str(s) for s in self.segments), self.n2)

    def reverse(self):
        segments = [s.reverse() for s in self.segments]
        return SegmentedInterface(self.n2, self.n1, segments, -self.sags)

    # TODO move to rt1
    # def make_profile(self):
    #     # TODO move to single dispatch in rt
    #     assert len(self.segments) == 2
    #     profiles = [s.make_profile() for s in self.segments]
    #     boundary = rt1.CircleBoundary(self.segments[0].radius*2)
    #     return rt1.BinaryProfile(profiles, boundary)

    def calc_sag(self, rho, derivative: bool = False):
        raise NotImplementedError()

# TODO make radius of each segment the outer radius rather than the incremental radius.
class SegmentedSurface(Surface):
    def __init__(self, segments: Sequence[Surface],  sags: Sequence[float]):
        radius = sum(s.radius for s in segments)
        Surface.__init__(self, segments[0].roc, radius)
        sags = np.asarray(sags)
        assert len(sags) == len(segments)
        self.segments = tuple(segments)
        self.sags = sags

    def __repr__(self):
        return f'SegmentedSurface({self.segments}, {self.sags})'

    #def __str__(self):
    #    return ', '.join(str(s) for s in self.segments)

    def reverse(self) -> 'SegmentedSurface':
        segments = [s.reverse() for s in self.segments]
        return SegmentedSurface(segments)

    def to_interface(self, n1, n2):
        return SegmentedInterface(n1, n2, self.segments, self.sags)

    def calc_sag(self, rho: float, derivative: bool = False):
        rho = float(rho)
        rho0 = 0
        for segment in self.segments:
            rho0 += segment.radius
            if rho <= rho0:
                return segment.calc_sag(rho)
        # TODO clamp?
        return self.segments[-1].calc_sag(rho)

    @property
    def sag_range(self) -> np.ndarray:
        rngs = [segment.sag_range for segment in self.segments]
        return np.array((min(rng[0] for rng in rngs), max(rng[1] for rng in rngs)))

@to_surface.register
def _(obj: SegmentedInterface, radius: float=None) -> SegmentedSurface:
    # TODO sort this out
    if radius is None:
        radius = obj.radius
    return SegmentedSurface(obj.segments, obj.sags)

class Singlet:
    """A singlet.

    Attributes:
        surfaces: Front and back surfaces.
        thickness: Center thickness.
        n: Defines internal refractive index.
    """
    surfaces: Tuple[Surface, Surface]
    thickness: float
    n: ri.Index
    radius: float

    def __init__(self, surfaces: Tuple[Surface, Surface], thickness: float, n: ri.Index):
        self.surfaces = surfaces
        self.thickness = float(thickness)
        self.n = n
        self.radius = surfaces[0].radius
        assert surfaces[1].radius == self.radius

    def make_interfaces(self, n1, n2=None):
        if n2 is None:
            n2 = n1
        return self.surfaces[0].to_interface(n1, self.n), self.surfaces[1].to_interface(self.n, n2)

    def __repr__(self):
        return f'Singlet(surfaces=({self.surfaces[0]}, {self.surfaces[1]}), thickness={self.thickness}, n={self.n})'

    def reverse(self) -> 'Singlet':
        return Singlet((self.surfaces[1].reverse(), self.surfaces[0].reverse()), self.thickness, self.n)

    @classmethod
    def from_focal_length(cls, f: float, n: ri.Index, center_thickness: float, radius: float, shape_factor: float = 0,
            n_external: ri.Index = ri.vacuum, lamb: float = None):
        rocs = paraxial.design_singlet(n(lamb), f, shape_factor, center_thickness, n_external(lamb))
        return cls.from_rocs(rocs, n, center_thickness, radius)

    @classmethod
    def from_rocs(cls, rocs: Tuple[float, float], n: ri.Index, center_thickness: float, radius: float):
        return cls(tuple(SphericalSurface(roc, radius) for roc in rocs), center_thickness, n)

    def to_train(self, n1: ri.Index, n2: ri.Index = None) -> Train:
        """Convert singlet to train.

        Args:
            n1: Initial external refractive index.
            n2: Final external refractive index. Defaults to n1.

        Returns:
            self as a train, with zero initial and final spaces.
        """
        if n2 is None:
            n2 = n1
        interfaces = self.surfaces[0].to_interface(n1, self.n), self.surfaces[1].to_interface(self.n, n2)
        return Train(interfaces, (0, self.thickness, 0))


class SingletSequence:
    def __init__(self, singlets: Sequence[Singlet], spaces: Sequence[float], n_external: ri.Index = ri.vacuum):
        """A sequence of singlets in a homogeneous medium with external and internal spaces.

        The first and last spaces must be included i.e. len(spaces) = len(singlets) + 1.
        """
        assert len(spaces) == len(singlets) + 1
        self.singlets = tuple(singlets)
        self.spaces = tuple(spaces)
        self.n_external = n_external
        self.center_length = sum(self.spaces) + sum(s.thickness for s in self.singlets)

    def to_train(self):
        interfaces = [i for singlet in self.singlets for i in singlet.make_interfaces(self.n_external)]
        spaces = self.spaces[:1] + tuple(
            s for singlet, space in zip(self.singlets, self.spaces[1:]) for s in (singlet.thickness, space))
        return Train(interfaces, spaces)

    def __add__(self, other):
        assert self.n_external == other.n_external
        spaces = self.spaces[:-1] + (self.spaces[-1] + other.spaces[0],) + other.spaces[1:]
        return SingletSequence(self.singlets + other.singlets, spaces, self.n_external)

    def __repr__(self):
        return f'SingletSequence(({", ".join(repr(e) for e in self.singlets)}), ({", ".join("%.3f mm"%(s*1e3) for s in self.spaces)}), {self.n_external})'

    def __str__(self):
        def format_space(s):
            return f'{s*1e3:.3g} mm'
        return '(' + format_space(self.spaces[0]) + ' / ' + ' / '.join(f'{singlet} / {format_space(space)}' for singlet, space in zip(self.singlets, self.spaces[1:])) + ') in ' + str(self.n_external)

    def reverse(self):
        singlets = tuple(s.reverse() for s in self.singlets[::-1])
        return SingletSequence(singlets, self.spaces[::-1], self.n_external)

    def split(self, index: int, frac: float = 0.5) -> Tuple['SingletSequence', 'SingletSequence']:
        before = SingletSequence(self.singlets[:index], self.spaces[:index] + (self.spaces[index]*frac,), self.n_external)
        after = SingletSequence(self.singlets[index:], (self.spaces[index]*(1 - frac),) + self.spaces[index+1:], self.n_external)
        return before, after

    @classmethod
    def from_train(cls, train: Train, radii='equal'):
        assert len(train.interfaces)%2 == 0
        n_external = train.interfaces[0].n1
        assert all(i.n1 == n_external for i in train.interfaces[::2])
        assert all(i.n2 == n_external for i in train.interfaces[1::2])
        singlets = []
        spaces = []
        for i1, i2, space1, space2 in zip(train.interfaces[::2], train.interfaces[1::2], train.spaces[::2],
                                          train.spaces[1::2]):
            spaces.append(space1)
            n = i1.n2
            thickness = space2
            if radii == 'max':
                radius = max((i1.radius, i2.radius))
            elif radii == 'equal':
                radius = i1.radius
                assert np.isclose(radius, i2.radius)
            else:
                radius = radii
            singlets.append(Singlet((to_surface(i1, radius=radius), to_surface(i2, radius=radius)), thickness, n))
        spaces.append(train.spaces[-1])
        return cls(singlets, spaces, n_external)

    @classmethod
    def from_train2(cls, train: Train, radii_mode: str = 'equal'):
        singlets = []
        spaces = []
        space = train.spaces[0]
        index0 = 0
        n_external = train.interfaces[0].n1
        while index0 < len(train.interfaces):
            interface0 = train.interfaces[index0]
            if interface0.n1 != interface0.n2 and interface0.n2 != n_external:
                spaces.append(space)
                space = 0

                radius = interface0.radius
                for index1 in range(index0 + 1, len(train.interfaces)):
                    interface1 = train.interfaces[index1]
                    if radii_mode == 'max':
                        radius = max(radius, interface1.radius)
                    elif radii_mode == 'equal':
                        if radius != interface1.radius:
                            raise ValueError(f'Singlet that starts at interface {index0} has radius {radius} but interface {index1} has radius {interface1.radius}.')
                    else:
                        raise ValueError(f'Unknown radii_mode {radii_mode}.')
                    if interface1.n2 != interface0.n2:
                        thickness = sum(train.spaces[index0+1 : index1+1])
                        # TODO tidy this up after reimplementing Interface as composition of Surface.
                        def pad(i: Interface):
                            outer_radius = radius - i.radius
                            if outer_radius > 1e-6: # TODO civilize
                                outer_surface = SphericalSurface(np.inf, outer_radius)
                                outer_sag = i.calc_sag(i.radius)
                                i = SegmentedInterface(i.n1, i.n2, (to_surface(i), outer_surface), (0, outer_sag))
                            return i
                        interface0 = pad(interface0)
                        interface1 = pad(interface1)
                        singlets.append(Singlet((to_surface(interface0, radius=radius), to_surface(interface1, radius=radius)), thickness, interface0.n2))
                        break
                else:
                    raise ValueError(f'Singlet that starts at interface {index0} does not finish.')
                index0 = index1
            else:
                index0 += 1
                space += train.spaces[index0]
        spaces.append(space)
        return cls(singlets, spaces, n_external)


    @classmethod
    def design_symmetric_singlet_transform(cls, n: ri.Index, f_transform: float, working_distance: float,
            min_thickness: float, field_radius: float, shape_factor: float = 0, n_external: ri.Index = ri.vacuum,
            lamb: float = None) -> 'SingletSequence':
        """Design a symmetric Fourier transform consisting of a pair of singlets.

        Args:
            n: Refractive index of singlets.
            f_transform: Focal length of transform.
            working_distance: Distance from focal planes to first piece of glass.
            min_thickness: Minimum thickness of singlets i.e. edge thickness since they are positive.
            field_radius: Radius of input field - determines lens sizes.
            shape_factor: Coddington shape factor - positive means convex-plano.
            n_external: Index of external medium.
            lamb: Design wavelength.

        Returns:
            Half of the transform i.e. first space is on-axis working distance and last space is
                to midpoint of transform pair.
        """
        tan_theta = field_radius/f_transform

        def make_half_transform_train(curvature):
            rocs = paraxial.calc_rocs(curvature, shape_factor)
            edge_propagation_distance = working_distance + functions.calc_sphere_sag(max(rocs[0], 0), field_radius)
            radius = field_radius + edge_propagation_distance*tan_theta
            thickness = paraxial.calc_center_thickness(rocs, radius, min_thickness)
            singlet = Singlet(tuple(SphericalSurface(roc, radius) for roc in rocs), thickness, n)
            sequence = SingletSequence((singlet,), (0, 0), n_external)
            train = sequence.to_train()
            half_transform_train = train.pad_to_half_transform(lamb, f=f_transform)
            return half_transform_train

        def calc_error(curvature):
            train = make_half_transform_train(curvature)
            space0 = working_distance - functions.calc_sphere_sag(min(train.interfaces[0].roc, 0), field_radius)
            error = train.spaces[0] - space0
            return error

        curvature_guess = 1/((n(lamb) - n_external(lamb))*f_transform)
        curvature = scipy.optimize.fsolve(calc_error, curvature_guess)

        train = make_half_transform_train(curvature)
        sequence = cls.from_train(train)
        return sequence

