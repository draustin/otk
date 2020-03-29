from typing import Tuple
import numpy as np
import mathx

class Mask:
    """Immutable class defining a two dimensional amplitude mask."""

    def eval(self, x:float, y:float)->float:
        """Should work with numpy broadcasting too."""
        raise NotImplementedError()

class LatticeMask(Mask):
    def __init__(self, lattice:mathx.Lattice, mask: Mask, indices:Tuple[int]=None):
        self.lattice = lattice
        self.mask = mask
        self.indices = indices

    def eval(self, x:float, y:float)->float:
        if self.indices is None:
            ox, oy = self.lattice.calc_offsets(x, y)
        else:
            cx, cy = self.lattice.calc_point(*self.indices)
            ox = x - cx
            oy = y - cy
        return self.mask.eval(ox, oy)

    def restrict(self, indices:Tuple[int]=None):
        return LatticeMask(self.lattice, self.mask, indices)


def calc_linear_apodization(x:float, size:float, apod_fraction:float):
    """Calculate apodization function on symmetric domain.

    Args:
        x: position relative to center.
        size: full size of domain.
        apod_fraction: Linear fraction of aperture taken up by the apodization at each edge. E.g. if
            apod_fraction = 0.1, then the apodization rises from 0 to 1 over (x/size) in [-0.5, -0.4], is 1 in (x/size)
            in [-0.4, 0.4], and falls to zero over [0.4, 0.5].

    Returns:
        Apodization at x.
    """
    x = np.asarray(x)
    if apod_fraction == 0:
        return abs(x) <= size/2
    else:
        return np.clip((0.5 - abs(x)/size)/apod_fraction, 0, 1)

class LinearSquareApodization(Mask):
    def __init__(self, size:float, apod_fraction:float):
        self.size=  size
        self.apod_fraction = apod_fraction

    def eval(self, x, y):
        return (calc_linear_apodization(x, self.size, self.apod_fraction)*
                calc_linear_apodization(x, self.size, self.apod_fraction))


# class IdealLens:
#     """A mask representing an ideal lens."""
#     def __init__(self, f, k, paraxial=False):
#         self.f = f
#         self.k = k
#         self.paraxial = paraxial
#
#     def eval(self, x, y):
#         if self.paraxial:
#             # return np.exp(-0.5j*(x**2+y**2)*self.k/self.f)
#             return calc_ideal_lens_phase_paraxial(x, y, self.k/self.f)
#         else:
#             # return np.exp(-1j*self.k*((x**2+y**2+self.f**2)**0.5-self.f))
#             return calc_ideal_lens_phase(x, y, self.k, self.f)
#
# class IdealSquareLensArray:
#     def __init__(self, f, k, pitch, paraxial=False):
#         self.f = f
#         self.k = k
#         self.paraxial = paraxial
#         self.pitch = pitch
#         self.unblocked_lenses = None
#
#     def __str__(self):
#         return 'IdealSquareLensArraySurface(f = %.3f mm,  pitch = %.3f mm)'%(self.f*1e3, self.pitch*1e3)
#
#     def eval(self, x, y):
#         # u = (x%self.pitch)-self.pitch/2
#         # v = (y%self.pitch)-self.pitch/2
#         if self.paraxial:
#             # amplitude = np.exp(-0.5j*(u**2+v**2)*self.k/self.f)
#             amplitude = calc_ideal_square_lens_array_phase_paraxial(x, y, self.pitch, self.k/self.f)
#         else:
#             # amplitude = np.exp(-1j*self.k*((u**2+v**2+self.f**2)**0.5-self.f))
#             amplitude = calc_ideal_square_lens_array_phase(x, y, self.pitch, self.k, self.f)
#         if self.unblocked_lenses is not None:
#             nx = np.floor(x/self.pitch).astype(int)
#             ny = np.floor(y/self.pitch).astype(int)
#             unblocked = np.any(
#                 [(nx == nx_unblocked) & (ny == ny_unblocked) for nx_unblocked, ny_unblocked in self.unblocked_lenses],
#                 axis=-1)
#             amplitude *= unblocked
#         return amplitude

