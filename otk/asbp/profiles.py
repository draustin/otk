import logging
import scipy
from typing import Tuple
import mathx
from mathx import matseq
import numpy as np
import opt_einsum
import pyqtgraph_extended as pg
from . import sa, math, fsq, source, plotting
from .. import bvar, rt1, trains

logger = logging.getLogger(__name__)


class NullProfileError(Exception):
    pass


def calc_quadratic_phase_mask(x, y, k_on_f):
    m = mathx.expj(-0.5*(x**2 + y**2)*k_on_f)
    gradxm = -m*x*k_on_f
    gradym = -m*y*k_on_f
    return m, (gradxm, gradym)


class Profile:
    """
    Attributes:
        phi_cs (pair of scalars): Phase of quadratic component at RMS distance from center. Positive means diverging.
    """

    def __init__(self, lamb: float, n: float, z_center: float, rs_support, Er, gradxyE, rs_center=(0, 0),
            qs_center=(0, 0), polarizationxy=(1, 0)):
        self.lamb = float(lamb)
        self.n = float(n)
        self.z_center = float(z_center)
        rs_support = sa.to_scalar_pair(rs_support)
        assert (rs_support > 0).all()
        self.rs_support = rs_support
        Er = np.asarray(Er).astype(complex)
        self.Er = Er
        assert not np.isnan(Er).any()
        assert len(gradxyE) == 2
        self.gradxyE = gradxyE
        self.rs_center = sa.to_scalar_pair(rs_center)
        self.qs_center = sa.to_scalar_pair(qs_center)

        self.Eq = math.fft2(Er)
        self.k = 2*np.pi*self.n/self.lamb

        self.Ir = mathx.abs_sqd(Er)
        sumIr = self.Ir.sum()
        if np.isclose(sumIr, 0):
            raise NullProfileError()
        self.Igradphi = fsq.calc_Igradphi(self.k, Er, gradxyE, self.Ir)
        x, y = sa.calc_xy(rs_support, Er.shape, rs_center)
        self.x = x
        self.y = y
        self.r_center_indices = abs(x - rs_center[0]).argmin(), abs(y - rs_center[1]).argmin()
        self.delta_x, self.delta_y = self.rs_support/Er.shape
        self.power = sumIr*self.delta_x*self.delta_y
        mean_x = mathx.moment(x, self.Ir, 1, sumIr)
        mean_y = mathx.moment(y, self.Ir, 1, sumIr)
        self.centroid_rs = mean_x, mean_y
        var_x = mathx.moment(x - rs_center[0], self.Ir, 2, sumIr)
        var_y = mathx.moment(y - rs_center[1], self.Ir, 2, sumIr)
        self.var_rs = np.asarray((var_x, var_y))

        # Calculate phase of quadratic component at RMS distance from center. Proportional to A in Siegman IEE J. Quantum Electronics Vol. 27
        # 1991. Positive means diverging.
        phi_cx = 0.5*((x - rs_center[0])*(self.Igradphi[0] - self.Ir*qs_center[0])).sum()/sumIr
        phi_cy = 0.5*((y - rs_center[1])*(self.Igradphi[1] - self.Ir*qs_center[1])).sum()/sumIr
        self.phi_cs = np.asarray((phi_cx, phi_cy))
        self.rocs = mathx.divide0(self.k*self.var_rs, 2*self.phi_cs, np.inf)

        xi, yi = np.unravel_index(self.Ir.argmax(), self.Ir.shape)
        self.peak_indices = xi, yi
        self.peak_Er = self.Er[xi, yi]
        self.peak_rs = np.asarray((self.x[xi], self.y[yi]))
        self.peak_qs = np.asarray(
            [mathx.divide0((gradxE[xi, yi]*Er[xi, yi].conj()).imag, self.Ir[xi, yi]) for gradxE in gradxyE])

        self.kz_center = math.calc_kz(self.k, *self.qs_center)

        # Calculate 3D vectors.
        vector_center = rt1.normalize(rt1.stack_xyzw(*self.qs_center, self.kz_center, 0))
        polarizationz = -(polarizationxy*vector_center[:2]).sum()/vector_center[2]
        origin_center = rt1.stack_xyzw(*self.rs_center, self.z_center, 1)
        polarization = polarizationxy[0], polarizationxy[1], polarizationz, 0
        y = rt1.cross(vector_center, polarization)
        self.frame = np.c_[
            polarization, y, vector_center, origin_center].T

        ## Want a vector perpendicular to vector_center lying in the same plane as vector_center and pol.
        #  n = rt.cross(self.vector_center, polarization)
        #  self.polarization = rt.normalize(rt.cross(n, self.vector_center))
        #  self.frame = rt.make_frame(vector_center, None, polarization, origin_center)

    def calc_quadratic_phase_factor(self, x, y):
        rs_center = self.rs_center
        phi_cx, phi_cy = self.phi_cs
        var_x, var_y = self.var_rs
        return mathx.expj((x - rs_center[0])**2*phi_cx/var_x)*mathx.expj((y - rs_center[1])**2*phi_cy/var_y)

    def mask_binary(self, fr):
        """Only for binary masks as assumes that fr is constant."""
        return self.change(Er=self.Er*fr, gradxyE=[g*fr for g in self.gradxyE])

    def mask(self, f:np.ndarray, gradxyf:tuple, n: float = None):
        """Return self with real-space mask applied.

        Args:
            f: Mask amplitude sampled at same points as self.Er.
            gradxyf: Mask gradients along x and y.
            n: New refractive index (defaults to self.n).
        """
        Er = self.Er*f
        gradxyE = [gradE*f + self.Er*gradf for gradE, gradf in zip(self.gradxyE, gradxyf)]
        Ir = mathx.abs_sqd(Er)
        Igradphi = fsq.calc_Igradphi(self.k, Er, gradxyE, Ir)
        # Mean transverse wavenumber is intensity-weighted average of transverse gradient of phase.
        qs_center = np.asarray([component.sum() for component in Igradphi[:2]])/Ir.sum()

        return self.change(Er = Er, gradxyE=gradxyE, n=n, qs_center=qs_center)

    def filter(self, fq):
        """Apply a Fourier-domain filter.

        This method only works correctly if the field is sampled finely enough to perform a regular DFT. Strongly curved
        wavefronts are not supported.

        Args:
            fq (2D array): Filter transmission amplitude, sampled at (self.kx, self.ky).

        Returns:
            Copy of self with filter applied.
        """
        Er = math.ifft2(math.fft2(self.Er)*fq)
        gradxyE = fsq.calc_gradxyE(self.rs_support, Er, self.qs_center)
        # return type(self)(self.lamb, self.n, self.z, self.rs_support, Er, gradxyE, self.rs_center, self.qs_center)
        return self.change(Er=Er, gradxyE=gradxyE)

    def recalc_gradxyE(self, gradphi):
        # See p116 Dane's logbook 2.
        gradxE = mathx.divide0((self.gradxyE[0]*self.Er.conj()).real, self.Er.conj()) + 1j*gradphi[0]*self.Er
        gradyE = mathx.divide0((self.gradxyE[1]*self.Er.conj()).real, self.Er.conj()) + 1j*gradphi[1]*self.Er
        return gradxE, gradyE

    def refract(self, normal, n, scale_Er=1, polarizationxy=(1, 0)):
        """

        Args:
            normal (3-tuple of 2D arrays): X, y and z components of normal vector, each sampled at (self.x, self.y).
            n:
            scale_Er: Multiplicative factor - must broadcast with self.Er.
            polarizationxy: Transverse components of polarization of the refracted beam.

        Returns:

        """
        k = 2*np.pi/self.lamb*n

        normal = [nc*np.sign(normal[2]) for nc in normal]
        Igradphi_tangent = matseq.project_onto_plane(self.Igradphi, normal)[0]
        Igradphi_normal = np.maximum((self.Ir*k)**2 - matseq.dot(Igradphi_tangent), 0)**0.5
        Igradphi = [tc + nc*Igradphi_normal for tc, nc in zip(Igradphi_tangent, normal)]

        gradphi = [mathx.divide0(c, self.Ir) for c in Igradphi[:2]]
        gradxE, gradyE = self.recalc_gradxyE(gradphi)

        # Mean transverse wavenumber is intensity-weighted average of transverse gradient of phase.
        qs_center = np.asarray([component.sum() for component in Igradphi[:2]])/self.Ir.sum()

        profile = self.change(n=n, gradxyE=(gradxE*scale_Er, gradyE*scale_Er), qs_center=qs_center, Er=self.Er*scale_Er,
                              polarizationxy=polarizationxy)
        for _ in range(3):
            profile = profile.center_q()
        return profile

    def reflect(self, normal, n, scale_Er=1, polarizationxy=(1, 0)):
        raise NotImplementedError()

    @property
    def title_str(self):
        return (
                   'num_pointss = (%d, %d), z_center = %.3f mm, power = %g, n = %g, rs_center = %.3f mm, %.3f mm, qs_center = %.3f rad/mm, %.3f rad/mm, '
                   'ROCs = %.6g mm, %.6g mm / phi_cs = %.3fpi, %.3fpi')%(
                   *self.Er.shape, self.z_center*1e3, self.power, self.n, *(self.rs_center*1e3), *(self.qs_center/1e3),
                   *(self.rocs*1e3), *(self.phi_cs/np.pi))

    def clip_points(self, inside, boundary_clip_r_support_factor=1.05):
        # The x and y axes are are DFT rolled, so just find the minimum and maximum point inside the boundary.
        x_lim = mathx.min_max(self.x[inside.any(1)])
        y_lim = mathx.min_max(self.y[inside.any(0)])
        lims = np.asarray(((x_lim[1] - x_lim[0]), (y_lim[1] - y_lim[0])))
        # if num_pointss is None:
        # In principle, can represent field using reduced number of points. In practice this isn't so
        # useful as subsequent propagation usually increases the required number again.
        num_pointss = self.Er.shape
        # num_pointss = ((np.ceil(lims/(masked_profile.rs_support/masked_profile.Er.shape)) + 1)*
        #               self.boundary_clip_num_points_factor).astype(int)
        num_pointss = sa.to_scalar_pair(num_pointss)
        # if rs_support is None:
        rs_support = lims/(num_pointss - 1)*num_pointss
        # if rs_center is None:
        # Changing the center breaks it the interpolation. Not exactly sure why.
        rs_center = self.rs_center
        #    rs_center = np.mean(x_lim), np.mean(y_lim)
        clipped_profile = self.interpolate(rs_support, num_pointss, rs_center)
        return clipped_profile

    def get_support(self):
        x_lim = mathx.min_max(self.x[self.Er.any(axis=1)])
        y_lim = mathx.min_max(self.y[self.Er.any(axis=0)])
        return x_lim + y_lim

    def crop_zeros(self):
        # TODO test this
        support = self.get_support()
        full_range = mathx.min_max(self.x) + mathx.min_max(self.y)
        if support == full_range:
            return
        rs_center = np.mean(support[:2]), np.mean(support[2:])
        lims = np.asarray(((support[1] - support[0]), (support[3] - support[2])))
        num_pointss = np.asarray(self.Er.shape)
        rs_support = lims/(num_pointss - 1)*num_pointss
        cropped = self.interpolate(rs_support, num_pointss, rs_center)
        return cropped

    def calc_points(self):
        """Return sampling points as nxmx4 array."""
        x, y, z = np.broadcast_arrays(self.x, self.y, self.z)
        return rt1.stack_xyzw(x, y, z, 1)

    def calc_normalized_wavevector(self):
        """Calculate nornmalized propagation vector"""
        vector = rt1.normalize(rt1.stack_xyzw(*self.Igradphi, 0))
        assert np.isfinite(vector).all()
        return vector

    def unroll_r(self, array=None):
        if array is None:
            array = self.Er
        x, y, arrayu = sa.unroll_r(self.rs_support, array, self.rs_center)
        return x, y, arrayu


class PlaneProfile(Profile):
    """A profile sampled on a plane transverse to the z axis i.e. at single z value.

    Because the field is sampled on a plane, we can propagate using (much more robust) non-iterative methods.
    """

    def __init__(self, lamb, n, z, rs_support, Er, gradxyE, rs_center=(0, 0), qs_center=(0, 0), polarizationxy=(1, 0)):
        assert np.isscalar(z)
        Profile.__init__(self, lamb, n, z, rs_support, Er, gradxyE, rs_center, qs_center, polarizationxy)
        self.z = z
        # Flat beam is beam with quadratic phase removed.
        self.Er_flat = self.Er*self.calc_quadratic_phase_factor(self.x, self.y).conj()
        self.Eq_flat = math.fft2(self.Er_flat)
        Iq_flat = mathx.abs_sqd(self.Eq_flat)
        sumIq_flat = Iq_flat.sum()
        self.qs_support = 2*np.pi*np.asarray(Er.shape)/self.rs_support
        kx, ky = sa.calc_kxky(rs_support, Er.shape, qs_center)
        self.kx = kx
        self.ky = ky
        self.q_center_indices = abs(kx - qs_center[0]).argmin(), abs(ky - qs_center[1]).argmin()
        mean_kx_flat = mathx.moment(kx, Iq_flat, 1, sumIq_flat)
        mean_ky_flat = mathx.moment(ky, Iq_flat, 1, sumIq_flat)
        self.centroid_qs_flat = mean_kx_flat, mean_ky_flat
        var_kx_flat = mathx.moment(kx - qs_center[0], Iq_flat, 2, sumIq_flat)
        var_ky_flat = mathx.moment(ky - qs_center[1], Iq_flat, 2, sumIq_flat)

        # Calculate angular variance (of plane beam) from flat beam.
        var_kx = bvar.infer_angular_variance_spherical(self.var_rs[0], self.phi_cs[0], var_kx_flat)
        var_ky = bvar.infer_angular_variance_spherical(self.var_rs[1], self.phi_cs[1], var_ky_flat)
        self.var_qs = np.asarray((var_kx, var_ky))

        dz_waists, self.var_r_waists, self.Msqds, self.z_R = np.asarray(
            [bvar.calc_waist(self.k, var_r, phi_c, var_q) for var_r, phi_c, var_q in
             zip(self.var_rs, self.phi_cs, self.var_qs)]).T
        self.z_waists = dz_waists + self.z
        self.z_waist = np.mean(dz_waists) + self.z

    def change(self, lamb=None, n=None, Er=None, gradxyE=None, rs_center=None, qs_center=None, polarizationxy=None):
        if lamb is None:
            lamb = self.lamb
        if n is None:
            n = self.n
        if Er is None:
            Er = self.Er
        if gradxyE is None:
            gradxyE = self.gradxyE
        if rs_center is None:
            rs_center = self.rs_center
        if qs_center is None:
            qs_center = self.qs_center
        if polarizationxy is None:
            polarizationxy = self.frame[0, :2]
        return PlaneProfile(lamb, n, self.z, self.rs_support, Er, gradxyE, rs_center, qs_center, polarizationxy)

    # Started this. Didn't get it working before realising I don't need it. It should work - will leave it for now.
    # def translate(self, x, y):
    #     Eq_flat = self.Eq_flat*mathx.expj(-(x*self.kx + y*self.ky))
    #     Er_flat = math.ifft2(Eq_flat)
    #     rs_center = self.rs_center + (x, y)
    #     x, y = sa.calc_xy(self.rs_support, Er_flat.shape, rs_center)
    #     Er = self.calc_quadratic_phase_factor(self.x, self.y)
    #     gradxyE = fsq.calc_gradxyE_spherical(self.k, self.rs_support, Er, self.rocs, rs_center, self.qs_center)
    #     return self.change(Er=Er, gradxyE=gradxyE, rs_center=rs_center)

    def fourier_transform(self, f, rs0=(0, 0), z=None):
        if z is None:
            z = self.z + 2*f
        # I *think* that there is an efficient implementation for highly curved wavefronts, analagous to
        # https://doi.org/10.1364/OE.24.025974

        # Derivation of normalization factor Dane's logbook 3 p81.
        norm_fac = self.delta_x*self.delta_y*self.k/(2*np.pi*f)*np.prod(self.Er.shape)**0.5
        Er = math.fft2(self.Er)*mathx.expj(rs0[0]*self.kx + rs0[1]*self.ky)*norm_fac
        rs_support = self.qs_support*f/self.k
        rs_center = self.qs_center*f/self.k
        qs_center = (self.rs_center - rs0)*self.k/f
        gradxyE = fsq.calc_gradxyE(rs_support, Er, qs_center)
        return PlaneProfile(self.lamb, self.n, z, rs_support, Er, gradxyE, rs_center, qs_center)

    def calc_propagation_roc(self, z, axis):
        """

        Args:
            z: Plane to propagate to.
            axis: 0 for x, 1 for y.

        Returns:

        """
        roc = bvar.calc_sziklas_siegman_roc(self.k, self.var_rs[axis], self.phi_cs[axis], self.var_qs[axis], z - self.z)
        return roc

    def calc_propagation_m(self, z, axis):
        """Calculate propagation magnification along an axis.

        Args:
            z: Plane to propagate to.
            axis: 0 for x, 1 for y.

        Returns:
            Magnification.
        """
        roc = self.calc_propagation_roc(z, axis)
        m = (z - self.z)/roc + 1
        # We always want positive magnification since we use it to calculate the next rs_support. I haven't proven that
        # the above always gives positive magnification. An alternative is to estimate the propagated variance using
        # var_rz = bvar.calc_propagated_variance_1d(self.k, self.var_rs[n], self.phi_cs[n], self.var_qs[n], z - self.z)[0]
        # and calculate from this. For now will put assert in.
        assert np.all(m > 0)
        return m

    def calc_propagation_ms(self, z):
        """Calculation propagation magnification along both axes.

        Args:
            z: Plane to propagate to.

        Returns:
            Pair of scalars: magnifications along x and y.
        """
        ms = np.asarray([self.calc_propagation_m(z, n) for n in (0, 1)])
        return ms

    def adjust_rs_center(self, zs, kz_mode='local_xy'):
        zs = sa.to_scalar_pair(zs)
        return math.adjust_r(self.k, self.rs_center, zs - self.z, self.qs_center, kz_mode)

    def interpolate(self, rs_support, num_pointss, rs_center=None):
        """Resample profile to new real space domain."""
        if rs_center is None:
            rs_center = self.rs_center
        rs_support = sa.to_scalar_pair(rs_support)
        num_pointss = sa.to_scalar_pair(num_pointss)
        assert np.isscalar(self.z)
        rs = [sa.calc_r(r_support1, num_points1, r_center1) for r_support1, num_points1, r_center1 in
              zip(rs_support, num_pointss, rs_center)]
        invTx, invTy = [math.make_ifft_arbitrary_matrix(r_support0, num_points0, q_center0, r1) for
                        r_support0, num_points0, q_center0, r1 in
                        zip(self.rs_support, self.Er.shape, self.qs_center, rs)]
        # Need to zero outside the original domain to prevent cyclic boundary conditions giving aliased copies.
        invTx *= abs(rs[0] - self.rs_center[0])[:, None] <= self.rs_support[0]/2
        invTy *= abs(rs[1] - self.rs_center[1])[:, None] <= self.rs_support[1]/2

        qpf0 = self.calc_quadratic_phase_factor(self.x, self.y)
        ft_Erp0 = math.fft2(self.Er*qpf0.conj())
        ft_gradxErp0 = math.fft2(self.gradxyE[0]*qpf0.conj())
        ft_gradyErp0 = math.fft2(self.gradxyE[1]*qpf0.conj())

        x, y = sa.calc_xy(rs_support, num_pointss, rs_center)
        qpf = self.calc_quadratic_phase_factor(x, y)
        # x=i, y=j, kx=k, ky=l
        Er = opt_einsum.contract('ik, jl, kl -> ij', invTx, invTy, ft_Erp0)*qpf
        gradxEr = opt_einsum.contract('ik, jl, kl -> ij', invTx, invTy, ft_gradxErp0)*qpf
        gradyEr = opt_einsum.contract('ik, jl, kl -> ij', invTx, invTy, ft_gradyErp0)*qpf

        return PlaneProfile(self.lamb, self.n, self.z, rs_support, Er, (gradxEr, gradyEr), rs_center, self.qs_center)

    def reflect(self, normal, n, scale_Er=1, polarizationxy=(1, 0)):
        # The reflection surface is by assumption the sampling plane.
        return self.change()

    def plot_r_q_polar(self, flat=False, tilt=False, show=True):
        """Plot amplitude and phase in real and angular space.

        To see self.Er and its transform exactly, set flat=False and tilt=True.

        Args:
            flat (bool): Plot Er_flat instead of Er (both domains).
            tilt (bool): Include the central tilt in the real and angular space phase plots.
            show:

        Returns:
            GraphicsLayoutWidget: Contains the four plots and a heading.
            plots (RQ tuple of AbsPhi tuples): The AlignedPlotItems.
        """
        Er = self.Er_flat if flat else self.Er
        Eq = math.fft2(Er)
        if not tilt:
            Er = Er*mathx.expj(-(self.qs_center[0]*self.x + self.qs_center[1]*self.y))
            Eq = Eq*mathx.expj(self.rs_center[0]*self.kx + self.rs_center[1]*self.ky)
        glw = pg.GraphicsLayoutWidget()
        glw.addLabel(self.title_str)
        glw.nextRow()
        gl = glw.addLayout()
        plots = plotting.plot_r_q_polar(self.rs_support, Er, self.rs_center, self.qs_center, gl, Eq)
        glw.resize(830, 675)
        if show:
            glw.show()
        return glw, plots

    def propagate_to_curved(self, rs_support, num_pointss, rs_center, zfun, roc_x=None, roc_y=None, kz_mode='local_xy',
            n_next=None):
        """Propagate to curved surface.

        Compared to the propagate_to_plane method, this method supports curved focal_surfaces (nonscalar z) as well as different
        Sziklas-Siegman magnification for each output point.

        Args:
            rs_support (scalar or pair of scalars): Real-space support in x and y of propagated profile.
            num_pointss (int or pair of ints): Number of points along x or y, here denoted M and N.
            rs_center (pair of scalars):
            zfun: Callable which accepts x and y and returns z.
            roc_x (scalar or M*N array): Input radius of curvature along x.
            roc_y (scalar or M*N array): Input radius of curvature along y.
            kz_mode:
            n_next:

        Returns:
            CurvedProfile object with Er.shape equal to (M, N).
        """
        rs_support = sa.to_scalar_pair(rs_support)
        assert (rs_support > 0).all()
        x, y = sa.calc_xy(rs_support, num_pointss, rs_center)
        z = zfun(x, y)

        # Remember points where no intersection occurred, but to avoid NaNs causing warnings in the propagation, set
        # z at these points to mean value.
        invalid = np.isnan(z)
        z[invalid] = np.nanmean(z)

        if roc_x is None:
            roc_x = self.calc_propagation_roc(z, 0)
        if roc_y is None:
            roc_y = self.calc_propagation_roc(z, 1)
        if n_next is None:
            n_next = self.n
        assert sa.is_scalar_pair(rs_center)

        Er, gradxyE = fsq.propagate_plane_to_curved_spherical_arbitrary(self.k, self.rs_support, self.Er, z - self.z, x,
                                                                        y, roc_x, roc_y, self.rs_center, self.qs_center,
                                                                        rs_center, kz_mode)

        # Zero points where no intersection occurred.
        Er[invalid] = 0
        gradxyE[0][invalid] = 0
        gradxyE[1][invalid] = 0

        profile = CurvedProfile(self.lamb, n_next, zfun, rs_support, Er, gradxyE, rs_center, self.qs_center)
        return profile

    def propagate_to_plane(self, z, rs_center=None, ms=None, kz_mode='local_xy', n_next=None):
        if rs_center is None:
            rs_center = self.adjust_rs_center(z, kz_mode)

        assert sa.is_scalar_pair(rs_center)
        if ms is None:
            ms = self.calc_propagation_ms(z)
        ms = sa.to_scalar_pair(ms)
        if n_next is None:
            n_next = self.n

        rs_support = self.rs_support*ms
        if np.isclose(z, self.z):
            # Handle special case as courtesy.
            assert np.allclose(ms, 1)
            Er = self.Er.copy()
            gradxyE = tuple(c.copy() for c in self.gradxyE)
        else:
            Er = fsq.propagate_plane_to_plane_spherical(self.k, self.rs_support, self.Er, z - self.z, ms,
                                                        self.rs_center, self.qs_center, rs_center, kz_mode)
            # Calculate radii of curvature at z.
            rocs = [bvar.calc_propagated_variance_1d(self.k, var_r, phi_c, var_q, z - self.z)[2] for var_r, phi_c, var_q
                    in zip(self.var_rs, self.phi_cs, self.var_qs)]
            gradxyE = fsq.calc_gradxyE_spherical(self.k, rs_support, Er, rocs, rs_center, self.qs_center)

        profile = PlaneProfile(self.lamb, n_next, z, rs_support, Er, gradxyE, rs_center, self.qs_center)
        return profile

    def center_q(self):
        return self.change(qs_center=self.centroid_qs_flat)

    @classmethod
    def make_gaussian(cls, lamb: float, n: float, waists: Tuple[float, float], rs_support: Tuple[float, float],
            num_points: int, rs_waist: Tuple[float, float] = (0, 0), qs_waist: Tuple[float, float] = (0, 0),
            z_waist: float = 0, rs_center=None, qs_center=None, z=None):
        """Make a PlaneProfile object sampling a paraxial Gaussian beam.

        Args:
            lamb (scalar): Wavelength.
            n (scalar): Refractive index.
            waists (scalar or pair of scalars): Waist sizes along x and y.
            rs_support (scalar or pair of scalars): Support along x and y.
            num_points (int or pair of ints): Number of points alond x and y.
            rs_waist (pair of scalars): Transverse coordinates of waist.
            qs_waist (pair of scalars): Center of angular distribution.
            z_waist (scalar): Axial location of waist.
            rs_center (pair of scalars or None): Transverse center of profile sampling. Defaults to beam center at z.
            qs_center (pair of scalars or None): Angular center of profile sampling. Defaults to qs_waist.
            z (numeric, shape broadcasts with num_points): Axial coordinates to sample - becomes profile.z

        Returns:
            Profile object.
        """
        assert np.isscalar(z_waist)
        k = 2*np.pi/lamb*n
        if z is None:
            z = z_waist
        rs_waist = np.asarray(rs_waist)
        assert sa.is_scalar_pair(rs_waist)
        qs_waist = np.asarray(qs_waist)
        assert sa.is_scalar_pair(qs_waist)
        if rs_center is None:
            rs_center = rs_waist + (np.mean(z) - z_waist)*qs_waist/k
        if qs_center is None:
            qs_center = qs_waist

        x, y = sa.calc_xy(rs_support, num_points, rs_center)
        Er, gradxyE = source.calc_gaussian(k, x, y, waists, z - z_waist, rs_waist, qs_waist, True)
        profile = cls(lamb=lamb, n=n, z=z, rs_support=rs_support, Er=Er, gradxyE=gradxyE, rs_center=rs_center,
                      qs_center=qs_center)
        return profile

    @classmethod
    def make_bessel(cls, lamb: float, n: float, radius: float, rs_support: Tuple[float, float],
            num_points: int, z_waist: float = 0):
        """Make a PlaneProfile object sampling zero-order Bessel beam.

        The normalization is performed using analytic formulae, so it becomes identically correct in the limit
        of infinite continuous domain.

        Args:
            lamb (scalar): Wavelength.
            n (scalar): Refractive index.
            waists (scalar or pair of scalars): Waist sizes along x and y.
            rs_support (scalar or pair of scalars): Support along x and y.
            num_points (int or pair of ints): Number of points alond x and y.
            z_waist (scalar): Axial location of waist. Also becomes profile.z.

        Returns:
            Profile object.
        """
        assert np.isscalar(z_waist)
        x, y = sa.calc_xy(rs_support, num_points)
        Er, gradxyE = source.calc_bessel(x, y, radius, True)
        profile = cls(lamb=lamb, n=n, z=z_waist, rs_support=rs_support, Er=Er, gradxyE=gradxyE)
        return profile

    def apply_interface_thin(self, interface: trains.Interface, rs_center=(0, 0), shape: str = 'circle'):
        """Return copy of self with interface applied as a thin phase mask.

        Args:
            interface:
            rs_center:
            shape: 'circle', 'square' or None.
        """
        dx = self.x - rs_center[0]
        dy = self.y - rs_center[1]
        rho = (dx**2 + dy**2)**0.5
        assert np.isclose(self.n, interface.n1(self.lamb))

        # if shape is not None:
        #     aperture = interface.calc_aperture(dx, dy, shape)
        #     apertured = self.mask_binary(aperture)
        #     cropped = apertured.crop_zeros()

        f, gradrf = interface.calc_mask(self.lamb, rho, True)
        if shape is not None:
            aperture = interface.calc_aperture(dx, dy, shape)
            f *= aperture
            gradrf *= aperture

        gradxyf = mathx.divide0(dx, rho)*gradrf, mathx.divide0(dy, rho)*gradrf

        return self.mask(f, gradxyf, interface.n2(self.lamb)).center_q()

    def apply_train_thin(self, train: trains.Train, rs_center=(0, 0), shape: str = 'circle'):
        profiles = []
        profiles.append(self.propagate_to_plane(self.z + train.spaces[0]))

        for num, (interface, space) in enumerate(zip(train.interfaces, train.spaces[1:])):
            logger.info(f'Applying interface {num:d}.')
            profiles.append(profiles[-1].apply_interface_thin(interface, rs_center, shape))
            profiles.append(profiles[-1].propagate_to_plane(profiles[-1].z + space))

        return profiles


class CurvedProfile(Profile):
    def __init__(self, lamb, n, zfun, rs_support, Er, gradxyE, rs_center=(0, 0), qs_center=(0, 0),
            polarizationxy=(1, 0)):
        z_center = zfun(*rs_center)
        Profile.__init__(self, lamb, n, z_center, rs_support, Er, gradxyE, rs_center, qs_center, polarizationxy)
        self.zfun = zfun
        self.z = zfun(self.x, self.y)
        self.valid = np.isfinite(self.z)
        self.z[~self.valid] = self.z[self.valid].mean()
        assert np.allclose(Er[~self.valid], 0)
        sumIr = self.Ir.sum()
        self.mean_z = (self.z*self.Ir).sum()/sumIr
        app_propagator = mathx.expj((self.mean_z - self.z)*mathx.divide0(self.Igradphi[2], self.Ir))
        app_propagator[np.isnan(self.z)] = 0
        Er_plane = Er*app_propagator
        gradxyE_plane = tuple(c*app_propagator for c in gradxyE)
        # Approximate plane profile.
        self.app = PlaneProfile(lamb, n, self.mean_z, rs_support, Er_plane, gradxyE_plane, rs_center, qs_center)

    def center_q(self):
        return self.change(qs_center=self.app.centroid_qs_flat)

    def change(self, lamb=None, n=None, Er=None, gradxyE=None, rs_center=None, qs_center=None, polarizationxy=None,
            zfun=None):
        if lamb is None:
            lamb = self.lamb
        if n is None:
            n = self.n
        if Er is None:
            Er = self.Er
        if gradxyE is None:
            gradxyE = self.gradxyE
        if rs_center is None:
            rs_center = self.rs_center
        if qs_center is None:
            qs_center = self.qs_center
        if polarizationxy is None:
            polarizationxy = self.frame[0, :2]
        if zfun is None:
            zfun = self.zfun
        return CurvedProfile(lamb, n, zfun, self.rs_support, Er, gradxyE, rs_center, qs_center, polarizationxy)

    def planarize(self, z=None, rs_support=None, num_pointss=None, kz_mode='local_xy', invert_kwargs=None):
        """Propagate to a flat plane."""
        # The only trick is figuring out the magnification for each sampling point (of the input). Since we are propagating
        # to mean_z, the magnifications are all about 1. But they need to be chosen carefully so that the (uniformly sampled)
        # result is well sampled after its (implied) radius of curvature is removed. Thus what we really need is the
        # ROC at z_plane.
        if z is None:
            z = self.mean_z

        if num_pointss is None:
            num_pointss = self.Er.shape

        # var_rzs = tuple(bvar.calc_propagated_variance_1d(self.k, var_r, phi_c, var_q, z - self.z)[0] for var_r, phi_c, var_q in
        #    zip(self.app.var_rs, self.app.phi_cs, self.app.var_qs))
        # mx = (var_rzs[0]/self.var_rs[0])**0.5
        # my = (var_rzs[1]/self.var_rs[1])**0.5

        roc_x, roc_y = tuple(
            bvar.calc_sziklas_siegman_roc(self.k, var_r, phi_c, var_q, z - self.z) for var_r, phi_c, var_q in
            zip(self.app.var_rs, self.app.phi_cs, self.app.var_qs))

        if rs_support is None:
            rs_support = self.rs_support

        # mx, my = self.app.calc_ms(z + self.app.z - self.z)
        Er = fsq.propagate_arbitrary_curved_to_plane_spherical(self.k, self.x, self.y, self.Er, roc_x, roc_y,
                                                               z - self.z, rs_support, num_pointss, self.rs_center,
                                                               self.qs_center, self.rs_center, kz_mode, invert_kwargs)

        gradxyE = fsq.calc_gradxyE_spherical(self.k, rs_support, Er, self.app.rocs, self.rs_center, self.qs_center)

        profile = PlaneProfile(self.lamb, self.n, z, rs_support, Er, gradxyE, self.rs_center, self.qs_center)

        return profile

    def interpolate(self, rs_support, num_pointss, rs_center=None):
        assert self.zfun is not None
        appi = self.app.interpolate(rs_support, num_pointss, rs_center)
        z = self.zfun(appi.x, appi.y)
        # Propagate from interpolated planar profile to resampled z.
        Ir = appi.Ir
        propagator = mathx.expj((z - appi.z)*mathx.divide0(appi.Igradphi[2], Ir))
        Er = appi.Er*propagator
        gradxyE = tuple(c*propagator for c in appi.gradxyE)
        return type(self)(self.lamb, self.n, self.zfun, appi.rs_support, Er, gradxyE, appi.rs_center, appi.qs_center)

    def reflect(self, normal, n, scale_Er=1, polarizationxy=(1, 0)):
        """

        Args:
            normal:
            n:
            scale_Er:
            polarizationxy:

        Returns:

        """
        Igradphi_tangent, Igradphi_normal = mathx.project_onto_plane(self.Igradphi, normal)
        Igradphi = [tc - nc*Igradphi_normal for tc, nc in zip(Igradphi_tangent, normal)]
        gradphi = [mathx.divide0(c, self.Ir) for c in Igradphi[:2]]
        gradxE, gradyE = self.recalc_gradxyE(gradphi)

        # Mean transverse wavenumber is intensity-weighted average of transverse gradient of phase.
        qs_center = np.asarray([component.sum() for component in Igradphi[:2]])/Ir.sum()

        # Don't understand. What is the new coordinate system? Surely should
        zfun = lambda x, y: 2*self.zfun(x, y) - self.zfun(*self.rs_center)

        profile = self.change(n=n, gradxyE=(gradxE*scale_Er, gradyE*scale_Er), qs_center=qs_center, Er=self.Er*scale_Er,
                              polarizationxy=polarizationxy, zfun=zfun)
        for _ in range(3):
            profile = profile.center_q()
        return profile

    def plot_r_q_polar(self, flat=False, tilt=False, show=True):
        """Plot approximate plane profile and surface z relative to z_mean."""
        app = self.app
        Er = app.Er_flat if flat else app.Er
        Eq = math.fft2(Er)
        if not tilt:
            Er = Er*mathx.expj(-(app.qs_center[0]*app.x + app.qs_center[1]*app.y))
            Eq = Eq*mathx.expj(app.rs_center[0]*app.kx + app.rs_center[1]*app.ky)

        glw = pg.GraphicsLayoutWidget()
        glw.addLabel(self.title_str)
        glw.nextRow()
        gl = glw.addLayout()
        plot = gl.addAlignedPlot(labels={'left': 'y (mm)', 'bottom': 'x (mm)'})
        x, y, zu = sa.unroll_r(self.rs_support, self.z, self.rs_center)
        image = plot.image((zu - self.mean_z)*1e3, rect=pg.axes_to_rect(x*1e3, y*1e3),
                           lut=pg.get_colormap_lut('bipolar'))
        gl.addHorizontalSpacer(10)
        gl.addColorBar(image=image, rel_row=2, label='Relative z (mm)')
        glw.nextRow()

        glw.addLabel('Approximate planar profile')
        glw.nextRow()
        gl = glw.addLayout()
        plots = plotting.plot_r_q_polar(app.rs_support, Er, app.rs_center, app.qs_center, gl, Eq)

        glw.resize(830, 675)
        if show:
            glw.show()
        return glw, plots

    @classmethod
    def make_gaussian(cls, lamb, n, waists, rs_support, num_pointss, rs_waist=(0, 0), qs_waist=(0, 0), z_waist=0,
            rs_center=None, qs_center=None, zfun=None):
        """Make a Profile object sampling a paraxial Gaussian beam.

        Args:
            lamb (scalar): Wavelength.
            n (scalar): Refractive index.
            waists (scalar or pair of scalars): Waist sizes along x and y.
            rs_support (scalar or pair of scalars): Support along x and y.
            num_pointss (int or pair of ints): Number of points alond x and y.
            rs_waist (pair of scalars): Transverse coordinates of waist.
            qs_waist (pair of scalars): Center of angular distribution.
            z_waist (scalar): Axial location of waist.
            rs_center (pair of scalars or None): Transverse center of profile sampling. Defaults to beam center at z.
            qs_center (pair of scalars or None): Angular center of profile sampling. Defaults to qs_waist.
            z (numeric, shape broadcasts with num_points): Axial coordinates to sample - becomes profile.z

        Returns:
            Profile object.
        """
        assert np.isscalar(z_waist)
        k = 2*np.pi/lamb*n
        if zfun is None:
            zfun = lambda x, y: np.full(np.broadcast(x, y).shape, z_waist)
        rs_waist = np.asarray(rs_waist)
        assert sa.is_scalar_pair(rs_waist)
        qs_waist = np.asarray(qs_waist)
        assert sa.is_scalar_pair(qs_waist)
        if rs_center is None:
            z_nominal = calc_z(k, rs_waist, zfun, qs_waist, 'paraxial', z_waist)
            rs_center = math.adjust_r(k, rs_waist, z_nominal - z_waist, qs_waist, 'paraxial')
        if qs_center is None:
            qs_center = qs_waist
        x, y = sa.calc_xy(rs_support, num_pointss, rs_center)
        z = zfun(x, y)
        Er, gradxyE = source.calc_gaussian(k, x, y, waists, z - z_waist, rs_waist, qs_waist, True)
        profile = cls(lamb, n, zfun, rs_support, Er, gradxyE, rs_center, qs_center)
        return profile


# TODO: this isn't actually used anywhere ... remove?
def calc_z(k, rs0, zfun, qs=(0, 0), kz_mode='local_xy', z0=0):
    z_tolerance = 1e-6
    max_iterations = 50
    z = z0
    last_z = z
    num_iterations = 0
    while num_iterations < max_iterations:
        rsz = math.adjust_r(k, rs0, z, qs, kz_mode)
        z = zfun(*rsz)
        if abs(z - last_z) <= z_tolerance:
            break
        last_z = z
        num_iterations += 1
    if num_iterations == max_iterations:
        logger.warning('z did not converge after %d iterations.', num_iterations)
    return z
