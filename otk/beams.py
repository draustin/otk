"""Standard beams e.g. Gaussian modes. Currently only fundamental Gaussian."""
import numpy as np
import scipy

def calc_super_gaussian(radius, order, r):
    """Calculate field of power/energy normalized super-Gaussian.

    Args:
        radius: Radius at exp(-2) of peak intensity.
        order: Exponent of r/radius i.e. for n=2 for fundamental Gaussian.
        r: Radial distance at which to sample.

    Returns:
        Field sampled at r.
    """
    r = np.asarray(r)
    # Normalization factor courtesy of
    # Shealy, D. L. & Hoffnagle, J. A. Laser beam shaping profiles and propagation, Applied Optics 2006, 45, 5118.
    norm_fac = (4**(1/order)*order/(2*np.pi*scipy.special.gamma(2/order)))**0.5
    return np.exp(-abs(r/radius)**order)*norm_fac/radius

class FundamentalGaussian:
    """lamb is in medium"""

    def __init__(self, lamb, **kwargs):
        """Create Gaussian beam.

        Can specify one of w_0 or z_R.

        Args:
            lamb: wavelength (in the medium)
            w_0: waist
            z_R: Rayleigh range
            flux: Tranverse integrated field-squared.
            absE_w: Field amplitude at waist.
        """
        self.lamb = np.asarray(lamb)
        self.k = 2*np.pi/lamb
        if 'w_0' in kwargs:
            self.w_0 = np.asarray(kwargs.pop('w_0'))
            self.z_R = np.pi*self.w_0**2/self.lamb
        elif 'z_R' in kwargs:
            self.z_R = np.asarray(kwargs.pop('z_R'))
            self.w_0 = (self.z_R*self.lamb/np.pi)**0.5
        self.area = np.pi*self.w_0**2/2
        if 'flux' in kwargs:
            self.flux = np.asarray(kwargs.pop('flux'))
            self.absE_w = (self.flux/self.area)**0.5
        elif 'absE_w' in kwargs:
            self.absE_w = np.asarray(kwargs.pop('absE_w'))
        else:
            self.absE_w = 1
        if len(kwargs) != 0:
            raise ValueError('Unknown keyword arguments %s'%list(kwargs.keys()))

    def __eq__(self, other):
        return self.lamb == other.lamb and self.w_0 == other.w_0 and self.absE_w == other.absE_w

    def roc(self, z):
        """Get radius of curvature

        Args:
            z: Axial distance from waist.

        Returns:
            Radius of curvature.
        """
        z = np.asarray(z)
        with np.errstate(divide='ignore'):
            return z + self.z_R**2/z

    # Not defined at z=0.
    # def drocdz(self, z):
    #     return 1 - mathx.divide0(self.z_R, z)**2

    def Gouy(self, z):
        """

        Args:
            z: Axial distance from waist.

        Returns:
            Gouy phase at z.
        """
        z = np.asarray(z)
        return np.arctan(z/self.z_R)

    def dGouydz(self, z):
        return self.z_R/(self.z_R**2 + z**2)

    def waist(self, z):
        z = np.asarray(z)
        return self.w_0*(1 + (z/self.z_R)**2)**0.5

    def calc_all(self, z, rho):
        z = np.asarray(z)
        rho = np.asarray(rho)
        w = self.waist(z)
        psi = self.Gouy(z)
        R = self.roc(z)
        absE = self.w_0/w*np.exp(-(rho/w)**2)*self.absE_w
        z_c = rho**2/(2*R)
        # try:
        #     z_c=rho**2/(2*R)
        # except ZeroDivisionError:
        #     z_c=0
        phi = self.k*z_c - psi  # leave k out, retarded reference frame
        return {'w':w, 'psi':psi, 'absE':absE, 'phi':phi, 'R':R}

    def dphidz(self, z, rho):
        z_R = self.z_R
        # Need an expression that is correct for z=0.
        dz_c_dz = -rho**2*(z**2 - z_R**2)/(2*(z**2 + z_R**2)**2)
        return self.k*dz_c_dz - self.dGouydz(z)

    def dphidrho(self, z, rho):
        return (rho*self.k)/self.roc(z)

    def absEphi(self, z, rho=0):
        s = self.calc_all(z, rho)
        return s['absE'], s['phi']

    def E(self, z, rho=0):
        absE, phi = self.absEphi(z, rho)
        return absE*np.exp(1j*phi)

    @classmethod
    def q_amplitude(cls, q, lamb):
        return cls(lamb, q.imag).absEphi(q.real)
