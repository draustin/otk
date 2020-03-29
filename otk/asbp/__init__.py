"""Numerical beam propagation using angular spectrum method.

# Options for kz_mode
Here kt is transverse component.

    paraxial: kz = k - kt**2/(2*k)
    local_xy: kz = k(kt) + grad k(kt)*(k-kt) + grad_xx k(kt)*(kx-ktx)**2/2 + grad_yy k(kt)*(ky-kty)**2/2
    local: Same but with xy component.
    exact: kz = (k**2 - kt**2)**0.5

"""
from .sa import calc_r, calc_q, calc_xy, calc_kxky, unroll_r, unroll_q, to_scalar_pair, calc_Eq_factor
from .source import calc_gaussian_1d, calc_gaussian
from .math import calc_curved_propagation_m, adjust_r, prepare_plane_to_curved_flat, prepare_plane_to_curved_spherical, \
    fft, ifft, fft2, ifft2, calc_propagation_m_1d, make_ifft_arbitrary_matrix, make_fft_matrix, \
    prepare_plane_to_curved_spherical_arbitrary, prepare_plane_to_curved_flat_arbitrary
from .fsq import calc_gradxyE, calc_gradxyE_spherical, propagate_plane_to_plane_flat_1d, propagate_plane_to_plane_flat, \
    propagate_plane_to_plane_spherical_1d, propagate_plane_to_plane_spherical_paraxial_1d, \
    propagate_plane_to_plane_spherical_paraxial_1dE, propagate_plane_to_plane_spherical, invert_plane_to_curved_flat, \
    propagate_plane_to_curved_flat, propagate_plane_to_curved_spherical, invert_plane_to_curved_spherical, \
    propagate_plane_to_curved_spherical_arbitrary, propagate_plane_to_curved_flat_arbitrary, \
    invert_plane_to_curved_spherical_arbitrary, propagate_arbitrary_curved_to_plane_spherical
from .profiles import NullProfileError, PlaneProfile, CurvedProfile, calc_quadratic_phase_mask
from .tracing import *
from .sbt import *
from .plotting import plot_r_q_polar, make_Eq_image_item, make_Er_image_item, plot_projection
from .widgets import PlaneCurvedProfileWidget, PlaneProfileWidget, CurvedProfileWidget, MultiProfileWidget, \
    make_profile_widget

"""
Early November 2017 -  Some proposed definitions.

An element is a 'straightened out view' of an rt-tracing entity. It may be 'thin' in which case it is
at a specific position z along this axis, or 'thick' in which case it occupies an interval along the axis. Free space
is a thick entity. An element may have 'knowledge' of the entity from which it came. A thin element may know the location
of its plane in the entities space; likewise a thick element may know the location of its input and output planes.

Given a beam with given wavelength (or several wavelengths simultaneously with broadcasting), sampled on a given grid,
an element can 'apply' itself (modifying the grid accordingly).

November 25/26 2017: Some development thoughts

Things we don't need, at least at first.

* figuring out which surface we have intersected with i.e. standard rt tracing. We will always know what the order is.
* calculation of rt deflections. can form a train of elements manually.
* out of plane grating diffraction
* arbitrary misalignments

Things we do need:
* proper dealing with curved focal_surfaces
* thin lens
* a simple way of repeating things e.g. microlens array
* interface - arbitrary shape? Maybe. Perhaps can use thin lens approximations - treat them as masks.
* fully accurate k_z=sqrt(k^2-k_trans^2). Probably easy.

For gratings, can think of two approaches:
* calculate field on the (inclinded) plane of the grating by summing over plane wave components, apply grating phase, and
calcuulate a uniformly sampled angular spectrum.
* Apply the grating equation to each of the incident plane waves, then accumulate the field in real space in a transverse
plane perpendicular to the outgoing central rt.

If grating diffraction is kept in one plane, then the transverse k component is the same. So we can remain in ky space,
and sum over kx components to build up a field E(x,ky), then transform along x.

** Interfaces with arbitrary curved shapes:
https://www.osapublishing.org/oe/abstract.cfm?uri=oe-22-10-12659

"""
