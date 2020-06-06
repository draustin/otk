"""Parabasal Gaussian. beams

Parabasal Gaussian beams are useful approximate solutions to the wave equation. A PBG mode is defined in 3D space
by a base ray, which defines its propagation direction, and two parabasal rays, which define its size and divergence.
The parabasal rays are said to be 'complex', a nomenclature that I found confusing initially. In terms of
regular ray tracing, if 'complex' ray is actually two regular rays, called the 'real' and the 'imaginary' ray. So
a parabasal Gaussian is actually defined by five regular rays - the base and the two pairs of parabasal regular rays.
The five rays are traced through the optical system as normal. The complex nature of the parabasal rays becomes
evident when we evaluate the field. Defining z as distance along the base ray, each parabasal ray is expressed as
h(z) = h(0) + u(z) where h and u are 2D complex vectors in the plane perpendicular to the base ray. The real and imaginary
parts of h(z) are the intersections of the parabasal ray real and imaginary parts with this plane.

There are some good writeups, especially by Greynolds. See the references.

Coordinate geometry and broadcasting follows the rt package convention i.e. coordinates run along the -1 axis. Other
axes are available for broadcasting i.e. multiple parabasal Gaussians.

References:
    Greynolds 1986:  A. W. Greynolds, “Vector Formulation Of The Ray-Equivalent Method For General Gaussian Beam
        Propagation,” Proc.SPIE , vol. 679. p. 679, 1986.
"""
from ._core import Segment, Mode