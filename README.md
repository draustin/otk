# otk - optics toolkit

Tools for doing optics in Python

![](screenshots/zemax_conic_telecentric_lens.png)

## Installation

In repository root folder, `pip install -e .` to install in editable mode.

## Contents

* `otk.sdb` - Geometry library based on signed distance bounds.
* `otk.rt1` - First attempt at ray tracing package. Superseded by otk.rt2.
* `otk.rt2` - Ray tracing package with flexible geometry based on otk.sdb. See also `otk.rt2_scalar_qt`.
* `otk.asbp` - Angular spectrum beam propagation.
* `otk.abcd` - 1D ray transfer matrix ("ABCD matrices") tools.
* `otk.rtm4` - abstractions for 1D ray transfer matrices, building upon `otk.abcd`.
* `otk.pgb` - parabasal Gaussian routines for doing wave optical calculations with ray tracing results. Builds upon `otk.rt1`.
* `otk.h4t` - homogeneous 4x4 transformation matrices.
* `otk.paraxial` - basic paraxial optics calculations.
* `otk.math` - various optics-specific math functions.
* `otk.pov` - tools for generating POV-Ray scenes of optical setups.
* `otk.pov` - for calculating properties of prisms.
* `otk.qt` - Qt-related utilities
* `otk.ri` - refractive index tools.
* `otk.trains` - axially symmetric optical systems
* `otk.v3` - operations on homogeneous vectors in 2D
* `otk.v4` - operations on homogeneous vectors in 3D
* `otk.v4b` - broadcasting operations on homogeneous vectors in 3D
* `otk.zemax` - reading Zemax files

## Contributing

Test framework uses `pytest` and `pytest-qt`.
