# otk - optics toolkit

Tools for doing optics in Python

<img src="screenshots/zemax_conic_telecentric_lens.png" width="400  " title="conic telecentric lens with rays">
<img src="screenshots/cell-phone-lens.png" width="160" title="cell phone lens">
<img src="screenshots/csg.png" width="160" title="cell phone lens">

## Installation

Installation methods include:

* Clone repository and interact with it using [Poetry](https://python-poetry.org/) e.g. `poetry run view-zmx <zemax-file>` or `poetry shell`.
* Install in development mode with pip: `pip install -e <path-to-local-repo>`.
* Install from package repository (e.g. PyPi) with pip: `pip install otk`.
* Development mode with [: `poetry add <path-to-local-repo>`.
* From package repository (e.g. PyPi) with Poetry: `poetry add otk`.

## Package management

otk uses [Poetry](https://python-poetry.org/) for package management. This means that dependencies, version, entry points etc are all defined in [`pyproject.toml`](./pyproject.toml). [`setup.py`](./setup.py) is generated using `dephell deps convert` to support pip development mode installation.

## Getting started

1. Check out the scripts in [examples](./examples).
2. View one of the lenses in [designs](./designs) with the command line tool `view-zmx`.

## Folder contents

* `otk` - the Python package itself.
* `examples` - example scripts.
* `properties` - material properties databases.
* `notes` - miscellaneous notes including derivations.

## Documentation

(Yep, this is it at the moment.)

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
