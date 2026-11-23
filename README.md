# radiomax

[![Tests](https://github.com/jejjohnson/radiomax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/radiomax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/radiomax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/radiomax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/radiomax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/radiomax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/radiomax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/radiomax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/radiomax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/radiomax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)


Observation geometry and radiometry on any array library.

radiomax is the GeoStack's library of observation geometry and radiometry: it converts and locates what a sensor saw. It holds the Sun's and the sensor's angles, the conversions between counts, radiance and reflectance, spectral responses, Planck's law and brightness temperature, and empirical formulas applied pixel by pixel.

It is written once, in the Python array API standard: NumPy in gives NumPy out, JAX in gives JAX out, and the same for CuPy and any other array-API library. An xarray DataArray in gives a DataArray out. Neither JAX nor xarray is imported unless you pass one of their arrays.

Nothing in radiomax models radiation moving through an atmosphere; that is [radtrax](https://github.com/jejjohnson/radtrax), which imports radiomax.

## Install

radiomax is not on PyPI yet. It builds on [geophysix](https://github.com/jejjohnson/geophysix), which is not either, so install both from GitHub:

```bash
uv pip install "geophysix @ git+https://github.com/jejjohnson/geophysix@v0.0.4" "radiomax @ git+https://github.com/jejjohnson/radiomax"
```

## Status

Wave 0: the package skeleton and its exception hierarchy. The public modules `geometry`, `radiometry`, `thermal`, `retrievals`, `sar` and `altimetry` are planned and built by the feature issues.

## Design

The backlog is [geostack#300](https://github.com/jejjohnson/geostack/issues/300), and the design page is in [geostack](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/radiomax.md).
