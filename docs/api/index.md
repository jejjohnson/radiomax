# API Reference

The public API is the set of public modules below, plus the exceptions that `radiomax.__all__` exports. Private code lives in `radiomax._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| [`radiomax.geometry`](geometry.md) | Sun and view angles, the Earth–Sun distance, phase and scattering angles, air masses, ray–ellipsoid intersection, parallax, geostationary navigation, whiskbroom scan geometry | available |
| [`radiomax.radiometry`](radiometry.md) | Counts, radiance, reflectance, band averages, response matrices, spectral densities | available |
| [`radiomax.thermal`](thermal.md) | Planck's law per wavenumber, its logarithm, and its views per µm and per GHz, monochromatic brightness temperature, and the band inverses sensors publish coefficients for | available |
| [`radiomax.retrievals`](retrievals.md) | The Ross–Li kernels, the kernel BRDF and its nadir adjustment, split-window temperatures, ocean-colour chlorophyll | available |
| `radiomax.sar` | SAR calibration | planned |
| `radiomax.altimetry` | Altimeter range corrections | planned |
| [Type aliases](typing.md) | `ArrayLike` | available |
| [Exceptions](exceptions.md) | `RadiomaxError`, `BackendError`, `ValidationError` | available |

## Package overview

::: radiomax
    options:
      members: false
      show_root_heading: false
      show_source: false
