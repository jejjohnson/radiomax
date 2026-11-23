# API Reference

The public API is the set of public modules below, plus the exceptions that `radiomax.__all__` exports. Private code lives in `radiomax._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| `radiomax.geometry` | Sun and view angles, the Earth–Sun distance, air masses, parallax, navigation, scan geometry | planned |
| `radiomax.radiometry` | Counts, radiance, reflectance, response matrices, band averages, spectral densities | planned |
| `radiomax.thermal` | Planck's law and brightness temperature | planned |
| `radiomax.retrievals` | Empirical formulas with given coefficients | planned |
| `radiomax.sar` | SAR calibration | planned |
| `radiomax.altimetry` | Altimeter range corrections | planned |
| [Type aliases](typing.md) | `ArrayLike` | available |
| [Exceptions](exceptions.md) | `RadiomaxError`, `ValidationError` | available |

## Package overview

::: radiomax
    options:
      members: false
      show_root_heading: false
      show_source: false
