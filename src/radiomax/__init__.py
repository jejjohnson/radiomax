"""Observation geometry and radiometry on any array library.

``radiomax`` converts and locates what a sensor saw: angles, unit conversions,
spectral responses, Planck's law, and empirical formulas with given
coefficients, pixel by pixel. Nothing in it models radiation moving through an
atmosphere.

## Modules

| Module | What it gives you |
|---|---|
| `radiomax.geometry` | Sun and view angles, the Earth–Sun distance, air masses |
| `radiomax.radiometry` | counts, radiance, reflectance and band averages |
| `radiomax.thermal` | Planck's law and brightness temperature |
| `radiomax.retrievals` | empirical formulas with given coefficients |
| `radiomax.sar` | SAR calibration |
| `radiomax.altimetry` | altimeter range corrections |
"""

from __future__ import annotations

from radiomax.exceptions import BackendError, RadiomaxError, ValidationError


__version__ = "0.0.7"

__all__ = [
    "BackendError",
    "RadiomaxError",
    "ValidationError",
    "__version__",
]
