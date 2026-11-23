---
title: Thermal
---

# Thermal

[`radiomax.thermal`](xref:api#radiomax.thermal) holds Planck's law. Its spectral axis is the wavenumber $\tilde\nu$ in cm⁻¹, in arguments named `wavenumber_cm`, and its radiance is per cm⁻¹.

## Planck's law

With $T$ the temperature in K, $c_1 = 2 h c^2$ and $c_2 = h c / k_B$,

$$
B(\tilde\nu, T) = \frac{c_1 \tilde\nu^3 e^{-x}}{1 - e^{-x}}
$$

where $x = c_2 \tilde\nu / T$.

```python
>>> import numpy as np
>>> import radiomax.thermal as thermal
>>> wavenumber = 1e4 / np.asarray([3.9, 10.8, 12.0])  # µm to cm⁻¹
>>> radiance = thermal.planck_radiance(wavenumber, 290.0)
>>> radiance.round(5)
array([0.0006 , 0.09661, 0.11216])
```

Sensors publish radiance per µm, or per GHz in the microwave. [`to_per_um`](xref:api#radiomax.thermal.to_per_um) and [`to_per_ghz`](xref:api#radiomax.thermal.to_per_ghz) convert, and each has its inverse. The conversion of a whole spectrum on a wavelength grid in nm is in the [radiometry guide](radiometry_guide.md).

```python
>>> thermal.to_per_um(radiance, wavenumber).round(3)
array([0.394, 8.283, 7.789])
```

## Where the radiance underflows

The radiance of a cold scene at a short wavelength falls below the smallest float32 long before it stops mattering to a retrieval that works with logarithms. [`log_planck_radiance`](xref:api#radiomax.thermal.log_planck_radiance) is computed without forming the radiance:

```python
>>> cold, short = np.float32(200.0), np.float32(25000.0)
>>> float(thermal.planck_radiance(short, cold))
0.0
>>> round(float(thermal.log_planck_radiance(short, cold)), 2)
-167.71
```

A temperature or a wavenumber that is not positive has no radiance, so those pixels are NaN, and the other pixels keep a finite gradient.
