---
title: Thermal
---

# Thermal

[`radiomax.thermal`](xref:api#radiomax.thermal) holds Planck's law and the brightness temperatures that invert it. Its spectral axis is the wavenumber $\tilde\nu$ in cm⁻¹, in arguments named `wavenumber_cm`, and its radiance is per cm⁻¹.

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

## Brightness temperature

The brightness temperature is the temperature of the black body that emits the radiance we measured,

$$
T_b = \frac{c_2 \tilde\nu}{\ln(1 + c_1 \tilde\nu^3 / L)}
$$

```python
>>> thermal.brightness_temperature(wavenumber, radiance).round(6)
array([290., 290., 290.])
```

The radiance must be per cm⁻¹. If it is per µm, [`from_per_um`](xref:api#radiomax.thermal.from_per_um) comes first:

```python
>>> per_um = np.asarray([0.4, 8.0, 7.5])
>>> per_cm = thermal.from_per_um(per_um, wavenumber)
>>> thermal.brightness_temperature(wavenumber, per_cm).round(2)
array([290.33, 287.85, 287.41])
```

Noise can make the radiance of a cold, dark pixel negative. Such a pixel has no brightness temperature and is NaN, and the gradient of its neighbours stays finite.

## Band coefficients

A sensor's band is wider than one wavenumber, and its operator publishes coefficients that correct for it. The record we pass says which formula they belong to: [`PlanckLinear`](xref:api#radiomax.thermal.PlanckLinear) for GOES and Landsat, [`PlanckQuadratic`](xref:api#radiomax.thermal.PlanckQuadratic) for Himawari.

$$
T = \frac{1}{b_2} \left(\frac{f_{k2}}{\ln(f_{k1} / L + 1)} - b_1\right)
$$

Landsat's $T = K_2 / \ln(K_1 / L + 1)$ is the same with the defaults $b_1 = 0$ and $b_2 = 1$. With Landsat 8's band 10:

```python
>>> band10 = thermal.PlanckLinear(fk1=774.8853, fk2=1321.0789)
>>> landsat = np.asarray([7.0, 9.0, 11.0, -0.1])
>>> thermal.brightness_temperature_band(landsat, band10).round(2)
array([280.14, 295.74, 309.46,    nan])
```

With several bands the fields are arrays that broadcast against the radiance, `(band, 1, 1)` against a `(band, y, x)` cube, or DataArrays on the band dimension, which align by name. Under JAX a record is a pytree, so a gradient with respect to a coefficient is a record of the same type:

```python
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
>>> sensitivity = jax.grad(lambda record: thermal.brightness_temperature_band(9.0, record))
>>> round(float(sensitivity(band10).fk2), 4)  # K per K
0.2239
```

`PlanckQuadratic` takes the radiance per metre, $10^6$ times the radiance per µm, and a Himawari file carries its own $h$, $c$ and $k_B$: we pass them as `constants=`, a [geophysix](https://github.com/jejjohnson/geophysix) `Constants` record.
