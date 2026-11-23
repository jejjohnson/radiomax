---
title: Radiometry
---

# Radiometry

[`radiomax.radiometry`](xref:api#radiomax.radiometry) converts what a sensor counts into what it measures, pixel by pixel. Coefficients, spectra and responses arrive as arrays: radiomax reads no files.

## From counts to reflectance

A band's counts $\mathrm{DN}$ become radiance $L$ through its gain $g$ and offset $o$, and radiance becomes top-of-atmosphere reflectance $\rho$ through the band's solar irradiance $E_0$, the solar zenith angle $\theta_s$ and the Earth–Sun distance $d$ in astronomical units:

$$
L = g \cdot \mathrm{DN} + o
$$

$$
\rho = \frac{\pi L d^2}{E_0 \cos\theta_s}
$$

```python
>>> import numpy as np
>>> import radiomax.geometry as geometry
>>> import radiomax.radiometry as radiometry
>>> day, seconds = geometry.utc_day_seconds("2024-06-21T10:30:00")
>>> zenith = geometry.solar_zenith(np.asarray([-5.0, 0.0, 5.0]), 45.0, day, seconds)
>>> distance = geometry.earth_sun_distance(day, seconds)
>>> counts = np.asarray([[9000, 9500, 10000], [7000, 7500, 8000]], dtype=np.uint16)
>>> gain = np.asarray([[1.6e-3], [5.4e-4]])
>>> radiance = radiometry.counts_to_radiance(counts, gain, -2.0)
>>> irradiance = np.asarray([[245.0], [85.0]])
>>> radiometry.reflectance(radiance, irradiance, zenith, distance).round(4)
array([[0.1923, 0.199 , 0.2064],
       [0.0795, 0.0891, 0.0986]])
```

The radiance and the irradiance must be per the same spectral unit, here per µm. The function cannot check it, and getting it wrong scales every pixel by the same factor, which is easy to miss.

As $\cos\theta_s$ goes to zero the reflectance blows up, so pixels whose zenith is 85° or more are NaN (`max_zenith` changes that), and the other pixels keep a finite gradient. [`radiance_from_reflectance`](xref:api#radiomax.radiometry.radiance_from_reflectance) goes back.

## Coefficients per band

A coefficient per band broadcasts as any array does, as `(band, 1, 1)` against a `(band, y, x)` cube. With xarray it is simpler: put it on the band dimension and it aligns by name.

```python
>>> import xarray as xr
>>> cube = xr.DataArray(counts, dims=("band", "x"), coords={"band": ["B6", "B7"]})
>>> per_band = xr.DataArray(gain[:, 0], dims="band", coords={"band": ["B6", "B7"]})
>>> radiometry.counts_to_radiance(cube, per_band, -2.0).dims
('band', 'x')
```

An array without names next to a DataArray raises, because its position would have to be guessed.

## Band averages

[`band_average`](xref:api#radiomax.radiometry.band_average) averages a spectrum over each band's spectral response, and [`band_irradiance`](xref:api#radiomax.radiometry.band_irradiance) is the same average of the solar spectrum, the $E_0$ above. With $w$ the weights of the trapezoid rule on the grid, $f$ the spectrum and $R_b$ the response of band $b$,

$$
\bar f_b = \frac{\sum w f R_b}{\sum w R_b}
$$

```python
>>> wavelength = np.linspace(1500.0, 1700.0, 201)
>>> centre = np.asarray([[1560.0], [1640.0]])
>>> responses = np.exp(-(((wavelength - centre) / 20.0) ** 2))
>>> solar = 0.25 - 1e-4 * (wavelength - 1600.0)
>>> radiometry.band_irradiance(solar, responses, wavelength).round(4)
array([0.254, 0.246])
```

A sample can be left out by a mask (`where`) or because it is NaN (`nan_policy="omit"`). Both sums then run over the samples that are left, so the average renormalises. With the default, `"propagate"`, any NaN among the kept samples makes every band NaN. The rules are integrix's, and the tests compare with it.

The weights are an argument, so any quadrature rule fits, such as `functax.quad.simpson(wavelength).weights`. The spectrum and the responses must be on one grid: interpolating a narrow response onto a coarse grid is wrong by up to 7 % (an 8.5 nm band on a 7.4 nm grid).

With DataArrays, `dim` names the wavelength dimension, and the result has the spectrum's other dimensions followed by the band's.

[`per_nm_to_per_cm`](xref:api#radiomax.radiometry.per_nm_to_per_cm) and [`per_cm_to_per_nm`](xref:api#radiomax.radiometry.per_cm_to_per_nm) convert a spectral density. They change the values only, so an ascending wavenumber axis also needs the arrays reversed.
