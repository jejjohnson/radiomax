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
