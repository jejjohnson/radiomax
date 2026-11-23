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

The weights are an argument, so any quadrature rule fits, such as `functax.quad.simpson(wavelength).weights`. The spectrum and the responses must be on one grid. For a response tabulated on another grid, use a response matrix.

With DataArrays, `dim` names the wavelength dimension, and the result has the spectrum's other dimensions followed by the band's.

[`per_nm_to_per_cm`](xref:api#radiomax.radiometry.per_nm_to_per_cm) and [`per_cm_to_per_nm`](xref:api#radiomax.radiometry.per_cm_to_per_nm) convert a spectral density. They change the values only, so an ascending wavenumber axis also needs the arrays reversed.

## Response matrices

A sensor's response is tabulated on one grid, usually every nanometre, and the spectrum lives on another: a hyperspectral sensor's bands, or a line-by-line grid in wavenumber. [`response_matrix`](xref:api#radiomax.radiometry.response_matrix) integrates one against the other exactly. With $R_b$ linear between the table's wavelengths and $\varphi_g$ the hat function of grid point $g$,

$$
A_{bg} = \frac{\int R_b \varphi_g \mathrm{d}\lambda}{\int R_b \mathrm{d}\lambda}
$$

so $\sum_g A_{bg} f_g$ is the band average of any spectrum that is linear between grid points, whatever the spacing.

```python
>>> table = np.linspace(1500.0, 1700.0, 201)
>>> narrow = radiometry.gaussian_response(table, np.asarray([1560.0, 1640.0]), 8.5)
>>> grid = np.arange(1500.0, 1700.0, 7.4)
>>> matrix = radiometry.response_matrix(narrow, table, grid)
>>> matrix.shape
(2, 28)
>>> radiometry.apply_response(matrix, grid).round(6)
array([1560., 1640.])
```

The two usual shortcuts each fail on one side of the table's spacing. Here is the worst relative error over 25 Gaussian bands and a spectrum of 400 random lines, with a 1 nm table, against the spectrum taken as linear between its grid points:

| Band, grid spacing | Response sampled on the grid | Sum over the table | `response_matrix` |
| --- | --- | --- | --- |
| 8.5 nm, 7.4 nm | 7.1 × 10⁻² | 1.8 × 10⁻³ | 6.3 × 10⁻⁹ |
| 8.5 nm, 0.01 nm | 9.1 × 10⁻⁸ | 7.3 × 10⁻² | 3.8 × 10⁻⁹ |
| 20 nm, 7.4 nm | 9.0 × 10⁻³ | 6.0 × 10⁻⁴ | 1.2 × 10⁻⁹ |
| 20 nm, 0.01 nm | 1.2 × 10⁻⁸ | 6.3 × 10⁻² | 4.9 × 10⁻¹⁰ |

No method recovers structure that the grid did not sample. On the table's own grid the exact matrix differs from a plain sum over the table, which is georeader's convention, by at most 4.1 × 10⁻⁵ relative for Landsat 8's B6 and B7 and a spectrum of lines.

A band that the grid covers partly is normalised over the covered part; [`response_coverage`](xref:api#radiomax.radiometry.response_coverage) gives the share. The matrix is built once, on the host, and [`apply_response`](xref:api#radiomax.radiometry.apply_response) and its adjoint run on any backend. A wavenumber grid is passed as its wavelengths, $10^7 / \nu$.

When the bands themselves move under a gradient, build the matrix on the responses' own grid with [`sampled_response_matrix`](xref:api#radiomax.radiometry.sampled_response_matrix), which traces, from [`gaussian_response`](xref:api#radiomax.radiometry.gaussian_response), [`box_response`](xref:api#radiomax.radiometry.box_response) or [`triangle_response`](xref:api#radiomax.radiometry.triangle_response).
