---
title: Retrievals
---

# Retrievals

[`radiomax.retrievals`](xref:api#radiomax.retrievals) holds empirical formulas: what a surface is, from what a sensor saw and coefficients that somebody fitted. Nothing here models the atmosphere, and nothing here is fitted: [radtrax](https://github.com/jejjohnson/radtrax) fits coefficients with its forward model, and we apply them.

## The BRDF kernels

A surface does not reflect the same in every direction. The MODIS model writes its reflectance as three terms,

$$
R = f_{iso} + f_{vol} K_{vol} + f_{geo} K_{geo}
$$

with the RossThick kernel $K_{vol}$ for the scattering of small leaves and the LiSparse-Reciprocal kernel $K_{geo}$ for the shadows of sparse crowns. Both depend on the three angles only, and both are 0 with the Sun and the sensor overhead.

```python
>>> import numpy as np
>>> import radiomax.retrievals as retrievals
>>> azimuth = np.asarray([0.0, 90.0, 180.0])
>>> retrievals.ross_thick(45.0, 30.0, azimuth).round(4)
array([ 0.1829, -0.0263, -0.1283])
>>> retrievals.li_sparse_reciprocal(45.0, 30.0, azimuth).round(4)
array([-0.2075, -1.2524, -1.5411])
```

The relative azimuth is the one of [`relative_azimuth`](xref:api#radiomax.geometry.relative_azimuth): 0 is backscatter, where the hotspot is. Some products count it the other way round, from the forward direction, and the kernels cannot tell.

A zenith of 90° or more is NaN. The gradient is finite everywhere else, at nadir and at the hotspot too, where the kernels have a kink; their second derivatives are not finite at nadir.

## Adjusting to nadir

Two images of one field on two days differ because the field changed and because it was seen from elsewhere. [`nadir_adjustment`](xref:api#radiomax.retrievals.nadir_adjustment) removes the second: it is the model's reflectance at nadir over the model's reflectance as observed, and we multiply the observation by it.

```python
>>> weights = (0.25, 0.15, 0.03)
>>> view = np.asarray([0.0, 20.0, 50.0])
>>> seen = retrievals.brdf_reflectance(*weights, 45.0, view, 0.0)
>>> seen.round(4)
array([0.2099, 0.247 , 0.3201])
>>> factor = retrievals.nadir_adjustment(*weights, 45.0, view, 0.0)
>>> (factor * seen).round(4)
array([0.2099, 0.2099, 0.2099])
```

Multiplying keeps the observation's detail, where the model at nadir would replace it. `reference_zenith=` also brings every date to one Sun; without it each pixel keeps its own. The weights per band go on the band dimension, and with noisy weights the model can fall to 0 or below, where the factor is NaN.

## Split window

Water vapour absorbs more at 12 µm than at 11 µm, so the difference of the two brightness temperatures says how much to add. Over the sea,

$$
T = a_0 + a_1 T_{11} + a_2 (T_{11} - T_{12}) + a_3 (\sec\theta_v - 1)
$$

```python
>>> t11, t12 = np.asarray([288.0, 295.0]), np.asarray([287.2, 292.5])
>>> coefficients = (0.5, 1.0, 2.0, 0.7)  # for illustration
>>> retrievals.split_window_sst(t11, t12, 30.0, coefficients).round(3)
array([290.208, 300.608])
```

Over land the emissivity enters, and [`split_window_lst`](xref:api#radiomax.retrievals.split_window_lst) is a weighted sum of features in one of two forms, `"quadratic"` (8 coefficients) and `"generalised"` (7). [`split_window_features`](xref:api#radiomax.retrievals.split_window_features) returns the features themselves, the design matrix of a fit. They are radtrax's, so coefficients fitted there apply here unchanged.

```python
>>> features = retrievals.split_window_features(t11, t12, 0.97, 0.005, 2.0, form="generalised")
>>> features.shape
(2, 7)
```

We ship no split-window coefficients, because they belong to a satellite, a region, a season and a version. They are a sequence, or an array with the coefficients on its last axis (so they may differ by pixel), or a DataArray with them on the dimension `feature`.

## Ocean colour

Phytoplankton absorb blue light, so chlorophyll follows from the ratio of blue to green reflectance. The OCx algorithms take the largest of several blue bands,

$$
\log_{10} \mathrm{chl} = \sum_i a_i r^i
$$

where $r = \log_{10}(\max R_{blue} / R_{green})$.

```python
>>> modis = retrievals.OCX_R2022["MODIS"]
>>> modis.blue_nm, modis.green_nm
((443.0, 488.0), 547.0)
>>> blue = [np.asarray([0.0050, 0.0030]), np.asarray([0.0046, 0.0036])]
>>> retrievals.band_ratio_chlorophyll(blue, 0.0030, modis.coefficients).round(3)
array([0.558, 1.153])
```

[`OCX_R2022`](xref:api#radiomax.retrievals.OCX_R2022) holds the coefficients of NASA's 2022 reprocessing for twelve sensors, each record with the algorithm's name, its bands and where the numbers were read. NASA's standard product uses this formula above 0.35 mg m⁻³; below 0.25 mg m⁻³ it uses another (the colour index), with a blend between. Any other coefficients are passed the same way, as a sequence or an array.
