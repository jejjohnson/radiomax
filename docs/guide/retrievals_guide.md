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
