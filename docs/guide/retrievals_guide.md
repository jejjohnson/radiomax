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
