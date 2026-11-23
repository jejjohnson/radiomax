---
title: Geometry
---

# Geometry

[`radiomax.geometry`](xref:api#radiomax.geometry) says where the Sun and the sensor are, as seen from each pixel. Nothing here follows light through the air.

## Time is two arrays

A function that depends on time takes `day`, the number of UTC midnights since 2000-01-01, and `seconds`, the second of the UTC day. [`utc_day_seconds`](xref:api#radiomax.geometry.utc_day_seconds) makes them on the host from timestamps.

```python
>>> import radiomax.geometry as geometry
>>> day, seconds = geometry.utc_day_seconds("2024-06-21T10:30:00")
>>> day, seconds
(8938.0, 37800.0)
```

We split time because JAX runs in float32 by default. There, a Julian date near $2.46 \times 10^6$ resolves a quarter of a day, and days since 2000 resolve 84 s, which is 0.35° of Earth rotation. The second of the day resolves 8 ms. One timestamp gives Python floats, so it adopts the dtype of the arrays it meets; an array of timestamps gives float64 arrays.

## The Sun at every pixel

[`solar_angles`](xref:api#radiomax.geometry.solar_angles) returns the zenith and the azimuth. Angles are degrees unless `degrees=False`, longitude comes before latitude, and the azimuth is clockwise from north.

```python
>>> import numpy as np
>>> lon = np.asarray([-5.0, 0.0, 5.0])
>>> zenith, azimuth = geometry.solar_angles(lon, 45.0, day, seconds)
>>> zenith.round(2)
array([31.33, 28.54, 26.06])
>>> azimuth.round(1)
array([124.1, 131.4, 139.9])
```

Across these 10° of longitude the zenith changes by more than 5°, so one angle per scene, at its centre, is not enough for a wide swath.

The formulas are the Astronomical Almanac's. Over 2000–2050 the zenith is within 0.0125° of NREL's SPA, in float64 and in float32. Refraction is ignored, and the night side is returned (a zenith above 90°), not masked.

## The Earth–Sun distance

[`earth_sun_distance`](xref:api#radiomax.geometry.earth_sun_distance) is the $d$ of the reflectance formula, in astronomical units.

```python
>>> round(float(geometry.earth_sun_distance(day, seconds)), 5)
1.01628
```

It is within $8.8 \times 10^{-5}$ AU of SPA. The formula on the day of the year that georeader uses, $d = 1 - 0.01673 \cos(0.0172 (t - 4))$, is off by up to $8.0 \times 10^{-4}$ AU, so a reflectance computed with it differs from ours by up to $1.5 \times 10^{-3}$ relative.
