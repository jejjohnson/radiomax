---
title: Navigation
---

# Navigation

Navigation says which point of the Earth a sensor looked at. These functions are in [`radiomax.geometry`](xref:api#radiomax.geometry), take any ellipsoid of [geophysix](https://github.com/jejjohnson/geophysix) as `earth=` (WGS84 by default), and return, next to the position, whether the sensor sees it.

## A ray and the Earth

[`earth_intersection`](xref:api#radiomax.geometry.earth_intersection) is where a ray from a satellite first meets the ellipsoid: the smaller root $t$ of

$$
\frac{x^2 + y^2}{a^2} + \frac{z^2}{b^2} = 1
$$

with $(x, y, z) = \vect{o} + t \vect{d}$, the satellite at $\vect{o}$ and the direction $\vect{d}$ of any length. A ray that misses the Earth gives NaN and `False`.

## Parallax

A plume or a cloud top at a height $H$ appears displaced, away from the satellite, by about $H \tan\theta_v$ on the ground. [`apparent_position`](xref:api#radiomax.geometry.apparent_position) follows the ray through the target to the ellipsoid, exactly:

```python
>>> import numpy as np
>>> import radiomax.geometry as geometry
>>> from geophysix.earth import geodetic_to_ecef
>>> satellite = geodetic_to_ecef(0.0, 0.0, 35_786_023.0)
>>> height = np.asarray([0.0, 2_000.0, 12_000.0])
>>> lon, lat, visible = geometry.apparent_position(2.17, 41.38, height, *satellite)
>>> lat.round(4)
array([41.38  , 41.3999, 41.4993])
```

The last target, at 12 km, appears about 13 km too far north. At height 0 the position is the pixel itself.

The satellite's position is Earth-centred, so the same function serves a polar orbiter, with one position per scan line. The height is per pixel, which is how a cloud-top height product is used. Moving the pixels to their true place (resampling) is the caller's job.

A target below the satellite's horizon is hidden, and one that stands out above the limb has no position on the Earth. Both are NaN, and the other pixels keep a finite gradient:

```python
>>> lon, lat, visible = geometry.apparent_position(np.asarray([60.0, 120.0]), 0.0, 5_000.0, *satellite)
>>> visible
array([ True, False])
```
