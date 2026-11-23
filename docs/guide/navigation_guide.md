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

## Geostationary scan angles

A geostationary imager's file holds scan angles $x$ (positive east) and $y$ (positive north) in radians. [`geostationary_lonlat`](xref:api#radiomax.geometry.geostationary_lonlat) turns them into positions, with the sub-satellite longitude and the satellite's height from the same file:

```python
>>> x = np.linspace(-0.08, 0.16, 4)
>>> goes = {"lon_0": -75.2, "height": 35_786_023.0, "sweep": "x"}
>>> lon, lat, visible = geometry.geostationary_lonlat(x, 0.08, **goes)
>>> lon.round(3)
array([-107.551,  -75.2  ,  -42.849,      nan])
>>> lat.round(3)
array([28.235, 27.489, 28.235,    nan])
```

The last angle misses the Earth, so it is NaN. `sweep` says which axis the instrument sweeps along, and it has no default, because the wrong one is off by kilometres and looks right: `"x"` for GOES ABI, `"y"` for Himawari AHI, MTG FCI and SEVIRI.

[`geostationary_scan_angles`](xref:api#radiomax.geometry.geostationary_scan_angles) is the inverse, for finding the pixel that sees a place. It is NaN where the satellite is below the place's horizon:

```python
>>> x, y, visible = geometry.geostationary_scan_angles(np.asarray([12.5, -103.2]), np.asarray([41.9, 31.8]), **goes)
>>> x.round(6), y.round(6)
(array([      nan, -0.067782]), array([     nan, 0.089174]))
```
