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

## Whiskbroom scans

A whiskbroom (MODIS, VIIRS) turns a mirror at a constant rate, so its columns are uniform in scan angle. [`scan_angles`](xref:api#radiomax.geometry.scan_angles) builds them, with the zones that VIIRS aggregates on board:

```python
>>> geometry.scan_angles(1354, 55.0)[[0, 676, 677, -1]].round(3)
array([-5.4959e+01, -4.1000e-02,  4.1000e-02,  5.4959e+01])
>>> viirs = ((1776, 3), (736, 2), (640, 1))  # (native samples, factor), from nadir
>>> geometry.scan_angles(3200, 56.28, viirs).shape
(3200,)
```

Towards the edge of the scan the pixel grows, faster across the track than along it. [`scan_pixel_growth`](xref:api#radiomax.geometry.scan_pixel_growth) returns both, and the growth along track in orbit angle, which is the one that matters for the overlap of successive scans:

```python
>>> angles = np.asarray([0.0, 30.0, 55.0])
>>> along, orbit, across = geometry.scan_pixel_growth(angles, 705e3)
>>> along.round(2), across.round(2)
(array([1.  , 1.18, 2.01]), array([1.  , 1.42, 4.83]))
```

Because the pixels grow along track, scans overlap away from nadir (the bowtie). [`bowtie_detector_positions`](xref:api#radiomax.geometry.bowtie_detector_positions) says which detector, as a fraction, sees each row of the nadir grid, and which detector of the neighbouring scan sees it too. With 10 detectors, the first two rows at the three angles above:

```python
>>> own, neighbour, step = geometry.bowtie_detector_positions(orbit, 10)
>>> own[:2].round(2)
array([[0.  , 0.68, 2.29],
       [1.  , 1.53, 2.78]])
>>> neighbour[:2].round(2)
array([[10.  ,  9.16,  7.2 ],
       [11.  , 10.01,  7.69]])
```

A position is inside the neighbour's footprint from −0.5 to 9.5, so at nadir the neighbour does not see row 0, and at 30° and 55° it does.
