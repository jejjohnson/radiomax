---
title: SAR
---

# SAR

[`radiomax.sar`](xref:api#radiomax.sar) turns a SAR image's digital numbers into backscatter. Speckle filters read a neighbourhood, so they are not here but in geoops.

## From digital numbers

A product comes with a table $A$ for each quantity, and the quantity is

$$
x = \left(\frac{DN}{A}\right)^2
$$

with $DN$ the amplitude of the pixel. The table decides whether $x$ is $\beta^0$, $\sigma^0$ or $\gamma^0$. This is the worked example of [ESA's note on Sentinel-1](https://sentinels.copernicus.eu/documents/247904/685163/S1-Radiometric-Calibration-V1.0.pdf), a small region whose mean intensity is 22142.71:

```python
>>> import numpy as np
>>> import radiomax.sar as sar
>>> sigma0 = sar.dn_to_backscatter(np.sqrt(22142.71), 319.946)
>>> round(float(10.0 * np.log10(sigma0)), 2)
-6.65
```

Average the intensity and not the decibels. For a complex image, $DN$ is $|I + iQ|$: take `abs` first. The thermal noise is the reader's to subtract, and the difference may be negative.

## Three areas

The three quantities divide one power by three areas: the slant range plane for $\beta^0$, the ground for $\sigma^0$, and the plane perpendicular to the slant range for $\gamma^0$. With $\theta_i$ the angle of incidence,

$$
\sigma^0 = \beta^0 \sin\theta_i = \gamma^0 \cos\theta_i
$$

```python
>>> incidence = np.asarray([30.0, 45.0])
>>> sar.sigma0_to_beta0(0.1, incidence).round(4)
array([0.2   , 0.1414])
>>> sar.sigma0_to_gamma0(0.1, incidence).round(4)
array([0.1155, 0.1414])
```

Nothing needs the angle when the product has all three tables, because it is in them: $\cos\theta_i = A_\gamma^2 / A_\sigma^2$. An angle that is not between 0° and 90° gives NaN. The angle to the ellipsoid leaves the terrain's slopes in the image.
