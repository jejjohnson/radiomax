---
title: Altimetry
---

# Altimetry

[`radiomax.altimetry`](xref:api#radiomax.altimetry) holds the corrections of an altimeter's range that have a closed form. A radar pulse is delayed on its way, so the measured range is too long.

## One sign

A correction is what we add to the range, as in the products: negative where the signal was delayed. The sea surface height is then

$$
SSH = H - \Big(R + \sum_i c_i\Big)
$$

with $H$ the altitude of the satellite above the ellipsoid, and $R$ the range.

```python
>>> import numpy as np
>>> import radiomax.altimetry as altimetry
>>> pressure = np.asarray([98000.0, 101100.0])
>>> dry = altimetry.dry_troposphere_correction(pressure, 45.0)
>>> wet = altimetry.wet_troposphere_correction(25.0, 275.0)
>>> ionosphere = altimetry.ionosphere_correction(50e16, 13.575e9)
>>> barometer = altimetry.inverse_barometer(pressure, 101100.0)
>>> for correction in (dry, wet, ionosphere, barometer):
...     print(correction.round(3))
[-2.231 -2.302]
-0.16
-0.109
[ 0.308 -0.   ]
>>> altitude, measured = 1_336_000.0, 1_335_962.0
>>> altimetry.sea_surface_height(
...     altitude, measured, dry, wet, ionosphere, barometer
... ).round(3)
array([40.193, 40.572])
```

The altitude is about $10^6$ m, where float32 steps by more than 0.06 m: keep it in float64.

## What each needs

| Correction | From | Formula |
| --- | --- | --- |
| [dry troposphere](xref:api#radiomax.altimetry.dry_troposphere_correction) | surface pressure, latitude | Saastamoinen's |
| [wet troposphere](xref:api#radiomax.altimetry.wet_troposphere_correction) | column water vapour, its mean temperature | Bevis and others' |
| [ionosphere](xref:api#radiomax.altimetry.ionosphere_correction) | electron content, frequency | first-order group delay |
| [ionosphere](xref:api#radiomax.altimetry.dual_frequency_ionosphere_correction) | two ranges at two frequencies | the same |
| [inverse barometer](xref:api#radiomax.altimetry.inverse_barometer) | surface pressure, its mean over the oceans | −9.948 mm per hPa |

The inverse barometer is a height and not a delay; subtracting it with the rest removes the sea's answer to the pressure. Its mean pressure varies in time, and the standard 1013.3 hPa in its place moves the height by 24 mm when the mean is 1010.9 hPa.

The sea state bias is empirical, and belongs to one retracker: take it from the product. The two ranges of the dual-frequency correction must already have it.
