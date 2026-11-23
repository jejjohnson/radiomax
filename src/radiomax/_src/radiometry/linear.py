"""The linear decode of counts, with a band's gain and offset."""

from __future__ import annotations

from typing import Any

from pipekit_array.functions import elementwise, floating_inputs


__all__ = [
    "counts_to_radiance",
    "counts_to_reflectance",
    "radiance_to_counts",
]


@elementwise()
def counts_to_radiance(counts: Any, gain: Any, offset: Any = 0.0) -> Any:
    """Radiance at the sensor from digital counts.

    L = g · DN + o

    Integer counts become floating before any arithmetic, so they cannot
    overflow. Fill values are the caller's to mask.

    Args:
        counts: DN, digital counts.
        gain: g, radiance per count, in the unit the radiance should have. Per
            band, as (band, 1, 1) or as a DataArray on the band dimension.
        offset: o, radiance at zero counts.

    Returns:
        L, in the gain's unit.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import counts_to_radiance
        >>> counts = np.asarray([0, 65535], dtype=np.uint16)
        >>> counts_to_radiance(counts, 2e-3, -10.0).round(6).tolist()
        [-10.0, 121.07]
    """
    xp, (counts, gain, offset) = floating_inputs(counts, gain, offset)
    # L = g · DN + o
    return xp.multiply(gain, counts) + offset


@elementwise()
def radiance_to_counts(radiance: Any, gain: Any, offset: Any = 0.0) -> Any:
    """Digital counts from radiance, the inverse of `counts_to_radiance`.

    DN = (L − o) / g

    Args:
        radiance: L, in the gain's unit.
        gain: g, radiance per count.
        offset: o, radiance at zero counts.

    Returns:
        DN, floating; round it to store counts.

    Examples:
        >>> from radiomax.radiometry import radiance_to_counts
        >>> round(float(radiance_to_counts(121.07, 2e-3, -10.0)))
        65535
    """
    xp, (radiance, gain, offset) = floating_inputs(radiance, gain, offset)
    # DN = (L − o) / g
    return xp.divide(radiance - offset, gain)


@elementwise()
def counts_to_reflectance(counts: Any, scale: Any, offset: Any = 0.0) -> Any:
    """Reflectance from the counts of a product that stores scaled reflectance.

    ρ = a · DN + b

    such as Sentinel-2 L1C (a = 1e-4, and b = −0.1 from processing baseline
    04.00 on) and Landsat Collection 2 surface reflectance (a = 2.75e-5,
    b = −0.2). A product's metadata has the last word.

    Args:
        counts: DN, digital counts.
        scale: a, reflectance per count.
        offset: b, reflectance at zero counts.

    Returns:
        ρ, dimensionless.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import counts_to_reflectance
        >>> counts = np.asarray([1000, 6000], dtype=np.int16)
        >>> counts_to_reflectance(counts, 1e-4, -0.1).round(6).tolist()
        [0.0, 0.5]
    """
    xp, (counts, scale, offset) = floating_inputs(counts, scale, offset)
    # ρ = a · DN + b
    return xp.multiply(scale, counts) + offset
