"""Top-of-atmosphere reflectance and the radiance it stands for."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

from pipekit_array.functions import elementwise, floating_inputs, like

from radiomax._src.numerics import to_radians


__all__ = [
    "radiance_from_reflectance",
    "reflectance",
]

_MAX_ZENITH_DEGREES = 85.0


@elementwise()
def reflectance(
    radiance: Any,
    irradiance: Any,
    solar_zenith: Any,
    distance: Any = 1.0,
    *,
    max_zenith: float | None = None,
    degrees: bool = True,
) -> Any:
    """Top-of-atmosphere reflectance from radiance.

    ρ = π L d² / (E₀ cos θₛ)

    The radiance and the irradiance must be per the same spectral unit, such
    as both per nm; the function cannot check it.

    Args:
        radiance: L, W m⁻² sr⁻¹ per spectral unit.
        irradiance: E₀, the band's solar irradiance at 1 AU, W m⁻² per the same
            spectral unit. Per band, as (band, 1, 1) or as a DataArray on the
            band dimension.
        solar_zenith: θₛ at each pixel, degrees (radians with
            ``degrees=False``).
        distance: d, the Earth–Sun distance, AU (`earth_sun_distance`).
        max_zenith: The zenith from which the result is NaN, in the unit of
            ``solar_zenith``; 85° by default, because cos θₛ → 0 makes ρ blow
            up.
        degrees: Whether the angles are in degrees.

    Returns:
        ρ, dimensionless; NaN where the zenith is ``max_zenith`` or more.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import reflectance
        >>> zenith = np.asarray([0.0, 60.0, 89.0])
        >>> reflectance(0.1, np.pi, zenith).round(6).tolist()
        [0.1, 0.2, nan]
        >>> limit = np.deg2rad(89.5)
        >>> radians = np.deg2rad(zenith)
        >>> reflectance(0.1, np.pi, radians, max_zenith=limit, degrees=False).round(
        ...     2
        ... )
        array([0.1 , 0.2 , 5.73])
    """
    xp, (radiance, irradiance, zenith, distance) = floating_inputs(
        radiance, irradiance, solar_zenith, distance
    )
    if max_zenith is None:
        max_zenith = _MAX_ZENITH_DEGREES
        max_zenith = max_zenith if degrees else math.radians(max_zenith)
    lit = xp.abs(zenith) < like(max_zenith, zenith, xp)
    # the inner where keeps the unused branch finite, and so its gradient
    safe_zenith = xp.where(lit, zenith, xp.zeros_like(zenith))
    cosine = _cosine(safe_zenith, xp, degrees=degrees)
    # ρ = π L d² / (E₀ cos θₛ)
    incoming = irradiance * cosine
    value = like(math.pi, radiance, xp) * radiance * distance**2 / incoming
    return xp.where(lit, value, like(xp.nan, value, xp))


@elementwise()
def radiance_from_reflectance(
    reflectance: Any,
    irradiance: Any,
    solar_zenith: Any,
    distance: Any = 1.0,
    *,
    degrees: bool = True,
) -> Any:
    """Radiance from top-of-atmosphere reflectance, the inverse of `reflectance`.

    L = ρ E₀ cos θₛ / (π d²)

    Args:
        reflectance: ρ, dimensionless.
        irradiance: E₀, the band's solar irradiance at 1 AU, W m⁻² per spectral
            unit.
        solar_zenith: θₛ at each pixel, degrees (radians with
            ``degrees=False``).
        distance: d, the Earth–Sun distance, AU (`earth_sun_distance`).
        degrees: Whether the angle is in degrees.

    Returns:
        L, W m⁻² sr⁻¹ per the irradiance's spectral unit. It is 0 at 90° and
        negative beyond, where a reflectance means nothing.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import radiance_from_reflectance
        >>> round(float(radiance_from_reflectance(0.2, np.pi, 60.0)), 12)
        0.1
    """
    xp, (reflectance, irradiance, zenith, distance) = floating_inputs(
        reflectance, irradiance, solar_zenith, distance
    )
    cosine = _cosine(zenith, xp, degrees=degrees)
    # L = ρ E₀ cos θₛ / (π d²)
    incoming = irradiance * cosine
    return reflectance * incoming / (like(math.pi, cosine, xp) * distance**2)


def _cosine(zenith: Any, xp: ModuleType, *, degrees: bool) -> Any:
    """cos θ."""
    return xp.cos(to_radians(zenith, xp, degrees=degrees))
