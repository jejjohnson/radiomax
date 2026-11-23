"""Geometric air masses: the plane-parallel one, and the spherical shell's."""

from __future__ import annotations

from typing import Any

from geophysix.earth import SPHERE

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise
from radiomax._src.numerics import full_turn, safe_sqrt, to_radians


__all__ = [
    "air_mass_factor",
    "shell_air_mass",
]


@elementwise()
def air_mass_factor(
    solar_zenith: Any, view_zenith: Any, *, degrees: bool = True
) -> Any:
    """The plane-parallel two-way air mass, Sun to surface to sensor.

    m = 1 / cos θₛ + 1 / cos θᵥ

    the path length relative to the vertical, both ways. It overestimates past
    about 75°, where `shell_air_mass` follows the curvature.

    Args:
        solar_zenith: Solar zenith θₛ, degrees (radians with
            ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        degrees: Whether the angles are in degrees.

    Returns:
        m, dimensionless, at least 2; NaN where either zenith is 90° or more.

    Examples:
        >>> import numpy as np
        >>> from radiomax.geometry import air_mass_factor
        >>> float(air_mass_factor(0.0, 0.0))
        2.0
        >>> air_mass_factor(np.asarray([60.0, 90.0]), 0.0).round(12).tolist()
        [3.0, nan]
    """
    xp, (solar, view) = floating_inputs(solar_zenith, view_zenith)
    quarter = like(full_turn(degrees=degrees) / 4.0, solar, xp)
    above = (xp.abs(solar) < quarter) & (xp.abs(view) < quarter)
    # the inner where keeps the unused branch finite, and so its gradient
    zero = xp.zeros_like(solar)
    safe_solar = to_radians(xp.where(above, solar, zero), xp, degrees=degrees)
    safe_view = to_radians(xp.where(above, view, zero), xp, degrees=degrees)
    # m = 1 / cos θₛ + 1 / cos θᵥ
    factor = 1.0 / xp.cos(safe_solar) + 1.0 / xp.cos(safe_view)
    return xp.where(above, factor, like(xp.nan, factor, xp))


@elementwise()
def shell_air_mass(
    cos_zenith: Any, z_bottom: Any, z_top: Any, *, radius: Any = SPHERE.semi_major
) -> Any:
    """The one-way air mass of a spherical shell, for a ray from the surface.

    m(μ) = (2R + z_t + z_b) / (√(R²μ² + 2R z_t + z_t²) + √(R²μ² + 2R z_b + z_b²))

    the straight slant path through the shell between z_b and z_t, over its
    thickness, with R the radius of the sphere through the surface. It tends
    to 1/μ as z/R → 0, stays finite at 90° for a shell above the surface, and
    does not divide by the thickness. Refraction is ignored. The two-way air
    mass is m(μₛ) + m(μᵥ).

    Args:
        cos_zenith: Cosine μ of the zenith angle at the surface.
        z_bottom: Height z_b of the shell's bottom above the surface, m.
        z_top: Height z_t of the shell's top above the surface, m.
        radius: R, m; the Earth's mean radius by default.

    Returns:
        m, dimensionless; NaN for a ray that points below the horizon (μ < 0).

    Examples:
        Two shells at three angles, broadcast to (3, 2):

        >>> import numpy as np
        >>> from radiomax.geometry import shell_air_mass
        >>> mu = np.asarray([[1.0], [0.5], [0.0]])
        >>> shell_air_mass(
        ...     mu, np.asarray([0.0, 1e4]), np.asarray([1e3, 2e4])
        ... ).round(2)
        array([[  1.  ,   1.  ],
               [  2.  ,   1.99],
               [112.88,  14.81]])
    """
    xp, (cosine, bottom, top) = floating_inputs(
        cos_zenith, z_bottom, z_top, params=(radius,)
    )
    r = like(radius, cosine, xp)
    upward = cosine >= 0
    safe_cosine = xp.where(upward, cosine, xp.ones_like(cosine))
    # R²μ²
    horizontal = (r * safe_cosine) ** 2
    # √(R²μ² + 2Rz + z²), whose gradient is 0 where the ray grazes the surface
    root_top = safe_sqrt(horizontal + top * (2.0 * r + top), xp)
    root_bottom = safe_sqrt(horizontal + bottom * (2.0 * r + bottom), xp)
    mass = (2.0 * r + top + bottom) / (root_top + root_bottom)
    return xp.where(upward, mass, like(xp.nan, mass, xp))
