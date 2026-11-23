"""The BRDF kernels of the MODIS model: RossThick and LiSparse-Reciprocal."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

from radiomax._src.backend import floating_inputs, like
from radiomax._src.geometry.angles import phase_radians
from radiomax._src.labelled import elementwise
from radiomax._src.numerics import full_turn, safe_sqrt, to_radians


__all__ = [
    "li_sparse_reciprocal",
    "ross_thick",
]


@elementwise()
def ross_thick(
    solar_zenith: Any, view_zenith: Any, relative_azimuth: Any, *, degrees: bool = True
) -> Any:
    """The RossThick kernel, volume scattering by a dense canopy of small leaves.

    K = ((π/2 − ξ) cos ξ + sin ξ) / (cos θₛ + cos θᵥ) − π/4

    with ξ the phase angle. It is 0 at nadir with the Sun overhead and
    symmetric in Sun and view. Second derivatives are not finite at nadir.

    Args:
        solar_zenith: Solar zenith θₛ, degrees (radians with
            ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        relative_azimuth: Relative azimuth ϕ, the same unit; 0 is backscatter,
            where the hotspot is.
        degrees: Whether the angles are in degrees.

    Returns:
        K, dimensionless; NaN where either zenith is 90° or more.

    References:
        Lucht, Schaaf and Strahler (2000), An algorithm for the retrieval of
        albedo from space using semiempirical BRDF models, IEEE Transactions
        on Geoscience and Remote Sensing 38, 977-998.

    Examples:
        Towards the hotspot, and away from it:

        >>> import numpy as np
        >>> from radiomax.retrievals import ross_thick
        >>> ross_thick(45.0, 45.0, np.asarray([0.0, 180.0])).round(4).tolist()
        [0.3253, -0.0783]
        >>> float(ross_thick(0.0, 0.0, 0.0))
        0.0
    """
    xp, valid, solar, view, azimuth = _angles(
        solar_zenith, view_zenith, relative_azimuth, degrees=degrees
    )
    phase = phase_radians(solar, view, azimuth, xp)
    # (π/2 − ξ) cos ξ + sin ξ
    lobe = (like(math.pi / 2.0, phase, xp) - phase) * xp.cos(phase) + xp.sin(phase)
    kernel = lobe / (xp.cos(solar) + xp.cos(view)) - like(math.pi / 4.0, phase, xp)
    return xp.where(valid, kernel, like(xp.nan, kernel, xp))


@elementwise()
def li_sparse_reciprocal(
    solar_zenith: Any,
    view_zenith: Any,
    relative_azimuth: Any,
    *,
    height_over_breadth: Any = 2.0,
    degrees: bool = True,
) -> Any:
    """The LiSparse-Reciprocal kernel, shadows cast by sparse crowns.

    K = O − sec θₛ − sec θᵥ + ½ (1 + cos ξ) sec θₛ sec θᵥ

    where the overlap of the view and Sun shadows is
    O = (t − sin t cos t) (sec θₛ + sec θᵥ) / π, with
    cos t = (h/b) √(D² + (tan θₛ tan θᵥ sin ϕ)²) / (sec θₛ + sec θᵥ) clipped
    at 1, D² = tan² θₛ + tan² θᵥ − 2 tan θₛ tan θᵥ cos ϕ, and ξ the phase
    angle. The crowns are spheres (b/r = 1, as in the MODIS product), so the
    angles need no transformation.

    Args:
        solar_zenith: Solar zenith θₛ, degrees (radians with
            ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        relative_azimuth: Relative azimuth ϕ, the same unit; 0 is backscatter.
        height_over_breadth: h/b, the height of a crown's centre over its
            vertical radius; 2 in the MODIS product.
        degrees: Whether the angles are in degrees.

    Returns:
        K, dimensionless; NaN where either zenith is 90° or more.

    References:
        Lucht, Schaaf and Strahler (2000), An algorithm for the retrieval of
        albedo from space using semiempirical BRDF models, IEEE Transactions
        on Geoscience and Remote Sensing 38, 977-998.

    Examples:
        >>> import numpy as np
        >>> from radiomax.retrievals import li_sparse_reciprocal
        >>> azimuth = np.asarray([0.0, 180.0])
        >>> li_sparse_reciprocal(45.0, 45.0, azimuth).round(4).tolist()
        [0.5858, -1.8284]
        >>> float(li_sparse_reciprocal(0.0, 0.0, 0.0))
        0.0
    """
    xp, valid, solar, view, azimuth = _angles(
        solar_zenith,
        view_zenith,
        relative_azimuth,
        degrees=degrees,
        params=(height_over_breadth,),
    )
    sec_solar, sec_view = 1.0 / xp.cos(solar), 1.0 / xp.cos(view)
    tan_solar, tan_view = xp.tan(solar), xp.tan(view)
    product = tan_solar * tan_view
    # D² with 1 − cos ϕ = 2 sin²(ϕ/2), which does not cancel at the hotspot
    distance_squared = (tan_solar - tan_view) ** 2 + 4.0 * product * (
        xp.sin(azimuth / 2.0) ** 2
    )
    reach = safe_sqrt(distance_squared + (product * xp.sin(azimuth)) ** 2, xp)
    # cos t, at most 1: past it the shadows do not overlap
    one = xp.ones_like(reach)
    cos_t = like(height_over_breadth, reach, xp) * reach / (sec_solar + sec_view)
    cos_t = xp.where(cos_t < one, cos_t, one)
    sin_t = safe_sqrt((1.0 - cos_t) * (1.0 + cos_t), xp)
    t = xp.atan2(sin_t, cos_t)
    # O = (t − sin t cos t) (sec θₛ + sec θᵥ) / π
    overlap = (t - sin_t * cos_t) * (sec_solar + sec_view) / like(math.pi, t, xp)
    cos_phase = xp.cos(phase_radians(solar, view, azimuth, xp))
    kernel = (
        overlap - sec_solar - sec_view + 0.5 * (1.0 + cos_phase) * sec_solar * sec_view
    )
    return xp.where(valid, kernel, like(xp.nan, kernel, xp))


def _angles(
    solar_zenith: Any,
    view_zenith: Any,
    relative_azimuth: Any,
    *,
    degrees: bool,
    params: tuple[Any, ...] = (),
) -> tuple[ModuleType, Any, Any, Any, Any]:
    """The namespace, where the zeniths are above the horizon, and the radians.

    An invalid pixel gets zeniths of 0, so that its gradient stays finite.
    """
    xp, (solar, view, azimuth) = floating_inputs(
        solar_zenith, view_zenith, relative_azimuth, params=params
    )
    quarter = like(full_turn(degrees=degrees) / 4.0, solar, xp)
    valid = (xp.abs(solar) < quarter) & (xp.abs(view) < quarter)
    zero = xp.zeros_like(solar)
    return (
        xp,
        valid,
        to_radians(xp.where(valid, solar, zero), xp, degrees=degrees),
        to_radians(xp.where(valid, view, zero), xp, degrees=degrees),
        to_radians(azimuth, xp, degrees=degrees),
    )
