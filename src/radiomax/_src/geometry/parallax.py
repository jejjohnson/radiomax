"""Where an elevated target appears on the ellipsoid, seen from a satellite."""

from __future__ import annotations

import math
from typing import Any

from geophysix.earth import WGS84, Ellipsoid, ecef_to_geodetic, geodetic_to_ecef
from pipekit_array.functions import elementwise, floating_inputs, like

from radiomax._src.geometry.intersection import near_root
from radiomax._src.numerics import to_radians


__all__ = [
    "apparent_position",
]


@elementwise(n_outputs=3)
def apparent_position(
    lon: Any,
    lat: Any,
    height: Any,
    satellite_x: Any,
    satellite_y: Any,
    satellite_z: Any,
    *,
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> tuple[Any, Any, Any]:
    """Where a target at a height above a pixel appears on the ellipsoid.

    q = p + H û, and s + t (q − s) on the ellipsoid, t the smaller root

    with p the pixel, û its geodetic normal and s the satellite. The target
    appears displaced away from the satellite, by about H tan θᵥ.

    Algorithm:
        1. The target's Earth-centred position q, and the ray d = q − s.
        2. The near root of the ray and the ellipsoid, measured from q, which
           keeps its precision for a satellite far away.
        3. The geodetic longitude and latitude of that point.

    Args:
        lon: Longitude λ of the pixel, degrees (radians with
            ``degrees=False``).
        lat: Geodetic latitude φ of the pixel, degrees (radians with
            ``degrees=False``).
        height: Height H of the target above the ellipsoid, m; it may be
            negative.
        satellite_x: The satellite's Earth-centred x, m.
        satellite_y: The satellite's Earth-centred y, m.
        satellite_z: The satellite's Earth-centred z, m.
        earth: The body; WGS84 by default.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        The apparent longitude in [−180°, 180°] and latitude, and whether the
        target is seen against the Earth; NaN where it is below the horizon
        or above the limb.

    Examples:
        A plume at 10 km over Barcelona, seen from a satellite above the
        equator at 0°, appears 11 km further north:

        >>> from geophysix.earth import geodetic_to_ecef
        >>> from radiomax.geometry import apparent_position
        >>> satellite = [float(v) for v in geodetic_to_ecef(0.0, 0.0, 35786023.0)]
        >>> lon, lat, visible = apparent_position(2.17, 41.38, 10e3, *satellite)
        >>> round(float(lon), 4), round(float(lat), 4), bool(visible)
        (2.1776, 41.4794, True)
    """
    xp, (lon, lat, height, x, y, z) = floating_inputs(
        lon,
        lat,
        height,
        satellite_x,
        satellite_y,
        satellite_z,
        params=(earth.semi_major, earth.flattening),
    )
    # q = p + H û
    target = geodetic_to_ecef(
        to_radians(lon, xp, degrees=degrees),
        to_radians(lat, xp, degrees=degrees),
        height,
        earth=earth,
        degrees=False,
    )
    # d = q − s
    ray = (target[0] - x, target[1] - y, target[2] - z)
    # a target on the far side moves away from the Earth's centre along the ray
    t, visible, _ = near_root(target, ray, earth, xp)
    point = [q + t * d for q, d in zip(target, ray, strict=True)]
    lon_seen, lat_seen, _ = ecef_to_geodetic(*point, earth=earth, degrees=False)
    unit = like(math.pi / 180.0, lon_seen, xp) if degrees else like(1.0, lon_seen, xp)
    nan = like(xp.nan, lon_seen, xp)
    return (
        xp.where(visible, lon_seen / unit, nan),
        xp.where(visible, lat_seen / unit, nan),
        visible,
    )
