"""Viewing angles of a satellite from its Earth-centred position."""

from __future__ import annotations

import math
from typing import Any

from geophysix.earth import WGS84, Ellipsoid, geodetic_to_ecef

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise
from radiomax._src.numerics import safe_atan2, safe_sqrt, to_radians, wrap_turn


__all__ = [
    "view_angles",
]


@elementwise(n_outputs=2)
def view_angles(
    lon: Any,
    lat: Any,
    satellite_x: Any,
    satellite_y: Any,
    satellite_z: Any,
    *,
    height: Any = 0.0,
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> tuple[Any, Any]:
    """The view zenith and azimuth of a satellite at each pixel.

    ℓ = s − p, cos θᵥ = ℓ · û / |ℓ|

    with p the pixel and s the satellite, both Earth-centred, and û the
    geodetic normal at the pixel (not the geocentric one, which differs by up
    to 0.19°).

    Algorithm:
        1. The pixel's Earth-centred position p and the line of sight ℓ.
        2. ℓ in the local east, north, up frame.
        3. The zenith and azimuth from atan2 of its components.

    Args:
        lon: Longitude λ of the pixel, degrees (radians with
            ``degrees=False``).
        lat: Geodetic latitude φ of the pixel, degrees (radians with
            ``degrees=False``).
        satellite_x: The satellite's Earth-centred x, m.
        satellite_y: The satellite's Earth-centred y, m.
        satellite_z: The satellite's Earth-centred z, m.
        height: Height of the pixel above the ellipsoid, m.
        earth: The body; WGS84 by default.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        The zenith θᵥ in [0°, 180°], above 90° where the satellite is below
        the horizon, and the azimuth in [0°, 360°), from the pixel towards the
        satellite, clockwise from north. Straight overhead the azimuth is 0.
        In float32, where an Earth-centred coordinate resolves 0.5 m, the
        zenith of a satellite at 700 km is good to about 2e-4°.

    Examples:
        A satellite 700 km above the pixel, and one 4° to the north:

        >>> import numpy as np
        >>> from geophysix.earth import geodetic_to_ecef
        >>> from radiomax.geometry import view_angles
        >>> overhead = geodetic_to_ecef(12.0, 34.0, 700e3)
        >>> round(float(view_angles(12.0, 34.0, *overhead)[0]), 6)
        0.0
        >>> zenith, azimuth = view_angles(
        ...     12.0, 34.0, *geodetic_to_ecef(12.0, 38.0, 700e3)
        ... )
        >>> round(float(zenith), 1), round(float(azimuth), 6)
        (35.8, 0.0)
    """
    xp, (lon, lat, x, y, z, height) = floating_inputs(
        lon,
        lat,
        satellite_x,
        satellite_y,
        satellite_z,
        height,
        params=(earth.semi_major, earth.flattening),
    )
    longitude = to_radians(lon, xp, degrees=degrees)
    latitude = to_radians(lat, xp, degrees=degrees)
    pixel = geodetic_to_ecef(longitude, latitude, height, earth=earth, degrees=False)
    # ℓ = s − p
    sight_x, sight_y, sight_z = x - pixel[0], y - pixel[1], z - pixel[2]
    sin_lon, cos_lon = xp.sin(longitude), xp.cos(longitude)
    sin_lat, cos_lat = xp.sin(latitude), xp.cos(latitude)
    # ℓ · ê, ê = (−sin λ, cos λ, 0)
    east = cos_lon * sight_y - sin_lon * sight_x
    # the part of ℓ in the meridian plane, away from the axis
    outward = cos_lon * sight_x + sin_lon * sight_y
    # ℓ · n̂, n̂ = (−sin φ cos λ, −sin φ sin λ, cos φ)
    north = cos_lat * sight_z - sin_lat * outward
    # ℓ · û, û = (cos φ cos λ, cos φ sin λ, sin φ)
    up = cos_lat * outward + sin_lat * sight_z
    # θᵥ = atan2(√((ℓ·ê)² + (ℓ·n̂)²), ℓ·û)
    zenith = xp.atan2(safe_sqrt(east**2 + north**2, xp), up)
    unit = like(math.pi / 180.0, up, xp) if degrees else like(1.0, up, xp)
    azimuth = wrap_turn(safe_atan2(east, north, xp) / unit, xp, degrees=degrees)
    return zenith / unit, azimuth
