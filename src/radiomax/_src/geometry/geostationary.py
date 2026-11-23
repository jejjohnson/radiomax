"""Geostationary navigation: scan angles to longitude and latitude, and back."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, Literal

from geophysix.earth import WGS84, Ellipsoid, ecef_to_geodetic, geodetic_to_ecef
from pipekit_array.functions import elementwise, floating_inputs, like

from radiomax._src.geometry.intersection import near_root
from radiomax._src.numerics import safe_sqrt, to_radians, wrap_turn
from radiomax.exceptions import ValidationError


__all__ = [
    "geostationary_lonlat",
    "geostationary_on_earth",
    "geostationary_scan_angles",
]


@elementwise(n_outputs=3)
def geostationary_lonlat(
    x: Any,
    y: Any,
    *,
    lon_0: Any,
    height: Any,
    sweep: Literal["x", "y"],
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> tuple[Any, Any, Any]:
    """Longitude and latitude of a geostationary imager's scan angles.

    d = (cos x cos y, sin x, cos x sin y) for sweep x (GOES ABI), and
    d = (cos x cos y, sin x cos y, sin y) for sweep y (AHI, FCI, SEVIRI)

    is the unit line of sight, with axes towards the Earth's centre, east and
    north. The pixel is where it first meets the ellipsoid.

    Algorithm:
        1. The line of sight d from the satellite at (a + h, 0, 0), in the
           Earth-fixed frame whose first axis goes through the sub-satellite
           longitude.
        2. The near intersection with the ellipsoid.
        3. Its geodetic position, with the sub-satellite longitude added.

    Args:
        x: Scan angle x, positive east, rad (always, as files hold it).
        y: Scan angle y, positive north, rad.
        lon_0: Sub-satellite longitude λ₀, degrees (radians with
            ``degrees=False``).
        height: The satellite's height h above the equator, m.
        sweep: The axis the instrument sweeps along, ``"x"`` for GOES ABI and
            ``"y"`` for AHI, FCI and SEVIRI. There is no default: the two
            differ by kilometres away from the axes.
        earth: The body; WGS84 by default. An imager's file names its own
            (GOES ABI's is GRS80).
        degrees: Whether the longitudes and the latitude are in degrees.

    Returns:
        The longitude in [−180°, 180°), the geodetic latitude, and whether the
        line of sight meets the Earth; NaN where it does not. In float32 the
        position is good to about 1e-4°.

    Raises:
        ValidationError: If ``sweep`` is neither ``"x"`` nor ``"y"``.

    Examples:
        The sub-satellite point of a satellite at 75.2° W, and space:

        >>> import numpy as np
        >>> from radiomax.geometry import geostationary_lonlat
        >>> lon, lat, visible = geostationary_lonlat(
        ...     np.asarray([0.0, 0.16]),
        ...     0.0,
        ...     lon_0=-75.2,
        ...     height=35786023.0,
        ...     sweep="x",
        ... )
        >>> lon.round(9).tolist(), lat.tolist(), visible.tolist()
        ([-75.2, nan], [0.0, nan], [True, False])
    """
    _check_sweep(sweep)
    xp, (x, y, lon_0, height) = floating_inputs(
        x, y, lon_0, height, params=(earth.semi_major, earth.flattening)
    )
    inward, east, north = _line_of_sight(x, y, sweep, xp)
    zero = xp.zeros_like(inward)
    # the satellite, at a + h on the first axis
    distance = zero + like(earth.semi_major, inward, xp) + height
    t, visible, _ = near_root((distance, zero, zero), (-inward, east, north), earth, xp)
    lon, lat, _ = ecef_to_geodetic(
        distance - t * inward, t * east, t * north, earth=earth, degrees=False
    )
    unit = like(math.pi / 180.0, lon, xp) if degrees else like(1.0, lon, xp)
    nan = like(xp.nan, lon, xp)
    half = like(180.0 if degrees else math.pi, lon, xp)
    # λ + λ₀ in [−180°, 180°)
    lon = wrap_turn(lon / unit + lon_0 + half, xp, degrees=degrees) - half
    return xp.where(visible, lon, nan), xp.where(visible, lat / unit, nan), visible


@elementwise(n_outputs=3)
def geostationary_scan_angles(
    lon: Any,
    lat: Any,
    *,
    lon_0: Any,
    height: Any,
    sweep: Literal["x", "y"],
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> tuple[Any, Any, Any]:
    """Scan angles at which a geostationary imager sees a position.

    x = asin d₂, y = atan2(d₃, d₁) for sweep x (GOES ABI), and
    y = asin d₃, x = atan2(d₂, d₁) for sweep y (AHI, FCI, SEVIRI)

    with d the unit line of sight from the satellite to the position, as
    `geostationary_lonlat` defines it, which this inverts.

    Args:
        lon: Longitude λ, degrees (radians with ``degrees=False``).
        lat: Geodetic latitude φ, degrees (radians with ``degrees=False``).
        lon_0: Sub-satellite longitude λ₀, the same unit.
        height: The satellite's height h above the equator, m.
        sweep: The axis the instrument sweeps along, ``"x"`` for GOES ABI and
            ``"y"`` for AHI, FCI and SEVIRI.
        earth: The body; WGS84 by default.
        degrees: Whether the longitudes and the latitude are in degrees.

    Returns:
        The scan angles x and y, rad, and whether the satellite is above the
        position's horizon; NaN where it is not.

    Raises:
        ValidationError: If ``sweep`` is neither ``"x"`` nor ``"y"``.

    Examples:
        >>> from radiomax.geometry import geostationary_scan_angles
        >>> x, y, visible = geostationary_scan_angles(
        ...     -75.2, 45.0, lon_0=-75.2, height=35786023.0, sweep="x"
        ... )
        >>> float(x), round(float(y), 6), bool(visible)
        (0.0, 0.118637, True)
    """
    _check_sweep(sweep)
    xp, (lon, lat, lon_0, height) = floating_inputs(
        lon, lat, lon_0, height, params=(earth.semi_major, earth.flattening)
    )
    longitude = to_radians(lon - lon_0, xp, degrees=degrees)
    latitude = to_radians(lat, xp, degrees=degrees)
    px, py, pz = geodetic_to_ecef(longitude, latitude, 0.0, earth=earth, degrees=False)
    distance = like(earth.semi_major, px, xp) + height
    # the line of sight, towards the Earth's centre, east and north
    inward = distance - px
    # û, the geodetic normal
    up_x = xp.cos(latitude) * xp.cos(longitude)
    up_y = xp.cos(latitude) * xp.sin(longitude)
    up_z = xp.sin(latitude)
    # the satellite is above the horizon when the line of sight faces û
    visible = inward * up_x - py * up_y - pz * up_z > 0
    if sweep == "x":
        # x = asin d₂ and y = atan2(d₃, d₁)
        x = xp.atan2(py, safe_sqrt(inward**2 + pz**2, xp))
        y = xp.atan2(pz, inward)
    else:
        # y = asin d₃ and x = atan2(d₂, d₁)
        x = xp.atan2(py, inward)
        y = xp.atan2(pz, safe_sqrt(inward**2 + py**2, xp))
    nan = like(xp.nan, x, xp)
    return xp.where(visible, x, nan), xp.where(visible, y, nan), visible


@elementwise()
def geostationary_on_earth(
    x: Any,
    y: Any,
    *,
    height: Any,
    sweep: Literal["x", "y"],
    earth: Ellipsoid = WGS84,
) -> Any:
    """Whether a geostationary imager's line of sight meets the Earth.

    (o̲·d̲)² ≥ |d̲|² (|o̲|² − 1), in units of the semi-axes

    with o the satellite and d the line of sight of `geostationary_lonlat`,
    whose third output this is, without the cost of the positions.

    Args:
        x: Scan angle x, positive east, rad.
        y: Scan angle y, positive north, rad.
        height: The satellite's height h above the equator, m.
        sweep: The axis the instrument sweeps along, ``"x"`` for GOES ABI and
            ``"y"`` for AHI, FCI and SEVIRI.
        earth: The body; WGS84 by default.

    Returns:
        True where the line of sight meets the Earth.

    Raises:
        ValidationError: If ``sweep`` is neither ``"x"`` nor ``"y"``.

    Examples:
        >>> import numpy as np
        >>> from radiomax.geometry import geostationary_on_earth
        >>> x = np.asarray([0.0, 0.151, 0.153])
        >>> geostationary_on_earth(x, 0.0, height=35786023.0, sweep="x").tolist()
        [True, True, False]
    """
    _check_sweep(sweep)
    xp, (x, y, height) = floating_inputs(
        x, y, height, params=(earth.semi_major, earth.flattening)
    )
    inward, east, north = _line_of_sight(x, y, sweep, xp)
    a, b = like(earth.semi_major, inward, xp), like(earth.semi_minor, inward, xp)
    # o̲ = ((a + h) / a, 0, 0), and d̲ in units of the semi-axes
    origin = (a + height) / a
    length_squared = (inward**2 + east**2) / a**2 + (north / b) ** 2
    along = origin * inward / a
    return along**2 >= length_squared * (origin**2 - 1.0)


def _line_of_sight(x: Any, y: Any, sweep: str, xp: ModuleType) -> tuple[Any, Any, Any]:
    """d, towards the Earth's centre, east and north."""
    sin_x, cos_x, sin_y, cos_y = xp.sin(x), xp.cos(x), xp.sin(y), xp.cos(y)
    if sweep == "x":
        return cos_x * cos_y, sin_x, cos_x * sin_y
    return cos_x * cos_y, sin_x * cos_y, sin_y


def _check_sweep(sweep: str) -> None:
    if sweep not in ("x", "y"):
        msg = f"'sweep' must be 'x' or 'y', got {sweep!r}"
        raise ValidationError(msg)
