"""The Sun's position per pixel, the Earth–Sun distance, and the two time arrays."""

from __future__ import annotations

import datetime
import math
from types import ModuleType
from typing import Any

import numpy as np
from pipekit_array.functions import (
    elementwise,
    floating_inputs,
    is_dataarray,
    like,
    safe_atan2,
    safe_sqrt,
)

from radiomax._src.numerics import to_radians, wrap_turn
from radiomax.exceptions import ValidationError


__all__ = [
    "earth_sun_distance",
    "solar_angles",
    "solar_declination",
    "solar_zenith",
    "utc_day_seconds",
]

_EPOCH = np.datetime64("2000-01-01", "us")
_DAY = np.timedelta64(86_400_000_000, "us")
_SECONDS_PER_DAY = 86_400.0

# Astronomical Almanac low-precision formulas, in degrees and days.
_MEAN_LONGITUDE, _MEAN_LONGITUDE_RATE = 280.460, 0.9856474
_MEAN_ANOMALY, _MEAN_ANOMALY_RATE = 357.528, 0.9856003
_CENTRE_1, _CENTRE_2 = 1.915, 0.020
_OBLIQUITY, _OBLIQUITY_RATE = 23.439, -4e-7
# θ_G at 2000-01-01 00:00: 280.46061837 − 360.98564736629 / 2, since J2000.0
# is at noon
_SIDEREAL, _SIDEREAL_RATE = 99.967794687, 0.98564736629
_DISTANCE_0, _DISTANCE_1, _DISTANCE_2 = 1.00014, -0.01671, -0.00014


def utc_day_seconds(time: Any) -> tuple[Any, Any]:
    """UTC midnights since 2000-01-01, and the second of the UTC day.

    Time enters radiomax as these two arrays. In float32, which JAX uses by
    default, a Julian date resolves a quarter of a day and days since 2000
    resolve 84 s; the second of the day resolves 8 ms.

    Runs on the host, and is not traceable.

    Args:
        time: UTC times: ``datetime64`` values, ISO strings or ``datetime``
            objects, an array of them or a DataArray. A ``datetime`` with a
            time zone is converted to UTC.

    Returns:
        The day and the seconds: Python floats for one time, so that they
        adopt the dtype of the arrays they meet; float64 NumPy arrays, or
        DataArrays, otherwise. NaT gives NaN.

    Raises:
        ValidationError: If ``time`` is not a date.

    Examples:
        >>> utc_day_seconds("2000-01-02T00:00:01.5")
        (1.0, 1.5)
        >>> import numpy as np
        >>> day, seconds = utc_day_seconds(
        ...     np.asarray(["1999-12-31T23:00"], "M8[s]")
        ... )
        >>> day.tolist(), seconds.tolist()
        ([-1.0], [82800.0])
    """
    if is_dataarray(time):
        labelled = type(time)
        day, seconds = utc_day_seconds(np.asarray(time.values))
        return (
            labelled(day, coords=time.coords, dims=time.dims),
            labelled(seconds, coords=time.coords, dims=time.dims),
        )
    if isinstance(time, datetime.datetime) and time.tzinfo is not None:
        time = time.astimezone(datetime.UTC).replace(tzinfo=None)
    values = np.asarray(time)
    if values.dtype.kind not in "MOSU":
        msg = f"'time' must hold dates, got dtype {values.dtype}"
        raise ValidationError(msg)
    try:
        stamps = values.astype("datetime64[us]")
    except (TypeError, ValueError) as err:
        msg = f"'time' must hold dates: {err}"
        raise ValidationError(msg) from err
    since = stamps - _EPOCH
    missing = np.isnat(stamps)
    # NaT divides to a warning and a value that `missing` replaces
    with np.errstate(invalid="ignore"):
        whole = np.floor_divide(since, _DAY)
    day = np.where(missing, np.nan, whole.astype(np.float64))
    # t − D, in µs, exactly
    remainder = (since - whole * _DAY).astype(np.float64)
    seconds = np.where(missing, np.nan, remainder / 1e6)
    if values.ndim == 0:
        return float(day), float(seconds)
    return day, seconds


@elementwise(n_outputs=2)
def solar_angles(
    lon: Any, lat: Any, day: Any, seconds: Any, *, degrees: bool = True
) -> tuple[Any, Any]:
    """The Sun's zenith and azimuth angles at each pixel.

    cos θₛ = sin φ sin δ + cos φ cos δ cos ω

    Algorithm:
        1. The declination δ and right ascension α from the Astronomical
           Almanac's low-precision formulas.
        2. The hour angle ω = θ_G + λ − α, with θ_G the sidereal angle.
        3. The Sun's direction in the local east, north, up frame.
        4. The zenith and azimuth from atan2 of its components.

    Within 0.0125° of NREL's SPA over 2000–2050, in float64 and float32.
    Refraction is ignored, which moves an apparent zenith near the horizon by
    up to about half a degree, and UT1 is taken as UTC (below 0.004°).

    Args:
        lon: Longitude λ, degrees (radians with ``degrees=False``).
        lat: Geodetic latitude φ, degrees (radians with ``degrees=False``).
        day: UTC midnights since 2000-01-01 (`utc_day_seconds`).
        seconds: Second of the UTC day, s.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        The zenith θₛ in [0°, 180°], the night side included, and the azimuth
        in [0°, 360°), clockwise from north. With the Sun exactly overhead or
        underfoot the azimuth is 0.

    References:
        Reda, I. and Andreas, A. (2004), Solar position algorithm for solar
        radiation applications, Solar Energy 76 (5), 577-589 (SPA, the
        reference of the tests).

    Examples:
        >>> from radiomax.geometry import solar_angles, utc_day_seconds
        >>> day, seconds = utc_day_seconds("2024-03-20T12:07:25")
        >>> zenith, azimuth = solar_angles(0.0, 45.0, day, seconds)
        >>> round(float(zenith)), round(float(azimuth))
        (45, 180)
    """
    xp, (lon, lat, day, seconds) = floating_inputs(lon, lat, day, seconds)
    east, north, up = _solar_direction(lon, lat, day, seconds, xp, degrees=degrees)
    unit = like(math.pi / 180.0, up, xp) if degrees else like(1.0, up, xp)
    azimuth = wrap_turn(safe_atan2(east, north, xp) / unit, xp, degrees=degrees)
    return _zenith(east, north, up, xp) / unit, azimuth


@elementwise()
def solar_zenith(
    lon: Any, lat: Any, day: Any, seconds: Any, *, degrees: bool = True
) -> Any:
    """The Sun's zenith angle at each pixel.

    cos θₛ = sin φ sin δ + cos φ cos δ cos ω

    The first output of `solar_angles`, whose gradient is also finite where
    the Sun is overhead.

    Args:
        lon: Longitude λ, degrees (radians with ``degrees=False``).
        lat: Geodetic latitude φ, degrees (radians with ``degrees=False``).
        day: UTC midnights since 2000-01-01 (`utc_day_seconds`).
        seconds: Second of the UTC day, s.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        θₛ in [0°, 180°], the night side included.

    Examples:
        >>> import numpy as np
        >>> from radiomax.geometry import solar_zenith
        >>> lat = np.asarray([0.0, 45.0], dtype=np.float32)
        >>> solar_zenith(0.0, lat, 8845.0, 43_645.0).dtype
        dtype('float32')
    """
    xp, (lon, lat, day, seconds) = floating_inputs(lon, lat, day, seconds)
    east, north, up = _solar_direction(lon, lat, day, seconds, xp, degrees=degrees)
    zenith = _zenith(east, north, up, xp)
    return zenith / like(math.pi / 180.0, up, xp) if degrees else zenith


@elementwise()
def solar_declination(day: Any, seconds: Any = 0.0, *, degrees: bool = True) -> Any:
    """The Sun's declination, its latitude on the celestial sphere.

    δ = asin(sin ε sin λ☉), with ε the obliquity and λ☉ the ecliptic longitude

    Args:
        day: UTC midnights since 2000-01-01 (`utc_day_seconds`).
        seconds: Second of the UTC day, s.
        degrees: Whether to return degrees.

    Returns:
        δ, between −23.44° and 23.44°.

    Examples:
        >>> from radiomax.geometry import solar_declination, utc_day_seconds
        >>> round(float(solar_declination(*utc_day_seconds("2024-06-20T20:51"))), 2)
        23.44
    """
    xp, (day, seconds) = floating_inputs(day, seconds)
    sin_declination, cos_declination, _ = _sun(day, seconds, xp)
    declination = xp.atan2(sin_declination, cos_declination)
    return declination / like(math.pi / 180.0, day, xp) if degrees else declination


@elementwise()
def earth_sun_distance(day: Any, seconds: Any = 0.0) -> Any:
    """The distance from the Earth to the Sun.

    d = 1.00014 − 0.01671 cos g − 0.00014 cos 2g, with g the mean anomaly

    Within 8.8 × 10⁻⁵ AU of NREL's SPA over 2000–2050. The formula
    d = 1 − 0.01673 cos(0.0172 (t − 4)) on the day of the year t, which
    georeader uses, is off by up to 8.0 × 10⁻⁴ AU; a reflectance made with
    it differs by up to 1.5 × 10⁻³ relative.

    Args:
        day: UTC midnights since 2000-01-01 (`utc_day_seconds`).
        seconds: Second of the UTC day, s.

    Returns:
        d, astronomical units.

    Examples:
        >>> from radiomax.geometry import earth_sun_distance, utc_day_seconds
        >>> round(float(earth_sun_distance(*utc_day_seconds("2024-01-03"))), 4)
        0.9833
    """
    xp, (day, seconds) = floating_inputs(day, seconds)
    anomaly = _mean_anomaly(day, _day_fraction(seconds, xp), xp)
    # d = 1.00014 − 0.01671 cos g − 0.00014 cos 2g
    first = like(_DISTANCE_1, day, xp) * xp.cos(anomaly)
    second = like(_DISTANCE_2, day, xp) * xp.cos(2.0 * anomaly)
    return like(_DISTANCE_0, day, xp) + first + second


def _day_fraction(seconds: Any, xp: ModuleType) -> Any:
    """f = s / 86400."""
    return seconds / like(_SECONDS_PER_DAY, seconds, xp)


def _advance(start: float, rate: float, day: Any, fraction: Any, xp: ModuleType) -> Any:
    """start + rate · n in radians, n = D + f − 1/2, reduced before f is added.

    Adding f to D first would lose it in float32, where D ≈ 9,800 resolves
    84 s.
    """
    whole = like(start, day, xp) + like(rate, day, xp) * day
    part = like(rate, day, xp) * (fraction - 0.5)
    angle = xp.remainder(whole, like(360.0, day, xp)) + part
    return to_radians(angle, xp, degrees=True)


def _mean_anomaly(day: Any, fraction: Any, xp: ModuleType) -> Any:
    """g = 357.528° + 0.9856003° n, radians."""
    return _advance(_MEAN_ANOMALY, _MEAN_ANOMALY_RATE, day, fraction, xp)


def _sun(day: Any, seconds: Any, xp: ModuleType) -> tuple[Any, Any, Any]:
    """sin δ, cos δ and the right ascension α (radians)."""
    fraction = _day_fraction(seconds, xp)
    # L = 280.460° + 0.9856474° n
    longitude = _advance(_MEAN_LONGITUDE, _MEAN_LONGITUDE_RATE, day, fraction, xp)
    anomaly = _mean_anomaly(day, fraction, xp)
    # λ☉ = L + 1.915° sin g + 0.020° sin 2g
    centre = like(_CENTRE_1, day, xp) * xp.sin(anomaly)
    centre = centre + like(_CENTRE_2, day, xp) * xp.sin(2.0 * anomaly)
    ecliptic = longitude + to_radians(centre, xp, degrees=True)
    # ε = 23.439° − 4 × 10⁻⁷° n, where f does not matter
    tilt = like(_OBLIQUITY, day, xp) + like(_OBLIQUITY_RATE, day, xp) * day
    obliquity = to_radians(tilt, xp, degrees=True)
    # the unit vector to the Sun on the celestial sphere, (x, y, sin δ)
    x = xp.cos(ecliptic)
    y = xp.cos(obliquity) * xp.sin(ecliptic)
    sin_declination = xp.sin(obliquity) * xp.sin(ecliptic)
    # cos δ = √(x² + y²), never 0 for the Sun
    cos_declination = xp.sqrt(x**2 + y**2)
    # α = atan2(cos ε sin λ☉, cos λ☉)
    return sin_declination, cos_declination, xp.atan2(y, x)


def _solar_direction(
    lon: Any, lat: Any, day: Any, seconds: Any, xp: ModuleType, *, degrees: bool
) -> tuple[Any, Any, Any]:
    """The unit vector to the Sun in the local east, north, up frame."""
    sin_declination, cos_declination, ascension = _sun(day, seconds, xp)
    # θ_G = 99.967794687° + 0.98564736629° D + 360.98564736629° f
    whole = like(_SIDEREAL, day, xp) + like(_SIDEREAL_RATE, day, xp) * day
    part = like(360.0 + _SIDEREAL_RATE, day, xp) * _day_fraction(seconds, xp)
    sidereal = xp.remainder(whole, like(360.0, day, xp)) + part
    # ω = θ_G + λ − α
    hour_angle = to_radians(sidereal, xp, degrees=True) - ascension
    hour_angle = hour_angle + to_radians(lon, xp, degrees=degrees)
    latitude = to_radians(lat, xp, degrees=degrees)
    sin_lat, cos_lat = xp.sin(latitude), xp.cos(latitude)
    cos_hour = cos_declination * xp.cos(hour_angle)
    # east = −cos δ sin ω
    east = -cos_declination * xp.sin(hour_angle)
    # north = sin δ cos φ − cos δ sin φ cos ω
    north = sin_declination * cos_lat - sin_lat * cos_hour
    # up = sin φ sin δ + cos φ cos δ cos ω
    up = sin_lat * sin_declination + cos_lat * cos_hour
    return east, north, up


def _zenith(east: Any, north: Any, up: Any, xp: ModuleType) -> Any:
    """θ = atan2(√(east² + north²), up), radians."""
    horizontal = safe_sqrt(east**2 + north**2, xp)
    return xp.atan2(horizontal, up)
