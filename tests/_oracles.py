"""Reference results, each skipped when its library is missing.

Every helper calls ``pytest.importorskip`` itself, so a test that uses an
oracle skips, rather than errors, in an environment without it. integrix and
functax need JAX and are installed only by the job with every extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


__all__ = [
    "RADTRAX_COMMIT",
    "functax_enu",
    "functax_trapezoid_weights",
    "georeader_band_irradiance",
    "georeader_reflectance",
    "integrix_band_mean",
    "proj_geostationary",
    "proj_view_angles",
    "radtrax_values",
    "spa_distance",
    "spa_sun",
]

#: The radtrax commit whose values are stored in ``tests/data``.
RADTRAX_COMMIT = "fadf2a6"

_EPOCH = np.datetime64("1970-01-01", "us")


def radtrax_values() -> dict[str, np.ndarray]:
    """The values stored from radtrax, by name (see ``make_radtrax_values.py``)."""
    path = Path(__file__).parent / "data" / f"radtrax_{RADTRAX_COMMIT}.npz"
    with np.load(path) as stored:
        return dict(stored)


def _spa_times(time: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """pvlib's SPA module, seconds since 1970 and ΔT, for UTC datetime64 values."""
    spa = pytest.importorskip("pvlib.spa")
    time = np.atleast_1d(np.asarray(time, dtype="datetime64[us]"))
    seconds = (time - _EPOCH) / np.timedelta64(1, "s")
    years = time.astype("datetime64[Y]").astype(int) + 1970
    months = time.astype("datetime64[M]").astype(int) % 12 + 1
    delta_t = np.asarray(spa.calculate_deltat(years, months), dtype=np.float64)
    return spa, seconds, delta_t


def spa_sun(time: Any, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """The Sun's zenith and azimuth (degrees) by SPA, without refraction.

    Args:
        time: UTC datetime64 values.
        lon: Longitude, degrees.
        lat: Latitude, degrees.

    Returns:
        The zenith, and the azimuth clockwise from north.
    """
    spa, seconds, delta_t = _spa_times(time)
    lon, lat = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (lon, lat)))
    out = spa.solar_position_numpy(
        seconds, lat, lon, 0.0, 1013.25, 12.0, delta_t, 0.0, 1
    )
    # rows: apparent zenith, zenith, apparent elevation, elevation, azimuth, …
    return np.asarray(out[1]), np.asarray(out[4])


def spa_distance(time: Any) -> np.ndarray:
    """The Earth–Sun distance (AU) by SPA, for UTC datetime64 values."""
    spa, seconds, delta_t = _spa_times(time)
    return np.asarray(spa.earthsun_distance(seconds, delta_t, 1))


def georeader_reflectance(
    radiance: np.ndarray, irradiance: np.ndarray, factor: float
) -> np.ndarray:
    """Top-of-atmosphere reflectance by georeader.

    Args:
        radiance: (band, y, x), W m⁻² sr⁻¹ nm⁻¹.
        irradiance: (band,), W m⁻² nm⁻¹.
        factor: π d² / cos θₛ.

    Returns:
        (band, y, x).
    """
    reflectance = pytest.importorskip("georeader.reflectance")
    out = reflectance.radiance_to_reflectance(
        radiance,
        irradiance,
        observation_date_corr_factor=factor,
        units="W/m2/sr/nm",
    )
    return np.asarray(out)


def georeader_band_irradiance(
    solar_spectrum: np.ndarray, responses: np.ndarray, wavelength_nm: np.ndarray
) -> np.ndarray:
    """Band irradiance by georeader, (W,), (B, W), (W,) → (B,)."""
    reflectance = pytest.importorskip("georeader.reflectance")
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame(responses.T, index=wavelength_nm)
    solar = pd.DataFrame(
        {"Nanometer": wavelength_nm, "Radiance(mW/m2/nm)": solar_spectrum}
    )
    out = reflectance.integrated_irradiance(table, solar_irradiance=solar)
    return np.asarray(out)


def proj_view_angles(
    lon: float, lat: float, height: float, satellite: tuple[float, float, float]
) -> tuple[float, float]:
    """View zenith and azimuth (degrees) of an Earth-centred point, by PROJ.

    Args:
        lon: Longitude of the pixel, degrees.
        lat: Geodetic latitude of the pixel, degrees.
        height: Height of the pixel above WGS84, m.
        satellite: The satellite's Earth-centred position, m.

    Returns:
        The zenith, and the azimuth clockwise from north.
    """
    pyproj = pytest.importorskip("pyproj")
    local = pyproj.Transformer.from_pipeline(
        f"+proj=topocentric +ellps=WGS84 +lon_0={lon!r} +lat_0={lat!r} +h_0={height!r}"
    )
    east, north, up = local.transform(*satellite)
    zenith = np.degrees(np.arctan2(np.hypot(east, north), up))
    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
    return float(zenith), float(azimuth)


def proj_geostationary(
    x: Any, y: Any, *, lon_0: float, height: float, sweep: str
) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude (degrees) of scan angles (radians), by PROJ.

    Off the Earth both are infinite.
    """
    pyproj = pytest.importorskip("pyproj")
    crs = pyproj.CRS.from_proj4(
        f"+proj=geos +lon_0={lon_0} +h={height} +ellps=WGS84 +sweep={sweep}"
    )
    inverse = pyproj.Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon, lat = inverse.transform(np.asarray(x) * height, np.asarray(y) * height)
    return np.asarray(lon), np.asarray(lat)


def integrix_band_mean(
    spectrum: np.ndarray, responses: np.ndarray, weights: np.ndarray, **options: Any
) -> np.ndarray:
    """integrix's ``weighted_mean``, (..., W), (B, W), (W,) → (..., B)."""
    rule = pytest.importorskip("integrix.rule")
    out = rule.weighted_mean(weights, spectrum, weight=responses, **options)
    return np.asarray(out)


def functax_trapezoid_weights(nodes: np.ndarray) -> np.ndarray:
    """functax's composite trapezoid weights on given nodes."""
    quad = pytest.importorskip("functax.quad")
    return np.asarray(quad.trapezoid(nodes).weights)


def functax_enu(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """functax's local frame, (N,), (N,) → (N, 3, 3) with rows east, north, up."""
    geometry = pytest.importorskip("functax.geometry")
    jnp = pytest.importorskip("jax.numpy")
    lonlat = jnp.stack([jnp.asarray(lon), jnp.asarray(lat)], axis=-1)
    return np.asarray(geometry.enu_basis(lonlat, unit="degrees"))
