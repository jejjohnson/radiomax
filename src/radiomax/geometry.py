"""Where the Sun and the sensor are, as seen from each pixel, and the path between."""

from __future__ import annotations

from radiomax._src.geometry.air_mass import air_mass_factor, shell_air_mass
from radiomax._src.geometry.angles import (
    phase_angle,
    relative_azimuth,
    scattering_angle,
)
from radiomax._src.geometry.geostationary import (
    geostationary_lonlat,
    geostationary_scan_angles,
)
from radiomax._src.geometry.intersection import earth_intersection
from radiomax._src.geometry.parallax import apparent_position
from radiomax._src.geometry.solar import (
    earth_sun_distance,
    solar_angles,
    solar_declination,
    solar_zenith,
    utc_day_seconds,
)
from radiomax._src.geometry.view import view_angles


__all__ = [
    "air_mass_factor",
    "apparent_position",
    "earth_intersection",
    "earth_sun_distance",
    "geostationary_lonlat",
    "geostationary_scan_angles",
    "phase_angle",
    "relative_azimuth",
    "scattering_angle",
    "shell_air_mass",
    "solar_angles",
    "solar_declination",
    "solar_zenith",
    "utc_day_seconds",
    "view_angles",
]
