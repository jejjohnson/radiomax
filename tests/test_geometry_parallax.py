"""The ray–ellipsoid intersection, and the parallax of an elevated target."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from geophysix.earth import WGS84, ecef_to_geodetic, geodetic_to_ecef, sphere
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.geometry import apparent_position, earth_intersection, view_angles


# geoops' defaults: a sphere of the equatorial radius, and a satellite above 0°
GEOOPS_RADIUS, GEOOPS_HEIGHT = 6_378_137.0, 35_786_023.0
GEOSTATIONARY = tuple(
    float(v) for v in geodetic_to_ecef(0.0, 0.0, GEOOPS_HEIGHT, earth=WGS84)
)


def _geoops(lon, lat, height, satellite_lon):
    """geoops ``_apparent_lonlat`` (geotoolz `3d05a9c`), on a sphere."""
    radius, altitude = GEOOPS_RADIUS, GEOOPS_HEIGHT
    lam, phi, lam_s = np.deg2rad(lon), np.deg2rad(lat), np.deg2rad(satellite_lon)
    up = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    elevated = up * (radius + height)
    satellite = np.stack([np.cos(lam_s), np.sin(lam_s), 0.0 * lam_s])
    satellite = (radius + altitude) * np.broadcast_to(satellite, (3,))[:, None]
    direction = elevated - satellite
    a = np.sum(direction**2, axis=0)
    b = 2.0 * np.sum(satellite * direction, axis=0)
    c = np.sum(satellite**2, axis=0) - radius**2
    near = (-b - np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    x, y, z = satellite + near * direction
    return np.rad2deg(np.arctan2(y, x)), np.rad2deg(np.arctan2(z, np.hypot(x, y)))


def _disk(n: int, *, seed: int = 0, reach: float = 70.0):
    """Pixels within ``reach`` degrees of the sub-satellite point."""
    rng = np.random.default_rng(seed)
    away, azimuth = rng.uniform(0.0, reach, n), rng.uniform(0.0, 2.0 * np.pi, n)
    east = np.sin(np.deg2rad(away)) * np.sin(azimuth)
    north = np.sin(np.deg2rad(away)) * np.cos(azimuth)
    lon = np.rad2deg(np.arctan2(east, np.cos(np.deg2rad(away))))
    return lon, np.rad2deg(np.arcsin(north))


class TestApparentPosition:
    def test_geoopss_formula_on_its_sphere(self) -> None:
        # gate G2-parallax
        lon, lat = _disk(300)
        height = np.random.default_rng(1).uniform(0.0, 18e3, 300)
        for satellite_lon in (0.0, 20.0):
            earth = sphere(GEOOPS_RADIUS)
            satellite = geodetic_to_ecef(satellite_lon, 0.0, GEOOPS_HEIGHT, earth=earth)
            expected = _geoops(lon + satellite_lon, lat, height, satellite_lon)
            got = apparent_position(
                lon + satellite_lon, lat, height, *satellite, earth=earth
            )
            assert got[2].all()
            for actual, wanted in zip(got[:2], expected, strict=True):
                assert_matches_oracle(actual, wanted, rtol=0.0, atol=1e-9)

    def test_height_zero_is_the_pixel(self) -> None:
        lon, lat = _disk(300)
        seen_lon, seen_lat, visible = apparent_position(lon, lat, 0.0, *GEOSTATIONARY)
        assert visible.all()
        assert_matches_oracle(seen_lon, lon, rtol=0.0, atol=1e-9)
        assert_matches_oracle(seen_lat, lat, rtol=0.0, atol=1e-9)

    def test_the_target_the_appearance_and_the_satellite_are_collinear(self) -> None:
        lon, lat = _disk(100)
        target = np.stack(geodetic_to_ecef(lon, lat, 12e3))
        seen_lon, seen_lat, _ = apparent_position(lon, lat, 12e3, *GEOSTATIONARY)
        seen = np.stack(geodetic_to_ecef(seen_lon, seen_lat, 0.0))
        ray = target - np.asarray(GEOSTATIONARY)[:, None]
        ray /= np.linalg.norm(ray, axis=0)
        across = np.cross(seen - target, ray, axis=0)
        assert np.linalg.norm(across, axis=0).max() < 1e-6

    def test_the_shift_is_h_tan_zenith_away_from_the_satellite(self) -> None:
        # to first order in H; the next term is of order H / R
        lon, lat = _disk(100, reach=60.0)
        height = 10.0
        zenith, azimuth = view_angles(lon, lat, *GEOSTATIONARY)
        seen_lon, seen_lat, _ = apparent_position(lon, lat, height, *GEOSTATIONARY)
        pixel = np.stack(geodetic_to_ecef(lon, lat, 0.0))
        seen = np.stack(geodetic_to_ecef(seen_lon, seen_lat, 0.0))
        shift = np.linalg.norm(seen - pixel, axis=0)
        expected = height * np.tan(np.deg2rad(zenith))
        assert_matches_oracle(shift, expected, rtol=1e-4, atol=1e-5)
        # the azimuth of the shift is opposite to the satellite's
        back = np.deg2rad(azimuth) + np.pi
        north = np.deg2rad(seen_lat - lat) * GEOOPS_RADIUS
        east = np.deg2rad(seen_lon - lon) * GEOOPS_RADIUS * np.cos(np.deg2rad(lat))
        far = shift > 1.0
        assert_matches_oracle(
            north[far], (shift * np.cos(back))[far], rtol=0.0, atol=0.15
        )
        assert_matches_oracle(
            east[far], (shift * np.sin(back))[far], rtol=0.0, atol=0.15
        )

    def test_a_low_orbit(self) -> None:
        # 45° off nadir a target of 1 km is displaced by about 1 km over its zenith
        satellite = geodetic_to_ecef(12.0, 44.0, 7e5)
        zenith, _ = view_angles(12.0, 40.0, *satellite)
        _, lat, _ = apparent_position(12.0, 40.0, 1e3, *satellite)
        shift = np.deg2rad(40.0 - lat) * 6.36e6
        assert_matches_oracle(shift, 1e3 * np.tan(np.deg2rad(zenith)), rtol=1e-2)

    def test_below_the_horizon_and_above_the_limb(self) -> None:
        lon = np.asarray([0.0, 80.0, 82.0, 100.0, 180.0])
        got = apparent_position(lon, 0.0, 0.0, *GEOSTATIONARY)
        # the limb of a geostationary disk is 81.3° away
        assert got[2].tolist() == [True, True, False, False, False]
        assert np.isnan(got[0]).tolist() == [False, False, True, True, True]
        # at 80.9° a target at 20 km stands out against space
        above = apparent_position(80.9, 0.0, 20e3, *GEOSTATIONARY)
        assert not above[2]
        assert np.isnan(above[0])

    def test_a_negative_height(self) -> None:
        lon, lat = _disk(50)
        down = apparent_position(lon, lat, -5.0, *GEOSTATIONARY)
        up = apparent_position(lon, lat, 5.0, *GEOSTATIONARY)
        assert down[2].all()
        # to first order the two shifts are opposite
        middle = 0.5 * (down[1] + up[1])
        assert_matches_oracle(middle, lat, rtol=0.0, atol=1e-7)

    def test_the_result_does_not_depend_on_the_unit(self) -> None:
        lon, lat = _disk(20)
        degrees = apparent_position(lon, lat, 1e4, *GEOSTATIONARY)
        radians = apparent_position(
            np.deg2rad(lon), np.deg2rad(lat), 1e4, *GEOSTATIONARY, degrees=False
        )
        for got, want in zip(radians[:2], degrees[:2], strict=True):
            assert_matches_oracle(np.rad2deg(got), want, rtol=0.0, atol=1e-11)

    def test_float32(self) -> None:
        # Earth-centred coordinates resolve 0.5 m, and 1/cos θᵥ stretches it
        lon, lat = _disk(300, reach=60.0)
        expected = apparent_position(lon, lat, 1e4, *GEOSTATIONARY)
        single = [np.asarray(v, dtype=np.float32) for v in (lon, lat, 1e4)]
        got = apparent_position(*single, *(np.float32(v) for v in GEOSTATIONARY))
        assert got[0].dtype == got[1].dtype == np.float32
        for actual, wanted in zip(got[:2], expected[:2], strict=True):
            assert_matches_oracle(actual, wanted, rtol=0.0, atol=1e-4)

    @given(st.floats(-60.0, 60.0), st.floats(-50.0, 50.0), st.floats(0.0, 2e4))
    def test_a_higher_target_appears_further(self, lon, lat, height) -> None:
        low = apparent_position(lon, lat, height, *GEOSTATIONARY)
        high = apparent_position(lon, lat, height + 100.0, *GEOSTATIONARY)
        if low[2] and high[2]:
            points = [np.stack(geodetic_to_ecef(*p[:2], 0.0)) for p in (low, high)]
            nadir = np.asarray([WGS84.semi_major, 0.0, 0.0])
            # further from the sub-satellite point, or nowhere at all
            near, far = (np.linalg.norm(p - nadir) for p in points)
            assert far >= near - 1e-6

    def test_gradient(self) -> None:
        satellite = np.asarray(GEOSTATIONARY)
        assert_grad_matches_fd(
            lambda x: sum(
                apparent_position(
                    x[0],
                    x[1],
                    1e4 * x[2],
                    *(satellite[i] + 1e5 * x[3] for i in range(3)),
                )[:2]
            ),
            np.asarray([20.0, 41.0, 1.0, 0.1]),
        )

    @needs_jax
    def test_the_gradient_beside_a_hidden_pixel_is_finite(self) -> None:
        import jax

        def total(lon):
            seen = apparent_position(lon, 0.0, 1e4, *GEOSTATIONARY)
            return jax.numpy.nansum(seen[0] + seen[1])

        gradient = jax.grad(total)(np.asarray([0.0, 30.0, 80.9, 100.0, 180.0]))
        assert np.isfinite(gradient).all()
        assert gradient[1] > 0.0

    def test_backends(self) -> None:
        lon, lat = (v.reshape(2, 3) for v in _disk(6, reach=50.0))
        assert_backends_agree(
            lambda lon, lat: apparent_position(lon, lat, 1e4, *GEOSTATIONARY)[:2],
            lon,
            lat,
            rtol=1e-11,
            rtol32=1e-4,
        )
        assert_traces(
            lambda lon, lat: apparent_position(lon, lat, 1e4, *GEOSTATIONARY),
            lon.ravel(),
            lat.ravel(),
            rtol=1e-11,
        )

    def test_dataarrays(self) -> None:
        xr = pytest.importorskip("xarray")
        lon = xr.DataArray(np.linspace(0.0, 5.0, 6).reshape(2, 3), dims=("y", "x"))
        height = xr.DataArray([0.0, 1e4], dims="level")
        lon_seen, _, visible = apparent_position(lon, 40.0, height, *GEOSTATIONARY)
        assert lon_seen.dims == visible.dims == ("y", "x", "level")
        assert visible.dtype == bool


class TestEarthIntersection:
    def test_straight_down_returns_the_point_below(self) -> None:
        lon, lat = _disk(50)
        above = np.stack(geodetic_to_ecef(lon, lat, 8e5))
        ground = np.stack(geodetic_to_ecef(lon, lat, 0.0))
        x, y, z, visible = earth_intersection(*above, *(ground - above))
        assert visible.all()
        assert_matches_oracle(np.stack([x, y, z]), ground, rtol=0.0, atol=1e-6)

    def test_a_point_on_the_near_side_is_met_first(self) -> None:
        # toward the antipode, and the near side comes first
        x, y, z, visible = earth_intersection(*GEOSTATIONARY, -1.0, 0.0, 0.0)
        assert abs(float(x) - WGS84.semi_major) < 1e-6
        assert (float(y), float(z)) == (0.0, 0.0)
        assert visible

    @given(st.floats(-80.0, 80.0), st.floats(-89.0, 89.0), st.floats(0.01, 100.0))
    def test_the_point_is_on_the_ellipsoid_and_the_ray(self, lon, lat, length) -> None:
        origin = np.asarray(GEOSTATIONARY)
        direction = length * (np.stack(geodetic_to_ecef(lon, lat, 0.0)) - origin)
        *point, visible = earth_intersection(*origin, *direction)
        if visible:
            _, _, height = ecef_to_geodetic(*point)
            assert abs(height) < 1e-5
            across = np.cross(np.stack(point) - origin, direction)
            assert np.linalg.norm(across) / np.linalg.norm(direction) < 1e-6

    def test_away_past_and_inside(self) -> None:
        # away from the Earth, past it, and from inside it
        origin = np.asarray([7e6, 7e6, 1e6])
        direction = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        x, _, _, visible = earth_intersection(origin, 0.0, 0.0, *direction.T, 0.0)
        assert visible.tolist() == [False, False, False]
        assert np.isnan(x).all()

    def test_gradient_and_backends(self) -> None:
        origin = 1e7 * np.asarray([[1.0, 0.3, 0.2], [0.9, -0.2, 0.4]])
        direction = -origin + 1e6 * np.asarray([[0.3, 0.2, -1.0], [1.0, 0.5, 0.2]])
        assert_backends_agree(
            lambda *v: earth_intersection(*v)[:3],
            *origin.T,
            *direction.T,
            rtol32=1e-4,
        )
        assert_traces(earth_intersection, *origin.T, *direction.T)
        assert_grad_matches_fd(
            lambda v: sum(earth_intersection(*(1e7 * v[:3]), *v[3:])[:3]) / 1e6,
            np.concatenate([origin[0] / 1e7, direction[0] / 1e7]),
        )
