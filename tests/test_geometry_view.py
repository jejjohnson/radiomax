"""Viewing angles from a position, and the angles between Sun and sensor."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import functax_enu, proj_view_angles
from geophysix.earth import SPHERE, geodetic_to_ecef
from hypothesis import given, strategies as st

from radiomax.geometry import (
    phase_angle,
    relative_azimuth,
    scattering_angle,
    view_angles,
)


ZENITH = st.floats(0.0, 89.0)
AZIMUTH = st.floats(-360.0, 360.0)


def _scene(n: int, altitude: float) -> tuple[np.ndarray, ...]:
    """Pixels, their heights, and a satellite near each one."""
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(-180.0, 180.0, n), rng.uniform(-89.0, 89.0, n)
    height = rng.uniform(-100.0, 8000.0, n)
    beside = np.clip(lat + rng.uniform(-8.0, 8.0, n), -90.0, 90.0)
    satellite = geodetic_to_ecef(lon + rng.uniform(-8.0, 8.0, n), beside, altitude)
    return lon, lat, height, *satellite


class TestViewAngles:
    @pytest.mark.parametrize("altitude", [4e5, 7e5, 3.58e7])
    def test_against_proj(self, altitude: float) -> None:
        # gate G1-view
        lon, lat, height, *satellite = _scene(60, altitude)
        zenith, azimuth = view_angles(lon, lat, *satellite, height=height)
        rows = zip(lon, lat, height, *satellite, strict=True)
        expected = np.asarray(
            [
                proj_view_angles(float(o), float(a), float(h), tuple(map(float, s)))
                for o, a, h, *s in rows
            ]
        )
        assert_matches_oracle(zenith, expected[:, 0], rtol=0.0, atol=1e-9)
        turned = np.abs((azimuth - expected[:, 1] + 180.0) % 360.0 - 180.0)
        # on the sky, an azimuth error shrinks with sin θᵥ
        assert (turned * np.sin(np.deg2rad(zenith))).max() < 1e-9

    def test_float32_is_good_to_a_few_arcseconds(self) -> None:
        # Earth-centred coordinates resolve 0.5 m: measured 1.1e-4° at 400 km
        lon, lat, _, *satellite = _scene(200, 4e5)
        single = [np.asarray(v, dtype=np.float32) for v in (lon, lat, *satellite)]
        zenith, azimuth = view_angles(*single)
        assert zenith.dtype == azimuth.dtype == np.float32
        expected, _ = view_angles(lon, lat, *satellite)
        assert_matches_oracle(zenith, expected, rtol=0.0, atol=3e-4)

    def test_the_frame_is_functaxs_local_frame(self) -> None:
        lon, lat = np.asarray([12.0, -100.0]), np.asarray([34.0, -70.0])
        frame = functax_enu(lon, lat)
        position = np.stack(geodetic_to_ecef(lon, lat, 0.0), axis=-1)
        # 700 km up and 500 km along each of east, north and up
        for axis, expected in ((0, 90.0), (1, 0.0)):
            satellite = position + 7e5 * frame[:, 2] + 5e5 * frame[:, axis]
            zenith, azimuth = view_angles(lon, lat, *satellite.T)
            tilt = np.rad2deg(np.arctan2(5.0, 7.0))
            assert_matches_oracle(zenith, np.full(2, tilt), rtol=1e-9)
            assert np.abs((azimuth - expected + 180.0) % 360.0 - 180.0).max() < 1e-7

    def test_overhead_and_below_the_horizon(self) -> None:
        zenith, azimuth = view_angles(12.0, 34.0, *geodetic_to_ecef(12.0, 34.0, 7e5))
        assert zenith < 1e-7
        assert 0.0 <= azimuth < 360.0
        below, _ = view_angles(12.0, 34.0, *geodetic_to_ecef(-168.0, -34.0, 7e5))
        assert 90.0 < below <= 180.0

    def test_the_geodetic_normal_is_used(self) -> None:
        # on the sphere the same longitude and latitude look elsewhere, by 0.19°
        satellite = geodetic_to_ecef(0.0, 45.0, 7e5)
        radial = np.stack(satellite) * 1.5
        zenith, _ = view_angles(0.0, 45.0, *radial)
        assert 0.1 < zenith < 0.2
        on_sphere = np.stack(geodetic_to_ecef(0.0, 45.0, 7e5, earth=SPHERE)) * 1.5
        assert view_angles(0.0, 45.0, *on_sphere, earth=SPHERE)[0] < 1e-6

    def test_radians_in_and_out(self) -> None:
        lon, lat, height, *satellite = _scene(20, 7e5)
        degrees = view_angles(lon, lat, *satellite, height=height)
        radians = view_angles(
            np.deg2rad(lon),
            np.deg2rad(lat),
            *satellite,
            height=height,
            degrees=False,
        )
        for got, want in zip(radians, degrees, strict=True):
            assert_matches_oracle(np.rad2deg(got), want, rtol=1e-9)

    @given(st.floats(-180.0, 180.0), st.floats(-90.0, 90.0), st.floats(0.0, 5e3))
    def test_a_satellite_straight_up_is_at_zenith_zero(self, lon, lat, height) -> None:
        satellite = geodetic_to_ecef(lon, lat, 7e5)
        zenith, azimuth = view_angles(lon, lat, *satellite, height=height)
        assert zenith < 1e-6
        assert 0.0 <= azimuth < 360.0

    def test_gradient(self) -> None:
        satellite = np.stack(geodetic_to_ecef(14.0, 36.0, 7e5))

        def angles(x):
            zenith, azimuth = view_angles(
                x[0], x[1], *(satellite[i] + 1e5 * x[2 + i] for i in range(3))
            )
            return zenith + 0.1 * azimuth

        assert_grad_matches_fd(angles, np.asarray([12.0, 34.0, 0.0, 0.0, 0.0]))

    @needs_jax
    def test_the_gradient_is_finite_straight_up(self) -> None:
        import jax

        # at the equator and the prime meridian every component is exact
        gradient = jax.grad(
            lambda lon, lat, x, y: sum(view_angles(lon, lat, x, y, 0.0)), (0, 1, 2, 3)
        )(0.0, 0.0, 7.1e6, 0.0)
        assert all(np.isfinite(float(g)) for g in gradient)

    def test_backends(self) -> None:
        lon, lat, _, *satellite = _scene(12, 7e5)
        assert_backends_agree(
            lambda *v: view_angles(*v)[0], lon, lat, *satellite, rtol32=1e-2
        )
        assert_traces(view_angles, lon, lat, *satellite, rtol=1e-11)

    def test_a_dataarray_scene_with_a_scalar_position(self) -> None:
        xr = pytest.importorskip("xarray")
        lon = xr.DataArray(np.linspace(0.0, 1.0, 6).reshape(2, 3), dims=("y", "x"))
        satellite = [float(v) for v in geodetic_to_ecef(0.5, 45.0, 7e5)]
        zenith, azimuth = view_angles(lon, lon + 45.0, *satellite, height=lon)
        assert zenith.dims == azimuth.dims == ("y", "x")


class TestRelativeAzimuth:
    def test_the_formula_of_radtraxs_readers(self) -> None:
        rng = np.random.default_rng(0)
        # azimuths in [−180°, 180°] and in [0°, 360°)
        view, solar = rng.uniform(-180.0, 180.0, 2000), rng.uniform(0.0, 360.0, 2000)
        expected = np.rad2deg(np.arccos(np.cos(np.deg2rad(view - solar))))
        inside = (expected > 0.1) & (expected < 179.9)
        assert_matches_oracle(
            relative_azimuth(view, solar), expected, rtol=0.0, atol=1e-10, where=inside
        )

    @given(AZIMUTH, AZIMUTH)
    def test_symmetric_and_periodic(self, view: float, solar: float) -> None:
        angle = float(relative_azimuth(view, solar))
        assert 0.0 <= angle <= 180.0
        assert abs(angle - float(relative_azimuth(solar, view))) < 1e-9
        assert abs(angle - float(relative_azimuth(view + 360.0, solar))) < 1e-9
        radians = relative_azimuth(np.deg2rad(view), np.deg2rad(solar), degrees=False)
        assert abs(np.rad2deg(radians) - angle) < 1e-9

    @needs_jax
    @pytest.mark.parametrize("view", [0.0, 180.0, -180.0, 360.0, 30.0])
    def test_the_gradient_is_one_in_size_everywhere(self, view: float) -> None:
        import jax

        gradient = jax.grad(relative_azimuth, (0, 1))(view, 0.0)
        assert [abs(float(g)) for g in gradient] == [1.0, 1.0]

    def test_gradient_and_backends(self) -> None:
        view = np.asarray([10.0, 200.0, -170.0])
        solar = np.asarray([350.0, 30.0, 45.0])
        assert_grad_matches_fd(lambda v: relative_azimuth(v, solar), view)
        assert_backends_agree(relative_azimuth, view, solar)
        assert_traces(relative_azimuth, view, solar)


class TestPhaseAndScattering:
    @given(ZENITH, ZENITH, st.floats(0.0, 180.0))
    def test_the_cosine_formula(self, solar, view, azimuth) -> None:
        s, v, a = np.deg2rad([solar, view, azimuth])
        cosine = np.cos(s) * np.cos(v) + np.sin(s) * np.sin(v) * np.cos(a)
        phase = phase_angle(solar, view, azimuth)
        assert_matches_oracle(np.cos(np.deg2rad(phase)), cosine, rtol=0.0, atol=1e-14)

    @given(ZENITH, ZENITH, st.floats(0.0, 180.0))
    def test_they_add_to_180_and_are_reciprocal(self, solar, view, azimuth) -> None:
        phase = float(phase_angle(solar, view, azimuth))
        assert abs(phase + scattering_angle(solar, view, azimuth) - 180.0) < 1e-12
        assert abs(phase - phase_angle(view, solar, azimuth)) < 1e-6

    @pytest.mark.parametrize(
        ("solar", "view", "azimuth", "expected"),
        [
            (30.0, 30.0, 0.0, 0.0),
            (0.0, 0.0, 77.0, 0.0),
            (30.0, 20.0, 180.0, 50.0),
            (90.0, 90.0, 180.0, 180.0),
            (60.0, 0.0, 10.0, 60.0),
        ],
    )
    def test_closed_forms(self, solar, view, azimuth, expected) -> None:
        assert_matches_oracle(
            phase_angle(solar, view, azimuth), expected, rtol=1e-13, atol=1e-13
        )

    @needs_jax
    @pytest.mark.parametrize("point", [(30.0, 30.0, 0.0), (0.0, 0.0, 10.0)])
    def test_gradients_are_finite_at_the_hotspot_and_nadir(self, point) -> None:
        import jax

        for function in (phase_angle, scattering_angle):
            gradient = jax.grad(function, (0, 1, 2))(*point)
            assert all(np.isfinite(float(g)) for g in gradient)

    def test_gradients_and_backends(self) -> None:
        solar = np.asarray([30.0, 55.0, 10.0])
        view = np.asarray([5.0, 40.0, 60.0])
        azimuth = np.asarray([20.0, 100.0, 170.0])
        for function in (phase_angle, scattering_angle):
            assert_grad_matches_fd(
                lambda x, f=function: f(x[0], x[1], x[2]),
                np.stack([solar, view, azimuth]),
            )
            assert_backends_agree(function, solar, view, azimuth)
            assert_traces(function, solar, view, azimuth)
            radians = function(*np.deg2rad([solar, view, azimuth]), degrees=False)
            degrees = function(solar, view, azimuth)
            assert_matches_oracle(np.rad2deg(radians), degrees, rtol=1e-12)
