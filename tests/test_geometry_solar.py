"""The Sun's position, the Earth–Sun distance and the two time arrays."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import radtrax_values, spa_distance, spa_sun
from hypothesis import given, strategies as st

from radiomax import ValidationError
from radiomax.geometry import (
    earth_sun_distance,
    solar_angles,
    solar_declination,
    solar_zenith,
    utc_day_seconds,
)


def _random_times(n: int) -> np.ndarray:
    """UTC times over 2000–2050."""
    microseconds = np.random.default_rng(0).uniform(0.0, 51 * 365.25 * 86_400e6, n)
    return np.datetime64("2000-01-01", "us") + microseconds.astype("timedelta64[us]")


@pytest.fixture
def pixels() -> tuple[np.ndarray, ...]:
    """Times, longitudes, latitudes, days and seconds."""
    rng = np.random.default_rng(1)
    time = _random_times(5000)
    lon, lat = rng.uniform(-180.0, 180.0, 5000), rng.uniform(-90.0, 90.0, 5000)
    return time, lon, lat, *utc_day_seconds(time)


def _separation(azimuth: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs((azimuth - reference + 180.0) % 360.0 - 180.0)


class TestTime:
    def test_the_epoch_and_its_neighbours(self) -> None:
        assert utc_day_seconds("2000-01-01") == (0.0, 0.0)
        assert utc_day_seconds("2000-01-01T12:00:00.000001") == (0.0, 43200.000001)
        assert utc_day_seconds("1999-12-31T23:59:59") == (-1.0, 86399.0)

    def test_one_time_gives_python_floats(self) -> None:
        day, seconds = utc_day_seconds(np.datetime64("2026-11-05T10:30"))
        assert type(day) is float
        assert type(seconds) is float

    @given(st.datetimes(datetime.datetime(1960, 1, 1), datetime.datetime(2100, 1, 1)))
    def test_round_trip_to_the_microsecond(self, time: datetime.datetime) -> None:
        day, seconds = utc_day_seconds(time)
        assert 0.0 <= seconds < 86_400.0
        back = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=day)
        assert back + datetime.timedelta(microseconds=round(seconds * 1e6)) == time

    def test_a_time_zone_is_converted_to_utc(self) -> None:
        zone = datetime.timezone(datetime.timedelta(hours=2))
        time = datetime.datetime(2000, 1, 2, 1, 30, tzinfo=zone)
        assert utc_day_seconds(time) == (0.0, 84600.0)

    def test_not_a_time_is_nan(self) -> None:
        day, seconds = utc_day_seconds(np.asarray(["NaT", "2000-01-03"], "M8[s]"))
        assert np.isnan(day[0]) and np.isnan(seconds[0])
        assert (day[1], seconds[1]) == (2.0, 0.0)

    @pytest.mark.parametrize("time", [5, 2451545.0, np.zeros(2), "noon", [1, 2]])
    def test_what_is_not_a_date_raises(self, time: object) -> None:
        with pytest.raises(ValidationError, match="must hold dates"):
            utc_day_seconds(time)

    def test_a_dataarray_keeps_its_coordinates(self) -> None:
        xr = pytest.importorskip("xarray")
        stamps = np.asarray(["2000-01-02", "2000-01-03"], "M8[ns]")
        time = xr.DataArray(stamps, {"scan": [7, 8]}, ("scan",), "time", {"a": 1})
        day, seconds = utc_day_seconds(time)
        assert day.scan.values.tolist() == [7, 8]
        assert day.values.tolist() == [1.0, 2.0]
        assert seconds.values.tolist() == [0.0, 0.0]
        assert (day.name, day.attrs) == (None, {})


class TestSolarAngles:
    def test_zenith_and_azimuth_against_spa(self, pixels) -> None:
        # gate G1-spa: the formulas' own error, at most 0.0116° over 20,000 times
        time, lon, lat, day, seconds = pixels
        zenith, azimuth = solar_angles(lon, lat, day, seconds)
        expected_zenith, expected_azimuth = spa_sun(time, lon, lat)
        assert_matches_oracle(zenith, expected_zenith, rtol=0.0, atol=0.015)
        # on the sky, an azimuth error shrinks with sin θₛ
        arc = _separation(azimuth, expected_azimuth)
        arc = arc * np.sin(np.deg2rad(expected_zenith))
        assert arc.max() < 0.015

    def test_float32_is_as_good(self, pixels) -> None:
        time, *inputs = pixels
        lon, lat = inputs[:2]
        single = [np.asarray(v, dtype=np.float32) for v in inputs]
        zenith, _ = solar_angles(*single)
        assert zenith.dtype == np.float32
        expected, _ = spa_sun(time, lon, lat)
        assert_matches_oracle(zenith, expected, rtol=0.0, atol=0.015)

    @needs_jax
    def test_jax_float32_is_as_good(self, pixels) -> None:
        import jax.numpy as jnp

        time, *inputs = pixels
        single = [jnp.asarray(v, dtype=jnp.float32) for v in inputs]
        zenith = solar_zenith(*single)
        assert zenith.dtype == jnp.float32
        expected, _ = spa_sun(time, *inputs[:2])
        assert_matches_oracle(zenith, expected, rtol=0.0, atol=0.015)

    def test_float32_pixels_stay_float32_with_one_time(self) -> None:
        lat = np.linspace(-80.0, 80.0, 5, dtype=np.float32)
        day, seconds = utc_day_seconds("2024-06-21T10:30")
        for angle in solar_angles(np.float32(3.0), lat, day, seconds):
            assert angle.dtype == np.float32

    def test_overhead_at_the_subsolar_point(self) -> None:
        # 2024-03-20 12:00 UTC: equation of time −7 min 25 s, so 1.85° E
        day, seconds = utc_day_seconds("2024-03-20T12:00")
        declination = solar_declination(day, seconds)
        zenith = solar_zenith(np.linspace(1.0, 3.0, 2001), declination, day, seconds)
        assert zenith.min() < 0.02

    def test_the_zenith_alone_is_the_first_angle(self, pixels) -> None:
        _, *inputs = pixels
        assert_matches_oracle(solar_zenith(*inputs), solar_angles(*inputs)[0], rtol=0.0)

    def test_radians_in_and_out(self, pixels) -> None:
        _, lon, lat, day, seconds = pixels
        radians = solar_angles(
            np.deg2rad(lon), np.deg2rad(lat), day, seconds, degrees=False
        )
        degrees = solar_angles(lon, lat, day, seconds)
        assert_matches_oracle(np.rad2deg(radians[0]), degrees[0], rtol=1e-10)
        assert _separation(np.rad2deg(radians[1]), degrees[1]).max() < 1e-7

    @given(
        st.floats(-180.0, 180.0),
        st.floats(-90.0, 90.0),
        st.floats(0.0, 18_000.0),
        st.floats(0.0, 86_400.0),
    )
    def test_ranges_and_a_turn_of_longitude(self, lon, lat, day, seconds) -> None:
        zenith, azimuth = solar_angles(lon, lat, np.floor(day), seconds)
        assert 0.0 <= zenith <= 180.0
        assert 0.0 <= azimuth < 360.0
        turned, _ = solar_angles(lon + 360.0, lat, np.floor(day), seconds)
        assert abs(turned - zenith) < 1e-6

    def test_at_the_pole_the_zenith_is_the_declination(self) -> None:
        day, seconds = utc_day_seconds("2024-06-20T20:51")
        declination = solar_declination(day, seconds)
        assert_matches_oracle(
            solar_zenith(0.0, 90.0, day, seconds), 90.0 - declination, rtol=1e-12
        )

    def test_gradients(self) -> None:
        day, seconds = utc_day_seconds("2024-06-21T10:30")
        point = np.asarray([3.0, 45.0, seconds])

        def angles(x):
            zenith, azimuth = solar_angles(x[0], x[1], day, x[2])
            return zenith**2 + azimuth

        assert_grad_matches_fd(angles, point, eps=1e-4)

    @needs_jax
    def test_gradients_are_finite_with_the_sun_overhead(self) -> None:
        import jax
        import jax.numpy as jnp

        from radiomax._src.geometry import solar

        def zenith(east, north):
            return solar._zenith(east, north, jnp.asarray(1.0), jnp)

        assert [float(g) for g in jax.grad(zenith, (0, 1))(0.0, 0.0)] == [0.0, 0.0]
        day, seconds = utc_day_seconds("2024-03-20T12:00")
        declination = float(solar_declination(day, seconds))
        gradient = jax.grad(solar_zenith, (0, 1, 3))(1.85, declination, day, seconds)
        assert all(np.isfinite(float(g)) for g in gradient)

    def test_backends(self, pixels) -> None:
        inputs = [v[:20] for v in pixels[1:]]
        # the azimuth near north wraps, and float32 resolves 3e-5° of it
        assert_backends_agree(solar_zenith, *inputs, rtol=1e-12, rtol32=1e-3)
        assert_backends_agree(solar_angles, *inputs, rtol=1e-11, rtol32=1e-2)
        assert_traces(solar_angles, *inputs, rtol=1e-11)

    def test_a_dataarray_scene_with_labelled_times(self) -> None:
        xr = pytest.importorskip("xarray")
        stamps = np.asarray(["2024-06-21T10:30", "2024-06-21T10:31"], "M8[ns]")
        scan = xr.DataArray(stamps, {"scan": [0, 1]}, ("scan",))
        lon = xr.DataArray(np.linspace(0.0, 1.0, 6).reshape(2, 3), dims=("scan", "x"))
        zenith, azimuth = solar_angles(lon, lon + 45.0, *utc_day_seconds(scan))
        assert zenith.dims == azimuth.dims == ("scan", "x")


class TestDeclination:
    def test_the_solstices_and_the_equinox(self) -> None:
        times = np.asarray(
            ["2024-06-20T20:51", "2024-12-21T09:20", "2024-03-20T03:06"], "M8[s]"
        )
        declination = solar_declination(*utc_day_seconds(times))
        assert_matches_oracle(declination, [23.44, -23.44, 0.0], rtol=0.0, atol=0.015)

    @given(st.floats(0.0, 18_000.0))
    def test_bounded_by_the_obliquity(self, day: float) -> None:
        assert abs(float(solar_declination(day))) <= 23.45
        radians = solar_declination(day, degrees=False)
        assert abs(np.rad2deg(radians) - solar_declination(day)) < 1e-10

    def test_gradient_and_backends(self) -> None:
        days = np.asarray([8938.0, 9000.0, 9100.0])
        seconds = np.asarray([1e3, 4e4, 8e4])
        assert_grad_matches_fd(lambda s: solar_declination(days, s), seconds, eps=10.0)
        assert_backends_agree(solar_declination, days, seconds, rtol32=1e-4)
        assert_traces(solar_declination, days, seconds)


class TestDistance:
    def test_against_spa(self) -> None:
        # gate G1-spa: the formula's own error, 8.8e-5 AU over 20,000 times
        time = _random_times(5000)
        distance = earth_sun_distance(*utc_day_seconds(time))
        assert_matches_oracle(distance, spa_distance(time), rtol=0.0, atol=1e-4)

    def test_the_difference_from_georeaders_formula(self) -> None:
        # the bound the docstring states, over one year of radtrax's values
        stored = radtrax_values()
        # 2024-01-01, day 1 of the year, is 8766 days after 2000-01-01
        day = 8766.0 + stored["distance_day_of_year"] - 1.0
        difference = np.abs(earth_sun_distance(day) - stored["distance"])
        assert 1e-4 < difference.max() < 8.0e-4

    @given(st.floats(0.0, 18_000.0))
    def test_between_perihelion_and_aphelion(self, day: float) -> None:
        assert 0.983 < float(earth_sun_distance(day)) < 1.017

    def test_gradient_and_backends(self) -> None:
        days = np.asarray([8938.0, 9000.0, 9100.0])
        seconds = np.asarray([1e3, 4e4, 8e4])
        assert_grad_matches_fd(
            lambda s: earth_sun_distance(days, s), seconds, eps=100.0, rtol=1e-5
        )
        assert_backends_agree(earth_sun_distance, days, seconds)
        assert_traces(earth_sun_distance, days, seconds)
