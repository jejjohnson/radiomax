"""The checks on a toy function, and the references they use.

That a check fails on a wrong function is tested in pipekit-array, which has it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _backends import assert_backends_agree
from _oracles import (
    RADTRAX_COMMIT,
    functax_enu,
    functax_trapezoid_weights,
    georeader_band_irradiance,
    georeader_reflectance,
    integrix_band_mean,
    proj_geostationary,
    proj_view_angles,
    radtrax_values,
    spa_distance,
    spa_sun,
)
from geophysix.constants import STANDARD
from geophysix.earth import geodetic_to_ecef
from hypothesis import given, strategies as st
from pipekit_array.arrays import floating_inputs, like
from pipekit_array.kernels import elementwise
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)


SLOPE = 0.3


@elementwise(n_outputs=2)
def toy(values, offset=0.0) -> Any:
    """sin(0.3 x) + o, and x² — what a radiomax function looks like."""
    xp, (values, offset) = floating_inputs(values, offset)
    slope = like(SLOPE, values, xp)
    return xp.sin(slope * values) + offset, values**2


X = np.linspace(0.1, 2.0, 6).reshape(2, 3)


class TestTheFourChecksOnAToyFunction:
    def test_oracle(self) -> None:
        assert_matches_oracle(toy(X)[0], np.sin(SLOPE * X), rtol=1e-15)

    @given(st.floats(-5.0, 5.0))
    def test_property(self, value: float) -> None:
        assert toy(value)[1] == toy(-value)[1]

    def test_gradient(self) -> None:
        assert_grad_matches_fd(lambda x: toy(x)[0], X)

    def test_backends(self) -> None:
        assert_backends_agree(toy, X, np.asarray([1.0, 2.0, 3.0]))
        assert_traces(toy, X, np.asarray([1.0, 2.0]))


class TestOracles:
    """Each reference against a closed form, so that a wrong call shows."""

    def test_spa_puts_the_sun_overhead_at_an_equinox_noon(self) -> None:
        # 2024-03-20, equation of time −7 min 25 s: overhead at 1.9° E at noon UTC
        zenith, _ = spa_sun(np.datetime64("2024-03-20T12:00:00"), 1.9, 0.1)
        assert zenith[0] < 0.3

    def test_spa_azimuth_is_clockwise_from_north(self) -> None:
        # morning at mid-northern latitude: the Sun is in the east
        _, azimuth = spa_sun(np.datetime64("2024-06-21T08:00:00"), 0.0, 45.0)
        assert 60.0 < azimuth[0] < 120.0

    def test_spa_distance_at_perihelion_and_aphelion(self) -> None:
        near, far = spa_distance(["2024-01-03T01:00", "2024-07-05T05:00"])
        assert_matches_oracle([near, far], [0.98331, 1.01673], rtol=1e-5)

    def test_georeader_reflectance_is_the_formula(self) -> None:
        radiance = np.full((2, 1, 1), 0.05)
        irradiance = np.asarray([0.9, 0.25])
        out = georeader_reflectance(radiance, irradiance, 3.5)
        # georeader keeps float32 or float64 of its input
        expected = radiance * 3.5 / irradiance[:, None, None]
        assert_matches_oracle(out, expected, rtol=1e-6)

    def test_georeader_band_irradiance_of_a_flat_spectrum(self) -> None:
        wavelength = np.arange(1500.0, 1701.0)
        response = np.exp(-(((wavelength - 1600.0) / 20.0) ** 2))[None]
        flat = np.full(wavelength.size, 2.0)
        out = georeader_band_irradiance(flat, response, wavelength)
        assert_matches_oracle(out, [2.0], rtol=1e-6)

    def test_proj_sees_a_satellite_overhead_and_one_to_the_east(self) -> None:
        overhead = tuple(float(v) for v in geodetic_to_ecef(12.0, 34.0, 7e5))
        assert proj_view_angles(12.0, 34.0, 0.0, overhead)[0] < 1e-9
        east = tuple(float(v) for v in geodetic_to_ecef(13.0, 34.0, 7e5))
        zenith, azimuth = proj_view_angles(12.0, 34.0, 0.0, east)
        assert 0.0 < zenith < 15.0
        assert 85.0 < azimuth < 95.0

    @pytest.mark.parametrize("sweep", ["x", "y"])
    def test_proj_geostationary_at_the_sub_satellite_point(self, sweep: str) -> None:
        options = {"lon_0": -75.0, "height": 35_786_023.0, "sweep": sweep}
        lon, lat = proj_geostationary([0.0, 0.16], [0.0, 0.0], **options)
        assert_matches_oracle([lon[0], lat[0]], [-75.0, 0.0], rtol=1e-12, atol=1e-9)
        assert not np.isfinite(lon[1])

    @pytest.mark.slow
    def test_integrix_mean_is_the_weighted_sum(self) -> None:
        rng = np.random.default_rng(0)
        spectrum, responses = rng.random((4, 9)), rng.random((2, 9))
        weights = rng.random(9)
        combined = responses * weights
        expected = (spectrum @ combined.T) / combined.sum(axis=1)
        out = integrix_band_mean(spectrum, responses, weights)
        assert_matches_oracle(out, expected, rtol=1e-14)

    def test_functax_weights_are_the_trapezoid(self) -> None:
        nodes = np.asarray([0.0, 1.0, 3.0])
        assert functax_trapezoid_weights(nodes).tolist() == [0.5, 1.5, 1.0]

    def test_functax_frame_is_east_north_up(self) -> None:
        frame = functax_enu(np.asarray([0.0]), np.asarray([0.0]))[0]
        expected = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        assert_matches_oracle(frame, expected, rtol=0.0, atol=1e-15)


class TestStoredRadtraxValues:
    """The file holds what its names say, by closed forms written here."""

    def test_the_file_is_small_and_finite(self) -> None:
        path = Path(__file__).parent / "data" / f"radtrax_{RADTRAX_COMMIT}.npz"
        assert path.stat().st_size < 200_000
        stored = radtrax_values()
        assert all(v.dtype == np.float64 for v in stored.values())
        assert all(np.isfinite(v).all() for v in stored.values())

    def test_air_masses(self) -> None:
        stored = radtrax_values()
        solar = np.cos(np.deg2rad(stored["air_mass_solar_zenith"]))
        view = np.cos(np.deg2rad(stored["air_mass_view_zenith"]))
        assert_matches_oracle(
            stored["air_mass_factor"], 1.0 / solar + 1.0 / view, rtol=1e-14
        )
        # the shell of the lowest 10 m at the zenith is plane-parallel
        assert_matches_oracle(stored["shell_air_mass"][0], np.ones(6), rtol=1e-12)
        assert (*stored["shell_cosine"].shape, 6) == stored["shell_air_mass"].shape

    def test_planck_against_the_constants(self) -> None:
        stored = radtrax_values()
        wavenumber = 100.0 * stored["planck_wavenumber_cm"][:, None]  # m⁻¹
        h, c, k = STANDARD.planck, STANDARD.speed_of_light, STANDARD.boltzmann
        x = h * c * wavenumber / (k * stored["planck_temperature"][None, :])
        # B = 2hc²ν³ / (eˣ − 1) per m⁻¹, times 100 per cm⁻¹
        planck = 100.0 * 2.0 * h * c**2 * wavenumber**3 / np.expm1(x)
        assert_matches_oracle(stored["planck_radiance"], planck, rtol=1e-12)
        assert_matches_oracle(stored["log_planck_radiance"], np.log(planck), rtol=1e-12)
        expected = np.broadcast_to(stored["planck_temperature"], planck.shape)
        assert_matches_oracle(stored["brightness_temperature"], expected, rtol=1e-12)

    def test_kernels_at_nadir_and_responses(self) -> None:
        stored = radtrax_values()
        for name in ("ross", "li"):
            nadir = stored[f"kernel_{name}"][10:14]
            assert_matches_oracle(nadir, np.zeros(4), rtol=0.0, atol=1e-15)
        assert_matches_oracle(
            stored["response_matrix"].sum(axis=1), np.ones(3), rtol=1e-14
        )
