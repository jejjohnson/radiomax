"""Counts, radiance and reflectance."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _oracles import georeader_reflectance
from hypothesis import example, given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.radiometry import (
    counts_to_radiance,
    counts_to_reflectance,
    radiance_from_reflectance,
    radiance_to_counts,
    reflectance,
)


FINITE = st.floats(-1e3, 1e3)
POSITIVE = st.floats(1e-3, 1e3)
COUNTS = np.asarray([0.0, 100.0, 40_000.0])
GAIN = np.asarray([2e-3, 1e-2, 5e-4])


class TestLinearDecode:
    def test_the_line(self) -> None:
        assert counts_to_radiance(100.0, 0.02, -1.0) == 1.0
        assert counts_to_radiance(100.0, 0.02) == 2.0
        assert radiance_to_counts(1.0, 0.02, -1.0) == 100.0
        assert counts_to_reflectance(6000.0, 1e-4, -0.1) == pytest.approx(0.5)

    @given(st.integers(0, 65_535), POSITIVE, FINITE)
    def test_round_trip(self, counts: int, gain: float, offset: float) -> None:
        radiance = counts_to_radiance(counts, gain, offset)
        back = radiance_to_counts(radiance, gain, offset)
        # L − o cancels when the offset dwarfs the counts
        scale = abs(offset) / gain + counts + 1.0
        assert abs(back - counts) <= 1e-14 * scale

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int64])
    def test_integers_do_not_overflow(self, dtype: type) -> None:
        counts = np.asarray([0, np.iinfo(dtype).max], dtype=dtype)
        for function in (counts_to_radiance, counts_to_reflectance):
            out = function(counts, 3, 1)
            assert out.dtype == np.float64
            expected = function(counts.astype(np.float64), 3.0, 1.0)
            assert_matches_oracle(out, expected, rtol=0.0)

    def test_integers_with_a_float32_gain_stay_float32(self) -> None:
        out = counts_to_radiance(np.uint16([1, 2]), np.float32([0.5, 0.5]), 1.0)
        assert out.dtype == np.float32

    def test_a_gain_per_band_aligns_by_name(self) -> None:
        xr = pytest.importorskip("xarray")
        band = {"band": ["B6", "B7"]}
        cube = xr.DataArray(np.ones((2, 3, 4)), band, ("band", "y", "x"))
        gain = xr.DataArray([2.0, 3.0], band, ("band",))
        out = counts_to_radiance(cube, gain, gain)
        assert out.dims == ("band", "y", "x")
        assert out.sel(band="B7").values.tolist() == np.full((3, 4), 6.0).tolist()

    def test_gradients_and_backends(self) -> None:
        for function in (counts_to_radiance, radiance_to_counts, counts_to_reflectance):
            assert_grad_matches_fd(
                lambda g, f=function: f(COUNTS + 1.0, g, 0.5), GAIN, eps=1e-8
            )
            assert_backends_agree(function, COUNTS, GAIN, np.asarray([1.0, -2.0, 0.0]))
            assert_traces(function, COUNTS, GAIN, GAIN)


class TestReflectance:
    def test_georeader(self) -> None:
        # gate G1-georeader: the same correction factor π d² / cos θₛ, per nm
        rng = np.random.default_rng(0)
        radiance = rng.uniform(0.0, 0.1, (3, 4, 5))
        irradiance = np.asarray([1.9, 0.9, 0.25])
        zenith, distance = 38.0, 1.0143
        factor = np.pi * distance**2 / np.cos(np.deg2rad(zenith))
        expected = georeader_reflectance(radiance, irradiance, factor)
        out = reflectance(radiance, irradiance[:, None, None], zenith, distance)
        rtol = 1e-12 if expected.dtype == np.float64 else 1e-6
        assert_matches_oracle(out, expected, rtol=rtol)

    def test_a_lambertian_white_surface_under_the_sun_is_one(self) -> None:
        # L = E₀ cos θₛ / π
        for zenith in (0.0, 30.0, 80.0):
            radiance = 1.5 * np.cos(np.deg2rad(zenith)) / np.pi
            assert_matches_oracle(reflectance(radiance, 1.5, zenith), 1.0, rtol=1e-14)

    @given(FINITE, POSITIVE, st.floats(0.0, 84.9), st.floats(0.9, 1.1))
    @example(2.225073858507203e-309, 1.0, 0.0, 1.0)
    def test_inverse_pair(self, radiance, irradiance, zenith, distance) -> None:
        rho = reflectance(radiance, irradiance, zenith, distance)
        back = radiance_from_reflectance(rho, irradiance, zenith, distance)
        # a subnormal radiance has no relative precision
        assert_matches_oracle(back, radiance, rtol=1e-14, atol=1e-300)

    @given(FINITE, FINITE, POSITIVE, st.floats(0.0, 84.9))
    def test_linear_in_the_radiance(self, first, second, irradiance, zenith) -> None:
        total = reflectance(first + second, irradiance, zenith)
        parts = (
            reflectance(first, irradiance, zenith),
            reflectance(second, irradiance, zenith),
        )
        scale = (abs(parts[0]) + abs(parts[1])) * 1e-13 + 1e-300
        assert abs(total - sum(parts)) <= scale

    def test_the_limit_is_85_degrees_in_either_unit(self) -> None:
        zenith = np.asarray([84.99, 85.0, 120.0])
        for rho in (
            reflectance(1.0, 1.0, zenith),
            reflectance(1.0, 1.0, np.deg2rad(zenith), degrees=False),
        ):
            assert np.isnan(rho).tolist() == [False, True, True]

    def test_the_limit_is_in_the_unit_of_the_zenith(self) -> None:
        wide = reflectance(1.0, 1.0, np.asarray([88.0, 89.5]), max_zenith=89.0)
        assert np.isnan(wide).tolist() == [False, True]
        radians = np.deg2rad([88.0, 89.5])
        limit = float(np.deg2rad(89.0))
        wide = reflectance(1.0, 1.0, radians, max_zenith=limit, degrees=False)
        assert np.isnan(wide).tolist() == [False, True]

    @needs_jax
    def test_a_dark_pixel_does_not_spoil_the_others_gradient(self) -> None:
        import jax
        import jax.numpy as jnp

        zenith = jnp.asarray([30.0, 90.0, 120.0])
        gradient = jax.grad(lambda z: jnp.nansum(reflectance(0.1, 1.0, z)))(zenith)
        assert np.isfinite(np.asarray(gradient)).all()
        assert gradient[0] > 0.0

    def test_gradients_and_backends(self) -> None:
        zenith = np.asarray([10.0, 45.0, 80.0])
        radiance = np.asarray([0.01, 0.05, 0.1])
        irradiance = np.asarray([1.9, 0.9, 0.25])
        distance = np.asarray([0.984, 1.0, 1.016])
        for function in (reflectance, radiance_from_reflectance):
            assert_grad_matches_fd(
                lambda x, f=function: f(x[0], irradiance, 100.0 * x[1], x[2]),
                np.stack([radiance, zenith / 100.0, distance]),
                rtol=1e-5,
            )
            assert_backends_agree(function, radiance, irradiance, zenith, distance)
            assert_traces(function, radiance, irradiance, zenith, distance)
