"""Altimeter corrections against the numbers of the documents they come from."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax, needs_xarray
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, strategies as st

from radiomax.altimetry import (
    dry_troposphere_correction,
    dual_frequency_ionosphere_correction,
    inverse_barometer,
    ionosphere_correction,
    sea_surface_height,
    wet_troposphere_correction,
)


KU, C = 13.575e9, 5.3e9
PRESSURE = st.floats(5e4, 1.1e5)
LATITUDE = st.floats(-90.0, 90.0)


class TestHeight:
    def test_the_sum(self) -> None:
        # the handbook's convention: corrected range = range + corrections
        height = sea_surface_height(1_336_000.0, 1_335_962.0, -2.25, -0.125, 0.5)
        assert float(height) == 1_336_000.0 - (1_335_962.0 - 2.25 - 0.125 + 0.5)

    def test_without_corrections(self) -> None:
        assert float(sea_surface_height(1000.0, 960.0)) == 40.0

    def test_the_inverse_barometer_is_subtracted(self) -> None:
        low = inverse_barometer(98000.0, 101100.0)
        assert float(low) > 0.0
        assert float(sea_surface_height(1000.0, 960.0, low)) < 40.0

    def test_float32_stays(self) -> None:
        altitude = np.asarray([1000.0], dtype=np.float32)
        got = sea_surface_height(altitude, 960.0, -2.0)
        assert got.dtype == np.float32
        # whichever argument carries the dtype
        got = sea_surface_height(1000.0, 960.0, np.float32(-2.0))
        assert got.dtype == np.float32

    def test_gradient_and_backends(self) -> None:
        values = [
            np.linspace(1000.0, 1010.0, 6).reshape(2, 3),
            np.linspace(960.0, 965.0, 6).reshape(2, 3),
            np.linspace(-2.4, -2.2, 6).reshape(2, 3),
            np.linspace(-0.4, -0.1, 6).reshape(2, 3),
        ]
        assert_backends_agree(sea_surface_height, *values, rtol32=1e-5)
        assert_traces(sea_surface_height, *(v.ravel() for v in values))
        assert_grad_matches_fd(
            lambda x: sea_surface_height(x[0], x[1], x[2], x[3]),
            np.asarray([1.2, 0.9, -0.2, -0.1]),
        )

    @needs_xarray
    def test_corrections_meet_by_name(self) -> None:
        import xarray as xr

        along = xr.DataArray([960.0, 961.0, 962.0], dims="time")
        model = xr.DataArray([-2.2, -2.3], dims="member")
        height = sea_surface_height(1000.0, along, model, -0.1)
        assert set(height.dims) == {"time", "member"}
        expected = 1000.0 - (962.0 - 2.3 - 0.1)
        assert_matches_oracle(
            height.isel(time=2, member=1).values, expected, rtol=1e-14
        )

    @needs_xarray
    def test_an_array_without_names_is_refused(self) -> None:
        import xarray as xr

        along = xr.DataArray([960.0, 961.0], dims="time")
        unnamed = np.asarray([-2.2, -2.3])
        with pytest.raises(ValueError, match="'corrections' has dimensions"):
            sea_surface_height(1000.0, along, -0.1, unnamed)
        with pytest.raises(ValueError, match="the DataArray 'corrections'"):
            sea_surface_height(unnamed, 960.0, along)


class TestDry:
    def test_about_minus_2_3_metres(self) -> None:
        # Jason-3 handbook, section 5.4.1
        got = dry_troposphere_correction(101325.0, np.asarray([0.0, 45.0, 90.0]))
        assert_matches_oracle(got, np.full(3, -2.3), rtol=0.0, atol=0.02)

    @given(PRESSURE, LATITUDE)
    def test_the_formula_in_its_own_units(self, pressure, lat) -> None:
        # millimetres and mbar
        mbar = pressure / 100.0
        expected = -2.277 * mbar * (1.0 + 0.0026 * np.cos(2.0 * np.deg2rad(lat)))
        got = 1000.0 * dry_troposphere_correction(pressure, lat)
        assert_matches_oracle(got, expected, rtol=1e-14)

    @given(PRESSURE, LATITUDE)
    def test_the_hemispheres_agree(self, pressure, lat) -> None:
        north = dry_troposphere_correction(pressure, lat)
        assert float(north) == float(dry_troposphere_correction(pressure, -lat))

    def test_radians_gradient_and_backends(self) -> None:
        radians = dry_troposphere_correction(1e5, np.deg2rad(37.0), degrees=False)
        assert_matches_oracle(
            radians, dry_troposphere_correction(1e5, 37.0), rtol=1e-15
        )
        assert_grad_matches_fd(
            lambda x: dry_troposphere_correction(1e5 * x[0], 40.0 * x[1]),
            np.asarray([1.01, 0.9]),
        )
        pressure = np.linspace(9.8e4, 1.03e5, 6).reshape(2, 3)
        lat = np.linspace(-60.0, 60.0, 6).reshape(2, 3)
        assert_backends_agree(dry_troposphere_correction, pressure, lat)
        assert_traces(dry_troposphere_correction, pressure.ravel(), lat.ravel())


class TestWet:
    def test_the_papers_one_term_form(self) -> None:
        # Bevis et al. 1992, equation 8: with the mean temperature near 260 K,
        # the two terms are one of (0.382 ± 0.004) K² mbar⁻¹, in 10⁶
        both = 3.776e5 + 17.0 * 260.0
        assert both == pytest.approx(0.382e6, abs=0.004e6)
        # W = (M / R) ∫ (e / T) dz, and the delay is 10⁻⁶ both ∫ (e / T²) dz
        column, temperature = 20.0, 260.0
        gas = 8.314462618 / 0.01801528
        integral = column * gas / temperature / 100.0  # ∫ (e / T²) dz, mbar m K⁻²
        got = wet_troposphere_correction(column, temperature)
        assert_matches_oracle(got, -1e-6 * both * integral, rtol=1e-9)

    def test_the_rule_of_thumb(self) -> None:
        # Bevis et al. 1992: the delay is roughly 6.4 times the precipitable water
        temperature = np.linspace(255.0, 290.0, 8)
        delay = -wet_troposphere_correction(25.0, temperature)
        assert_matches_oracle(delay / 0.025, np.full(8, 6.4), rtol=0.1)

    def test_the_constants_are_arguments(self) -> None:
        # with k₂′ alone it is linear in the column, without the temperature
        got = wet_troposphere_correction(10.0, np.asarray([250.0, 290.0]), k3=0.0)
        assert float(got[0]) == float(got[1])
        doubled = wet_troposphere_correction(20.0, 250.0, k3=0.0)
        assert_matches_oracle(doubled, 2.0 * got[0], rtol=1e-15)

    def test_no_temperature_is_nan(self) -> None:
        got = wet_troposphere_correction(25.0, np.asarray([270.0, 0.0, -1.0, np.nan]))
        assert np.isnan(got).tolist() == [False, True, True, True]

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        gradient = jax.grad(
            lambda t: jax.numpy.nansum(wet_troposphere_correction(25.0, t))
        )(np.asarray([270.0, 0.0, -1.0]))
        assert np.isfinite(gradient).all()
        assert gradient[0] > 0.0

    @needs_jax
    def test_a_traced_constant_keeps_jax(self) -> None:
        import jax

        slope = jax.grad(lambda k: wet_troposphere_correction(25.0, 270.0, k3=k))(
            3776.0
        )
        step = wet_troposphere_correction(25.0, 270.0, k2=0.0, k3=1.0)
        assert_matches_oracle(slope, step, rtol=1e-14)

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: wet_troposphere_correction(25.0 * x[0], 270.0 * x[1]),
            np.asarray([1.1, 0.98]),
        )
        column = np.linspace(5.0, 60.0, 6).reshape(2, 3)
        temperature = np.linspace(255.0, 288.0, 6).reshape(2, 3)
        assert_backends_agree(wet_troposphere_correction, column, temperature)
        assert_traces(wet_troposphere_correction, column.ravel(), temperature.ravel())


class TestIonosphere:
    def test_the_conventions_example(self) -> None:
        # IERS Conventions (2010), equation 9.24: 300 TECU is 1.2 × 10²⁰ / f² m
        got = ionosphere_correction(300e16, 1.0)
        assert float(got) == pytest.approx(-1.2e20, rel=0.01)

    def test_the_constant_is_the_electrons(self) -> None:
        # equation 9.19: q² / (8 π² mₑ ε₀), with the conventions' constants
        charge, mass, vacuum = 1.602176565e-19, 9.10938291e-31, 8.854187817e-12
        expected = charge**2 / (8.0 * np.pi**2 * mass * vacuum)
        assert_matches_oracle(-ionosphere_correction(1.0, 1.0), expected, rtol=1e-5)

    @given(st.floats(0.0, 3e18), st.floats(1e9, 4e10))
    def test_the_formula(self, content, frequency) -> None:
        got = ionosphere_correction(content, frequency)
        expected = -40.308 * content / frequency**2
        assert_matches_oracle(got, expected, rtol=1e-14, atol=1e-15)

    def test_the_handbooks_range_at_ku(self) -> None:
        # "typically 0.2 to 20 cm at 13.6 GHz"
        got = ionosphere_correction(np.asarray([1e16, 90e16]), 13.6e9)
        assert_matches_oracle(got, np.asarray([-0.002, -0.2]), rtol=0.1)

    def test_no_frequency_is_nan(self) -> None:
        got = ionosphere_correction(1e17, np.asarray([KU, 0.0, -C, np.nan]))
        assert np.isnan(got).tolist() == [False, True, True, True]

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: ionosphere_correction(5e17 * x[0], KU * x[1]),
            np.asarray([1.1, 0.9]),
        )
        content = np.linspace(1e16, 1e18, 6).reshape(2, 3)
        frequency = np.linspace(5e9, 3.6e10, 6).reshape(2, 3)
        assert_backends_agree(ionosphere_correction, content, frequency)
        assert_traces(ionosphere_correction, content.ravel(), frequency.ravel())


class TestTwoFrequencies:
    @given(st.floats(0.0, 3e18), st.floats(5e9, 4e10), st.floats(1.1, 5.0))
    def test_it_recovers_the_delay(self, content, f2, ratio) -> None:
        # two ranges to one surface, each delayed by the single-frequency formula
        f1 = ratio * f2
        delay_1 = -ionosphere_correction(content, f1)
        delay_2 = -ionosphere_correction(content, f2)
        got = dual_frequency_ionosphere_correction(delay_1, delay_2, f1, f2)
        # the absolute part is for subnormal delays
        assert_matches_oracle(got, -delay_1, rtol=1e-12, atol=1e-15)

    def test_the_other_band(self) -> None:
        ku, c = 100.2, 101.3
        first = dual_frequency_ionosphere_correction(ku, c, KU, C)
        second = dual_frequency_ionosphere_correction(c, ku, C, KU)
        assert_matches_oracle(second, first * (KU / C) ** 2, rtol=1e-13)
        # both corrected ranges are one
        assert_matches_oracle(ku + first, c + second, rtol=1e-14)

    def test_at_the_satellites_altitude(self) -> None:
        surface, delay = 1_335_962.0, 0.1
        first = surface + delay
        second = surface + delay * (KU / C) ** 2
        got = dual_frequency_ionosphere_correction(first, second, KU, C)
        # float64 steps by 2 × 10⁻¹⁰ m there
        assert_matches_oracle(got, -delay, rtol=0.0, atol=1e-9)

    def test_frequencies_that_cannot_tell_are_nan(self) -> None:
        f1 = np.asarray([KU, KU, 0.0, KU, -KU, np.nan])
        f2 = np.asarray([C, KU, C, 0.0, C, C])
        got = dual_frequency_ionosphere_correction(100.1, 100.6, f1, f2)
        assert np.isnan(got).tolist() == [False, True, True, True, True, True]

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        for argument in (2, 3):
            gradient = jax.grad(
                lambda *v: jax.numpy.nansum(dual_frequency_ionosphere_correction(*v)),
                argnums=argument,
            )(100.1, 100.6, np.asarray([KU, KU, 0.0]), np.asarray([C, KU, C]))
            assert np.isfinite(gradient).all()
            assert gradient[0] != 0.0

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: dual_frequency_ionosphere_correction(
                x[0], x[1], KU * x[2], C * x[3]
            ),
            np.asarray([1.1, 1.6, 1.0, 1.0]),
        )
        first = np.linspace(0.1, 0.2, 6).reshape(2, 3)
        second = np.linspace(0.8, 1.3, 6).reshape(2, 3)
        assert_backends_agree(
            lambda a, b: dual_frequency_ionosphere_correction(a, b, KU, C),
            first,
            second,
        )
        assert_traces(
            lambda a, b: dual_frequency_ionosphere_correction(a, b, KU, C),
            first.ravel(),
            second.ravel(),
        )


class TestBarometer:
    def test_the_handbooks_two_means(self) -> None:
        # section 5.4.6.1: −9.948 (1010.9 − 1013.3) = 23.9 mm
        varying = inverse_barometer(100000.0, 101090.0)
        static = inverse_barometer(100000.0, 101330.0)
        assert 1000.0 * float(static - varying) == pytest.approx(23.9, abs=0.05)

    def test_one_hectopascal_is_about_a_centimetre(self) -> None:
        assert float(inverse_barometer(101100.0, 101000.0)) == pytest.approx(
            -0.01, 0.01
        )

    @given(PRESSURE, PRESSURE)
    def test_the_formula_in_its_own_units(self, pressure, mean) -> None:
        expected = -9.948 * (pressure / 100.0 - mean / 100.0)
        got = 1000.0 * inverse_barometer(pressure, mean)
        assert_matches_oracle(got, expected, rtol=1e-12, atol=1e-9)

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: inverse_barometer(1e5 * x[0], 1e5 * x[1]), np.asarray([1.0, 1.01])
        )
        pressure = np.linspace(9.8e4, 1.03e5, 6).reshape(2, 3)
        mean = np.linspace(1.009e5, 1.0115e5, 6).reshape(2, 3)
        # a difference of two pressures keeps fewer digits in float32
        assert_backends_agree(inverse_barometer, pressure, mean, rtol32=1e-3)
        assert_traces(inverse_barometer, pressure.ravel(), mean.ravel())
