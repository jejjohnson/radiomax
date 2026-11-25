"""Planck's law against radtrax's values, its limits and its integral."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _oracles import radtrax_values
from geophysix.constants import STANDARD
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.thermal import (
    from_per_ghz,
    from_per_um,
    log_planck_radiance,
    planck_radiance,
    to_per_ghz,
    to_per_um,
)


H, C, K = STANDARD.planck, STANDARD.speed_of_light, STANDARD.boltzmann
C1, C2 = 2.0 * H * C**2 * 1e8, H * C / K * 1e2

WAVENUMBER = st.floats(600.0, 25000.0)
TEMPERATURE = st.floats(150.0, 2500.0)


def _stored() -> dict[str, np.ndarray]:
    stored = radtrax_values()
    stored["nu"] = stored["planck_wavenumber_cm"][:, None]
    stored["t"] = stored["planck_temperature"][None, :]
    return stored


class TestAgainstRadtrax:
    def test_the_radiance(self) -> None:
        # gate G2-planck
        stored = _stored()
        radiance = planck_radiance(stored["nu"], stored["t"])
        assert_matches_oracle(radiance, stored["planck_radiance"], rtol=1e-13)

    def test_its_logarithm(self) -> None:
        stored = _stored()
        log_radiance = log_planck_radiance(stored["nu"], stored["t"])
        assert_matches_oracle(log_radiance, stored["log_planck_radiance"], rtol=1e-13)

    @needs_jax
    def test_its_derivative(self) -> None:
        # gate G2-planck: autodiff against radtrax's hand-written ∂B/∂T
        import jax

        stored = _stored()
        nu, t = np.broadcast_arrays(stored["nu"], stored["t"])
        gradient = jax.vmap(jax.grad(planck_radiance, argnums=1))
        derivative = np.asarray(gradient(nu.ravel(), t.ravel())).reshape(nu.shape)
        assert_matches_oracle(derivative, stored["planck_dt"], rtol=1e-13)

    def test_the_views(self) -> None:
        stored = _stored()
        radiance = stored["planck_radiance"]
        per_um = to_per_um(radiance, stored["nu"])
        assert_matches_oracle(per_um, stored["planck_per_um"], rtol=1e-14)
        per_ghz = to_per_ghz(radiance)
        assert_matches_oracle(per_ghz, stored["planck_per_ghz"], rtol=1e-14)


class TestPlanckRadiance:
    def test_the_rayleigh_jeans_limit(self) -> None:
        # B = c₁ ν̃² T / c₂ (1 − x/2 + O(x²))
        nu, t = np.asarray([1e-3, 1e-2, 1e-1]), 300.0
        x = C2 * nu / t
        ratio = planck_radiance(nu, t) / (C1 * nu**2 * t / C2)
        assert_matches_oracle(ratio, 1.0 - x / 2.0, rtol=0.0, atol=1e-7)

    def test_the_wien_limit(self) -> None:
        # B = c₁ ν̃³ e⁻ˣ (1 + O(e⁻ˣ))
        nu, t = np.asarray([5e3, 1e4]), 250.0
        x = C2 * nu / t
        expected = C1 * nu**3 * np.exp(-x)
        assert_matches_oracle(planck_radiance(nu, t), expected, rtol=2e-12)

    @pytest.mark.parametrize("temperature", [150.0, 300.0, 2500.0])
    def test_the_stefan_boltzmann_law(self, temperature: float) -> None:
        # π ∫ B dν̃ = σ T⁴, σ = 2π⁵ k⁴ / (15 h³ c²)
        sigma = 2.0 * math.pi**5 * K**4 / (15.0 * H**3 * C**2)
        # ∫ over ln x, where the trapezoid rule converges geometrically
        log_x = np.linspace(-18.0, 6.0, 4001)
        nu = np.exp(log_x) * temperature / C2
        total = np.trapezoid(planck_radiance(nu, temperature) * nu, log_x)
        assert_matches_oracle(math.pi * total, sigma * temperature**4, rtol=1e-8)

    def test_the_wien_displacement_law(self) -> None:
        # the peak of B per wavenumber is at x = 3 + W(−3e⁻³) = 2.8214393721
        nu = np.linspace(580.0, 596.0, 160001)
        peak = nu[np.argmax(planck_radiance(nu, 300.0))]
        assert abs(C2 * peak / 300.0 - 2.8214393721) < 1e-5

    @given(WAVENUMBER, TEMPERATURE, st.floats(1.0001, 2.0))
    def test_it_increases_with_temperature(self, nu, t, factor) -> None:
        assert planck_radiance(nu, t * factor) > planck_radiance(nu, t) > 0.0

    @given(WAVENUMBER, TEMPERATURE)
    def test_the_logarithm_is_the_logarithm(self, nu, t) -> None:
        log_radiance = log_planck_radiance(nu, t)
        assert abs(log_radiance - np.log(planck_radiance(nu, t))) < 1e-11

    def test_invalid_arguments_are_nan(self) -> None:
        nu = np.asarray([1000.0, 0.0, -1.0, 1000.0, 1000.0, np.nan])
        t = np.asarray([300.0, 300.0, 300.0, 0.0, -1.0, 300.0])
        for function in (planck_radiance, log_planck_radiance):
            assert np.isnan(function(nu, t)).tolist() == [False] + [True] * 5

    def test_float32_over_the_solar_and_thermal_range(self) -> None:
        stored = _stored()
        nu, t = (stored[k].astype(np.float32) for k in ("nu", "t"))
        log_radiance = log_planck_radiance(nu, t)
        assert log_radiance.dtype == np.float32
        assert np.isfinite(log_radiance).all()
        assert_matches_oracle(log_radiance, stored["log_planck_radiance"], rtol=4e-6)
        radiance = planck_radiance(nu, t)
        assert np.isfinite(radiance).all()
        visible = stored["planck_radiance"] > 1e-30
        assert_matches_oracle(
            radiance, stored["planck_radiance"], rtol=1e-5, where=visible
        )

    def test_other_constants(self) -> None:
        doubled = dataclasses.replace(STANDARD, boltzmann=2.0 * K)
        # only k T enters
        got = planck_radiance(1000.0, 150.0, constants=doubled)
        assert_matches_oracle(got, planck_radiance(1000.0, 300.0), rtol=1e-14)

    @needs_jax
    def test_a_traced_constant(self) -> None:
        import jax

        def radiance(boltzmann):
            constants = dataclasses.replace(STANDARD, boltzmann=boltzmann)
            return planck_radiance(1000.0, 300.0, constants=constants)

        # ∂B/∂ln k = ∂B/∂ln T
        by_k = jax.grad(radiance)(K) * K
        by_t = jax.grad(planck_radiance, argnums=1)(1000.0, 300.0) * 300.0
        assert_matches_oracle(by_k, by_t, rtol=1e-12)

    def test_gradient(self) -> None:
        for function in (planck_radiance, log_planck_radiance):
            assert_grad_matches_fd(
                lambda x, f=function: f(1e3 * x[:3], 300.0 * x[3:]),
                np.asarray([0.7, 1.0, 2.5, 0.8, 1.0, 1.1]),
            )

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        for function in (planck_radiance, log_planck_radiance):
            gradient = jax.grad(lambda t, f=function: jax.numpy.nansum(f(1000.0, t)))(
                np.asarray([0.0, -5.0, 300.0])
            )
            assert np.isfinite(gradient).all()
            assert gradient[2] > 0.0

    def test_backends(self) -> None:
        nu, t = np.linspace(700.0, 2500.0, 6).reshape(2, 3), np.full((2, 3), 290.0)
        assert_backends_agree(planck_radiance, nu, t, rtol32=5e-6)
        assert_backends_agree(log_planck_radiance, nu, t, rtol32=5e-6)
        batch = np.linspace(700.0, 2500.0, 5), np.linspace(220.0, 330.0, 5)
        assert_traces(planck_radiance, *batch)
        assert_traces(log_planck_radiance, *batch)


class TestViews:
    @given(st.floats(1e-12, 1e6), WAVENUMBER)
    def test_they_are_inverses(self, radiance, nu) -> None:
        back = from_per_um(to_per_um(radiance, nu), nu)
        assert_matches_oracle(back, radiance, rtol=1e-15)
        assert_matches_oracle(from_per_ghz(to_per_ghz(radiance)), radiance, rtol=1e-15)

    def test_planck_per_wavelength(self) -> None:
        # B_λ = 2hc² / λ⁵ / (exp(hc / λkT) − 1), per metre, at 10 µm
        wavelength, t = 10e-6, 300.0
        x = H * C / (wavelength * K * t)
        per_metre = 2.0 * H * C**2 / wavelength**5 / np.expm1(x)
        per_um = to_per_um(planck_radiance(1000.0, t), 1000.0)
        assert_matches_oracle(per_um, per_metre * 1e-6, rtol=1e-13)

    def test_planck_per_frequency(self) -> None:
        # B_f = 2hf³/c² / (exp(hf / kT) − 1), per Hz, at 183.31 GHz
        frequency, t = 183.31e9, 250.0
        per_hz = 2.0 * H * frequency**3 / C**2 / np.expm1(H * frequency / (K * t))
        per_ghz = to_per_ghz(planck_radiance(frequency / C / 100.0, t))
        assert_matches_oracle(per_ghz, per_hz * 1e9, rtol=1e-13)

    def test_no_wavenumber_is_nan(self) -> None:
        assert np.isnan(from_per_um(1.0, np.asarray([0.0, -1.0]))).all()

    def test_backends_and_gradient(self) -> None:
        nu = np.asarray([700.0, 1000.0, 2500.0])
        radiance = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        for function in (to_per_um, from_per_um):
            assert_backends_agree(function, radiance, nu)
            assert_traces(function, np.asarray([0.1, 0.2]), np.asarray([7e2, 9e2]))
            assert_grad_matches_fd(
                lambda x, f=function: f(x[0], 1e3 * x[1]), np.asarray([0.3, 1.1])
            )
        for function in (to_per_ghz, from_per_ghz):
            assert_backends_agree(function, radiance)
            assert_traces(function, np.asarray([0.1, 0.2]))
            assert_grad_matches_fd(function, np.asarray([0.3, 1.1]))

    def test_a_dataarray_aligns_by_name(self) -> None:
        xr = pytest.importorskip("xarray")
        nu = xr.DataArray(np.asarray([700.0, 1000.0]), dims="band")
        t = xr.DataArray(np.full((2, 3), 300.0), dims=("y", "x"))
        radiance = planck_radiance(nu, t)
        assert radiance.dims == ("band", "y", "x")
        assert to_per_um(radiance, nu).dims == ("band", "y", "x")
