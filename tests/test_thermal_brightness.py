"""The monochromatic brightness temperature against radtrax and its inverse."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import radtrax_values
from geophysix.constants import STANDARD
from hypothesis import assume, given, strategies as st

from radiomax.thermal import brightness_temperature, planck_radiance


C2 = STANDARD.planck * STANDARD.speed_of_light / STANDARD.boltzmann * 1e2


class TestBrightnessTemperature:
    def test_against_radtrax(self) -> None:
        stored = radtrax_values()
        nu = stored["planck_wavenumber_cm"][:, None]
        temperature = brightness_temperature(nu, stored["planck_radiance"])
        assert_matches_oracle(temperature, stored["brightness_temperature"], rtol=1e-13)

    @given(st.floats(600.0, 25000.0), st.floats(150.0, 2500.0))
    def test_it_inverts_planck(self, nu, t) -> None:
        # gate G2-inverse
        radiance = planck_radiance(nu, t)
        assume(radiance > 1e-300)
        assert_matches_oracle(brightness_temperature(nu, radiance), t, rtol=1e-12)

    @given(st.floats(1e-3, 600.0), st.floats(2.0, 2500.0))
    def test_it_inverts_planck_in_the_microwave(self, nu, t) -> None:
        radiance = planck_radiance(nu, t)
        assume(radiance > 1e-300)
        # at small x, T_b ∝ L, so the relative error is L's
        assert_matches_oracle(brightness_temperature(nu, radiance), t, rtol=1e-12)

    @given(st.floats(600.0, 25000.0), st.floats(1e-20, 1e3), st.floats(1.001, 10.0))
    def test_it_increases_with_radiance(self, nu, radiance, factor) -> None:
        low = brightness_temperature(nu, radiance)
        assert 0.0 < low < brightness_temperature(nu, radiance * factor)

    def test_the_rayleigh_jeans_limit(self) -> None:
        # T_b = L c₂ / (c₁ ν̃²) + c₂ ν̃ / 2 + O(x)
        nu, t = 1e-2, 300.0
        radiance = planck_radiance(nu, t)
        c1 = 2.0 * STANDARD.planck * STANDARD.speed_of_light**2 * 1e8
        linear = radiance * C2 / (c1 * nu**2)
        assert abs(linear + 0.5 * C2 * nu - t) < 1e-6
        assert abs(brightness_temperature(nu, radiance) - t) < 1e-9

    def test_a_radiance_that_is_not_positive_is_nan(self) -> None:
        radiance = np.asarray([0.1, 0.0, -1e-3, np.nan])
        temperature = brightness_temperature(1000.0, radiance)
        assert np.isnan(temperature).tolist() == [False, True, True, True]
        assert np.isnan(brightness_temperature(np.asarray([0.0, -1.0]), 0.1)).all()

    def test_float32(self) -> None:
        stored = radtrax_values()
        nu = stored["planck_wavenumber_cm"][:, None]
        seen = stored["planck_radiance"] > 1e-30
        temperature = brightness_temperature(
            nu.astype(np.float32), stored["planck_radiance"].astype(np.float32)
        )
        assert temperature.dtype == np.float32
        expected = stored["brightness_temperature"]
        assert_matches_oracle(temperature, expected, rtol=2e-6, where=seen)

    @needs_jax
    def test_its_derivative_is_the_inverse_of_plancks(self) -> None:
        import jax

        for nu, t in ((700.0, 220.0), (2500.0, 300.0), (25000.0, 2500.0)):
            by_t = jax.grad(planck_radiance, argnums=1)(nu, t)
            by_l = jax.grad(brightness_temperature, argnums=1)(
                nu, planck_radiance(nu, t)
            )
            assert_matches_oracle(by_l, 1.0 / by_t, rtol=1e-12)

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        gradient = jax.grad(
            lambda radiance: jax.numpy.nansum(brightness_temperature(1e3, radiance))
        )(np.asarray([0.0, -1.0, 0.1]))
        assert np.isfinite(gradient).all()
        assert gradient[2] > 0.0

    def test_gradient(self) -> None:
        assert_grad_matches_fd(
            lambda x: brightness_temperature(1e3 * x[:3], 0.1 * x[3:]),
            np.asarray([0.7, 1.0, 2.5, 0.8, 1.0, 1.1]),
        )

    def test_backends(self) -> None:
        nu = np.linspace(700.0, 2500.0, 6).reshape(2, 3)
        radiance = planck_radiance(nu, 290.0)
        assert_backends_agree(brightness_temperature, nu, radiance, rtol32=5e-6)
        assert_traces(brightness_temperature, nu.ravel(), radiance.ravel())

    def test_a_dataarray_aligns_by_name(self) -> None:
        xr = pytest.importorskip("xarray")
        nu = xr.DataArray(np.asarray([700.0, 1000.0]), dims="band")
        radiance = xr.DataArray(np.full((3, 2), 0.1), dims=("y", "band"))
        assert set(brightness_temperature(nu, radiance).dims) == {"band", "y"}
