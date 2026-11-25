"""SAR calibration against ESA's note, and between its three quantities."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.sar import (
    beta0_to_sigma0,
    dn_to_backscatter,
    gamma0_to_sigma0,
    sigma0_to_beta0,
    sigma0_to_gamma0,
)


INCIDENCE = st.floats(0.01, 89.99)
PAIRS = pytest.mark.parametrize(
    ("forward", "back", "factor"),
    [
        (sigma0_to_beta0, beta0_to_sigma0, np.sin),
        (sigma0_to_gamma0, gamma0_to_sigma0, np.cos),
    ],
    ids=["beta0", "gamma0"],
)


class TestDigitalNumbers:
    def test_the_example_of_esas_note(self) -> None:
        # ESA-EOPG-CSCOP-TN-0002, section 4.1: −6.65 dB
        sigma0 = dn_to_backscatter(np.sqrt(22142.71), 319.946)
        assert 10.0 * np.log10(float(sigma0)) == pytest.approx(-6.65, abs=5e-3)

    def test_the_tables_of_esas_note_agree_with_the_angle(self) -> None:
        # one calibration vector of the note: cos α = A_γ² / A_σ², and the angle
        # is inside Sentinel-1's interferometric swath
        sigma_table = np.asarray([332.0240, 331.9609, 319.9946, 319.8001])
        gamma_table = np.asarray([307.9890, 307.9099, 292.6021, 292.3477])
        angle = np.rad2deg(np.arccos(gamma_table**2 / sigma_table**2))
        assert (angle > 29.0).all()
        assert (angle < 46.0).all()
        dn = np.asarray([10.0, 200.0, 3000.0, 65535.0])
        gamma0 = sigma0_to_gamma0(dn_to_backscatter(dn, sigma_table), angle)
        assert_matches_oracle(gamma0, dn_to_backscatter(dn, gamma_table), rtol=1e-14)

    def test_integers_do_not_overflow(self) -> None:
        dn = np.asarray([0, 65535], dtype=np.uint16)
        got = dn_to_backscatter(dn, 300.0)
        assert_matches_oracle(got, np.asarray([0.0, 65535.0 / 300.0]) ** 2, rtol=1e-15)

    def test_float32_keeps_its_range(self) -> None:
        dn = np.asarray([1e25], dtype=np.float32)
        got = dn_to_backscatter(dn, np.asarray([1e24], dtype=np.float32))
        assert got.dtype == np.float32
        assert_matches_oracle(got, np.asarray([100.0]), rtol=1e-6)

    def test_a_table_value_that_is_not_positive_is_nan(self) -> None:
        got = dn_to_backscatter(100.0, np.asarray([300.0, 0.0, -1.0, np.nan]))
        assert np.isnan(got).tolist() == [False, True, True, True]

    def test_complex_numbers_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must be real"):
            dn_to_backscatter(np.asarray([1.0 + 1.0j]), 300.0)

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: dn_to_backscatter(100.0 * x[0], 300.0 * x[1]),
            np.asarray([1.1, 0.9]),
        )
        dn = np.linspace(0.0, 500.0, 6).reshape(2, 3)
        table = np.linspace(290.0, 330.0, 6).reshape(2, 3)
        assert_backends_agree(dn_to_backscatter, dn, table)
        assert_traces(dn_to_backscatter, dn.ravel(), table.ravel())

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        gradient = jax.grad(
            lambda a: jax.numpy.nansum(dn_to_backscatter(100.0, a)), argnums=0
        )(np.asarray([300.0, 0.0, -1.0]))
        assert np.isfinite(gradient).all()
        assert gradient[0] < 0.0


@given(st.floats(1e-6, 1e3), INCIDENCE)
def test_one_sigma0_from_both(sigma0, incidence) -> None:
    # gate G4-sar: σ⁰ = β⁰ sin θᵢ = γ⁰ cos θᵢ
    angle = np.deg2rad(incidence)
    beta0 = sigma0_to_beta0(sigma0, incidence)
    gamma0 = sigma0_to_gamma0(sigma0, incidence)
    assert_matches_oracle(beta0 * np.sin(angle), sigma0, rtol=1e-15)
    assert_matches_oracle(gamma0 * np.cos(angle), sigma0, rtol=1e-15)
    assert_matches_oracle(gamma0, beta0 * np.tan(angle), rtol=2e-15)


@PAIRS
class TestAreas:
    @given(sigma0=st.floats(-10.0, 1e3), incidence=INCIDENCE)
    def test_there_and_back(self, forward, back, factor, sigma0, incidence) -> None:
        # a σ⁰ below the noise is negative, and stays
        got = back(forward(sigma0, incidence), incidence)
        assert_matches_oracle(got, sigma0, rtol=1e-15)

    @given(incidence=INCIDENCE)
    def test_the_factor(self, forward, back, factor, incidence) -> None:
        assert_matches_oracle(
            back(1.0, incidence), factor(np.deg2rad(incidence)), rtol=1e-15
        )

    def test_nan_outside_the_quarter(self, forward, back, factor) -> None:
        incidence = np.asarray([-10.0, 0.0, 1e-3, 45.0, 89.999, 90.0, 120.0, np.nan])
        expected = [True, True, False, False, False, True, True, True]
        assert np.isnan(forward(0.1, incidence)).tolist() == expected
        assert np.isnan(back(0.1, incidence)).tolist() == expected
        radians = np.deg2rad(incidence)
        assert np.isnan(forward(0.1, radians, degrees=False)).tolist() == expected

    def test_radians(self, forward, back, factor) -> None:
        got = forward(0.1, np.deg2rad(37.0), degrees=False)
        assert_matches_oracle(got, forward(0.1, 37.0), rtol=1e-15)

    def test_gradient_and_backends(self, forward, back, factor) -> None:
        sigma0 = np.linspace(0.01, 0.5, 6).reshape(2, 3)
        incidence = np.linspace(20.0, 46.0, 6).reshape(2, 3)
        for function in (forward, back):
            assert_grad_matches_fd(
                lambda x, f=function: f(x[0], 40.0 * x[1]), np.asarray([0.2, 0.9])
            )
            assert_backends_agree(function, sigma0, incidence)
            assert_traces(function, sigma0.ravel(), incidence.ravel())

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(
        self, forward, back, factor
    ) -> None:
        import jax

        for function in (forward, back):
            gradient = jax.grad(lambda t, f=function: jax.numpy.nansum(f(0.1, t)))(
                np.asarray([0.0, 40.0, 90.0])
            )
            assert np.isfinite(gradient).all()
            assert gradient[1] != 0.0
