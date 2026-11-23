"""Band averages, band irradiance, quadrature weights and spectral densities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import (
    functax_trapezoid_weights,
    georeader_band_irradiance,
    integrix_band_mean,
)
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from radiomax import ValidationError
from radiomax.radiometry import (
    band_average,
    band_irradiance,
    per_cm_to_per_nm,
    per_nm_to_per_cm,
    trapezoid_weights,
)


RNG = np.random.default_rng(0)
#: An uneven grid, three bands on it, and spectra that differ per pixel.
GRID = np.sort(RNG.uniform(2000.0, 2500.0, 60))
CENTRES = np.asarray([2150.0, 2250.0, 2350.0])
RESPONSES = np.exp(-(((GRID - CENTRES[:, None]) / 40.0) ** 2))
SPECTRA = RNG.uniform(0.5, 2.0, (7, 60))


def _with_nans() -> np.ndarray:
    spectra = SPECTRA.copy()
    spectra[RNG.random(spectra.shape) < 0.1] = np.nan
    return spectra


class TestTrapezoidWeights:
    def test_functax(self) -> None:
        for grid in (GRID, GRID[::-1], GRID[:1]):
            expected = functax_trapezoid_weights(grid)
            assert_matches_oracle(trapezoid_weights(grid), expected, rtol=1e-15)

    def test_exact_for_a_line_and_negated_on_a_descending_grid(self) -> None:
        weights = trapezoid_weights(GRID)
        assert_matches_oracle(weights.sum(), GRID[-1] - GRID[0], rtol=1e-14)
        integral = (GRID[-1] ** 2 - GRID[0] ** 2) / 2.0
        assert_matches_oracle(weights @ GRID, integral, rtol=1e-14)
        assert_matches_oracle(trapezoid_weights(GRID[::-1])[::-1], -weights, rtol=1e-15)

    def test_integers_become_floating_and_other_shapes_raise(self) -> None:
        assert trapezoid_weights(np.asarray([0, 1, 3])).tolist() == [0.5, 1.5, 1.0]
        with pytest.raises(ValidationError, match="must be 1-D"):
            trapezoid_weights(np.ones((2, 2)))

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(trapezoid_weights, GRID[:5])
        assert_backends_agree(
            lambda grid: trapezoid_weights(getattr(grid, "values", grid)) + 0.0 * grid,
            GRID[:5],
            rtol32=1e-3,
        )


class TestBandAverage:
    @pytest.mark.slow
    def test_integrix(self) -> None:
        weights = trapezoid_weights(GRID)
        out = band_average(SPECTRA, RESPONSES, GRID)
        expected = integrix_band_mean(SPECTRA, RESPONSES, weights)
        assert_matches_oracle(out, expected, rtol=1e-14)

        spectra = _with_nans()
        for policy in ("propagate", "omit"):
            out = band_average(spectra, RESPONSES, GRID, nan_policy=policy)
            expected = integrix_band_mean(
                spectra, RESPONSES, weights, nan_policy=policy
            )
            assert_matches_oracle(out, expected, rtol=1e-14)

        mask = RNG.random(SPECTRA.shape) < 0.7
        out = band_average(spectra, RESPONSES, GRID, where=mask, nan_policy="omit")
        expected = integrix_band_mean(
            spectra, RESPONSES, weights, where=mask, nan_policy="omit"
        )
        assert_matches_oracle(out, expected, rtol=1e-14)

    @pytest.mark.slow
    @pytest.mark.parametrize("policy", ["propagate", "omit"])
    def test_integrixs_gradient(self, policy: str) -> None:
        jax = pytest.importorskip("jax")
        rule = pytest.importorskip("integrix.rule")
        jnp = jax.numpy
        spectra = jnp.asarray(_with_nans() if policy == "omit" else SPECTRA)
        weights = jnp.asarray(trapezoid_weights(GRID))
        responses = jnp.asarray(RESPONSES)

        def ours(f: Any) -> Any:
            return jnp.sum(band_average(f, responses, GRID, nan_policy=policy))

        def theirs(f: Any) -> Any:
            mean = rule.weighted_mean(weights, f, weight=responses, nan_policy=policy)
            return jnp.sum(mean)

        gradient = jax.grad(ours)(spectra)
        assert np.isfinite(np.asarray(gradient)).all()
        assert_matches_oracle(
            gradient, jax.grad(theirs)(spectra), rtol=1e-13, atol=1e-16
        )

    def test_functaxs_simpson_weights_plug_in(self) -> None:
        quad = pytest.importorskip("functax.quad")
        weights = np.asarray(quad.simpson(GRID).weights)
        out = band_average(SPECTRA, RESPONSES, GRID, weights=weights)
        combined = RESPONSES * weights
        expected = SPECTRA @ combined.T / combined.sum(axis=1)
        assert_matches_oracle(out, expected, rtol=1e-12)

    @given(
        st.floats(-1e3, 1e3),
        arrays(np.bool_, 60),
        arrays(np.float64, (3, 60), elements=st.floats(0.0, 1.0)),
    )
    def test_a_flat_spectrum_is_its_value(self, value, mask, responses) -> None:
        # gate G1-band
        out = band_average(np.full(60, value), responses, GRID, where=mask)
        covered = (responses * trapezoid_weights(GRID) * mask).sum(axis=1) != 0
        expected = np.where(covered, value, np.nan)
        assert_matches_oracle(out, expected, rtol=1e-13, atol=1e-13)

    def test_a_line_over_a_symmetric_response_is_its_centre_value(self) -> None:
        grid = np.linspace(1500.0, 1700.0, 201)
        response = np.exp(-(((grid - 1600.0) / 15.0) ** 2))
        out = band_average(3.0 + 0.01 * (grid - 1600.0), response, grid)
        assert_matches_oracle(out, 3.0, rtol=1e-12)

    def test_grid_order_does_not_matter(self) -> None:
        forward = band_average(SPECTRA, RESPONSES, GRID)
        backward = band_average(SPECTRA[:, ::-1], RESPONSES[:, ::-1], GRID[::-1])
        assert_matches_oracle(backward, forward, rtol=1e-13)

    def test_one_response_drops_the_band_axis(self) -> None:
        one = band_average(SPECTRA, RESPONSES[1], GRID)
        assert one.shape == (7,)
        assert_matches_oracle(
            one, band_average(SPECTRA, RESPONSES, GRID)[:, 1], rtol=1e-14
        )
        assert band_average(SPECTRA[0], RESPONSES[1], GRID).shape == ()
        cube = SPECTRA.reshape(7, 1, 1, 60)
        assert band_average(cube, RESPONSES, GRID).shape == (7, 1, 1, 3)

    def test_negative_tails_count_as_zero(self) -> None:
        tails = np.where(RESPONSES < 1e-3, -0.1, RESPONSES)
        clipped = np.where(RESPONSES < 1e-3, 0.0, RESPONSES)
        assert_matches_oracle(
            band_average(SPECTRA, tails, GRID),
            band_average(SPECTRA, clipped, GRID),
            rtol=0.0,
        )

    def test_nothing_left_is_nan_without_a_warning(self) -> None:
        with np.errstate(all="raise"):
            out = band_average(SPECTRA, np.zeros((2, 60)), GRID)
            assert np.isnan(out).all()
            masked = band_average(SPECTRA, RESPONSES, GRID, where=np.zeros(60, bool))
            assert np.isnan(masked).all()

    def test_a_masked_nan_does_not_propagate(self) -> None:
        spectra = SPECTRA.copy()
        spectra[:, 3] = np.nan
        keep = np.arange(60) != 3
        out = band_average(spectra, RESPONSES, GRID, where=keep)
        omitted = band_average(spectra, RESPONSES, GRID, nan_policy="omit")
        assert_matches_oracle(out, omitted, rtol=0.0)
        assert np.isfinite(out).all()
        assert np.isnan(band_average(spectra, RESPONSES, GRID)).all()

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"nan_policy": "drop"}, "nan_policy"),
            ({"where": np.ones(60)}, "must be boolean"),
            ({"weights": np.ones(59)}, "do not fit"),
            ({"dim": "w"}, "'dim' is for DataArrays"),
        ],
    )
    def test_bad_options_raise(self, options: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            band_average(SPECTRA, RESPONSES, GRID, **options)

    @pytest.mark.parametrize(
        ("spectrum", "responses"),
        [
            (np.ones(59), RESPONSES),
            (SPECTRA, RESPONSES[:, :-1]),
            (SPECTRA, RESPONSES[None]),
            (np.float64(1.0), RESPONSES),
        ],
    )
    def test_shapes_that_do_not_fit_raise(self, spectrum, responses) -> None:
        with pytest.raises(ValidationError, match="do not fit"):
            band_average(spectrum, responses, GRID)

    def test_gradients_and_backends(self) -> None:
        def average(f, r):
            return band_average(f, r, GRID[:8])

        spectra, responses = SPECTRA[:2, :8], RESPONSES[:, :8] + 0.1
        assert_grad_matches_fd(lambda f: average(f, responses), spectra)
        assert_grad_matches_fd(lambda r: average(spectra, r), responses)
        assert_traces(lambda f: average(f, responses), spectra)

    @pytest.mark.parametrize("policy", ["propagate", "omit"])
    def test_array_kinds(self, policy: str) -> None:
        strict = pytest.importorskip("array_api_strict")
        inputs = (SPECTRA, RESPONSES, GRID)
        expected = band_average(*inputs, nan_policy=policy)
        out = band_average(*map(strict.asarray, inputs), nan_policy=policy)
        assert_matches_oracle(np.asarray(out), expected, rtol=1e-13)
        single = band_average(*(np.float32(v) for v in inputs), nan_policy=policy)
        assert single.dtype == np.float32
        assert_matches_oracle(single, expected, rtol=1e-5)

    @needs_jax
    def test_jax_under_jit_with_a_mask(self) -> None:
        import jax
        import jax.numpy as jnp

        mask = jnp.asarray(RNG.random(SPECTRA.shape) < 0.7)

        def average(f, m):
            return band_average(f, RESPONSES, GRID, where=m)

        expected = average(SPECTRA, np.asarray(mask))
        out = jax.jit(average)(jnp.asarray(SPECTRA), mask)
        assert isinstance(out, jax.Array)
        assert_matches_oracle(out, expected, rtol=1e-13)


class TestLabelledBandAverage:
    @pytest.fixture
    def scene(self) -> tuple[Any, ...]:
        xr = pytest.importorskip("xarray")
        grid = xr.DataArray(GRID, dims="wavelength")
        responses = xr.DataArray(
            RESPONSES, {"band": ["a", "b", "c"]}, ("band", "wavelength")
        )
        # the wavelength is not the last dimension of the cube
        cube = xr.DataArray(
            SPECTRA.reshape(7, 1, 60).transpose(0, 2, 1),
            {"y": np.arange(7)},
            ("y", "wavelength", "x"),
            name="radiance",
        )
        return cube, responses, grid

    def test_dimensions_and_values(self, scene) -> None:
        cube, responses, grid = scene
        out = band_average(cube, responses, grid, dim="wavelength")
        assert out.dims == ("y", "x", "band")
        assert out.band.values.tolist() == ["a", "b", "c"]
        assert (out.name, out.attrs) == (None, {})
        expected = band_average(SPECTRA, RESPONSES, GRID)
        assert_matches_oracle(out.isel(x=0).values, expected, rtol=1e-14)

    def test_mask_weights_and_one_band(self, scene) -> None:
        cube, responses, grid = scene
        mask = cube > 0.7
        weights = grid * 0.0 + 1.0
        out = band_average(
            cube, responses, grid, weights=weights, where=mask, dim="wavelength"
        )
        expected = band_average(
            SPECTRA, RESPONSES, GRID, weights=np.ones(60), where=SPECTRA > 0.7
        )
        assert_matches_oracle(out.isel(x=0).values, expected, rtol=1e-14)
        one = band_average(cube, responses.isel(band=0), grid, dim="wavelength")
        assert one.dims == ("y", "x")

    def test_irradiance(self, scene) -> None:
        cube, responses, grid = scene
        solar = cube.isel(y=0, x=0, drop=True)
        out = band_irradiance(solar, responses, grid, dim="wavelength")
        assert out.dims == ("band",)

    def test_dask_stays_lazy(self, scene) -> None:
        da = pytest.importorskip("dask.array")
        cube, responses, grid = scene
        out = band_average(
            cube.chunk({"y": 2}), responses, grid, dim="wavelength", nan_policy="omit"
        )
        assert isinstance(out.data, da.Array)
        expected = band_average(cube, responses, grid, dim="wavelength")
        assert_matches_oracle(out.compute().values, expected.values, rtol=1e-13)

    def test_what_cannot_be_aligned_raises(self, scene) -> None:
        cube, responses, grid = scene
        with pytest.raises(ValidationError, match="need 'dim'"):
            band_average(cube, responses, grid)
        with pytest.raises(ValidationError, match="'responses' must be a DataArray"):
            band_average(cube, RESPONSES, grid, dim="wavelength")
        with pytest.raises(ValidationError, match="'wavelength' must be a DataArray"):
            band_average(cube, responses, GRID, dim="wavelength")
        with pytest.raises(ValidationError, match="no dimension 'lam'"):
            band_average(cube, responses, grid, dim="lam")
        with pytest.raises(ValidationError, match="already has the dimension 'band'"):
            band_average(responses, responses, grid, dim="wavelength")


class TestBandIrradiance:
    def test_georeader_on_a_1_nm_grid(self) -> None:
        # georeader drops responses below 1e-4 of the sum: measured 3.2e-7
        grid = np.arange(1500.0, 1701.0)
        centres = np.asarray([[1560.0], [1600.0], [1640.0]])
        responses = np.exp(-(((grid - centres) / 20.0) ** 2))
        solar = 1.0 + 0.2 * np.sin(grid / 7.0)
        expected = georeader_band_irradiance(solar, responses, grid)
        out = band_irradiance(solar, responses, grid)
        assert_matches_oracle(out, expected, rtol=1e-6)

    def test_only_one_spectrum(self) -> None:
        with pytest.raises(ValidationError, match="must be 1-D"):
            band_irradiance(SPECTRA, RESPONSES, GRID)


class TestSpectralDensities:
    WAVELENGTH = st.floats(200.0, 2e4)

    @given(st.floats(-1e3, 1e3), WAVELENGTH)
    def test_inverse_pair(self, density: float, wavelength: float) -> None:
        there = per_nm_to_per_cm(density, wavelength)
        back = per_cm_to_per_nm(there, 1e7 / wavelength)
        assert_matches_oracle(back, density, rtol=1e-15)

    def test_the_integral_is_conserved(self) -> None:
        wavelength = np.linspace(1500.0, 1700.0, 10_001)
        density = 1.0 + 0.5 * np.sin(wavelength / 20.0)
        wavenumber = 1e7 / wavelength
        per_cm = per_nm_to_per_cm(density, wavelength)
        # the wavenumber grid descends, so its weights are negative
        total = -trapezoid_weights(wavenumber) @ per_cm
        expected = trapezoid_weights(wavelength) @ density
        assert_matches_oracle(total, expected, rtol=1e-6)

    def test_gradients_and_backends(self) -> None:
        density = np.asarray([0.1, 0.2, 0.3])
        for function, axis in (
            (per_nm_to_per_cm, np.asarray([400.0, 1600.0, 2300.0])),
            (per_cm_to_per_nm, np.asarray([25_000.0, 6250.0, 4347.0])),
        ):
            assert_grad_matches_fd(
                lambda x, f=function, a=axis: f(x[0], a * x[1]),
                np.stack([density, np.ones(3)]),
            )
            assert_backends_agree(function, density, axis)
            assert_traces(function, density, axis)
