"""Responses, their exact and sampled matrices, and what the matrices do."""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from _backends import needs_jax
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import radtrax_values
from hypothesis import assume, given, settings, strategies as st

from radiomax import ValidationError
from radiomax.radiometry import (
    apply_response,
    apply_response_adjoint,
    band_average,
    band_centre,
    box_response,
    equivalent_width,
    gaussian_response,
    response_coverage,
    response_matrix,
    sampled_response_matrix,
    trapezoid_weights,
    triangle_response,
)


TABLE = np.arange(2000.0, 2501.0)
CENTRES = np.linspace(2190.0, 2310.0, 5)
NARROW = gaussian_response(TABLE, CENTRES, 8.5)
SHAPES = (gaussian_response, box_response, triangle_response)


def _integral(r: np.ndarray, table: np.ndarray, f: np.ndarray, grid: np.ndarray):
    """∫ R f / ∫ R on the union of both grids, without hat functions."""
    union = np.union1d(table, grid)
    low, high = max(table[0], grid[0]), min(table[-1], grid[-1])
    union = union[(union >= low) & (union <= high)]
    r, f = np.interp(union, table, r), np.interp(union, grid, f)
    h = np.diff(union)
    products = 2 * r[:-1] * f[:-1] + r[:-1] * f[1:] + r[1:] * f[:-1] + 2 * r[1:] * f[1:]
    return (h * products).sum() / 6.0 / (r @ trapezoid_weights(union))


def _lines(seed: int, low: float, high: float):
    """A spectrum of 600 random narrow lines."""
    rng = np.random.default_rng(seed)
    centre, depth = rng.uniform(low, high, 600), rng.uniform(0.1, 0.9, 600)
    half_width = rng.uniform(0.02, 0.1, 600)

    def spectrum(wavelength: np.ndarray) -> np.ndarray:
        tau = np.zeros_like(wavelength)
        for c, d, w in zip(centre, depth, half_width, strict=True):
            near = np.abs(wavelength - c) < 60.0 * w
            tau[near] += d / (1.0 + ((wavelength[near] - c) / w) ** 2)
        return np.exp(-tau)

    return spectrum


class TestShapes:
    def test_gaussian_is_half_at_half_its_fwhm(self) -> None:
        values = gaussian_response(np.asarray([-5.0, 0.0, 5.0]), 0.0, 10.0)
        assert_matches_oracle(values, [0.5, 1.0, 0.5], rtol=1e-15)

    def test_box_and_triangle_edges(self) -> None:
        wavelength = np.asarray([-1.5, -1.0, -0.5, 0.0, 1.0])
        assert box_response(wavelength, 0.0, 2.0).tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
        triangle = triangle_response(wavelength, 0.0, 2.0)
        assert triangle.tolist() == [0.0, 0.0, 0.5, 1.0, 0.0]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_dimensions(self, shape) -> None:
        centres, widths = np.asarray([2100.0, 2200.0, 2300.0]), np.full(3, 20.0)
        assert shape(TABLE, centres, widths).shape == (3, 501)
        assert shape(TABLE, centres, 20.0).shape == (3, 501)
        assert shape(TABLE, 2100.0, 20.0).shape == (501,)
        single = shape(np.float32(TABLE), np.float32(centres), 20.0)
        assert single.dtype == np.float32
        with pytest.raises(ValidationError, match=r"must be \(W,\)"):
            shape(TABLE[None], 2100.0, 20.0)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_array_kinds(self, shape) -> None:
        strict = pytest.importorskip("array_api_strict")
        inputs = (TABLE[::50], CENTRES, np.full(5, 60.0))
        out = shape(*map(strict.asarray, inputs))
        assert_matches_oracle(np.asarray(out), shape(*inputs), rtol=1e-14)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_dataarrays_put_the_bands_first(self, shape) -> None:
        xr = pytest.importorskip("xarray")
        grid = xr.DataArray(TABLE, dims="wavelength")
        centres = xr.DataArray(CENTRES, {"band": list("abcde")}, ("band",))
        out = shape(grid, centres, 20.0)
        assert out.dims == ("band", "wavelength")
        assert_matches_oracle(out.values, shape(TABLE, CENTRES, 20.0), rtol=0.0)

    def test_gradients(self) -> None:
        grid = np.linspace(-1.0, 1.0, 5) + 0.03

        def moved(shape, x):
            return shape(grid, x[0], 2.0 + x[1])

        for shape in (gaussian_response, triangle_response):
            assert_grad_matches_fd(lambda x, s=shape: moved(s, x), np.zeros(2))
        assert_traces(
            lambda c: gaussian_response(grid, c, 2.0), np.asarray([0.0, 0.1, 0.2])
        )


class TestExactMatrix:
    @settings(max_examples=60)
    @given(st.integers(0, 10_000), st.integers(3, 7), st.integers(2, 9), st.booleans())
    def test_rational_arithmetic(self, seed, n_table, n_grid, descending) -> None:
        # gate G1-response: uneven grids, coarser and finer than the table
        rng = np.random.default_rng(seed)
        table = np.cumsum(rng.integers(1, 64, n_table) / 32.0)
        low = rng.uniform(-1.0, 1.0)
        grid = low + np.cumsum(rng.integers(1, 64, n_grid) / 32.0)
        response = rng.integers(0, 16, n_table) / 16.0 + 1 / 16
        f = rng.integers(-16, 16, n_grid) / 4.0
        low, high = max(table[0], grid[0]), min(table[-1], grid[-1])
        assume(low < high)

        def at(x: Fraction, xs: np.ndarray, ys: np.ndarray) -> Fraction:
            k = min(
                max(int(np.searchsorted(xs, float(x), "right")) - 1, 0), len(xs) - 2
            )
            a, b = Fraction(float(xs[k])), Fraction(float(xs[k + 1]))
            y0, y1 = Fraction(float(ys[k])), Fraction(float(ys[k + 1]))
            return y0 + (y1 - y0) * (x - a) / (b - a)

        nodes = sorted(
            {Fraction(float(v)) for v in (*table, *grid) if low <= v <= high}
        )
        numerator = denominator = Fraction(0)
        for u, v in itertools.pairwise(nodes):
            r0, r1 = at(u, table, response), at(v, table, response)
            f0, f1 = at(u, grid, f), at(v, grid, f)
            numerator += (v - u) * (2 * r0 * f0 + r0 * f1 + r1 * f0 + 2 * r1 * f1) / 6
            denominator += (v - u) * (r0 + r1) / 2
        step = -1 if descending else 1
        matrix = response_matrix(response[::step], table[::step], grid[::step])
        value = matrix @ f[::step]
        scale = np.abs(f).max() + 1.0
        assert abs(value - float(numerator / denominator)) <= 1e-12 * scale

    @pytest.mark.parametrize("spacing", [0.01, 0.3, 1.0, 7.4, 25.0])
    @pytest.mark.parametrize("order", [1, -1])
    def test_exact_at_every_spacing(self, spacing: float, order: int) -> None:
        grid = np.arange(2000.0, 2500.0 + 1e-9, spacing)
        f = _lines(0, 2000.0, 2500.0)(grid)
        matrix = response_matrix(NARROW, TABLE, grid[::order])
        expected = [_integral(row, TABLE, f, grid) for row in NARROW]
        assert_matches_oracle(matrix @ f[::order], expected, rtol=1e-12)
        assert_matches_oracle(matrix.sum(axis=1), np.ones(5), rtol=1e-13)
        if spacing <= 1.0:
            # every grid point inside the band has a weight
            assert np.all(matrix[0, np.abs(grid[::order] - CENTRES[0]) < 10.0] > 0.0)

    def test_a_table_in_descending_order(self) -> None:
        grid = np.arange(2000.0, 2500.0, 3.7)
        expected = response_matrix(NARROW, TABLE, grid)
        flipped = response_matrix(NARROW[:, ::-1], TABLE[::-1], grid)
        assert_matches_oracle(flipped, expected, rtol=0.0)

    @pytest.mark.parametrize(
        ("band", "low", "high"), [("b6", 1480, 1760), ("b7", 2050, 2370)]
    )
    def test_landsat_on_a_line_by_line_grid(self, band, low, high) -> None:
        # what radtrax's projection got wrong by up to 2.6 % (radtrax#322);
        # the trapezoid error of the grid is left, 1.3e-7 and 4.8e-7 over 20 seeds
        stored = radtrax_values()
        table = stored[f"landsat_{band}_wavelength_nm"]
        response = stored[f"landsat_{band}_response"]
        spectrum = _lines(1, low, high)
        fine = np.arange(table[0], table[-1], 0.002)
        weight = np.interp(fine, table, np.maximum(response, 0.0))
        reference = (weight * spectrum(fine)).sum() / weight.sum()
        # 10⁷ / ν descends
        grid = 1e7 / np.arange(1e7 / high - 20.0, 1e7 / low + 20.0, 0.01)
        value = response_matrix(response, table, grid) @ spectrum(grid)
        assert_matches_oracle(value, reference, rtol=1e-6)

    @pytest.mark.parametrize(
        ("band", "low", "high"), [("b6", 1480, 1760), ("b7", 2050, 2370)]
    )
    def test_the_difference_from_georeader_on_the_table(self, band, low, high) -> None:
        # georeader sums over the table: the difference the docstring states
        stored = radtrax_values()
        table = stored[f"landsat_{band}_wavelength_nm"]
        response = np.maximum(stored[f"landsat_{band}_response"], 0.0)
        matrix = response_matrix(response, table, table)
        differences = []
        for seed in range(20):
            f = _lines(seed, low, high)(table)
            node_sum = (response @ f) / response.sum()
            differences.append(abs(matrix @ f / node_sum - 1.0))
        assert 1e-6 < max(differences) < 1e-4

    def test_the_mass_matrix_on_the_tables_own_grid(self) -> None:
        table, response = np.arange(6.0), np.asarray([0.0, 1.0, 4.0, 2.0, 1.0, 0.0])
        padded = np.concatenate([[0.0], response, [0.0]])
        rows = (padded[:-2] + 4.0 * padded[1:-1] + padded[2:]) / 6.0
        matrix = response_matrix(response, table, table)
        assert_matches_oracle(matrix, rows / rows.sum(), rtol=1e-14)

    def test_one_band_and_partial_cover(self) -> None:
        one = response_matrix(NARROW[0], TABLE, TABLE[::7])
        assert_matches_oracle(
            one, response_matrix(NARROW, TABLE, TABLE[::7])[0], rtol=0
        )
        half = response_matrix(NARROW[0], TABLE, np.arange(CENTRES[0], 2400.0, 2.0))
        assert_matches_oracle(half.sum(), 1.0, rtol=1e-14)

    def test_negative_tails_count_as_zero(self) -> None:
        tails = np.where(NARROW < 1e-3, -0.01, NARROW)
        clipped = np.where(NARROW < 1e-3, 0.0, NARROW)
        grid = TABLE[::3]
        assert_matches_oracle(
            response_matrix(tails, TABLE, grid),
            response_matrix(clipped, TABLE, grid),
            rtol=0.0,
        )

    def test_a_band_without_response_raises_and_names_it(self) -> None:
        responses = NARROW.copy()
        responses[3] = 0.0
        with pytest.raises(ValidationError, match=r"bands \[3\]"):
            response_matrix(responses, TABLE, TABLE)
        with pytest.raises(ValidationError, match=r"bands \[0, 1, 2, 3, 4\]"):
            response_matrix(NARROW, TABLE, np.asarray([3000.0, 3100.0]))

    @pytest.mark.parametrize(
        ("responses", "table", "grid", "match"),
        [
            (NARROW, TABLE, TABLE[:1], "two points"),
            (NARROW, TABLE, TABLE[None], "must be 1-D"),
            (NARROW, TABLE, np.asarray([1.0, 3.0, 2.0]), "strictly monotone"),
            (NARROW, TABLE, np.asarray([1.0, np.nan]), "strictly monotone"),
            (NARROW[:, :-1], TABLE, TABLE, "do not fit"),
            (NARROW[None], TABLE, TABLE, "do not fit"),
            (NARROW * np.nan, TABLE, TABLE, "must be finite"),
        ],
    )
    def test_bad_input_raises(self, responses, table, grid, match) -> None:
        with pytest.raises(ValidationError, match=match):
            response_matrix(responses, table, grid)

    def test_dataarrays(self) -> None:
        xr = pytest.importorskip("xarray")
        band = list("abcde")
        responses = xr.DataArray(
            NARROW.T, {"band": band, "fwhm": ("band", np.full(5, 8.5))}, ("lam", "band")
        )
        table = xr.DataArray(TABLE, dims="lam")
        nu = np.linspace(4100.0, 4600.0, 40)
        grid = xr.DataArray(1e7 / nu, {"nu": nu}, ("nu",))
        out = response_matrix(responses, table, grid)
        assert out.dims == ("band", "nu")
        assert set(out.coords) == {"band", "fwhm", "nu"}
        expected = response_matrix(NARROW, TABLE, 1e7 / nu)
        assert_matches_oracle(out.values, expected, rtol=0.0)
        one = response_matrix(responses.isel(band=0), table, grid)
        assert one.dims == ("nu",)
        with pytest.raises(ValidationError, match="must all be DataArrays"):
            response_matrix(responses, table, 1e7 / nu)
        with pytest.raises(ValidationError, match="must be 1-D"):
            response_matrix(responses, responses, grid)
        with pytest.raises(ValidationError, match="must be on 'lam'"):
            response_matrix(responses.rename(lam="other"), table, grid)


class TestCoverage:
    def test_half_whole_and_none(self) -> None:
        symmetric = triangle_response(TABLE, 2250.0, 100.0)
        assert_matches_oracle(
            response_coverage(symmetric, TABLE, np.asarray([2250.0, 2400.0])),
            0.5,
            rtol=1e-14,
        )
        assert response_coverage(symmetric, TABLE, TABLE) == 1.0
        assert response_coverage(symmetric, TABLE, np.asarray([1.0, 2.0])) == 0.0
        assert np.isnan(response_coverage(0.0 * symmetric, TABLE, TABLE))
        assert np.isnan(response_coverage(0.0 * symmetric, TABLE, [1.0, 2.0]))

    def test_bands_and_order(self) -> None:
        grid = np.arange(2200.0, 2290.0)
        share = response_coverage(NARROW, TABLE, grid)
        assert share.shape == (5,)
        assert np.all((share >= 0.0) & (share <= 1.0))
        assert_matches_oracle(
            share,
            response_coverage(NARROW[:, ::-1], TABLE[::-1], grid[::-1]),
            rtol=1e-13,
        )

    def test_dataarrays_are_refused(self) -> None:
        xr = pytest.importorskip("xarray")
        with pytest.raises(ValidationError, match="takes arrays"):
            response_coverage(NARROW, xr.DataArray(TABLE), TABLE)


class TestSampledMatrix:
    def test_radtraxs_gaussian_instrument(self) -> None:
        stored = radtrax_values()
        wavelength = 1e7 / stored["response_wavenumber_cm"]
        responses = gaussian_response(
            wavelength, stored["response_centre_nm"], stored["response_fwhm_nm"]
        )
        matrix = sampled_response_matrix(responses, wavelength)
        assert_matches_oracle(matrix, stored["response_matrix"], rtol=1e-11)

    def test_the_band_average_is_the_matrix(self) -> None:
        f = _lines(3, 2000.0, 2500.0)(TABLE)
        matrix = sampled_response_matrix(NARROW, TABLE)
        expected = band_average(f, NARROW, TABLE)
        assert_matches_oracle(apply_response(matrix, f), expected, rtol=1e-13)
        weights = np.ones(501)
        matrix = sampled_response_matrix(NARROW, TABLE, weights=weights)
        expected = band_average(f, NARROW, TABLE, weights=weights)
        assert_matches_oracle(apply_response(matrix, f), expected, rtol=1e-13)

    def test_close_to_the_exact_one_on_the_table(self) -> None:
        # the trapezoid's O(h² R″ f″): a smooth spectrum, 1 nm, 8.5 nm bands
        f = 1.0 + 0.3 * np.sin(TABLE / 37.0)
        sampled = apply_response(sampled_response_matrix(NARROW, TABLE), f)
        exact = apply_response(response_matrix(NARROW, TABLE, TABLE), f)
        assert_matches_oracle(sampled, exact, rtol=1e-4)

    def test_an_empty_band_is_nan_and_shapes_are_checked(self) -> None:
        responses = np.stack([NARROW[0], 0.0 * NARROW[0]])
        with np.errstate(divide="raise", invalid="raise"):
            matrix = sampled_response_matrix(responses, TABLE)
        assert np.isfinite(matrix[0]).all()
        assert np.isnan(matrix[1]).all()
        with pytest.raises(ValidationError, match="do not fit"):
            sampled_response_matrix(NARROW, TABLE[:-1])
        with pytest.raises(ValidationError, match="'dim' is for DataArrays"):
            sampled_response_matrix(NARROW, TABLE, dim="w")

    def test_gradients_through_a_moving_band(self) -> None:
        grid = np.linspace(2240.0, 2260.0, 21)
        f = 1.0 + 0.5 * np.sin(grid)

        def band(x: Any) -> Any:
            responses = gaussian_response(grid, 2250.0 + x[0], 8.0 + x[1])
            return apply_response(sampled_response_matrix(responses, grid), f)

        assert_grad_matches_fd(band, np.zeros(2))

    def test_dataarrays(self) -> None:
        xr = pytest.importorskip("xarray")
        responses = xr.DataArray(NARROW, {"band": list("abcde")}, ("band", "lam"))
        out = sampled_response_matrix(
            responses, xr.DataArray(TABLE, dims="lam"), dim="lam"
        )
        assert out.dims == ("band", "lam")
        assert_matches_oracle(
            out.values, sampled_response_matrix(NARROW, TABLE), rtol=1e-14
        )


class TestApply:
    MATRIX = response_matrix(NARROW, TABLE, TABLE[::10])

    def test_adjoint(self) -> None:
        n_bands, n_grid = self.MATRIX.shape
        assert_adjoint(
            lambda f: apply_response(self.MATRIX, f),
            lambda y: apply_response_adjoint(self.MATRIX, y),
            n_grid,
            n_bands,
        )
        row = self.MATRIX[0]
        assert_adjoint(
            lambda f: np.atleast_1d(apply_response(row, f)),
            lambda y: apply_response_adjoint(row, y[0]),
            n_grid,
            1,
        )

    def test_batches(self) -> None:
        cube = np.random.default_rng(0).random((4, 3, self.MATRIX.shape[1]))
        out = apply_response(self.MATRIX, cube)
        assert out.shape == (4, 3, 5)
        assert_matches_oracle(out[1, 2], self.MATRIX @ cube[1, 2], rtol=1e-14)
        assert apply_response(self.MATRIX[0], cube).shape == (4, 3)
        assert apply_response_adjoint(self.MATRIX, out).shape == cube.shape
        assert apply_response_adjoint(self.MATRIX[0], out[..., 0]).shape == cube.shape

    def test_array_kinds(self) -> None:
        strict = pytest.importorskip("array_api_strict")
        f = np.random.default_rng(0).random((2, self.MATRIX.shape[1]))
        matrix = strict.asarray(self.MATRIX)
        out = apply_response(matrix, strict.asarray(f))
        assert_matches_oracle(np.asarray(out), f @ self.MATRIX.T, rtol=1e-13)
        back = apply_response_adjoint(matrix, out)
        assert_matches_oracle(
            np.asarray(back), np.asarray(out) @ self.MATRIX, rtol=1e-13
        )

    @needs_jax
    def test_jax_gradient_is_the_adjoint(self) -> None:
        import jax
        import jax.numpy as jnp

        f = jnp.asarray(np.random.default_rng(0).random(self.MATRIX.shape[1]))
        y = jnp.arange(5.0)
        _, pullback = jax.vjp(lambda x: apply_response(self.MATRIX, x), f)
        adjoint = apply_response_adjoint(self.MATRIX, y)
        assert_matches_oracle(pullback(y)[0], adjoint, rtol=1e-13)
        assert_traces(lambda x: apply_response(self.MATRIX, x), jnp.stack([f, 2 * f]))

    def test_dataarrays_and_their_adjoint(self) -> None:
        xr = pytest.importorskip("xarray")
        grid = TABLE[::10]
        matrix = xr.DataArray(self.MATRIX, {"band": list("abcde")}, ("band", "lam"))
        cube = xr.DataArray(
            np.random.default_rng(0).random((grid.size, 3)), dims=("lam", "x")
        )
        out = apply_response(matrix, cube, dim="lam")
        assert out.dims == ("x", "band")
        assert_matches_oracle(out.values, cube.values.T @ self.MATRIX.T, rtol=1e-13)
        back = apply_response_adjoint(matrix, out, dim="band")
        assert back.dims == ("x", "lam")
        assert_matches_oracle(back.values, out.values @ self.MATRIX, rtol=1e-13)
        with pytest.raises(ValidationError, match="'matrix' must be a DataArray"):
            apply_response(self.MATRIX, cube, dim="lam")

    def test_shapes_that_do_not_fit_raise(self) -> None:
        with pytest.raises(ValidationError, match="do not fit"):
            apply_response(self.MATRIX, np.ones(3))
        with pytest.raises(ValidationError, match="do not fit"):
            apply_response_adjoint(self.MATRIX, np.ones(3))
        with pytest.raises(ValidationError, match="'dim' is for DataArrays"):
            apply_response(self.MATRIX, np.ones(51), dim="w")
        with pytest.raises(ValidationError, match="'dim' is for DataArrays"):
            apply_response_adjoint(self.MATRIX, np.ones(5), dim="w")


class TestSummaries:
    def test_centres_and_widths_of_the_shapes(self) -> None:
        assert_matches_oracle(band_centre(NARROW, TABLE), CENTRES, rtol=1e-12)
        triangle = triangle_response(TABLE, CENTRES, 40.0)
        assert_matches_oracle(
            equivalent_width(triangle, TABLE), np.full(5, 20.0), rtol=1e-12
        )
        # a Gaussian's integral over its peak is σ √(2π)
        sigma = 8.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        width = np.full(5, sigma * np.sqrt(2.0 * np.pi))
        assert_matches_oracle(equivalent_width(7.0 * NARROW, TABLE), width, rtol=1e-12)

    def test_grid_order_and_empty_bands(self) -> None:
        flipped = NARROW[:, ::-1], TABLE[::-1]
        assert_matches_oracle(band_centre(*flipped), CENTRES, rtol=1e-12)
        assert_matches_oracle(
            equivalent_width(*flipped), equivalent_width(NARROW, TABLE), rtol=1e-12
        )
        empty = np.zeros(501)
        assert np.isnan(equivalent_width(empty, TABLE))
        assert np.isnan(band_centre(empty, TABLE))
        assert band_centre([0.0, 1.0, 0.0], [1.0, 2.0, 3.0]) == 2.0
        with pytest.raises(ValidationError, match="'dim' is for DataArrays"):
            equivalent_width(NARROW, TABLE, dim="w")

    def test_dataarrays(self) -> None:
        xr = pytest.importorskip("xarray")
        responses = xr.DataArray(NARROW, {"band": list("abcde")}, ("band", "lam"))
        grid = xr.DataArray(TABLE, dims="lam")
        centre = band_centre(responses, grid, dim="lam")
        width = equivalent_width(responses, grid, dim="lam")
        assert centre.dims == width.dims == ("band",)
        assert_matches_oracle(centre.values, CENTRES, rtol=1e-12)

    @needs_jax
    def test_jax(self) -> None:
        import jax
        import jax.numpy as jnp

        out = jax.jit(lambda r: equivalent_width(r, jnp.asarray(TABLE)))(NARROW)
        assert_matches_oracle(out, equivalent_width(NARROW, TABLE), rtol=1e-12)
