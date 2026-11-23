"""Spectral responses, their matrices on a grid, and what the matrices do."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from pipekit_array.functions import elementwise, floating_inputs, is_dataarray, like

from radiomax._src.labelled import apply_along
from radiomax._src.radiometry.band import band_average, trapezoid_weights
from radiomax.exceptions import ValidationError


__all__ = [
    "apply_response",
    "apply_response_adjoint",
    "band_centre",
    "box_response",
    "equivalent_width",
    "gaussian_response",
    "response_coverage",
    "response_matrix",
    "sampled_response_matrix",
    "triangle_response",
]

#: FWHM / σ of a Gaussian, exactly: 2 √(2 ln 2).
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


# --- shapes ---


def gaussian_response(wavelength: Any, centre: Any, fwhm: Any) -> Any:
    """Gaussian responses with a peak of 1.

    R_b(λ) = exp(−(λ − c_b)² / (2 σ_b²)), σ = FWHM / (2 √(2 ln 2))

    Args:
        wavelength: λ, (W,).
        centre: c, the bands' centres, (B,), or one number.
        fwhm: The bands' full widths at half maximum, (B,), or one number.

    Returns:
        R, (B, W), or (W,) for one band; with DataArrays, the bands' dimensions
        and then the wavelength's.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import gaussian_response
        >>> wavelength = np.asarray([1600.0, 1605.0, 1610.0])
        >>> gaussian_response(wavelength, 1600.0, 10.0).round(6).tolist()
        [1.0, 0.5, 0.0625]
    """
    return _on_bands(_gaussian, wavelength, centre, fwhm)


def box_response(wavelength: Any, centre: Any, width: Any) -> Any:
    """Box responses: 1 within half a width of the centre, 0 outside.

    R_b(λ) = 1 where |λ − c_b| ≤ width_b / 2

    Args:
        wavelength: λ, (W,).
        centre: c, the bands' centres, (B,), or one number.
        width: The bands' full widths, (B,), or one number.

    Returns:
        R, (B, W), or (W,) for one band; with DataArrays, the bands' dimensions
        and then the wavelength's.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import box_response
        >>> box_response(np.asarray([0.0, 1.0, 2.0, 3.0]), 1.0, 2.0).tolist()
        [1.0, 1.0, 1.0, 0.0]
    """
    return _on_bands(_box, wavelength, centre, width)


def triangle_response(wavelength: Any, centre: Any, width: Any) -> Any:
    """Triangular responses, 1 at the centre and 0 half a width from it.

    R_b(λ) = max(0, 1 − 2 |λ − c_b| / width_b)

    Args:
        wavelength: λ, (W,).
        centre: c, the bands' centres, (B,), or one number.
        width: The bands' full widths at the base, (B,), or one number.

    Returns:
        R, (B, W), or (W,) for one band; with DataArrays, the bands' dimensions
        and then the wavelength's.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import triangle_response
        >>> triangle_response(np.asarray([0.0, 0.5, 1.0, 3.0]), 1.0, 2.0).tolist()
        [0.0, 0.5, 1.0, 0.0]
    """
    return _on_bands(_triangle, wavelength, centre, width)


def _on_bands(
    shape: Callable[..., Any], wavelength: Any, centre: Any, width: Any
) -> Any:
    """A response shape on every band, bands first."""
    if any(is_dataarray(value) for value in (wavelength, centre, width)):
        return shape(centre, width, wavelength)
    _, (grid, centre, width) = floating_inputs(wavelength, centre, width)
    if grid.ndim != 1 or centre.ndim > 1 or width.ndim > 1:
        msg = (
            "'wavelength' must be (W,) and the centres and widths (B,) or numbers, "
            f"got {grid.shape}, {centre.shape}, {width.shape}"
        )
        raise ValidationError(msg)
    return shape(centre[..., None], width[..., None], grid)


@elementwise()
def _gaussian(centre: Any, fwhm: Any, wavelength: Any) -> Any:
    xp, (c, f, x) = floating_inputs(centre, fwhm, wavelength)
    # z = (λ − c) / σ
    z = (x - c) * like(_FWHM_PER_SIGMA, x, xp) / f
    return xp.exp(-0.5 * z**2)


@elementwise()
def _box(centre: Any, width: Any, wavelength: Any) -> Any:
    xp, (c, w, x) = floating_inputs(centre, width, wavelength)
    inside = xp.abs(x - c) <= 0.5 * w
    return xp.where(inside, like(1.0, x, xp), like(0.0, x, xp))


@elementwise()
def _triangle(centre: Any, width: Any, wavelength: Any) -> Any:
    xp, (c, w, x) = floating_inputs(centre, width, wavelength)
    height = 1.0 - 2.0 * xp.abs(x - c) / w
    return xp.where(height > 0, height, xp.zeros_like(height))


# --- matrices ---


def sampled_response_matrix(
    responses: Any, wavelength: Any, *, weights: Any = None, dim: str | None = None
) -> Any:
    """The response matrix on the responses' own grid, by a quadrature rule.

    A_bg = w_g R_bg / Σ w R_b

    It traces and is differentiable in the responses, so a band that moves
    under a gradient uses it. A response tabulated on another grid needs
    `response_matrix`.

    Args:
        responses: R, (B, G) or (G,). Negative values count as 0.
        wavelength: λ, (G,), ascending or descending.
        weights: Quadrature weights w, (G,); the trapezoid rule by default.
        dim: The wavelength dimension, when the inputs are DataArrays.

    Returns:
        A, in the shape of the responses, rows summing to 1; a band without
        response is a row of NaN.

    Raises:
        ValidationError: If the shapes do not fit, or ``dim`` is missing for
            DataArrays or given for arrays.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import sampled_response_matrix
        >>> responses = np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        >>> grid = np.asarray([0.0, 1.0, 2.0])
        >>> sampled_response_matrix(responses, grid).round(3).tolist()
        [[0.333, 0.667, 0.0], [0.0, 0.667, 0.333]]
    """
    arrays = {"responses": responses, "wavelength": wavelength, "weights": weights}
    if any(is_dataarray(value) for value in arrays.values()):
        core = dict.fromkeys(arrays, (dim,))
        return apply_along(_sampled, arrays, core, dim=dim, output_dims=(dim,))
    _require_no_dim(dim)
    return _sampled(**arrays)


def _sampled(responses: Any, wavelength: Any, weights: Any = None) -> Any:
    """`sampled_response_matrix` on arrays."""
    xp, weighted = _weighted_responses(responses, wavelength, weights)
    total = xp.sum(weighted, axis=-1, keepdims=True)
    covered = total != 0
    safe = xp.where(covered, total, xp.ones_like(total))
    return xp.where(covered, weighted / safe, like(xp.nan, weighted, xp))


def _weighted_responses(
    responses: Any, wavelength: Any, weights: Any
) -> tuple[ModuleType, Any]:
    """w_g R_bg, with negative responses as 0."""
    if weights is None:
        weights = trapezoid_weights(wavelength)
    xp, (r, w) = floating_inputs(responses, weights, params=(wavelength,))
    if w.ndim != 1 or r.shape[-1:] != w.shape:
        msg = f"'responses' (..., G) and the grid (G,) do not fit: {r.shape}, {w.shape}"
        raise ValidationError(msg)
    return xp, xp.where(r > 0, r, xp.zeros_like(r)) * w


def response_matrix(responses: Any, table_wavelength: Any, grid: Any) -> Any:
    """The matrix of tabulated responses on any grid, as an exact integral.

    A_bg = ∫ R_b φ_g dλ / ∫ R_b dλ

    with R_b linear between the table's wavelengths and 0 outside, and φ_g the
    hat function of grid point g: 1 there, 0 at its neighbours, linear
    between. ``A @ f`` is then exactly ∫ R_b f dλ / ∫ R_b dλ for every f that
    is linear between grid points, on a grid coarser or finer than the table.

    Runs on the host, because it merges two grids; the result applies on any
    backend (`apply_response`).

    Algorithm:
        1. The sorted union u of both grids, inside both ranges.
        2. On each interval of u both R_b and the two hats of its grid cell
           are linear, so with end values R₀, R₁ and φ₀, φ₁ the integral is
           h (2 R₀ φ₀ + R₀ φ₁ + R₁ φ₀ + 2 R₁ φ₁) / 6.
        3. Sum onto the grid points, and divide by the row sum.

    A grid that covers a band partly normalises over the covered part; see
    `response_coverage`. On the table's own uniform grid a row is
    (R_{g−1} + 4 R_g + R_{g+1}) / 6 before normalisation, not R_g, so it
    differs from a sum over the table, which is georeader's convention: by at
    most 4.1e-5 relative for a spectrum of lines on Landsat 8's 1 nm tables.

    Args:
        responses: R, (B, T) or (T,). Negative values, as some published tails
            have, count as 0. Rows that were dropped from a table because they
            are small must be restored as 0, or the gap is interpolated across.
        table_wavelength: λ of the table, (T,), ascending or descending.
        grid: λ of the grid, (G,), ascending or descending, in the same unit.
            A wavenumber grid is passed as its wavelengths.

    Returns:
        A, (B, G) or (G,), float64 NumPy, columns in the grid's order and rows
        summing to 1; a DataArray when every input is one.

    Raises:
        ValidationError: If a grid is not strictly monotone with two points or
            more, a shape does not fit, a value is not finite, a band has no
            response on the grid, or only some inputs are DataArrays.

    Examples:
        A box from 1 to 3, on a coarse grid and on a fine one:

        >>> import numpy as np
        >>> from radiomax.radiometry import response_matrix
        >>> table = np.asarray([0.9, 1.0, 3.0, 3.1])
        >>> box = np.asarray([0.0, 1.0, 1.0, 0.0])
        >>> response_matrix(box, table, np.asarray([0.0, 2.0, 4.0])).round(
        ...     3
        ... ).tolist()
        [0.131, 0.737, 0.131]
        >>> fine = np.linspace(0.0, 4.0, 401)
        >>> round(float(response_matrix(box, table, fine) @ fine), 12)
        2.0
    """
    labelled = [is_dataarray(v) for v in (responses, table_wavelength, grid)]
    if any(labelled):
        if not all(labelled):
            msg = "'responses', 'table_wavelength' and 'grid' must all be DataArrays"
            raise ValidationError(msg)
        return _labelled_matrix(responses, table_wavelength, grid)
    r, table, squeeze = _host_table(responses, table_wavelength)
    nodes, order = _host_grid(grid, "grid")
    matrix = _exact_integrals(r, table, nodes)
    total = np.sum(matrix, axis=1, keepdims=True)
    empty = np.flatnonzero(total <= 0.0)
    if empty.size:
        msg = f"no response on the grid for the bands {empty.tolist()}"
        raise ValidationError(msg)
    matrix = matrix[:, order] / total
    return matrix[0] if squeeze else matrix


def response_coverage(responses: Any, table_wavelength: Any, grid: Any) -> np.ndarray:
    """The share of each band's response that a grid covers.

    ∫ R_b dλ over the grid's range, over ∫ R_b dλ

    `response_matrix` normalises over the covered part, so a share below 1
    biases the band's average; what share is enough is the caller's choice.

    Args:
        responses: R, (B, T) or (T,). Negative values count as 0.
        table_wavelength: λ of the table, (T,), ascending or descending.
        grid: λ of the grid, (G,), ascending or descending, in the same unit.

    Returns:
        The shares, (B,) or a number, in [0, 1], float64 NumPy; NaN for a band
        without any response.

    Raises:
        ValidationError: As `response_matrix`, or if an input is a DataArray.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import response_coverage
        >>> table = np.asarray([0.0, 1.0, 2.0])
        >>> triangle = np.asarray([0.0, 1.0, 0.0])
        >>> float(response_coverage(triangle, table, np.asarray([1.0, 5.0])))
        0.5
    """
    if any(is_dataarray(v) for v in (responses, table_wavelength, grid)):
        msg = "'response_coverage' takes arrays; pass the DataArrays' values"
        raise ValidationError(msg)
    r, table, squeeze = _host_table(responses, table_wavelength)
    nodes, _ = _host_grid(grid, "grid")
    low, high = max(table[0], nodes[0]), min(table[-1], nodes[-1])
    total = r @ trapezoid_weights(table)
    if low >= high:
        share = np.where(total > 0.0, 0.0, np.nan)
        return share[0] if squeeze else share
    inner = table[(table > low) & (table < high)]
    union = np.concatenate([[low], inner, [high]])
    inside = _interpolate(r, table, union) @ trapezoid_weights(union)
    safe = np.where(total > 0.0, total, 1.0)
    share = np.where(total > 0.0, inside / safe, np.nan)
    return share[0] if squeeze else share


def _labelled_matrix(responses: Any, table_wavelength: Any, grid: Any) -> Any:
    """`response_matrix` on DataArrays, as one on the bands and the grid."""
    if table_wavelength.ndim != 1 or grid.ndim != 1:
        msg = "'table_wavelength' and 'grid' must be 1-D"
        raise ValidationError(msg)
    (table_dim,), (grid_dim,) = table_wavelength.dims, grid.dims
    if table_dim not in responses.dims or responses.ndim > 2:
        msg = f"'responses' must be on {table_dim!r} and at most one band dimension"
        raise ValidationError(msg)
    bands = [name for name in responses.dims if name != table_dim]
    on_table = responses.transpose(*bands, table_dim)
    matrix = response_matrix(on_table.values, table_wavelength.values, grid.values)
    kept = {
        name: coordinate
        for name, coordinate in responses.coords.items()
        if table_dim not in coordinate.dims
    }
    coords = kept | dict(grid.coords.items())
    return type(responses)(matrix, coords=coords, dims=(*bands, grid_dim))


def _host_table(
    responses: Any, table_wavelength: Any
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Non-negative responses (B, T), their ascending table, and if B was 1-D."""
    table, order = _host_grid(table_wavelength, "table_wavelength")
    r = np.asarray(responses, dtype=np.float64)
    if r.ndim not in (1, 2) or r.shape[-1:] != table.shape:
        msg = (
            "'responses' (B, T) or (T,) and the table (T,) do not fit: "
            f"{r.shape}, {table.shape}"
        )
        raise ValidationError(msg)
    if not np.all(np.isfinite(r)):
        msg = "'responses' must be finite"
        raise ValidationError(msg)
    rows = np.atleast_2d(r)[:, order]
    return np.maximum(rows, 0.0), table, r.ndim == 1


def _host_grid(grid: Any, name: str) -> tuple[np.ndarray, Any]:
    """An ascending grid, and the index that puts results in the caller's order."""
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        msg = f"'{name}' must be 1-D with two points or more, got shape {values.shape}"
        raise ValidationError(msg)
    steps = np.diff(values)
    if not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        msg = f"'{name}' must be finite and strictly monotone"
        raise ValidationError(msg)
    if steps[0] > 0.0:
        return values, slice(None)
    return values[::-1], slice(None, None, -1)


def _interpolate(r: np.ndarray, table: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """R at nodes inside the table, linear, (B, K)."""
    return np.stack([np.interp(nodes, table, row) for row in r])


def _exact_integrals(r: np.ndarray, table: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """∫ R_b φ_g dλ, (B, G), on ascending grids."""
    low, high = max(table[0], grid[0]), min(table[-1], grid[-1])
    if low >= high:
        return np.zeros((r.shape[0], grid.size))
    # --- the union of both grids, inside both ranges ---
    union = np.union1d(table, grid)
    union = union[(union >= low) & (union <= high)]
    h = np.diff(union)
    # --- end values on each interval ---
    at_nodes = _interpolate(r, table, union)
    r0, r1 = at_nodes[:, :-1], at_nodes[:, 1:]
    # the grid cell g of each interval, and the right hat's end values
    cell = np.searchsorted(grid, union[:-1], side="right") - 1
    width = grid[cell + 1] - grid[cell]
    t0 = (union[:-1] - grid[cell]) / width
    t1 = (union[1:] - grid[cell]) / width
    # --- h (2 R₀ φ₀ + R₀ φ₁ + R₁ φ₀ + 2 R₁ φ₁) / 6, for both hats ---
    sixth = h / 6.0
    right = sixth * (r0 * (2.0 * t0 + t1) + r1 * (t0 + 2.0 * t1))
    # the two hats add to 1 on the cell
    both = 3.0 * sixth * (r0 + r1)
    left = both - right
    return np.stack(
        [
            np.bincount(cell, weights=lo, minlength=grid.size)
            + np.bincount(cell + 1, weights=hi, minlength=grid.size)
            for lo, hi in zip(left, right, strict=True)
        ]
    )


# --- what a matrix does ---


def apply_response(matrix: Any, spectrum: Any, *, dim: str | None = None) -> Any:
    """Band values of a spectrum, through a response matrix.

    y_b = Σ_g A_bg f_g

    Args:
        matrix: A, (B, G) or (G,), as `response_matrix` or
            `sampled_response_matrix` gives it.
        spectrum: f, (..., G).
        dim: The wavelength dimension, when the inputs are DataArrays.

    Returns:
        y, (..., B), or (...) for one band; DataArrays give the spectrum's
        dimensions and then the bands'.

    Raises:
        ValidationError: If the shapes do not fit, or ``dim`` is missing for
            DataArrays or given for arrays.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import apply_response
        >>> matrix = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
        >>> apply_response(matrix, np.asarray([1.0, 2.0, 4.0])).tolist()
        [1.5, 3.0]
    """
    if is_dataarray(matrix) or is_dataarray(spectrum):
        if not is_dataarray(matrix):
            msg = "'matrix' must be a DataArray when the spectrum is one"
            raise ValidationError(msg)
        bands = tuple(name for name in matrix.dims if name != dim)
        return apply_along(
            _apply_matrix,
            {"matrix": matrix, "spectrum": spectrum},
            {"matrix": (*bands, dim), "spectrum": (dim,)},
            dim=dim,
            output_dims=bands,
        )
    _require_no_dim(dim)
    return _apply_matrix(matrix, spectrum)


def apply_response_adjoint(matrix: Any, bands: Any, *, dim: str | None = None) -> Any:
    """The adjoint of `apply_response`, from bands back to the grid.

    g_g = Σ_b A_bg y_b

    It is the transpose, which reverse-mode derivatives and gradients of a
    misfit need, and not an inverse.

    Args:
        matrix: A, (B, G) or (G,).
        bands: y, (..., B), or (...) for one band.
        dim: The band dimension, when the inputs are DataArrays.

    Returns:
        g, (..., G); DataArrays give the bands' other dimensions and then the
        grid's.

    Raises:
        ValidationError: If the shapes do not fit, or ``dim`` is missing for
            DataArrays or given for arrays.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import apply_response_adjoint
        >>> matrix = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
        >>> apply_response_adjoint(matrix, np.asarray([2.0, 4.0])).tolist()
        [1.0, 3.0, 2.0]
    """
    if is_dataarray(matrix) or is_dataarray(bands):
        # with names, the adjoint sums the same product over the other dimension
        return apply_response(matrix, bands, dim=dim)
    _require_no_dim(dim)
    xp, (a, y) = floating_inputs(matrix, bands)
    if a.ndim == 1:
        return y[..., None] * a
    if a.ndim != 2 or y.shape[-1:] != a.shape[:1]:
        msg = f"'matrix' (B, G) and 'bands' (..., B) do not fit: {a.shape}, {y.shape}"
        raise ValidationError(msg)
    return xp.matmul(y, a)


def _apply_matrix(matrix: Any, spectrum: Any) -> Any:
    """`apply_response` on arrays, wavelength last."""
    xp, (a, f) = floating_inputs(matrix, spectrum)
    if a.ndim not in (1, 2) or f.shape[-1:] != a.shape[-1:]:
        msg = (
            "'matrix' (B, G) or (G,) and 'spectrum' (..., G) do not fit: "
            f"{a.shape}, {f.shape}"
        )
        raise ValidationError(msg)
    if a.ndim == 1:
        return xp.matmul(f, a)
    return xp.matmul(f, xp.permute_dims(a, (1, 0)))


def _require_no_dim(dim: str | None) -> None:
    """Refuse a dimension name for arrays that have none."""
    if dim is not None:
        msg = f"'dim' is for DataArrays, got dim={dim!r} and arrays without names"
        raise ValidationError(msg)


# --- summaries of a response ---


def band_centre(
    responses: Any, wavelength: Any, *, weights: Any = None, dim: str | None = None
) -> Any:
    """The response-weighted mean wavelength of each band.

    c_b = ∫ λ R_b dλ / ∫ R_b dλ

    Args:
        responses: R, (B, W) or (W,). Negative values count as 0.
        wavelength: λ, (W,), ascending or descending.
        weights: Quadrature weights, (W,); the trapezoid rule by default.
        dim: The wavelength dimension, when the inputs are DataArrays.

    Returns:
        c, (B,) or a number, in the wavelength's unit; NaN for a band without
        response.

    Raises:
        ValidationError: As `band_average`.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import band_centre
        >>> round(float(band_centre([0.0, 1.0, 0.0], [1.0, 2.0, 3.0])), 12)
        2.0
    """
    return band_average(wavelength, responses, wavelength, weights=weights, dim=dim)


def equivalent_width(
    responses: Any, wavelength: Any, *, weights: Any = None, dim: str | None = None
) -> Any:
    """The width of a box as high as each band's peak, with the same integral.

    ∫ R_b dλ / max R_b

    Args:
        responses: R, (B, W) or (W,). Negative values count as 0.
        wavelength: λ, (W,), ascending or descending.
        weights: Quadrature weights, (W,); the trapezoid rule by default.
        dim: The wavelength dimension, when the inputs are DataArrays.

    Returns:
        The widths, (B,) or a number, in the wavelength's unit; NaN for a band
        without response.

    Raises:
        ValidationError: If the shapes do not fit, or ``dim`` is missing for
            DataArrays or given for arrays.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import equivalent_width
        >>> wavelength = np.linspace(-1.0, 1.0, 201)
        >>> triangle = 5.0 * (1.0 - np.abs(wavelength))
        >>> round(float(equivalent_width(triangle, wavelength)), 12)
        1.0
    """
    arrays = {"responses": responses, "wavelength": wavelength, "weights": weights}
    if any(is_dataarray(value) for value in arrays.values()):
        core = dict.fromkeys(arrays, (dim,))
        return apply_along(_width, arrays, core, dim=dim, output_dims=())
    _require_no_dim(dim)
    return _width(**arrays)


def _width(responses: Any, wavelength: Any, weights: Any = None) -> Any:
    """`equivalent_width` on arrays."""
    xp, weighted = _weighted_responses(responses, wavelength, weights)
    _, (r, _) = floating_inputs(responses, weighted)
    peak = xp.max(r, axis=-1)
    covered = peak > 0
    safe = xp.where(covered, peak, xp.ones_like(peak))
    # a descending grid has negative weights
    total = xp.abs(xp.sum(weighted, axis=-1))
    return xp.where(covered, total / safe, like(xp.nan, total, xp))
