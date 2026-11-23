"""Averages of a spectrum over a band's response, with masks and missing samples."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any, Literal

from radiomax._src.backend import (
    device_of,
    floating_inputs,
    is_host,
    like,
    place,
)
from radiomax._src.labelled import apply_along, is_dataarray
from radiomax.exceptions import ValidationError


__all__ = [
    "band_average",
    "band_irradiance",
    "trapezoid_weights",
]


def trapezoid_weights(wavelength: Any) -> Any:
    """The weights of the composite trapezoid rule on a grid, uniform or not.

    w₁ = (λ₂ − λ₁)/2, wᵢ = (λᵢ₊₁ − λᵢ₋₁)/2, w_W = (λ_W − λ_{W−1})/2

    so that Σ w f is ∫ f dλ for f linear between grid points. A descending
    grid gives negative weights, whose sign cancels in an average.

    Args:
        wavelength: λ, (W,), in any unit.

    Returns:
        w, (W,), in the same unit.

    Raises:
        ValidationError: If the grid is not 1-D.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import trapezoid_weights
        >>> trapezoid_weights(np.asarray([0.0, 1.0, 3.0])).tolist()
        [0.5, 1.5, 1.0]
    """
    xp, (grid,) = floating_inputs(wavelength)
    if grid.ndim != 1:
        msg = f"'wavelength' must be 1-D, got shape {grid.shape}"
        raise ValidationError(msg)
    half = 0.5 * (grid[1:] - grid[:-1])
    zero = xp.zeros_like(grid[:1])
    # each interval gives half its width to the two nodes that bound it
    return xp.concat([half, zero]) + xp.concat([zero, half])


def band_average(
    spectrum: Any,
    responses: Any,
    wavelength: Any,
    *,
    weights: Any = None,
    where: Any = None,
    nan_policy: Literal["propagate", "omit"] = "propagate",
    dim: str | None = None,
) -> Any:
    """The response-weighted average of a spectrum, per band.

    f̄_b = Σ w f R_b / Σ w R_b ≈ ∫ f R_b dλ / ∫ R_b dλ

    Both sums run over the kept samples only, so the average renormalises
    where samples are left out. The spectrum and the responses are on one
    grid; a response tabulated on another grid needs a response matrix.

    The result has the common dtype of the inputs, so a float32 cube stays
    float32 when the responses and the grid are float32 too.

    Args:
        spectrum: f, (..., W).
        responses: R, (B, W), or (W,) for one band. Negative values, as some
            published tails have, count as 0.
        wavelength: λ, (W,), ascending or descending, in any unit.
        weights: Quadrature weights w, (W,); the trapezoid rule by default. Any
            rule's weights fit, such as functax's.
        where: A boolean mask that broadcasts against the spectrum; samples
            where it is false are left out.
        nan_policy: ``"propagate"`` lets a NaN among the kept samples make
            every band's average NaN, also where the response is 0;
            ``"omit"`` leaves NaN samples out.
        dim: The wavelength dimension, when the inputs are DataArrays.

    Returns:
        f̄, (..., B), or (...) for one band; NaN where no response or no sample
        is left. DataArrays give the spectrum's dimensions and then the
        responses', without ``dim``.

    Raises:
        ValidationError: If a shape does not fit, ``where`` is not boolean,
            ``nan_policy`` is unknown, or ``dim`` is missing for DataArrays
            or given for arrays.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import band_average
        >>> wavelength = np.asarray([1.0, 2.0, 3.0, 4.0])
        >>> responses = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        >>> spectrum = np.asarray([1.0, 4.0, 5.0, np.nan])
        >>> band_average(spectrum, responses, wavelength).tolist()
        [nan, nan]
        >>> band_average(
        ...     spectrum, responses, wavelength, nan_policy="omit"
        ... ).tolist()
        [3.0, 5.0]
    """
    if nan_policy not in ("propagate", "omit"):
        msg = f"'nan_policy' must be 'propagate' or 'omit', got {nan_policy!r}"
        raise ValidationError(msg)
    arrays = {
        "spectrum": spectrum,
        "responses": responses,
        "wavelength": wavelength,
        "weights": weights,
        "where": where,
    }
    if any(is_dataarray(value) for value in arrays.values()):
        return _labelled_average(arrays, nan_policy, dim)
    if dim is not None:
        msg = f"'dim' is for DataArrays, got dim={dim!r} and arrays without names"
        raise ValidationError(msg)
    return _average(nan_policy=nan_policy, **arrays)


def band_irradiance(
    solar_spectrum: Any,
    responses: Any,
    wavelength: Any,
    *,
    weights: Any = None,
    dim: str | None = None,
) -> Any:
    """The solar irradiance of each band at 1 AU.

    E₀ = ∫ E☉ R dλ / ∫ R dλ

    the `band_average` of one solar spectrum, as `reflectance` takes it.

    Args:
        solar_spectrum: E☉ at 1 AU, (W,), W m⁻² per spectral unit.
        responses: R, (B, W), or (W,) for one band.
        wavelength: λ, (W,), ascending or descending, in any unit.
        weights: Quadrature weights, (W,); the trapezoid rule by default.
        dim: The wavelength dimension, when the inputs are DataArrays.

    Returns:
        E₀, (B,), in the spectrum's unit.

    Raises:
        ValidationError: If the solar spectrum is not 1-D, or as
            `band_average`.

    Examples:
        >>> import numpy as np
        >>> from radiomax.radiometry import band_irradiance
        >>> wavelength = np.linspace(1500.0, 1700.0, 201)
        >>> response = np.exp(-(((wavelength - 1600.0) / 30.0) ** 2))
        >>> solar = 0.25 - 1e-4 * (wavelength - 1600.0)
        >>> round(float(band_irradiance(solar, response, wavelength)), 12)
        0.25
    """
    if getattr(solar_spectrum, "ndim", 1) != 1:
        msg = f"'solar_spectrum' must be 1-D, got shape {solar_spectrum.shape}"
        raise ValidationError(msg)
    return band_average(solar_spectrum, responses, wavelength, weights=weights, dim=dim)


def _average(
    spectrum: Any,
    responses: Any,
    wavelength: Any,
    weights: Any = None,
    where: Any = None,
    *,
    nan_policy: str,
) -> Any:
    """`band_average` on arrays, wavelength last."""
    if weights is None:
        weights = trapezoid_weights(wavelength)
    xp, (f, r, w) = floating_inputs(
        spectrum, responses, weights, params=(wavelength, where)
    )
    size = w.shape[-1] if w.ndim == 1 else -1
    if r.ndim not in (1, 2) or r.shape[-1] != size or f.shape[-1:] != (size,):
        msg = (
            "'spectrum' (..., W), 'responses' (B, W) or (W,) and the grid (W,) do "
            f"not fit: {f.shape}, {r.shape}, {w.shape}"
        )
        raise ValidationError(msg)
    # c_bn = wₙ R_bn
    combined = xp.where(r > 0, r, xp.zeros_like(r)) * w
    kept = _kept(f, where, nan_policy, xp)
    if kept is None:
        samples = f
        # Σ w R, the same for every spectrum
        denominator = xp.sum(combined, axis=-1)
    else:
        # 0 in place of a dropped sample: it adds nothing and has no gradient
        samples = xp.where(kept, f, xp.zeros_like(f))
        counted = xp.where(kept, xp.ones_like(f), xp.zeros_like(f))
        # Σ w R over the kept samples
        denominator = _contract(counted, combined, xp)
    # Σ w f R
    numerator = _contract(samples, combined, xp)
    covered = denominator != 0
    safe = xp.where(covered, denominator, xp.ones_like(denominator))
    return xp.where(covered, numerator / safe, like(xp.nan, numerator, xp))


def _kept(f: Any, where: Any, nan_policy: str, xp: ModuleType) -> Any:
    """The samples to keep, in the spectrum's shape, or ``None`` for all."""
    kept = None
    if where is not None:
        dtype = xp.asarray(where).dtype if is_host(where) else where.dtype
        if not xp.isdtype(dtype, "bool"):
            msg = f"'where' must be boolean, got dtype {dtype}"
            raise ValidationError(msg)
        mask = place(where, xp, dtype, device_of(f, xp))
        kept = xp.broadcast_to(mask, f.shape)
    if nan_policy == "omit":
        present = ~xp.isnan(f)
        kept = present if kept is None else kept & present
    return kept


def _contract(samples: Any, combined: Any, xp: ModuleType) -> Any:
    """Σₙ f_…n c_bn, (..., W) and (B, W) or (W,), to (..., B) or (...)."""
    if combined.ndim == 1:
        return xp.matmul(samples, combined)
    return xp.matmul(samples, xp.permute_dims(combined, (1, 0)))


def _labelled_average(arrays: dict[str, Any], nan_policy: str, dim: str | None) -> Any:
    """`band_average` on DataArrays, along ``dim``."""
    responses = arrays["responses"]
    if not is_dataarray(responses):
        msg = "'responses' must be a DataArray when another input is one"
        raise ValidationError(msg)
    band_dims = tuple(name for name in responses.dims if name != dim)
    core_dims = dict.fromkeys(arrays, (dim,)) | {"responses": (*band_dims, dim)}
    return apply_along(
        functools.partial(_average, nan_policy=nan_policy),
        arrays,
        core_dims,
        dim=dim,
        output_dims=band_dims,
    )
