"""Split-window surface temperature, over the sea and over land."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Literal

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise, is_dataarray
from radiomax._src.numerics import full_turn, to_radians
from radiomax._src.retrievals.coefficients import unpack_coefficients
from radiomax.exceptions import ValidationError


__all__ = [
    "split_window_features",
    "split_window_lst",
    "split_window_sst",
]

Form = Literal["quadratic", "generalised"]


def split_window_sst(
    t11: Any, t12: Any, view_zenith: Any, coefficients: Any, *, degrees: bool = True
) -> Any:
    """Sea-surface temperature from two thermal bands and the view angle.

    T = a₀ + a₁ T₁₁ + a₂ (T₁₁ − T₁₂) + a₃ (sec θᵥ − 1)

    The difference of the two bands grows with the water vapour in the path,
    and the secant with its length.

    Args:
        t11: Brightness temperature near 11 µm, T₁₁, K.
        t12: Brightness temperature near 12 µm, T₁₂, K.
        view_zenith: View zenith θᵥ, degrees (radians with
            ``degrees=False``).
        coefficients: (a₀, a₁, a₂, a₃): a sequence, an array with them on the
            last axis, or a DataArray with them on ``"feature"``. The unit of
            a₀ decides between K and °C.
        degrees: Whether the view zenith is in degrees.

    Returns:
        T, in the unit of a₀; NaN where the view zenith is 90° or more.

    Raises:
        ValidationError: If there are not four coefficients.

    Examples:
        Coefficients for illustration only:

        >>> from radiomax.retrievals import split_window_sst
        >>> round(
        ...     float(split_window_sst(290.0, 288.0, 60.0, (1.0, 1.0, 2.0, 0.5))), 9
        ... )
        295.5
    """
    a0, a1, a2, a3 = unpack_coefficients(coefficients, 4)
    return _sst(t11, t12, view_zenith, a0, a1, a2, a3, degrees=degrees)


def split_window_lst(
    t11: Any,
    t12: Any,
    emissivity_mean: Any,
    emissivity_difference: Any,
    water_vapour: Any,
    coefficients: Any,
    *,
    form: Form,
) -> Any:
    """Land-surface temperature from two thermal bands and their emissivities.

    T = ∑_f a_f φ_f(T₁₁, T₁₂, ε̄, Δε, W)

    with the features φ of `split_window_features`.

    Args:
        t11: Brightness temperature near 11 µm, T₁₁, K.
        t12: Brightness temperature near 12 µm, T₁₂, K.
        emissivity_mean: Mean emissivity of the two bands, ε̄.
        emissivity_difference: Emissivity at 11 µm minus that at 12 µm, Δε.
        water_vapour: Column water vapour W, cm of precipitable water (the
            ``"generalised"`` form does not use it).
        coefficients: The F coefficients a: a sequence, an array with them on
            the last axis (so they may differ by pixel), or a DataArray with
            them on ``"feature"``.
        form: Which features, ``"quadratic"`` (F = 8) or ``"generalised"``
            (F = 7).

    Returns:
        T, K; NaN where the mean emissivity is not positive, for the
        ``"generalised"`` form, which divides by it.

    Raises:
        ValidationError: If the form is unknown, or there are not F
            coefficients.

    Examples:
        >>> from radiomax.retrievals import split_window_lst
        >>> only_t11 = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> float(
        ...     split_window_lst(
        ...         300.0, 298.0, 0.97, 0.0, 2.0, only_t11, form="quadratic"
        ...     )
        ... )
        300.0
    """
    values = unpack_coefficients(coefficients, _feature_count(form))
    features = _features(
        t11, t12, emissivity_mean, emissivity_difference, water_vapour, form
    )
    # T = ∑_f a_f φ_f
    total = _times(values[0], features[0])
    for coefficient, feature in zip(values[1:], features[1:], strict=True):
        total = total + _times(coefficient, feature)
    return total


def split_window_features(
    t11: Any,
    t12: Any,
    emissivity_mean: Any,
    emissivity_difference: Any,
    water_vapour: Any,
    *,
    form: Form,
) -> Any:
    """The features whose weighted sum is a split-window land temperature.

    With T̄ = (T₁₁ + T₁₂)/2 and D = (T₁₁ − T₁₂)/2,

    "quadratic":   (1, T₁₁, 2D, (2D)², 1 − ε̄, W(1 − ε̄), Δε, WΔε), F = 8
    "generalised": (1, T̄, T̄(1 − ε̄)/ε̄, T̄Δε/ε̄², D, D(1 − ε̄)/ε̄, DΔε/ε̄²), F = 7

    They are the design matrix of a fit, and radtrax's, so coefficients fitted
    there apply here. Neither is attributed to a published algorithm: that has
    not been checked against the primary sources.

    Args:
        t11: Brightness temperature near 11 µm, T₁₁, K.
        t12: Brightness temperature near 12 µm, T₁₂, K.
        emissivity_mean: Mean emissivity of the two bands, ε̄.
        emissivity_difference: Emissivity at 11 µm minus that at 12 µm, Δε.
        water_vapour: Column water vapour W, cm of precipitable water (the
            ``"generalised"`` form does not use it).
        form: ``"quadratic"`` or ``"generalised"``.

    Returns:
        φ, with the features on the last axis, or on the dimension
        ``"feature"`` of a DataArray.

    Raises:
        ValidationError: If the form is unknown.

    Examples:
        >>> from radiomax.retrievals import split_window_features
        >>> split_window_features(
        ...     300.0, 298.0, 1.0, 0.0, 2.0, form="generalised"
        ... ).tolist()
        [1.0, 299.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    """
    _feature_count(form)
    features = _features(
        t11, t12, emissivity_mean, emissivity_difference, water_vapour, form
    )
    return _stack_last(features, dim="feature")


@elementwise()
def _sst(
    t11: Any,
    t12: Any,
    view_zenith: Any,
    a0: Any,
    a1: Any,
    a2: Any,
    a3: Any,
    *,
    degrees: bool,
) -> Any:
    xp, (t11, t12, view, a0, a1, a2, a3) = floating_inputs(
        t11, t12, view_zenith, a0, a1, a2, a3
    )
    quarter = like(full_turn(degrees=degrees) / 4.0, view, xp)
    valid = xp.abs(view) < quarter
    # the inner where keeps the unused branch finite, and so its gradient
    safe = to_radians(xp.where(valid, view, xp.zeros_like(view)), xp, degrees=degrees)
    path = 1.0 / xp.cos(safe) - 1.0
    # T = a₀ + a₁ T₁₁ + a₂ (T₁₁ − T₁₂) + a₃ (sec θᵥ − 1)
    temperature = a0 + a1 * t11 + a2 * (t11 - t12) + a3 * path
    return xp.where(valid, temperature, like(xp.nan, temperature, xp))


def _feature_count(form: str) -> int:
    """F, after checking the form."""
    counts = {"quadratic": 8, "generalised": 7}
    if form not in counts:
        msg = f"'form' must be 'quadratic' or 'generalised', got {form!r}"
        raise ValidationError(msg)
    return counts[form]


def _features(t11: Any, t12: Any, mean: Any, diff: Any, water: Any, form: str) -> Any:
    """The features, one array each, of one shape."""
    if form == "quadratic":
        return _quadratic(t11, t12, mean, diff, water)
    return _generalised(t11, t12, mean, diff)


@elementwise(n_outputs=8)
def _quadratic(t11: Any, t12: Any, mean: Any, diff: Any, water: Any) -> Any:
    xp, (t11, t12, mean, diff, water) = floating_inputs(t11, t12, mean, diff, water)
    split, dark = t11 - t12, 1.0 - mean
    terms = (xp.ones_like(t11), t11, split, split**2, dark, water * dark)
    return tuple(_broadcast((*terms, diff, water * diff), xp))


@elementwise(n_outputs=7)
def _generalised(t11: Any, t12: Any, mean: Any, diff: Any) -> Any:
    xp, (t11, t12, mean, diff) = floating_inputs(t11, t12, mean, diff)
    valid = mean > 0
    # the inner where keeps the unused branch finite, and so its gradient
    safe = xp.where(valid, mean, xp.ones_like(mean))
    nan = like(xp.nan, mean, xp)
    # (1 − ε̄) / ε̄ and Δε / ε̄²
    dark = xp.where(valid, (1.0 - safe) / safe, nan)
    tilt = xp.where(valid, diff / safe**2, nan)
    half_sum, half_split = 0.5 * (t11 + t12), 0.5 * (t11 - t12)
    terms = [xp.ones_like(t11)]
    for part in (half_sum, half_split):
        terms += [part, part * dark, part * tilt]
    return tuple(_broadcast(terms, xp))


def _broadcast(arrays: Any, xp: ModuleType) -> list[Any]:
    return list(xp.broadcast_arrays(*arrays))


@elementwise()
def _times(coefficient: Any, feature: Any) -> Any:
    _, (coefficient, feature) = floating_inputs(coefficient, feature)
    return coefficient * feature


def _stack_last(arrays: Any, *, dim: str) -> Any:
    """Stack arrays of one shape on a new last axis, or DataArrays on a dimension."""
    if is_dataarray(arrays[0]):
        # a DataArray exists, so xarray is imported
        stacked = sys.modules["xarray"].concat(arrays, dim=dim)
        return stacked.transpose(..., dim)
    xp, values = floating_inputs(*arrays)
    return xp.stack(values, axis=-1)
