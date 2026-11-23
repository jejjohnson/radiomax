"""Coefficients of an empirical formula, one value each, on any kind of array."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit_array.functions import is_dataarray

from radiomax.exceptions import ValidationError


__all__ = [
    "unpack_coefficients",
]


def unpack_coefficients(
    coefficients: Any, count: int | None, *, dim: str = "feature"
) -> tuple[Any, ...]:
    """The coefficients of a formula as a tuple, one number or array each.

    Args:
        coefficients: A sequence of numbers or arrays, or an array whose last
            axis, or a DataArray whose dimension ``dim``, runs over them.
        count: How many the formula needs, or ``None`` for at least one.
        dim: The name of that dimension.

    Returns:
        The coefficients, in order.

    Raises:
        ValidationError: If there are too few or too many, or a DataArray
            lacks ``dim``.

    Examples:
        >>> import numpy as np
        >>> unpack_coefficients([1.0, 2.0], 2)
        (1.0, 2.0)
        >>> per_pixel = np.ones((4, 5, 3))
        >>> [c.shape for c in unpack_coefficients(per_pixel, 3)]
        [(4, 5), (4, 5), (4, 5)]
    """
    if isinstance(coefficients, Sequence):
        values = tuple(coefficients)
    elif is_dataarray(coefficients):
        if dim not in coefficients.dims:
            msg = (
                f"the coefficients have no dimension {dim!r}, only {coefficients.dims}"
            )
            raise ValidationError(msg)
        # the coordinate of a single feature would conflict between coefficients
        values = tuple(
            coefficients.isel({dim: index}, drop=True)
            for index in range(coefficients.sizes[dim])
        )
    elif getattr(coefficients, "ndim", 0) > 0:
        values = tuple(
            coefficients[..., index] for index in range(coefficients.shape[-1])
        )
    else:
        msg = "the coefficients must be a sequence or an array with one axis or more"
        raise ValidationError(msg)
    if not values or (count is not None and len(values) != count):
        wanted = "at least 1" if count is None else count
        msg = f"the formula needs {wanted} coefficients, got {len(values)}"
        raise ValidationError(msg)
    return values
