"""Labelled arrays: a DataArray in gives a DataArray out, aligned by dimension name."""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable
from typing import Any

from radiomax.exceptions import ValidationError


__all__ = [
    "elementwise",
    "is_dataarray",
]


def is_dataarray(value: Any) -> bool:
    """Say whether a value is an xarray DataArray, without importing xarray.

    A DataArray can only exist once xarray is imported.

    Examples:
        >>> is_dataarray(1.0)
        False
    """
    xarray = sys.modules.get("xarray")
    return xarray is not None and isinstance(value, xarray.DataArray)


def elementwise[**P, R](
    *, n_outputs: int = 1
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Let an elementwise function take and return DataArrays.

    Algorithm:
        1. With no DataArray among the arguments, call the function as it is.
        2. Otherwise hand every DataArray to ``xarray.apply_ufunc``, which
           aligns them by dimension name and calls the function on their data.
           The other arguments are passed through.
        3. Return DataArrays with the inputs' coordinates, and without a name
           or attributes, because the quantity and its unit have changed.

    An array without dimension names cannot be aligned with one that has
    them, so it is refused, unless it has no dimensions.

    Args:
        n_outputs: How many arrays the function returns; above 1, a tuple.

    Returns:
        The decorator.

    Raises:
        ValidationError: (the decorated function) if a DataArray meets an
            array with dimensions but no names.

    Examples:
        >>> import numpy as np
        >>> @elementwise()
        ... def twice(x):
        ...     return 2.0 * x
        >>> twice(np.asarray([1.0, 2.0])).tolist()
        [2.0, 4.0]
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = signature.bind(*args, **kwargs)
            labelled = [k for k, v in bound.arguments.items() if is_dataarray(v)]
            if not labelled:
                return function(*args, **kwargs)
            return _apply(function, bound, labelled, n_outputs)

        return wrapper

    return decorate


def _apply(
    function: Callable[..., Any],
    bound: inspect.BoundArguments,
    labelled: list[str],
    n_outputs: int,
) -> Any:
    """Call ``function`` through ``xarray.apply_ufunc`` on its labelled arguments."""
    xarray = sys.modules["xarray"]
    for name, value in bound.arguments.items():
        if name not in labelled and _has_dimensions(value):
            msg = (
                f"'{name}' has dimensions but no names, and cannot be aligned with "
                f"the DataArray '{labelled[0]}'; make it a DataArray too"
            )
            raise ValidationError(msg)

    def on_data(*data: Any) -> Any:
        bound.arguments.update(zip(labelled, data, strict=True))
        return function(*bound.args, **bound.kwargs)

    result = xarray.apply_ufunc(
        on_data,
        *(bound.arguments[name] for name in labelled),
        output_core_dims=[()] * n_outputs,
        dask="allowed",
        keep_attrs=False,
    )
    if n_outputs == 1:
        return result.rename(None)
    return tuple(array.rename(None) for array in result)


def _has_dimensions(value: Any) -> bool:
    """Whether a value is an array or a sequence, rather than a number or an option."""
    if isinstance(value, (list, tuple)):
        return True
    return getattr(value, "ndim", 0) > 0
