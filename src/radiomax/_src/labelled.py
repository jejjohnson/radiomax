"""Functions that consume a named dimension, on DataArrays.

Elementwise functions go through ``pipekit_array.functions.elementwise``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any

from pipekit_array.functions import is_dataarray

from radiomax.exceptions import ValidationError


__all__ = [
    "apply_along",
]


def apply_along(
    function: Callable[..., Any],
    arguments: Mapping[str, Any],
    core_dims: Mapping[str, tuple[str | None, ...]],
    *,
    dim: str | None,
    output_dims: tuple[str | None, ...],
) -> Any:
    """Call a function that consumes one named dimension on DataArrays.

    Args:
        function: Takes the arguments' data as keywords, each with its core
            dimensions last.
        arguments: The arguments by name; one that is ``None`` is left out.
        core_dims: The core dimensions of each argument, ``dim`` last.
        dim: The dimension that is consumed, or ``None`` if the caller gave
            none.
        output_dims: The core dimensions of the result.

    Returns:
        A DataArray without a name or attributes.

    Raises:
        ValidationError: If ``dim`` is missing, an argument lacks it or is not
            a DataArray, or one already has an output dimension.

    Examples:
        >>> apply_along(abs, {"x": 1.0}, {"x": (None,)}, dim=None, output_dims=())
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        radiomax.exceptions.ValidationError: DataArrays need 'dim', …
    """
    xarray = sys.modules.get("xarray")
    if dim is None or xarray is None:
        msg = "DataArrays need 'dim', the name of the dimension to average over"
        raise ValidationError(msg)
    given = {name: value for name, value in arguments.items() if value is not None}
    for name, value in given.items():
        if not is_dataarray(value):
            msg = f"'{name}' must be a DataArray when another input is one"
            raise ValidationError(msg)
        if dim not in value.dims:
            msg = f"'{name}' has no dimension {dim!r}, only {value.dims}"
            raise ValidationError(msg)
        loop = [d for d in value.dims if d not in core_dims[name]]
        clash = [d for d in loop if d in output_dims and d != dim]
        if clash:
            msg = f"'{name}' already has the dimension {clash[0]!r} of the result"
            raise ValidationError(msg)

    def on_data(*data: Any) -> Any:
        return function(**dict(zip(given, data, strict=True)))

    result = xarray.apply_ufunc(
        on_data,
        *given.values(),
        input_core_dims=[core_dims[name] for name in given],
        output_core_dims=[output_dims],
        dask="allowed",
        # for the coordinates: False would drop their attributes too
        keep_attrs="drop_conflicts",
    )
    result = result.rename(None)
    result.attrs = {}
    return result
