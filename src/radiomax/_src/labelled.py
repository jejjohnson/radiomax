"""Labelled arrays: a DataArray in gives a DataArray out, aligned by dimension name."""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from radiomax._src.backend import array_namespace
from radiomax.exceptions import ValidationError


__all__ = [
    "apply_along",
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
           The other arguments are passed through, and an output that does
           not depend on every input is broadcast.
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
            # positions and keywords, with a variadic argument spread out
            values = {**dict(enumerate(bound.args)), **bound.kwargs}
            labelled = [k for k, v in values.items() if is_dataarray(v)]
            if not labelled:
                return function(*args, **kwargs)
            return _apply(function, signature, values, labelled, n_outputs)

        return wrapper

    return decorate


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
        keep_attrs=False,
    )
    return result.rename(None)


def _apply(
    function: Callable[..., Any],
    signature: inspect.Signature,
    values: dict[int | str, Any],
    labelled: list[int | str],
    n_outputs: int,
) -> Any:
    """Call ``function`` through ``xarray.apply_ufunc`` on its labelled arguments."""
    xarray = sys.modules["xarray"]
    for key, value in values.items():
        if key not in labelled and _has_dimensions(value):
            msg = (
                f"'{_name(key, signature)}' has dimensions but no names, and cannot "
                f"be aligned with the DataArray '{_name(labelled[0], signature)}'; "
                "make it a DataArray too"
            )
            raise ValidationError(msg)

    def on_data(*data: Any) -> Any:
        given = values | dict(zip(labelled, data, strict=True))
        args = [value for key, value in given.items() if isinstance(key, int)]
        kwargs = {key: value for key, value in given.items() if isinstance(key, str)}
        result = function(*args, **kwargs)
        # xarray needs every output in the inputs' common shape, and one that
        # does not depend on an input lacks that input's dimensions
        shape = np.broadcast_shapes(*(array.shape for array in data))
        outputs = result if n_outputs > 1 else (result,)
        outputs = tuple(
            array_namespace(output).broadcast_to(output, shape) for output in outputs
        )
        return outputs if n_outputs > 1 else outputs[0]

    result = xarray.apply_ufunc(
        on_data,
        *(values[key] for key in labelled),
        output_core_dims=[()] * n_outputs,
        dask="allowed",
        keep_attrs=False,
    )
    if n_outputs == 1:
        return result.rename(None)
    return tuple(array.rename(None) for array in result)


def _name(key: int | str, signature: inspect.Signature) -> str:
    """The parameter at a position, or of a keyword."""
    if isinstance(key, str):
        return key
    parameters = list(signature.parameters.values())
    for parameter in parameters[:key]:
        # it takes every position from its own on
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return parameter.name
    return parameters[key].name


def _has_dimensions(value: Any) -> bool:
    """Whether a value is an array or a sequence, rather than a number or an option."""
    if isinstance(value, (list, tuple)):
        return True
    return getattr(value, "ndim", 0) > 0
