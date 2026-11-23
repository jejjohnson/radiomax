"""Array plumbing: one namespace and one floating dtype per call, on any array library.

A function computes in the namespace of its inputs: NumPy in gives NumPy out,
JAX in gives JAX out, and the same for every array-API library that
`array_api_compat` knows. Only functions of the array API standard are called
on a namespace. No array library but NumPy is imported here.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import array_api_compat
import array_api_compat.numpy as numpy_namespace
import numpy as np

from radiomax.exceptions import BackendError, ValidationError


__all__ = [
    "array_namespace",
    "floating_inputs",
    "is_host",
    "like",
]

# Values that live on the host and join whatever library the others are from.
_HOST_TYPES = (bool, int, float, complex, np.ndarray, np.generic, list, tuple)


def is_host(value: Any) -> bool:
    """Say whether a value is ``None``, a Python number, a sequence or NumPy's.

    Examples:
        >>> is_host(1.0), is_host(None), is_host([1.0, 2.0])
        (True, True, True)
        >>> is_host(object())
        False
    """
    return value is None or isinstance(value, _HOST_TYPES)


def array_namespace(*values: Any) -> ModuleType:
    """The array namespace that a computation on ``values`` runs in.

    Python numbers, ``None``, sequences and NumPy arrays join the library of
    the other arrays; on their own they resolve to NumPy.

    Args:
        *values: Arguments of the function being dispatched.

    Returns:
        The namespace, such as ``array_api_compat.numpy`` or ``jax.numpy``.

    Raises:
        BackendError: If a value is not an array of a known library, or if two
            libraries other than NumPy meet in one call.

    Examples:
        >>> import numpy as np
        >>> array_namespace(np.zeros(3), 2.0, None).__name__
        'array_api_compat.numpy'
        >>> array_namespace("text")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        radiomax.exceptions.BackendError: str is not an array of a known library
    """
    found: dict[str, ModuleType] = {}
    for value in values:
        if is_host(value):
            continue
        try:
            namespace = array_api_compat.array_namespace(value)
        except TypeError as err:
            msg = f"{type(value).__name__} is not an array of a known library"
            raise BackendError(msg) from err
        found[namespace.__name__] = namespace
    if len(found) > 1:
        msg = f"arrays of several libraries in one call: {', '.join(sorted(found))}"
        raise BackendError(msg)
    if found:
        return next(iter(found.values()))
    return numpy_namespace


def floating_inputs(
    *values: Any, params: tuple[Any, ...] = ()
) -> tuple[ModuleType, list[Any]]:
    """The namespace of a call, and its inputs as arrays of one floating dtype.

    The dtype is the common one of the floating arrays (float32 stays
    float32). Python numbers and integer or boolean arrays adopt it; with no
    floating array, everything takes the namespace's default float.

    Args:
        *values: The array inputs: arrays, sequences or Python numbers.
        params: Record fields. They help to choose the namespace, so a traced
            JAX scalar in a record selects JAX, and never decide the dtype.

    Returns:
        The namespace, and ``values`` as arrays of one floating dtype.

    Raises:
        BackendError: As `array_namespace`.
        ValidationError: If an input is complex, a date or not numeric.

    Examples:
        >>> import numpy as np
        >>> xp, (x, y) = floating_inputs(np.zeros(2, dtype=np.float32), 1.5)
        >>> x.dtype, y.dtype
        (dtype('float32'), dtype('float32'))
        >>> xp, (counts,) = floating_inputs(np.asarray([1, 2], dtype=np.uint16))
        >>> counts.dtype
        dtype('float64')
    """
    xp = array_namespace(*values, *params)
    arrays = [value if _is_number(value) else xp.asarray(value) for value in values]
    floating = []
    for array in arrays:
        if _is_number(array):
            continue
        if xp.isdtype(array.dtype, "real floating"):
            floating.append(array.dtype)
        elif not xp.isdtype(array.dtype, ("bool", "integral")):
            msg = f"inputs must be real numbers, got dtype {array.dtype}"
            raise ValidationError(msg)
    default = xp.asarray(0.0).dtype
    dtype = xp.result_type(*floating) if floating else default
    return xp, [xp.asarray(array, dtype=dtype) for array in arrays]


def like(value: Any, ref: Any, xp: ModuleType) -> Any:
    """``value`` (a number or a 0-d array) as an array of ``ref``'s dtype.

    Examples:
        >>> import numpy as np
        >>> like(0.1, np.zeros(2, dtype=np.float32), np).dtype
        dtype('float32')
    """
    return xp.asarray(value, dtype=ref.dtype)


def _is_number(value: Any) -> bool:
    """A Python real number, which adopts the dtype of the arrays.

    A NumPy scalar carries its own dtype, though ``np.float64`` is a float.
    """
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)
