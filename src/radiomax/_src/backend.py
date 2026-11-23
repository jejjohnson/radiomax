"""Array plumbing: one namespace, dtype and device per call, on any array library.

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
    "device_of",
    "floating_inputs",
    "is_host",
    "like",
    "place",
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


def device_of(value: Any, xp: ModuleType) -> Any:
    """The device an array is on, or ``None`` if it does not say where to compute.

    Host values join any device. So do JAX's arrays: a tracer has no device,
    and a constant that JAX has not committed to one goes where it is used.

    Examples:
        >>> import array_api_strict as strict
        >>> device_of(
        ...     strict.asarray([1.0], device=strict.Device("device1")), strict
        ... )
        array_api_strict.Device('device1')
        >>> device_of(1.0, strict) is None
        True
    """
    if is_host(value) or array_api_compat.is_jax_namespace(xp):
        return None
    return array_api_compat.device(value)


def place(value: Any, xp: ModuleType, dtype: Any, device: Any) -> Any:
    """``value`` as an array of a dtype, on a device.

    An array that already belongs to the library keeps its device and, where
    the dtype matches, its identity: ``asarray`` would detach a torch tensor
    from its graph in some versions (2.5 and 2.10 among them).

    Args:
        value: A number, a sequence, a NumPy array or an array of ``xp``.
        xp: The namespace.
        dtype: The dtype of the result.
        device: Where a host value goes; ``None`` for the default device.

    Returns:
        The array.

    Examples:
        >>> import numpy as np
        >>> place([1, 2], np, np.float32, None)
        array([1., 2.], dtype=float32)
    """
    if not is_host(value):
        return xp.astype(value, dtype, copy=False)
    if device is None:
        return xp.asarray(value, dtype=dtype)
    return xp.asarray(value, dtype=dtype, device=device)


def floating_inputs(
    *values: Any, params: tuple[Any, ...] = ()
) -> tuple[ModuleType, list[Any]]:
    """The namespace of a call, and its inputs as arrays of one floating dtype.

    The dtype is the common one of the floating arrays (float32 stays
    float32). Python numbers and integer or boolean arrays adopt it; with no
    floating array, everything takes the namespace's default float. Host
    values go to the device of the other arrays.

    Args:
        *values: The array inputs: arrays, sequences or Python numbers.
        params: Record fields. They help to choose the namespace and the
            device, so a traced JAX scalar in a record selects JAX, and never
            decide the dtype.

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
    devices = (device_of(value, xp) for value in (*values, *params))
    device = next((found for found in devices if found is not None), None)
    floating = []
    for value in values:
        if _is_number(value):
            continue
        dtype = xp.asarray(value).dtype if is_host(value) else value.dtype
        if xp.isdtype(dtype, "real floating"):
            floating.append(dtype)
        elif not xp.isdtype(dtype, ("bool", "integral")):
            msg = f"inputs must be real numbers, got dtype {dtype}"
            raise ValidationError(msg)
    dtype = xp.result_type(*floating) if floating else xp.asarray(0.0).dtype
    return xp, [place(value, xp, dtype, device) for value in values]


def like(value: Any, ref: Any, xp: ModuleType) -> Any:
    """``value`` (a number or an array) with ``ref``'s dtype, on ``ref``'s device.

    Examples:
        >>> import numpy as np
        >>> like(0.1, np.zeros(2, dtype=np.float32), np).dtype
        dtype('float32')
    """
    return place(value, xp, ref.dtype, device_of(ref, xp))


def _is_number(value: Any) -> bool:
    """A Python real number, which adopts the dtype of the arrays.

    A NumPy scalar carries its own dtype, though ``np.float64`` is a float.
    """
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)
