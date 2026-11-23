"""The kinds of array a function must accept, and the check that they agree.

JAX, xarray and `array_api_strict` are each optional here: a kind whose
library is missing is left out, so the job without JAX and xarray runs the rest.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _harness import assert_matches_oracle


__all__ = [
    "HAS_JAX",
    "HAS_TORCH",
    "HAS_XARRAY",
    "KINDS",
    "SECOND_DEVICE",
    "ArrayKind",
    "assert_backends_agree",
    "needs_jax",
    "needs_xarray",
]

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_XARRAY = importlib.util.find_spec("xarray") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_STRICT = importlib.util.find_spec("array_api_strict") is not None

#: For ``waits``: functions that call geophysix coordinates, until a release of
#: geophysix puts its constants on the input's device (geophysix#56).
SECOND_DEVICE = ("second device",)

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
needs_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray is not installed")


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayKind:
    """One kind of array: how to make it, what comes back, and how to read it.

    ``host`` takes an array back to NumPy, and checks where it was.
    """

    name: str
    make: Callable[[np.ndarray], Any]
    type: Any
    host: Callable[[Any], np.ndarray] = np.asarray

    def read(self, array: Any) -> np.ndarray:
        """The array as NumPy, after checking that it is of this kind."""
        assert isinstance(array, self.type), (
            f"{self.name} in must give {self.name} out, got {type(array).__name__}"
        )
        return self.host(array)


def _dims(array: np.ndarray) -> tuple[str, ...]:
    """Dimension names from the right, so that arrays broadcast as NumPy does."""
    return tuple(f"dim{array.ndim - axis}" for axis in range(array.ndim))


def _kinds() -> list[ArrayKind]:
    kinds = [ArrayKind("numpy", np.asarray, (np.ndarray, np.generic))]
    if HAS_STRICT:
        import array_api_strict

        strict_type = type(array_api_strict.asarray(0.0))
        kinds.append(ArrayKind("strict", array_api_strict.asarray, strict_type))
        second = array_api_strict.Device("device1")

        def on_second(array: np.ndarray) -> Any:
            return array_api_strict.asarray(array, device=second)

        def from_second(array: Any) -> np.ndarray:
            assert array.device == second, f"came back on {array.device}"
            return np.asarray(array.to_device(array_api_strict.Device("CPU_DEVICE")))

        kinds.append(ArrayKind("second device", on_second, strict_type, from_second))
    if HAS_JAX:
        import jax
        import jax.numpy as jnp

        kinds.append(ArrayKind("jax", jnp.asarray, jax.Array))
    if HAS_TORCH:
        import torch

        def detached(array: Any) -> np.ndarray:
            return array.detach().numpy()

        kinds.append(ArrayKind("torch", torch.asarray, torch.Tensor, detached))
    if HAS_XARRAY:
        import xarray as xr

        def labelled(array: np.ndarray) -> Any:
            return xr.DataArray(array, dims=_dims(np.asarray(array)))

        kinds.append(ArrayKind("xarray", labelled, xr.DataArray))
    return kinds


KINDS = _kinds()


def assert_backends_agree(
    fn: Callable[..., Any],
    *args: Any,
    rtol: float = 1e-12,
    rtol32: float = 1e-5,
    waits: tuple[str, ...] = (),
) -> None:
    """Check that every array kind gives NumPy's numbers, kind and dtype.

    For each kind in `KINDS`: the same kind comes back, float64 values agree
    with NumPy's to ``rtol``, float32 in gives float32 out and agrees to
    ``rtol32``. JAX is also checked under ``jit``.

    Args:
        fn: The function under test; it returns an array or a tuple of them.
        *args: Float64 NumPy arrays to call it with.
        rtol: Relative tolerance in float64.
        rtol32: Relative tolerance of float32 against float64.
        waits: Names of kinds on which the function raises for now, because a
            dependency does. One that stops raising fails the check, so it is
            taken off the list.

    Raises:
        AssertionError: If a check fails; the message names the kind.
    """
    arrays = [np.asarray(arg, dtype=np.float64) for arg in args]
    expected = _outputs(fn(*arrays))
    for kind in KINDS:
        if kind.name in waits:
            with pytest.raises(Exception):  # noqa: B017
                fn(*(kind.make(array) for array in arrays))
            continue
        variants = {kind.name: fn}
        if kind.name == "jax":
            import jax

            variants["jax.jit"] = jax.jit(fn)
        for what, variant in variants.items():
            for dtype, tolerance in ((np.float64, rtol), (np.float32, rtol32)):
                inputs = [kind.make(array.astype(dtype)) for array in arrays]
                outputs = _outputs(variant(*inputs))
                for got, want in zip(outputs, expected, strict=True):
                    values = kind.read(got)
                    label = f"{what} {np.dtype(dtype).name}"
                    assert values.dtype == dtype, f"{label}: {values.dtype} came back"
                    try:
                        assert_matches_oracle(values, want, rtol=tolerance)
                    except AssertionError as err:
                        raise AssertionError(f"{label}: {err}") from err


def _outputs(result: Any) -> tuple[Any, ...]:
    return result if isinstance(result, tuple) else (result,)
