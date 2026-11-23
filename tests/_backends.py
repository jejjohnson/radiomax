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
    "HAS_XARRAY",
    "KINDS",
    "ArrayKind",
    "assert_backends_agree",
    "needs_jax",
    "needs_xarray",
]

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_XARRAY = importlib.util.find_spec("xarray") is not None
HAS_STRICT = importlib.util.find_spec("array_api_strict") is not None

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
needs_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray is not installed")


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayKind:
    """One kind of array: how to make it, what comes back, and how to read it."""

    name: str
    make: Callable[[np.ndarray], Any]
    type: Any

    def read(self, array: Any) -> np.ndarray:
        """The array as NumPy, after checking that it is of this kind."""
        assert isinstance(array, self.type), (
            f"{self.name} in must give {self.name} out, got {type(array).__name__}"
        )
        return np.asarray(array)


def _dims(array: np.ndarray) -> tuple[str, ...]:
    """Dimension names from the right, so that arrays broadcast as NumPy does."""
    return tuple(f"dim{array.ndim - axis}" for axis in range(array.ndim))


def _kinds() -> list[ArrayKind]:
    kinds = [ArrayKind("numpy", np.asarray, (np.ndarray, np.generic))]
    if HAS_STRICT:
        import array_api_strict

        strict_type = type(array_api_strict.asarray(0.0))
        kinds.append(ArrayKind("strict", array_api_strict.asarray, strict_type))
    if HAS_JAX:
        import jax
        import jax.numpy as jnp

        kinds.append(ArrayKind("jax", jnp.asarray, jax.Array))
    if HAS_XARRAY:
        import xarray as xr

        def labelled(array: np.ndarray) -> Any:
            return xr.DataArray(array, dims=_dims(np.asarray(array)))

        kinds.append(ArrayKind("xarray", labelled, xr.DataArray))
    return kinds


KINDS = _kinds()


def assert_backends_agree(
    fn: Callable[..., Any], *args: Any, rtol: float = 1e-12, rtol32: float = 1e-5
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

    Raises:
        AssertionError: If a check fails; the message names the kind.
    """
    arrays = [np.asarray(arg, dtype=np.float64) for arg in args]
    expected = _outputs(fn(*arrays))
    for kind in KINDS:
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
