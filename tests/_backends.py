"""The kinds of array a function must accept, and what a test needs to run.

The kinds, and the check that they agree, are pipekit-array's
(`pipekit_array.testing`). JAX, xarray, dask, torch and `array_api_strict` are each
optional: a kind whose library is missing is left out, so the job without JAX and
xarray runs the rest.
"""

from __future__ import annotations

import functools
import importlib.util

import pytest
from pipekit_array import testing


__all__ = [
    "HAS_JAX",
    "HAS_TORCH",
    "KINDS",
    "assert_backends_agree",
    "needs_jax",
    "needs_xarray",
]

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_XARRAY = importlib.util.find_spec("xarray") is not None

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
needs_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray is not installed")

KINDS = testing.array_kinds()

#: Every radiomax function is elementwise, so it takes DataArrays too.
assert_backends_agree = functools.partial(testing.assert_backends_agree, labelled=True)
