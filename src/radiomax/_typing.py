"""Shared type aliases.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+). Array arguments are typed loosely because
a function computes in the array namespace of its inputs, so arrays of any
array-API library, DataArrays and Python floats are all accepted.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "ArrayLike",
]


type ArrayLike = Any
"""A float, or an array of any array-API library (NumPy, JAX, CuPy, …) or xarray.

Functions return an array of the same kind and floating dtype as their inputs.
"""
