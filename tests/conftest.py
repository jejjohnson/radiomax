"""Shared pytest configuration.

JAX is optional. When it is importable, 64-bit mode is switched on so that
float64 reference comparisons are meaningful; when it is not, nothing else is
skipped here (tests that need JAX or xarray skip themselves).
"""

from __future__ import annotations


try:
    import jax
except ImportError:  # pragma: no cover - exercised in the job without JAX
    pass
else:
    jax.config.update("jax_enable_x64", True)
