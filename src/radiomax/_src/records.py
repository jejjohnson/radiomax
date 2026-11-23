"""Frozen records of parameters, registered as JAX pytrees once JAX loads.

A record is a frozen dataclass of numbers or arrays. Under JAX its fields are
the leaves of a pytree, so a record crosses ``jit`` and a coefficient can be
fitted. Importing radiomax never imports JAX, so a record is registered at
once if JAX is already loaded, and otherwise when something first imports it,
by a finder on ``sys.meta_path``. The mechanism is geophysix's
(``_src/_records.py`` at ``aed2dab``), which keeps it private.
"""

from __future__ import annotations

import dataclasses
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Any, cast


__all__ = [
    "register_record",
]

# records waiting for JAX
_PENDING: list[type] = []


def register_record[T: type](cls: T) -> T:
    """Register a frozen dataclass as a JAX pytree, now or once JAX loads.

    Without JAX the record is never registered, and works as it is.

    Args:
        cls: A frozen dataclass whose every field is a number or an array.

    Returns:
        ``cls`` itself.

    Examples:
        >>> @register_record
        ... @dataclasses.dataclass(frozen=True, slots=True)
        ... class Gain:
        ...     slope: float
        >>> Gain(2.0)
        Gain(slope=2.0)
    """
    jax = sys.modules.get("jax")
    # a jax that is still being imported has no tree_util yet
    if jax is not None and hasattr(jax, "tree_util"):
        _register(jax, cls)
        return cls
    _PENDING.append(cls)
    if not any(isinstance(finder, _JaxImportHook) for finder in sys.meta_path):
        sys.meta_path.insert(0, _JaxImportHook())
    return cls


def _register(jax: ModuleType, cls: type) -> None:
    """Register one frozen dataclass with ``jax.tree_util``."""
    names = [field.name for field in dataclasses.fields(cast("Any", cls))]
    jax.tree_util.register_dataclass(cls, data_fields=names, meta_fields=[])


class _AfterJaxLoads(importlib.abc.Loader):
    """Wrap the ``jax`` loader so the pending records register once it has run."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        while _PENDING:
            _register(module, _PENDING.pop(0))


class _JaxImportHook(importlib.abc.MetaPathFinder):
    """Finds nothing itself; when ``jax`` is imported, wraps its loader."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != "jax":
            return None
        # let the other finders locate jax, then remove this hook for good
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec("jax")
        if spec is not None and spec.loader is not None:
            spec.loader = _AfterJaxLoads(spec.loader)
        return spec
