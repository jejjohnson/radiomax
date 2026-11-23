"""Guards on the public API surface.

These tests are deliberately about *contracts*, not behaviour: if a symbol is
removed from ``__all__`` or stops being importable from the top level, that is
a breaking change and should fail loudly here.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import radiomax


SUBMODULES = [
    "_typing",
    "exceptions",
]


def test_version_is_a_dotted_string() -> None:
    assert isinstance(radiomax.__version__, str)
    major, minor, patch = radiomax.__version__.split(".")[:3]
    assert all(part.isdigit() for part in (major, minor, patch))


def test_all_is_sorted_and_unique() -> None:
    assert radiomax.__all__ == sorted(radiomax.__all__)
    assert len(radiomax.__all__) == len(set(radiomax.__all__))


@pytest.mark.parametrize("name", radiomax.__all__)
def test_every_exported_name_is_importable(name: str) -> None:
    assert hasattr(radiomax, name), f"{name} is in __all__ but not defined"


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports_cleanly(module: str) -> None:
    assert importlib.import_module(f"radiomax.{module}") is not None


def test_no_unexpected_submodules() -> None:
    """A new module must be added to SUBMODULES (and given an API doc page)."""
    found = {
        info.name for info in pkgutil.iter_modules(radiomax.__path__) if not info.ispkg
    }
    assert found == set(SUBMODULES)


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodules_declare_all(module: str) -> None:
    imported = importlib.import_module(f"radiomax.{module}")
    assert hasattr(imported, "__all__"), f"radiomax.{module} is missing __all__"
    assert imported.__all__ == sorted(imported.__all__)


def test_package_is_typed() -> None:
    """PEP 561 marker must ship so downstream type checkers see annotations."""
    marker = Path(radiomax.__path__[0]) / "py.typed"
    assert marker.is_file(), f"missing PEP 561 marker at {marker}"
