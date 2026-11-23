"""Records become JAX pytrees, whichever of JAX and radiomax is imported first."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from _backends import HAS_JAX, needs_jax


CHECK = """
    import jax
    from radiomax.thermal import PlanckLinear
    leaves = jax.tree_util.tree_leaves(PlanckLinear(1.0, 2.0, 3.0, 4.0))
    assert leaves == [1.0, 2.0, 3.0, 4.0], leaves
"""


def _run(code: str) -> None:
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_importing_radiomax_does_not_import_jax() -> None:
    _run("import sys, radiomax.thermal; assert 'jax' not in sys.modules")


@needs_jax
@pytest.mark.slow
def test_jax_imported_first() -> None:
    _run(CHECK)


@needs_jax
@pytest.mark.slow
def test_radiomax_imported_first() -> None:
    _run("    import radiomax.thermal" + CHECK)


@needs_jax
def test_the_hook_removes_itself() -> None:
    _run(
        """
        import sys
        import radiomax.thermal
        import jax
        names = [type(finder).__name__ for finder in sys.meta_path]
        # geophysix has its own hook, which it removes the same way
        assert "_JaxImportHook" not in names, names
        """
    )


@pytest.mark.skipif(HAS_JAX, reason="JAX is installed")
def test_without_jax_a_record_is_a_plain_dataclass() -> None:
    from radiomax.thermal import PlanckLinear

    assert PlanckLinear(1.0, 2.0).bc2 == 1.0
