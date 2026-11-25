"""radiomax on the array backend of `pipekit_array.arrays`, which tests itself."""

from __future__ import annotations

import subprocess
import sys

import array_api_strict as strict
import numpy as np
import pytest

from radiomax import BackendError
from radiomax.geometry import solar_declination


class Unknown:
    """An array-like of no known library, as a pandas Series is."""

    shape = (2,)
    ndim = 1
    dtype = np.dtype(np.float64)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.zeros(2)


@pytest.mark.parametrize("value", [Unknown(), object(), "text", {"a": 1}])
def test_an_unknown_array_raises_and_is_never_numpy(value: object) -> None:
    with pytest.raises(BackendError, match="not an array of a known library"):
        solar_declination(value)


def test_two_libraries_in_one_call_raise() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from radiomax.geometry import air_mass_factor

    with pytest.raises(BackendError, match=r"array_api_strict, jax\.numpy"):
        air_mass_factor(jnp.zeros(2), strict.zeros(2))


def test_values_that_are_not_real_numbers_raise() -> None:
    with pytest.raises(ValueError, match="real numbers"):
        solar_declination(np.asarray([1j]))


def test_importing_the_package_imports_neither_jax_nor_xarray() -> None:
    code = (
        "import sys, numpy as np, radiomax.geometry, radiomax.radiometry;"
        "import radiomax.thermal, radiomax.retrievals;"
        "radiomax.geometry.solar_declination(np.zeros(2));"
        "sys.exit(bool({'jax', 'xarray'} & set(sys.modules)))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
