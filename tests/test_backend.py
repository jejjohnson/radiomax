"""The array namespace and dtype of a call."""

from __future__ import annotations

import subprocess
import sys

import array_api_compat.numpy as numpy_namespace
import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import given, strategies as st

from radiomax import BackendError, ValidationError
from radiomax._src.backend import array_namespace, floating_inputs, is_host, like


class Unknown:
    """An array-like of no known library, as a torch tensor is without compat."""

    shape = (2,)
    ndim = 1
    dtype = np.dtype(np.float64)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.zeros(2)


def affine(values, slope, offset=0.0):
    """The toy function: sin(slope · values) + offset, through the backend."""
    xp, (values, slope, offset) = floating_inputs(values, slope, offset)
    return xp.sin(slope * values) + offset


@pytest.mark.parametrize(
    "value", [1, 1.0, True, None, [1.0], (1.0, 2.0), np.zeros(2), np.float64(1.0)]
)
def test_host_values_alone_resolve_to_numpy(value: object) -> None:
    assert is_host(value)
    assert array_namespace(value) is numpy_namespace


def test_no_arguments_resolve_to_numpy() -> None:
    assert array_namespace() is numpy_namespace


def test_numpy_joins_the_other_library() -> None:
    assert array_namespace(np.zeros(2), strict.asarray([1.0])) is strict
    assert array_namespace(strict.asarray([1.0]), 2.0, None) is strict


def test_jax_wins_over_numpy_and_meets_no_third_library() -> None:
    jnp = pytest.importorskip("jax.numpy")
    assert array_namespace(np.zeros(2), jnp.zeros(2)) is jnp
    with pytest.raises(BackendError, match=r"array_api_strict, jax\.numpy"):
        array_namespace(jnp.zeros(2), strict.asarray([1.0]))


@pytest.mark.parametrize("value", [Unknown(), object(), "text", {"a": 1}])
def test_an_unknown_array_raises_and_is_never_numpy(value: object) -> None:
    with pytest.raises(BackendError, match="not an array of a known library"):
        array_namespace(np.zeros(2), value)


def test_float32_stays_float32_next_to_numbers_and_integers() -> None:
    x = np.zeros(2, dtype=np.float32)
    _, arrays = floating_inputs(x, 2, 1.5, True, np.asarray([1, 2]), np.asarray(True))
    assert [a.dtype for a in arrays] == [np.float32] * 6


def test_float32_and_float64_give_float64() -> None:
    _, arrays = floating_inputs(np.float32([1.0]), np.float64([1.0]))
    assert [a.dtype for a in arrays] == [np.float64] * 2


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint16, np.int64, np.bool_])
def test_integers_become_the_default_float(dtype: type) -> None:
    _, (x,) = floating_inputs(np.asarray([0, 1], dtype=dtype))
    assert x.dtype == np.float64


def test_numbers_alone_give_the_default_float() -> None:
    _, (x, y) = floating_inputs(1, 2.5)
    assert x.dtype == y.dtype == np.float64
    assert x.shape == ()


def test_a_numpy_scalar_carries_its_dtype() -> None:
    _, (x, y) = floating_inputs(np.float32([1.0]), np.float64(1.0))
    assert x.dtype == y.dtype == np.float64


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1j]),
        3j,
        np.asarray(["2020-01-01"], dtype="datetime64[D]"),
        np.asarray(["a"]),
        np.asarray([object()]),
    ],
)
def test_values_that_are_not_real_numbers_raise(value: object) -> None:
    with pytest.raises(ValidationError, match="real numbers"):
        floating_inputs(value, 1.0)


def test_params_choose_the_namespace_and_never_the_dtype() -> None:
    record = strict.asarray(1.0, dtype=strict.float64)
    xp, (x,) = floating_inputs(np.float32([1.0]), params=(record, 2.0))
    assert xp is strict
    assert x.dtype == strict.float32


def test_like_takes_the_dtype_of_the_reference() -> None:
    assert like(0.1, np.float32([1.0]), np).dtype == np.float32
    assert like(0.1, strict.asarray([1.0]), strict).dtype == strict.float64


def test_the_toy_function_runs_on_the_standard_only() -> None:
    x = strict.asarray([0.1, 0.2])
    out = affine(x, strict.asarray(2, dtype=strict.int32), 1.0)
    assert out.dtype == strict.float64
    expected = np.sin(2.0 * np.asarray([0.1, 0.2])) + 1.0
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-15)


@given(
    st.lists(st.floats(-10.0, 10.0), min_size=1, max_size=5),
    st.sampled_from([np.float32, np.float64]),
)
def test_every_input_comes_back_in_one_dtype(values: list[float], dtype: type) -> None:
    x = np.asarray(values, dtype=dtype)
    _, arrays = floating_inputs(x, 1, x[:1], values[0])
    assert {a.dtype for a in arrays} == {np.dtype(dtype)}
    np.testing.assert_array_equal(arrays[0], x)


def test_transforms_go_through_the_backend() -> None:
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    x = jnp.asarray([0.1, 0.2, 0.3])
    eager = jnp.stack([affine(value, 2.0) for value in x])
    np.testing.assert_allclose(jax.jit(affine)(x, 2.0), eager, rtol=1e-15)
    np.testing.assert_allclose(jax.vmap(affine, (0, None))(x, 2.0), eager, rtol=1e-15)
    slope = jax.grad(lambda value: affine(value, 2.0))(0.1)
    np.testing.assert_allclose(slope, 2.0 * np.cos(0.2), rtol=1e-15)


def test_jax_float32_stays_float32() -> None:
    jnp = pytest.importorskip("jax.numpy")
    assert affine(jnp.float32([1.0]), 2.0, 1).dtype == jnp.float32


def test_a_traced_record_field_selects_jax() -> None:
    jax = pytest.importorskip("jax")

    def scaled(field):
        xp, (x,) = floating_inputs(np.asarray([1.0, 2.0]), params=(field,))
        return xp.sum(like(field, x, xp) * x)

    assert float(jax.grad(scaled)(2.0)) == 3.0


def test_importing_the_package_imports_neither_jax_nor_xarray() -> None:
    code = (
        "import sys, radiomax, radiomax._src.backend, radiomax._src.labelled;"
        "import numpy as np; radiomax._src.backend.floating_inputs(np.zeros(2));"
        "sys.exit(bool({'jax', 'xarray'} & set(sys.modules)))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
