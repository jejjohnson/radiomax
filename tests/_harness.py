"""Shared checks: an oracle, an adjoint, the JAX transforms and the gradient.

Each helper raises `AssertionError` with the worst index and the values there,
so a failing test says where it fails, not just that it does. They come from
functax (`tests/_harness.py` at `17deb26`, MIT), with JAX imported inside the
helpers that need it, so the suite also runs where JAX is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


__all__ = [
    "assert_adjoint",
    "assert_grad_matches_fd",
    "assert_matches_oracle",
    "assert_traces",
]


def _jax() -> Any:
    """JAX in 64-bit mode, or skip the test."""
    jax = pytest.importorskip("jax")
    if not jax.config.read("jax_enable_x64"):
        raise AssertionError(
            "the harness needs float64: enable jax_enable_x64 "
            "(conftest.py does this for the suite)"
        )
    return jax


def assert_matches_oracle(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float = 0.0,
    where: Any = None,
) -> None:
    """Assert ``|actual − expected| ≤ atol + rtol·|expected|`` wherever ``where`` holds.

    NaNs match only NaNs at the same index. Boolean and integer arrays must
    match exactly, whatever the tolerances.

    Args:
        actual: The values under test.
        expected: The reference values, the same shape.
        rtol: The relative tolerance.
        atol: The absolute tolerance, for references that cross zero.
        where: A boolean mask; only ``True`` entries are compared. ``None``
            compares every entry.

    Raises:
        AssertionError: If the shapes differ or any compared entry is out of
            tolerance.

    Examples:
        >>> import numpy as np
        >>> assert_matches_oracle(
        ...     np.array([1.0, np.nan]), np.array([1.0, np.nan]), rtol=1e-12
        ... )
    """
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    if actual_values.shape != expected_values.shape:
        raise AssertionError(
            f"shape mismatch: actual {actual_values.shape}, "
            f"expected {expected_values.shape}"
        )
    if where is None:
        mask = np.ones(actual_values.shape, dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(where, dtype=bool), actual_values.shape)
    floating = np.issubdtype(actual_values.dtype, np.inexact) or np.issubdtype(
        expected_values.dtype, np.inexact
    )
    if not floating:
        # Booleans and integers (masks, indices) must match exactly.
        actual_values = actual_values.astype(np.float64)
        expected_values = expected_values.astype(np.float64)
        rtol, atol = 0.0, 0.0

    # e = |a − b|; a NaN on one side only counts as an infinite error
    with np.errstate(invalid="ignore"):
        error = np.abs(actual_values - expected_values)
    both_nan = np.isnan(actual_values) & np.isnan(expected_values)
    # equal values (infinities included) have no error, and no inf − inf
    error = np.where(both_nan | (actual_values == expected_values), 0.0, error)
    error = np.where(np.isnan(error), np.inf, error)
    # τ = atol + rtol·|b|; a NaN reference contributes no tolerance
    reference_scale = np.where(np.isnan(expected_values), 0.0, expected_values)
    tolerance = atol + rtol * np.abs(reference_scale)
    failing = mask & (error > tolerance)
    if not failing.any():
        return

    # The worst entry is the one furthest past its own tolerance.
    excess = np.where(failing, error - tolerance, -np.inf)
    worst = np.unravel_index(np.argmax(excess), excess.shape)
    raise AssertionError(
        f"{int(failing.sum())} of {int(mask.sum())} compared entries out of "
        f"tolerance; worst at index {tuple(int(i) for i in worst)}: "
        f"actual {actual_values[worst]!r}, expected {expected_values[worst]!r}, "
        f"error {error[worst]:.3e} > tolerance {tolerance[worst]:.3e} "
        f"(rtol={rtol}, atol={atol})"
    )


def assert_adjoint(
    apply: Callable[[Any], Any],
    apply_adjoint: Callable[[Any], Any],
    n_source: int,
    n_target: int,
    *,
    seed: int = 0,
    rtol: float = 1e-12,
) -> None:
    """Assert the dot-product test ⟨Hx, r⟩ = ⟨x, Gr⟩ on random NumPy x and r.

    Args:
        apply: H, a linear map from ℝᴺ to ℝᴹ.
        apply_adjoint: G, the claimed adjoint Hᵀ, from ℝᴹ to ℝᴺ.
        n_source: N, the length of x.
        n_target: M, the length of Hx.
        seed: The seed of x and r.
        rtol: The tolerance, relative to max(‖Hx‖ ‖r‖, ‖x‖ ‖Gr‖).

    Raises:
        AssertionError: If an output has the wrong shape, or the relative error
            exceeds ``rtol``.

    Examples:
        >>> import numpy as np
        >>> H = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
        >>> assert_adjoint(lambda x: H @ x, lambda r: H.T @ r, 3, 2)
    """
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(n_source)
    residual = rng.standard_normal(n_target)
    forward = np.asarray(apply(source))
    backward = np.asarray(apply_adjoint(residual))
    if forward.shape != (n_target,):
        raise AssertionError(f"H x has shape {forward.shape}, expected ({n_target},)")
    if backward.shape != (n_source,):
        raise AssertionError(f"Hᵀ r has shape {backward.shape}, expected ({n_source},)")
    # ⟨Hx, r⟩ and ⟨x, Gr⟩
    lhs = float(forward @ residual)
    rhs = float(source @ backward)
    # the larger product, so a G that is too large also fails
    scale = max(
        float(np.linalg.norm(forward) * np.linalg.norm(residual)),
        float(np.linalg.norm(source) * np.linalg.norm(backward)),
    )
    if abs(lhs - rhs) > rtol * scale:
        raise AssertionError(
            f"dot-product test failed: <Hx, r> = {lhs!r}, <x, H^T r> = {rhs!r}, "
            f"relative error {abs(lhs - rhs) / scale:.3e} > rtol {rtol}"
        )


def assert_traces(fn: Callable[..., Any], *args: Any, rtol: float = 1e-12) -> None:
    """Assert ``jit(fn)``, ``vmap(fn)`` and an eager loop agree over a batch.

    Every array of ``args`` carries a leading batch axis of the same length B.
    The reference is the eager loop over the batch. Skips without JAX.

    Args:
        fn: The function under test; it returns an array or a tuple of them.
        *args: The batched arguments.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If any variant disagrees with the eager loop; the
            message names the variant.
    """
    jax = _jax()
    jnp = jax.numpy
    arrays = [jnp.asarray(arg) for arg in args]
    batch = {array.shape[0] for array in arrays}
    if len(batch) != 1:
        raise AssertionError(f"every argument needs the same batch axis, got {batch}")
    size = batch.pop()

    def stack(outputs: list[Any]) -> Any:
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *outputs)

    slices = [tuple(array[index] for array in arrays) for index in range(size)]
    reference = stack([fn(*one) for one in slices])
    jitted = jax.jit(fn)
    variants = {
        "jit(fn)": stack([jitted(*one) for one in slices]),
        "vmap(fn)": jax.vmap(fn)(*arrays),
        "jit(vmap(fn))": jax.jit(jax.vmap(fn))(*arrays),
    }
    for what, got in variants.items():
        pairs = zip(jax.tree.leaves(got), jax.tree.leaves(reference), strict=True)
        for index, (actual, expected) in enumerate(pairs):
            try:
                assert_matches_oracle(actual, expected, rtol=rtol)
            except AssertionError as err:
                raise AssertionError(f"{what}, output {index}: {err}") from err


def assert_grad_matches_fd(
    fn: Callable[[Any], Any],
    x: Any,
    *,
    eps: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Assert ``jax.jacrev(fn)`` agrees with central finite differences at x.

    The tolerance is relative to the largest entry of the finite-difference
    Jacobian, since an entry near zero carries the O(ε²) truncation error at
    full size. Skips without JAX.

    Args:
        fn: The differentiable function under test, of one array.
        x: The point, a float64 array.
        eps: The step ε.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If the Jacobians differ, or one is not finite; the
            message gives the worst (output, input) index.
    """
    jax = _jax()
    jnp = jax.numpy
    point = jnp.asarray(x)
    flat_point = jnp.reshape(point, (-1,))

    def flat_fn(flat: Any) -> Any:
        return jnp.reshape(fn(jnp.reshape(flat, point.shape)), (-1,))

    jacobian = np.asarray(jax.jit(jax.jacrev(flat_fn))(flat_point))
    steps = eps * np.eye(flat_point.shape[0])
    # (f(x + ε eₖ) − f(x − ε eₖ)) / 2ε, one column per input entry
    columns = [
        (flat_fn(flat_point + step) - flat_fn(flat_point - step)) / (2.0 * eps)
        for step in steps
    ]
    finite_difference = np.stack([np.asarray(c) for c in columns], axis=1)
    scale = float(np.max(np.abs(finite_difference)))
    assert_matches_oracle(jacobian, finite_difference, rtol=0.0, atol=rtol * scale)
