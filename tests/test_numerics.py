"""Angle units."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from radiomax._src.numerics import full_turn, to_radians, wrap_turn


def test_to_radians_by_the_argument_and_never_by_size() -> None:
    x = np.float32([1.5, 180.0])
    assert to_radians(x, np, degrees=False) is x
    converted = to_radians(x, np, degrees=True)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, np.deg2rad(x), rtol=1e-6)


@pytest.mark.parametrize("degrees", [True, False])
@given(st.floats(-1e4, 1e4))
def test_wrapped_angles_are_in_range_and_the_same_angle(
    degrees: bool, angle: float
) -> None:
    turn = full_turn(degrees=degrees)
    wrapped = float(wrap_turn(np.asarray(angle), np, degrees=degrees))
    assert 0.0 <= wrapped < turn
    scale = 2.0 * math.pi / turn
    assert math.isclose(
        math.cos(scale * wrapped), math.cos(scale * angle), abs_tol=1e-9
    )
    assert math.isclose(
        math.sin(scale * wrapped), math.sin(scale * angle), abs_tol=1e-9
    )


def test_a_tiny_negative_angle_wraps_to_zero() -> None:
    tiny = np.asarray([-1e-30, -0.0], dtype=np.float32)
    assert wrap_turn(tiny, np, degrees=True).tolist() == [0.0, 0.0]


def test_a_missing_angle_stays_missing() -> None:
    angles = np.asarray([np.nan, 370.0])
    np.testing.assert_allclose(wrap_turn(angles, np, degrees=True), [np.nan, 10.0])
