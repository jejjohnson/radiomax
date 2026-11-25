"""Angle units, which every module shares."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

from pipekit_array.functions import like


__all__ = [
    "full_turn",
    "to_radians",
    "wrap_turn",
]

_DEG = math.pi / 180.0


def to_radians(angle: Any, xp: ModuleType, *, degrees: bool) -> Any:
    """An angle in radians, from degrees or radians.

    Examples:
        >>> import numpy as np
        >>> round(float(to_radians(np.asarray(180.0), np, degrees=True)), 12)
        3.14159265359
    """
    return angle * like(_DEG, angle, xp) if degrees else angle


def full_turn(*, degrees: bool) -> float:
    """A full turn, 360° or 2π."""
    return 360.0 if degrees else 2.0 * math.pi


def wrap_turn(angle: Any, xp: ModuleType, *, degrees: bool) -> Any:
    """An angle modulo a full turn, in [0°, 360°) or [0, 2π).

    Rounding can take a tiny negative angle to the full turn itself, which is
    sent to 0. NaN stays NaN.

    Examples:
        >>> import numpy as np
        >>> wrap_turn(np.asarray([-90.0, 360.0, -1e-30]), np, degrees=True).tolist()
        [270.0, 0.0, 0.0]
    """
    turn = full_turn(degrees=degrees)
    wrapped = xp.remainder(angle, like(turn, angle, xp))
    # not `wrapped < turn`, which is False for NaN and would send it to 0
    return xp.where(wrapped >= turn, xp.zeros_like(wrapped), wrapped)
