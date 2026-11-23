"""Small numerics every module shares: angle units, and roots with finite gradients."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

from pipekit_array.functions import like


__all__ = [
    "full_turn",
    "safe_atan2",
    "safe_sqrt",
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
    sent to 0.

    Examples:
        >>> import numpy as np
        >>> wrap_turn(np.asarray([-90.0, 360.0, -1e-30]), np, degrees=True).tolist()
        [270.0, 0.0, 0.0]
    """
    turn = full_turn(degrees=degrees)
    wrapped = xp.remainder(angle, like(turn, angle, xp))
    return xp.where(wrapped < turn, wrapped, xp.zeros_like(wrapped))


def safe_sqrt(value: Any, xp: ModuleType) -> Any:
    """√x for x > 0, and 0 with a zero gradient at x ≤ 0.

    Examples:
        >>> import numpy as np
        >>> safe_sqrt(np.asarray([4.0, 0.0, -1.0]), np).tolist()
        [2.0, 0.0, 0.0]
    """
    positive = value > 0
    # the inner where keeps the unused branch finite, and so its gradient
    safe = xp.where(positive, value, xp.ones_like(value))
    return xp.where(positive, xp.sqrt(safe), xp.zeros_like(value))


def safe_atan2(y: Any, x: Any, xp: ModuleType) -> Any:
    """atan2(y, x), and 0 with a zero gradient at the origin.

    Examples:
        >>> import numpy as np
        >>> zero = np.asarray(0.0)
        >>> float(safe_atan2(zero, zero, np))
        0.0
        >>> round(float(safe_atan2(np.asarray(1.0), zero, np)), 12)
        1.570796326795
    """
    away = (x != 0) | (y != 0)
    safe_x = xp.where(away, x, xp.ones_like(x))
    safe_y = xp.where(away, y, xp.zeros_like(y))
    return xp.atan2(safe_y, safe_x)
