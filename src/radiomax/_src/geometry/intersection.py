"""Where a ray first meets the Earth's ellipsoid."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from geophysix.earth import WGS84, Ellipsoid
from pipekit_array.arrays import floating_inputs, like, safe_sqrt
from pipekit_array.kernels import elementwise


__all__ = [
    "earth_intersection",
    "near_root",
]


@elementwise(n_outputs=4)
def earth_intersection(
    origin_x: Any,
    origin_y: Any,
    origin_z: Any,
    direction_x: Any,
    direction_y: Any,
    direction_z: Any,
    *,
    earth: Ellipsoid = WGS84,
) -> tuple[Any, Any, Any, Any]:
    """The point where a ray from outside the Earth first meets the ellipsoid.

    (x² + y²) / a² + z² / b² = 1, (x, y, z) = o + t d, t > 0

    the smaller root t of a quadratic, written so that it does not cancel for
    an origin far away. The ray may be a line of sight in any direction of
    any length.

    Args:
        origin_x: Earth-centred x of the origin o, m.
        origin_y: Earth-centred y of the origin, m.
        origin_z: Earth-centred z of the origin, m.
        direction_x: x component of the direction d, any unit.
        direction_y: y component of the direction.
        direction_z: z component of the direction.
        earth: The body; WGS84 by default.

    Returns:
        The point's Earth-centred x, y and z, m, and whether the ray meets the
        Earth; NaN where it does not, or where the origin is inside.

    Examples:
        Straight down from above the equator, and past the Earth:

        >>> import numpy as np
        >>> from radiomax.geometry import earth_intersection
        >>> x, y, z, visible = earth_intersection(
        ...     2e7, 0.0, 0.0, -1.0, np.asarray([0.0, 1.0]), 0.0
        ... )
        >>> x.round(6).tolist(), visible.tolist()
        ([6378137.0, nan], [True, False])
    """
    xp, (ox, oy, oz, dx, dy, dz) = floating_inputs(
        origin_x,
        origin_y,
        origin_z,
        direction_x,
        direction_y,
        direction_z,
        params=(earth.semi_major, earth.flattening),
    )
    t, meets, above = near_root((ox, oy, oz), (dx, dy, dz), earth, xp)
    visible = meets & (above >= 0)
    nan = like(xp.nan, t, xp)
    x, y, z = (
        xp.where(visible, o + t * d, nan) for o, d in ((ox, dx), (oy, dy), (oz, dz))
    )
    return x, y, z, visible


def near_root(
    origin: tuple[Any, Any, Any],
    direction: tuple[Any, Any, Any],
    earth: Ellipsoid,
    xp: ModuleType,
) -> tuple[Any, Any, Any]:
    """The smaller root t of |o + t d| = 1, in coordinates where the Earth is a sphere.

    In units of the semi-axes, with o̲ = (oₓ/a, o_y/a, o_z/b) and d̲ alike,

    t = (|o̲|² − 1) / (−o̲·d̲ + √Δ), Δ = |d̲|² − |o̲ × d̲|²

    The root is negative for an origin inside the ellipsoid that the ray enters
    the far half of.

    Returns:
        t; whether the line meets the ellipsoid ahead of the origin, or just
        behind one inside it; and |o̲|² − 1, which is negative inside.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import sphere
        >>> one, zero = np.asarray(1.0), np.asarray(0.0)
        >>> t, meets, above = near_root(
        ...     (3.0 * one, zero, zero), (-one, zero, zero), sphere(1.0), np
        ... )
        >>> float(t), bool(meets), float(above)
        (2.0, True, 8.0)
    """
    ref = origin[0]
    axes = [like(earth.semi_major, ref, xp)] * 2 + [like(earth.semi_minor, ref, xp)]
    ox, oy, oz = (
        component / axis for component, axis in zip(origin, axes, strict=True)
    )
    dx, dy, dz = (
        component / axis for component, axis in zip(direction, axes, strict=True)
    )
    dot = ox * dx + oy * dy + oz * dz
    cross = (oy * dz - oz * dy, oz * dx - ox * dz, ox * dy - oy * dx)
    # Δ = |d̲|² − |o̲ × d̲|²; (o̲·d̲)² − |d̲|² (|o̲|² − 1) cancels
    discriminant = dx**2 + dy**2 + dz**2 - sum(c**2 for c in cross)
    above = ox**2 + oy**2 + oz**2 - like(1.0, ref, xp)
    meets = (discriminant >= 0) & (dot < 0)
    denominator = safe_sqrt(discriminant, xp) - dot
    # the inner where keeps the unused branch finite, and so its gradient
    t = above / xp.where(meets, denominator, xp.ones_like(denominator))
    return t, meets, above
