"""Whiskbroom scan geometry: scan angles, pixel growth and bowtie positions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from geophysix.earth import SPHERE

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise, is_dataarray
from radiomax._src.numerics import safe_sqrt, to_radians
from radiomax.exceptions import ValidationError


__all__ = [
    "bowtie_detector_positions",
    "scan_angles",
    "scan_pixel_growth",
]


def scan_angles(
    n_columns: int,
    max_scan_angle: float,
    aggregation_zones: Sequence[tuple[int, int]] | None = None,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """The scan angle of every column of a whiskbroom swath.

    θ_c = θ_max (2c + 1 − C) / C

    The mirror turns at a constant rate, so native samples are uniform in
    angle, and θ_max is the outer edge of the outermost one. With on-board
    aggregation each half-scan is a sequence of zones from nadir outward, and
    a column averages a zone's factor of consecutive native samples; the two
    halves mirror each other. A host-side builder, in float64 NumPy.

    Args:
        n_columns: Number of columns C of the swath.
        max_scan_angle: θ_max, degrees (radians with ``degrees=False``).
        aggregation_zones: (native samples, factor) of each zone, from nadir
            outward. VIIRS M-bands are ((1776, 3), (736, 2), (640, 1)).
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        (C,) scan angles, increasing, 0 at nadir.

    Raises:
        ValidationError: If there are no columns, a zone's samples are not a
            positive multiple of its factor, or the zones do not make
            ``n_columns``.

    Examples:
        >>> from radiomax.geometry import scan_angles
        >>> scan_angles(3, 60.0).round(9).tolist()
        [-40.0, 0.0, 40.0]
        >>> viirs = ((1776, 3), (736, 2), (640, 1))
        >>> scan_angles(3200, 56.28, viirs)[-2:].round(4).tolist()
        [56.2532, 56.2711]
    """
    if n_columns < 1:
        msg = f"'n_columns' must be positive, got {n_columns}"
        raise ValidationError(msg)
    if aggregation_zones is None:
        columns = np.arange(n_columns, dtype=np.float64)
        # θ_c = θ_max (2c + 1 − C) / C
        return max_scan_angle * (2.0 * columns + 1.0 - n_columns) / n_columns
    for native, factor in aggregation_zones:
        if factor < 1 or native < 1 or native % factor:
            msg = (
                f"aggregation zone ({native}, {factor}) must hold a positive, whole "
                "number of aggregated samples"
            )
            raise ValidationError(msg)
    # Δθ = θ_max / n
    step = max_scan_angle / sum(native for native, _ in aggregation_zones)
    half, start = [], 0
    for native, factor in aggregation_zones:
        # θᵢ = (i + ½) Δθ
        centres = (start + np.arange(native, dtype=np.float64) + 0.5) * step
        half.append(centres.reshape(-1, factor).mean(axis=1))
        start += native
    right = np.concatenate(half)
    if 2 * right.size != n_columns:
        msg = f"'aggregation_zones' make {2 * right.size} columns, got {n_columns}"
        raise ValidationError(msg)
    return np.concatenate([-right[::-1], right])


@elementwise(n_outputs=3)
def scan_pixel_growth(
    scan_angle: Any,
    altitude: Any,
    *,
    radius: Any = SPHERE.semi_major,
    degrees: bool = True,
) -> tuple[Any, Any, Any]:
    """How much a whiskbroom's pixel grows from nadir, over a spherical Earth.

    s/h = (k + 1) / (k cos θ + r), g = (s/h) / cos β, and (s/h) / r

    with k = (R + h)/R, r = √(1 − k² sin² θ), s the slant range and
    β = asin(k sin θ) − θ the Earth central angle from nadir, whose cosine is
    r cos θ + k sin² θ. The slant range is (R + h) cos θ − R r, written here
    without its cancellation.

    Args:
        scan_angle: Scan angle θ off nadir, degrees (radians with
            ``degrees=False``); its sign does not matter.
        altitude: Altitude h of the sensor, m.
        radius: Radius R of the Earth, m; its mean radius by default.
        degrees: Whether the scan angle is in degrees.

    Returns:
        The growth along track on the ground, along track in orbit angle, and
        across track on the ground, each 1 at nadir; NaN from the horizon on,
        where k |sin θ| ≥ 1.

    Examples:
        MODIS, 705 km up, at the edge of its scan:

        >>> from radiomax.geometry import scan_pixel_growth
        >>> along, orbit, across = scan_pixel_growth(55.0, 705e3)
        >>> round(float(along), 2), round(float(orbit), 2), round(float(across), 2)
        (2.01, 2.04, 4.83)
    """
    xp, (angle, altitude) = floating_inputs(scan_angle, altitude, params=(radius,))
    theta = to_radians(angle, xp, degrees=degrees)
    sin_theta, cos_theta = xp.sin(theta), xp.cos(theta)
    # k = (R + h) / R
    k = 1.0 + altitude / like(radius, theta, xp)
    # r² = 1 − k² sin² θ, as a product that does not cancel near the horizon
    root_squared = (1.0 - k * sin_theta) * (1.0 + k * sin_theta)
    valid = (root_squared > 0) & (cos_theta > 0)
    # the inner where keeps the unused branch finite, and so its gradient
    root = safe_sqrt(xp.where(valid, root_squared, xp.ones_like(root_squared)), xp)
    safe_cos = xp.where(valid, cos_theta, xp.ones_like(cos_theta))
    # s / h = (k + 1) / (k cos θ + r)
    along = (k + 1.0) / (k * safe_cos + root)
    # cos β = r cos θ + k sin² θ
    cos_beta = root * safe_cos + k * sin_theta**2
    nan = like(xp.nan, along, xp)
    return (
        xp.where(valid, along, nan),
        xp.where(valid, along / cos_beta, nan),
        xp.where(valid, along / root, nan),
    )


def bowtie_detector_positions(
    along_orbit_growth: Any,
    detectors_per_scan: int,
    *,
    detector_dim: str = "detector",
) -> tuple[Any, Any, Any]:
    """The detector that sees each row of a scan once its overlap is removed.

    k₀ = c + (m − c) / g, k₁ = c + (m − c ∓ N) / g, c = (N − 1) / 2

    Successive scans of N detectors overlap by N (g − 1) rows where the pixels
    have grown by g in orbit angle. Row m of the nadir grid is seen by
    detector k₀ of its own scan, and where they overlap by detector k₁ of the
    neighbour on its side, the next scan for m > c and the previous one
    otherwise. A position is inside a scan where −½ ≤ k ≤ N − ½.

    Args:
        along_orbit_growth: Growth g along track in orbit angle, of any shape
            (usually one per column), from `scan_pixel_growth`.
        detectors_per_scan: Number of detectors N.
        detector_dim: The name of the rows' dimension, for a DataArray.

    Returns:
        k₀ and k₁, with a leading axis of N rows, and the offset ±1 of the
        neighbouring scan, (N,).

    Raises:
        ValidationError: If there are no detectors, or the growth already has
            the dimension ``detector_dim``.

    Examples:
        >>> import numpy as np
        >>> from radiomax.geometry import bowtie_detector_positions
        >>> own, neighbour, step = bowtie_detector_positions(
        ...     np.asarray([1.0, 2.0]), 4
        ... )
        >>> own.T.tolist()
        [[0.0, 1.0, 2.0, 3.0], [0.75, 1.25, 1.75, 2.25]]
        >>> neighbour.T.tolist()
        [[4.0, 5.0, -2.0, -1.0], [2.75, 3.25, -0.25, 0.25]]
        >>> step.tolist()
        [-1, -1, 1, 1]
    """
    if detectors_per_scan < 1:
        msg = f"'detectors_per_scan' must be positive, got {detectors_per_scan}"
        raise ValidationError(msg)
    centre = (detectors_per_scan - 1) / 2.0
    # m − c
    offset = np.arange(detectors_per_scan, dtype=np.float64) - centre
    step = np.where(offset > 0, 1, -1)
    # m − c ∓ N
    shifted = offset - detectors_per_scan * step
    if is_dataarray(along_orbit_growth):
        if detector_dim in along_orbit_growth.dims:
            msg = f"the growth already has the dimension {detector_dim!r}"
            raise ValidationError(msg)
        # a DataArray, without importing xarray
        dataarray = type(along_orbit_growth)
        offset, shifted, step = (
            dataarray(rows, dims=detector_dim) for rows in (offset, shifted, step)
        )
    else:
        xp, (along_orbit_growth,) = floating_inputs(along_orbit_growth)
        shape = (detectors_per_scan,) + (1,) * along_orbit_growth.ndim
        offset, shifted = (
            xp.reshape(like(rows, along_orbit_growth, xp), shape)
            for rows in (offset, shifted)
        )
        # through the growth's dtype, which puts it where the growth is
        step = xp.astype(like(step, along_orbit_growth, xp), xp.int32)
    # k₀ = c + (m − c) / g
    own = _position(offset, along_orbit_growth, centre)
    # k₁ = c + (m − c ∓ N) / g
    neighbour = _position(shifted, along_orbit_growth, centre)
    return own, neighbour, step


@elementwise()
def _position(offset: Any, growth: Any, centre: float) -> Any:
    xp, (growth,) = floating_inputs(growth)
    return like(centre, growth, xp) + like(offset, growth, xp) / growth
