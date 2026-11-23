"""The angles between the Sun's and the sensor's directions at a pixel."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

from pipekit_array.functions import elementwise, floating_inputs, like

from radiomax._src.numerics import full_turn, safe_sqrt, to_radians


__all__ = [
    "phase_angle",
    "phase_radians",
    "relative_azimuth",
    "scattering_angle",
]


@elementwise()
def relative_azimuth(
    view_azimuth: Any, solar_azimuth: Any, *, degrees: bool = True
) -> Any:
    """The azimuth between the sensor and the Sun, folded to [0°, 180°].

    ϕ = arccos(cos(φᵥ − φₛ))

    0 is backscatter, the hotspot's side: the sensor in the Sun's direction.
    The fold is computed without arccos, whose derivative is infinite at both
    ends; the gradient is ±1 everywhere.

    Args:
        view_azimuth: Azimuth φᵥ from the pixel towards the sensor, degrees
            (radians with ``degrees=False``), from any origin.
        solar_azimuth: Azimuth φₛ from the pixel towards the Sun, from the
            same origin and in the same direction.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        ϕ in [0°, 180°].

    Examples:
        >>> from radiomax.geometry import relative_azimuth
        >>> (
        ...     float(relative_azimuth(350.0, 20.0)),
        ...     float(relative_azimuth(-10.0, 170.0)),
        ... )
        (30.0, 180.0)
    """
    xp, (view, solar) = floating_inputs(view_azimuth, solar_azimuth)
    turn = like(full_turn(degrees=degrees), view, xp)
    half = 0.5 * turn
    # φᵥ − φₛ, in one turn
    difference = xp.remainder(view - solar, turn)
    # ϕ = 180° − |(φᵥ − φₛ) − 180°|
    return half - xp.abs(difference - half)


@elementwise()
def phase_angle(
    solar_zenith: Any, view_zenith: Any, relative_azimuth: Any, *, degrees: bool = True
) -> Any:
    """The angle at the pixel between the directions to the Sun and the sensor.

    cos ξ = cos θₛ cos θᵥ + sin θₛ sin θᵥ cos ϕ

    Args:
        solar_zenith: Solar zenith θₛ, degrees (radians with
            ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        relative_azimuth: Relative azimuth ϕ, the same unit; 0 is backscatter.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        ξ in [0°, 180°]; 0 at the hotspot, where its gradient is 0.

    Examples:
        >>> from radiomax.geometry import phase_angle
        >>> round(float(phase_angle(30.0, 30.0, 0.0)), 9)
        0.0
        >>> round(float(phase_angle(30.0, 20.0, 180.0)), 9)
        50.0
    """
    xp, (solar, view, azimuth) = floating_inputs(
        solar_zenith, view_zenith, relative_azimuth
    )
    solar, view, azimuth = (
        to_radians(v, xp, degrees=degrees) for v in (solar, view, azimuth)
    )
    angle = phase_radians(solar, view, azimuth, xp)
    return angle / like(math.pi / 180.0, angle, xp) if degrees else angle


@elementwise()
def scattering_angle(
    solar_zenith: Any, view_zenith: Any, relative_azimuth: Any, *, degrees: bool = True
) -> Any:
    """The angle by which a photon from the Sun turns to reach the sensor.

    Θ = 180° − ξ, with ξ the phase angle (`phase_angle`)

    Args:
        solar_zenith: Solar zenith θₛ, degrees (radians with
            ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        relative_azimuth: Relative azimuth ϕ, the same unit; 0 is backscatter.
        degrees: Whether the angles are in degrees, in and out.

    Returns:
        Θ in [0°, 180°]; 180° at the hotspot.

    Examples:
        >>> from radiomax.geometry import scattering_angle
        >>> round(float(scattering_angle(30.0, 0.0, 0.0)), 9)
        150.0
    """
    xp, (solar, view, azimuth) = floating_inputs(
        solar_zenith, view_zenith, relative_azimuth
    )
    solar, view, azimuth = (
        to_radians(v, xp, degrees=degrees) for v in (solar, view, azimuth)
    )
    angle = phase_radians(solar, view, azimuth, xp)
    angle = like(math.pi, angle, xp) - angle
    return angle / like(math.pi / 180.0, angle, xp) if degrees else angle


def phase_radians(solar: Any, view: Any, azimuth: Any, xp: ModuleType) -> Any:
    """ξ in radians from angles in radians, by the cross and dot products.

    The Sun is at (sin θₛ, 0, cos θₛ) and the sensor at
    (sin θᵥ cos ϕ, sin θᵥ sin ϕ, cos θᵥ).

    Examples:
        >>> import numpy as np
        >>> solar, view, azimuth = np.asarray(0.5), np.asarray(0.2), np.asarray(0.0)
        >>> round(float(phase_radians(solar, view, azimuth, np)), 12)
        0.3
    """
    sin_solar, cos_solar = xp.sin(solar), xp.cos(solar)
    sin_view, cos_view = xp.sin(view), xp.cos(view)
    along, across = sin_view * xp.cos(azimuth), sin_view * xp.sin(azimuth)
    # cos ξ = cos θₛ cos θᵥ + sin θₛ sin θᵥ cos ϕ
    dot = cos_solar * cos_view + sin_solar * along
    # sin ξ = |s × v|; its vertical and cos θₛ parts add to (sin θᵥ sin ϕ)²
    tilt = cos_solar * along - sin_solar * cos_view
    cross = safe_sqrt(across**2 + tilt**2, xp)
    return xp.atan2(cross, dot)
