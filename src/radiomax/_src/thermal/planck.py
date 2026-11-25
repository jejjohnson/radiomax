"""Planck's law per wavenumber, and the views of a radiance per µm and per GHz."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from geophysix.constants import STANDARD, Constants
from pipekit_array.arrays import floating_inputs, like
from pipekit_array.kernels import elementwise


__all__ = [
    "first_radiation_constant",
    "from_per_ghz",
    "from_per_um",
    "log_planck_radiance",
    "planck_radiance",
    "second_radiation_constant",
    "to_per_ghz",
    "to_per_um",
]


def first_radiation_constant(constants: Constants) -> Any:
    """c₁ = 2hc² in W m⁻² sr⁻¹ cm⁴, for radiance per cm⁻¹ of wavenumber.

    Examples:
        >>> from geophysix.constants import STANDARD
        >>> round(first_radiation_constant(STANDARD) * 1e8, 9)
        1.191042972
    """
    # 10⁸ cm⁴ per m⁴
    return 2.0 * constants.planck * constants.speed_of_light**2 * 1e8


def second_radiation_constant(constants: Constants) -> Any:
    """c₂ = hc/k in cm K.

    Examples:
        >>> from geophysix.constants import STANDARD
        >>> round(second_radiation_constant(STANDARD), 8)
        1.43877688
    """
    # 10² cm per m
    return constants.planck * constants.speed_of_light / constants.boltzmann * 1e2


@elementwise()
def planck_radiance(
    wavenumber_cm: Any, temperature: Any, *, constants: Constants = STANDARD
) -> Any:
    """Planck's law: the radiance of a black body per wavenumber.

    B(ν̃, T) = c₁ ν̃³ e⁻ˣ / (1 − e⁻ˣ), x = c₂ ν̃ / T

    with c₁ = 2hc² and c₂ = hc/k. Written in e⁻ˣ it cannot overflow, and 1 − e⁻ˣ is
    computed without cancelling, so it holds from the Rayleigh–Jeans limit to
    the Wien limit. Where x is large e⁻ˣ underflows to 0 (beyond x = 708 in
    float64 and x = 87 in float32); `log_planck_radiance` does not.

    Args:
        wavenumber_cm: Wavenumber ν̃, cm⁻¹.
        temperature: Temperature T, K.
        constants: h, c and k; CODATA 2018 by default.

    Returns:
        B, W m⁻² sr⁻¹ (cm⁻¹)⁻¹; NaN where either argument is not positive.

    Examples:
        At 10 µm, the radiance of a scene at 300 K:

        >>> from radiomax.thermal import planck_radiance
        >>> round(float(planck_radiance(1000.0, 300.0)), 6)
        0.09924
        >>> import numpy as np
        >>> planck_radiance(1000.0, np.asarray([0.0, 300.0])).round(4).tolist()
        [nan, 0.0992]
    """
    xp, (nu, t) = floating_inputs(wavenumber_cm, temperature, params=_fields(constants))
    nu, t, valid = _valid(nu, t, xp)
    # x = c₂ ν̃ / T
    x = like(second_radiation_constant(constants), nu, xp) * nu / t
    c1 = like(first_radiation_constant(constants), nu, xp)
    # B = c₁ ν̃³ e⁻ˣ / (1 − e⁻ˣ)
    radiance = c1 * nu**3 * xp.exp(-x) / (-xp.expm1(-x))
    return xp.where(valid, radiance, like(xp.nan, radiance, xp))


@elementwise()
def log_planck_radiance(
    wavenumber_cm: Any, temperature: Any, *, constants: Constants = STANDARD
) -> Any:
    """The logarithm of Planck's law, where the radiance itself underflows.

    ln B = ln c₁ + 3 ln ν̃ − x − ln(1 − e⁻ˣ), x = c₂ ν̃ / T

    Args:
        wavenumber_cm: Wavenumber ν̃, cm⁻¹.
        temperature: Temperature T, K.
        constants: h, c and k; CODATA 2018 by default.

    Returns:
        ln B, with B in W m⁻² sr⁻¹ (cm⁻¹)⁻¹; NaN where either argument is not
        positive.

    Examples:
        At 0.4 µm the radiance at 150 K is about 10⁻⁹⁹:

        >>> import numpy as np
        >>> from radiomax.thermal import log_planck_radiance
        >>> cold = log_planck_radiance(np.float32(25000.0), np.float32(150.0))
        >>> round(float(cold), 2)
        -227.66
    """
    xp, (nu, t) = floating_inputs(wavenumber_cm, temperature, params=_fields(constants))
    nu, t, valid = _valid(nu, t, xp)
    # x = c₂ ν̃ / T
    x = like(second_radiation_constant(constants), nu, xp) * nu / t
    c1 = like(first_radiation_constant(constants), nu, xp)
    # ln B = ln c₁ + 3 ln ν̃ − x − ln(1 − e⁻ˣ)
    log_radiance = xp.log(c1) + 3.0 * xp.log(nu) - x - xp.log(-xp.expm1(-x))
    return xp.where(valid, log_radiance, like(xp.nan, log_radiance, xp))


@elementwise()
def to_per_um(radiance: Any, wavenumber_cm: Any) -> Any:
    """Radiance per µm of wavelength from radiance per cm⁻¹ of wavenumber.

    L_λ = L_ν̃ |dν̃/dλ| = L_ν̃ ν̃² / 10⁴

    Args:
        radiance: L_ν̃, W m⁻² sr⁻¹ (cm⁻¹)⁻¹.
        wavenumber_cm: Wavenumber ν̃, cm⁻¹.

    Returns:
        L_λ, W m⁻² sr⁻¹ µm⁻¹.

    Examples:
        >>> from radiomax.thermal import to_per_um
        >>> float(to_per_um(0.1, 1000.0))
        10.0
    """
    xp, (radiance, nu) = floating_inputs(radiance, wavenumber_cm)
    # L_λ = L_ν̃ ν̃² / 10⁴
    return radiance * nu**2 / like(1e4, nu, xp)


@elementwise()
def from_per_um(radiance_um: Any, wavenumber_cm: Any) -> Any:
    """Radiance per cm⁻¹ of wavenumber from radiance per µm of wavelength.

    L_ν̃ = 10⁴ L_λ / ν̃²

    Args:
        radiance_um: L_λ, W m⁻² sr⁻¹ µm⁻¹.
        wavenumber_cm: Wavenumber ν̃, cm⁻¹.

    Returns:
        L_ν̃, W m⁻² sr⁻¹ (cm⁻¹)⁻¹; NaN where the wavenumber is not positive.

    Examples:
        >>> from radiomax.thermal import from_per_um
        >>> float(from_per_um(10.0, 1000.0))
        0.1
    """
    xp, (radiance, nu) = floating_inputs(radiance_um, wavenumber_cm)
    positive = nu > 0
    safe = xp.where(positive, nu, xp.ones_like(nu))
    # L_ν̃ = 10⁴ L_λ / ν̃²
    converted = radiance * like(1e4, nu, xp) / safe**2
    return xp.where(positive, converted, like(xp.nan, converted, xp))


@elementwise()
def to_per_ghz(radiance: Any, *, constants: Constants = STANDARD) -> Any:
    """Radiance per GHz of frequency from radiance per cm⁻¹ of wavenumber.

    L_f = L_ν̃ / c, with c in GHz cm

    Args:
        radiance: L_ν̃, W m⁻² sr⁻¹ (cm⁻¹)⁻¹.
        constants: c; CODATA 2018 by default.

    Returns:
        L_f, W m⁻² sr⁻¹ GHz⁻¹.

    Examples:
        One wavenumber is 29.9792458 GHz:

        >>> from radiomax.thermal import to_per_ghz
        >>> round(float(to_per_ghz(29.9792458)), 12)
        1.0
    """
    xp, (radiance,) = floating_inputs(radiance, params=(constants.speed_of_light,))
    # L_f = L_ν̃ / c; 10⁻⁷ GHz cm per m s⁻¹
    return radiance / like(constants.speed_of_light * 1e-7, radiance, xp)


@elementwise()
def from_per_ghz(radiance_ghz: Any, *, constants: Constants = STANDARD) -> Any:
    """Radiance per cm⁻¹ of wavenumber from radiance per GHz of frequency.

    L_ν̃ = c L_f, with c in GHz cm

    Args:
        radiance_ghz: L_f, W m⁻² sr⁻¹ GHz⁻¹.
        constants: c; CODATA 2018 by default.

    Returns:
        L_ν̃, W m⁻² sr⁻¹ (cm⁻¹)⁻¹.

    Examples:
        >>> from radiomax.thermal import from_per_ghz
        >>> round(float(from_per_ghz(1.0)), 7)
        29.9792458
    """
    xp, (radiance,) = floating_inputs(radiance_ghz, params=(constants.speed_of_light,))
    # L_ν̃ = c L_f; 10⁻⁷ GHz cm per m s⁻¹
    return radiance * like(constants.speed_of_light * 1e-7, radiance, xp)


def _fields(constants: Constants) -> tuple[Any, ...]:
    """The constants that may be traced, for choosing the namespace."""
    return constants.planck, constants.speed_of_light, constants.boltzmann


def _valid(nu: Any, t: Any, xp: ModuleType) -> tuple[Any, Any, Any]:
    """ν̃ and T with 1 in place of invalid values, and where both are valid."""
    valid = (nu > 0) & (t > 0)
    # the inner where keeps the unused branch finite, and so its gradient
    one = like(1.0, nu, xp)
    return xp.where(valid, nu, one), xp.where(valid, t, one), valid
