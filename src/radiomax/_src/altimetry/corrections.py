"""Altimeter range corrections with a closed form, and the height they correct."""

from __future__ import annotations

from typing import Any

from geophysix.constants import H2O, STANDARD, Constants, Gas
from pipekit_array.functions import elementwise, floating_inputs, like

from radiomax._src.numerics import to_radians


__all__ = [
    "dry_troposphere_correction",
    "dual_frequency_ionosphere_correction",
    "inverse_barometer",
    "ionosphere_correction",
    "sea_surface_height",
    "wet_troposphere_correction",
]

# Jason-3 Products Handbook 3.1: 2.277 mm mbar⁻¹ and 0.0026 (section 5.4.1.1),
# 9.948 mm mbar⁻¹ (section 5.4.6.1)
_DRY_PER_PASCAL = 2.277e-5
_DRY_LATITUDE = 0.0026
_BAROMETER_PER_PASCAL = 9.948e-5
# IERS Conventions (2010), chapter 9, equation 9.21, m³ s⁻²
_IONOSPHERE = 40.308


@elementwise()
def sea_surface_height(altitude: Any, altimeter_range: Any, *corrections: Any) -> Any:
    """Sea surface height above the ellipsoid from an altimeter's range.

    SSH = H − (R + ∑ c)

    The altitude is about 10⁶ m, which float32 resolves to 0.06 m or worse:
    give it in float64.

    Args:
        altitude: H, the satellite's altitude above the ellipsoid, m.
        altimeter_range: R, the measured range, m.
        *corrections: c, what is added to the range, m: negative where the
            signal was delayed, as every function of this module returns it.
            The inverse barometer, which is a height, goes here too.

    Returns:
        SSH, m.

    References:
        Jason-3 Products Handbook, SALP-MU-M-OP-16118-CN, issue 3.1 (2026),
        sections 1.4.2 and 4.2.

    Examples:
        >>> from radiomax.altimetry import sea_surface_height
        >>> round(
        ...     float(sea_surface_height(1_336_000.0, 1_335_962.0, -2.3, -0.2)), 6
        ... )
        40.5
    """
    _, (altitude, corrected, *values) = floating_inputs(
        altitude, altimeter_range, *corrections
    )
    for value in values:
        corrected = corrected + value
    # SSH = H − (R + ∑ c)
    return altitude - corrected


@elementwise()
def dry_troposphere_correction(pressure: Any, lat: Any, *, degrees: bool = True) -> Any:
    """Range correction for the dry gases of the troposphere, after Saastamoinen.

    c = −2.277 × 10⁻⁵ p (1 + 0.0026 cos 2φ)

    Args:
        pressure: p, the pressure at the surface, Pa.
        lat: φ, the latitude, degrees (radians with ``degrees=False``).
        degrees: Whether the latitude is in degrees.

    Returns:
        c, m, negative: about −2.3 at the sea surface.

    References:
        Jason-3 Products Handbook, SALP-MU-M-OP-16118-CN, issue 3.1 (2026),
        section 5.4.1.1.

    Examples:
        >>> from radiomax.altimetry import dry_troposphere_correction
        >>> round(float(dry_troposphere_correction(101325.0, 45.0)), 4)
        -2.3072
    """
    xp, (pressure, lat) = floating_inputs(pressure, lat)
    # c = −2.277 × 10⁻⁵ p (1 + 0.0026 cos 2φ)
    angle = 2.0 * to_radians(lat, xp, degrees=degrees)
    return -_DRY_PER_PASCAL * pressure * (1.0 + _DRY_LATITUDE * xp.cos(angle))


@elementwise()
def wet_troposphere_correction(
    water_vapour: Any,
    mean_temperature: Any,
    *,
    k2: float = 0.17,
    k3: float = 3776.0,
    vapour: Gas = H2O,
    constants: Constants = STANDARD,
) -> Any:
    """Range correction for the water vapour of the troposphere.

    c = −10⁻⁶ R_v (k₂′ + k₃ / Tₘ) W,  R_v = R / M

    Tₘ is the mean of the temperature weighted by the vapour's density,
    Tₘ = ∫ (e / T) dz / ∫ (e / T²) dz, with e the vapour pressure. Cloud liquid
    water is not in it.

    Args:
        water_vapour: W, the column water vapour, kg m⁻².
        mean_temperature: Tₘ, K.
        k2: k₂′, the refractivity's constant, K Pa⁻¹; (17 ± 10) K mbar⁻¹ by
            default.
        k3: k₃, the refractivity's constant, K² Pa⁻¹; 3.776 × 10⁵ K² mbar⁻¹ by
            default.
        vapour: The gas, for M; geophysix's ``H2O`` by default.
        constants: The constants record, for R; ``STANDARD`` by default.

    Returns:
        c, m, negative: about −0.06 to −0.4 over the sea; NaN where Tₘ is not
        positive.

    References:
        Bevis, Businger, Herring, Rocken, Anthes and Ware (1992), *GPS
        Meteorology: Remote Sensing of Atmospheric Water Vapor Using the
        Global Positioning System*, J. Geophys. Res. 97 (D14), 15787–15801,
        equations 4, 7 and 12.

    Examples:
        >>> from radiomax.altimetry import wet_troposphere_correction
        >>> round(float(wet_troposphere_correction(25.0, 275.0)), 4)
        -0.1604
    """
    xp, (column, temperature) = floating_inputs(
        water_vapour,
        mean_temperature,
        params=(k2, k3, vapour.molar_mass, constants.gas_constant),
    )
    valid = temperature > 0
    # the inner where keeps the unused branch finite, and so its gradient
    safe = xp.where(valid, temperature, xp.ones_like(temperature))
    scale = like(1e-6 * constants.gas_constant / vapour.molar_mass, safe, xp)
    # c = −10⁻⁶ R_v (k₂′ + k₃ / Tₘ) W
    refractivity = like(k2, safe, xp) + like(k3, safe, xp) / safe
    correction = -scale * refractivity * column
    return xp.where(valid, correction, like(xp.nan, correction, xp))


@elementwise()
def ionosphere_correction(electron_content: Any, frequency: Any) -> Any:
    """Range correction for the ionosphere from its electron content.

    c = −40.308 N / f²

    It is the first-order term of the group delay.

    Args:
        electron_content: N, electrons per m² between the satellite and the
            surface; 1 TECU is 10¹⁶.
        frequency: f, the radar's frequency, Hz.

    Returns:
        c, m, negative; NaN where the frequency is not positive.

    References:
        IERS Conventions (2010), chapter 9 as of 10 June 2013, equation 9.21.

    Examples:
        >>> from radiomax.altimetry import ionosphere_correction
        >>> round(float(ionosphere_correction(50e16, 13.6e9)), 4)
        -0.109
    """
    xp, (content, frequency) = floating_inputs(electron_content, frequency)
    valid = frequency > 0
    safe = xp.where(valid, frequency, xp.ones_like(frequency))
    # c = −40.308 N / f², with 1 / f² never formed
    correction = -_IONOSPHERE * (content / safe) / safe
    return xp.where(valid, correction, like(xp.nan, correction, xp))


@elementwise()
def dual_frequency_ionosphere_correction(
    range_1: Any, range_2: Any, frequency_1: Any, frequency_2: Any
) -> Any:
    """Range correction for the ionosphere from two ranges at two frequencies.

    c₁ = (R₁ − R₂) f₂² / (f₁² − f₂²)

    Both ranges see one distance and the delay a / f², which solves for a. The
    ranges must be free of everything else that depends on the frequency, the
    sea state bias first. Give them in float64: the difference is centimetres
    of 10⁶ m.

    Args:
        range_1: R₁, the range at the first frequency, m.
        range_2: R₂, the range at the second frequency, m.
        frequency_1: f₁, Hz.
        frequency_2: f₂, Hz.

    Returns:
        c₁, the correction to R₁, m; NaN where a frequency is not positive or
        the two are equal. The correction to R₂ is c₁ (f₁ / f₂)².

    Examples:
        The Ku and C bands, delayed by 0.1 m and more:

        >>> from radiomax.altimetry import dual_frequency_ionosphere_correction
        >>> ku, c = 13.575e9, 5.3e9
        >>> delay = 0.1 * (ku / c) ** 2
        >>> round(
        ...     float(
        ...         dual_frequency_ionosphere_correction(10.1, 10.0 + delay, ku, c)
        ...     ),
        ...     9,
        ... )
        -0.1
    """
    xp, (first, second, f1, f2) = floating_inputs(
        range_1, range_2, frequency_1, frequency_2
    )
    valid = (f1 > 0) & (f2 > 0) & (f1 != f2)
    # the inner where keeps the unused branch finite, and so its gradient
    top = xp.where(valid, f1, like(2.0, f1, xp))
    bottom = xp.where(valid, f2, xp.ones_like(f2))
    # c₁ = (R₁ − R₂) / ((f₁ / f₂)² − 1), factored so that close frequencies keep
    # their digits
    spread = (top - bottom) / bottom * ((top + bottom) / bottom)
    correction = (first - second) / spread
    return xp.where(valid, correction, like(xp.nan, correction, xp))


@elementwise()
def inverse_barometer(pressure: Any, mean_pressure: Any) -> Any:
    """Height of the sea surface above its mean, under the air's pressure.

    h = −9.948 × 10⁻⁵ (p − p̄)

    It is a height, subtracted from the sea surface height: pass it to
    `sea_surface_height` with the corrections.

    Args:
        pressure: p, the pressure at the surface, Pa.
        mean_pressure: p̄, the mean pressure over the oceans at that time, Pa.
            A constant 101330 in its place moves the height by centimetres.

    Returns:
        h, m: positive under a low.

    References:
        Jason-3 Products Handbook, SALP-MU-M-OP-16118-CN, issue 3.1 (2026),
        section 5.4.6.1.

    Examples:
        >>> from radiomax.altimetry import inverse_barometer
        >>> round(float(inverse_barometer(98000.0, 101100.0)), 4)
        0.3084
    """
    _, (pressure, mean) = floating_inputs(pressure, mean_pressure)
    # h = −9.948 × 10⁻⁵ (p − p̄)
    return -_BAROMETER_PER_PASCAL * (pressure - mean)
