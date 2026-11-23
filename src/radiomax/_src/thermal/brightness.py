"""The monochromatic brightness temperature."""

from __future__ import annotations

from typing import Any

from geophysix.constants import STANDARD, Constants

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise
from radiomax._src.thermal.planck import (
    first_radiation_constant,
    second_radiation_constant,
)


__all__ = [
    "brightness_temperature",
]


@elementwise()
def brightness_temperature(
    wavenumber_cm: Any, radiance: Any, *, constants: Constants = STANDARD
) -> Any:
    """The temperature of the black body that emits a given radiance.

    T_b = c₂ ν̃ / ln(1 + c₁ ν̃³ / L)

    the inverse of `planck_radiance`, monochromatic. The logarithm is
    logaddexp(0, ln(c₁ ν̃³ / L)), which neither overflows nor cancels from the
    Rayleigh–Jeans limit to the Wien limit. A band's brightness temperature
    is `brightness_temperature_band`, or a solve through its response.

    Args:
        wavenumber_cm: Wavenumber ν̃, cm⁻¹.
        radiance: Radiance L, W m⁻² sr⁻¹ (cm⁻¹)⁻¹; `from_per_um` converts.
        constants: h, c and k; CODATA 2018 by default.

    Returns:
        T_b, K; NaN where the radiance or the wavenumber is not positive, as
        noise makes it in a cold, dark band.

    Examples:
        >>> import numpy as np
        >>> from radiomax.thermal import brightness_temperature, planck_radiance
        >>> radiance = planck_radiance(1000.0, np.asarray([220.0, 300.0]))
        >>> brightness_temperature(1000.0, radiance).round(9).tolist()
        [220.0, 300.0]
        >>> float(brightness_temperature(1000.0, -1e-3))
        nan
    """
    xp, (nu, radiance) = floating_inputs(
        wavenumber_cm,
        radiance,
        params=(constants.planck, constants.speed_of_light, constants.boltzmann),
    )
    valid = (nu > 0) & (radiance > 0)
    # the inner where keeps the unused branch finite, and so its gradient
    one = like(1.0, nu, xp)
    safe_nu = xp.where(valid, nu, one)
    safe_radiance = xp.where(valid, radiance, one)
    c1 = like(first_radiation_constant(constants), nu, xp)
    c2 = like(second_radiation_constant(constants), nu, xp)
    # ln(c₁ ν̃³ / L)
    log_ratio = xp.log(c1) + 3.0 * xp.log(safe_nu) - xp.log(safe_radiance)
    # T_b = c₂ ν̃ / ln(1 + c₁ ν̃³ / L)
    temperature = c2 * safe_nu / xp.logaddexp(xp.zeros_like(log_ratio), log_ratio)
    return xp.where(valid, temperature, like(xp.nan, temperature, xp))
