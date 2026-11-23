"""Band brightness temperatures from the coefficients sensors publish."""

from __future__ import annotations

import dataclasses
from typing import Any

from geophysix.constants import STANDARD, Constants
from pipekit_array.functions import elementwise, floating_inputs, like

from radiomax._src.records import register_record


__all__ = [
    "PlanckLinear",
    "PlanckQuadratic",
    "brightness_temperature_band",
]


@register_record
@dataclasses.dataclass(frozen=True, slots=True)
class PlanckLinear:
    """A band's inverse Planck function, with a linear correction.

    T = (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂

    the form in GOES ABI files. Landsat's T = K₂ / ln(K₁ / L + 1) is the same
    with the default b₁ = 0 and b₂ = 1.

    Attributes:
        fk1: fk₁ (Landsat K₁), in the radiance's unit, whatever that is.
        fk2: fk₂ (Landsat K₂), K.
        bc1: Offset b₁ of the correction, K.
        bc2: Slope b₂ of the correction.

    Examples:
        >>> from radiomax.thermal import PlanckLinear
        >>> PlanckLinear(fk1=774.8853, fk2=1321.0789)
        PlanckLinear(fk1=774.8853, fk2=1321.0789, bc1=0.0, bc2=1.0)
    """

    fk1: Any
    fk2: Any
    bc1: Any = 0.0
    bc2: Any = 1.0


@register_record
@dataclasses.dataclass(frozen=True, slots=True)
class PlanckQuadratic:
    """A band's inverse Planck function, with a quadratic correction.

    Tₑ = (hc / kλ) / ln(2hc² / (λ⁵ L) + 1), T = c₀ + c₁ Tₑ + c₂ Tₑ²

    the form in Himawari AHI files, whose radiance is per µm: multiply it by 10⁶
    for the radiance per metre that the formula takes. By default there is no
    correction.

    Attributes:
        wavelength: Central wavelength λ, m.
        c0: Constant c₀ of the correction, K.
        c1: Slope c₁ of the correction.
        c2: Curvature c₂ of the correction, K⁻¹.

    Examples:
        >>> from radiomax.thermal import PlanckQuadratic
        >>> PlanckQuadratic(wavelength=1.04e-5)
        PlanckQuadratic(wavelength=1.04e-05, c0=0.0, c1=1.0, c2=0.0)
    """

    wavelength: Any
    c0: Any = 0.0
    c1: Any = 1.0
    c2: Any = 0.0


def brightness_temperature_band(
    radiance: Any,
    coefficients: PlanckLinear | PlanckQuadratic,
    *,
    constants: Constants = STANDARD,
) -> Any:
    """A band's brightness temperature from its published coefficients.

    The record says which formula: `PlanckLinear` (GOES ABI, Landsat) or
    `PlanckQuadratic` (Himawari AHI). The monochromatic one is
    `brightness_temperature`.

    Args:
        radiance: Radiance L: in the unit of ``fk1`` for `PlanckLinear`, and
            W m⁻² sr⁻¹ m⁻¹ for `PlanckQuadratic`.
        coefficients: The band's coefficients; with several bands, arrays that
            broadcast against the radiance, or DataArrays.
        constants: h, c and k, which `PlanckQuadratic` needs; a file that
            carries its own should pass them.

    Returns:
        T, K; NaN where the radiance is not positive.

    Examples:
        Landsat 8's band 10, with K₁ and K₂ from its metadata:

        >>> from radiomax.thermal import PlanckLinear, brightness_temperature_band
        >>> band10 = PlanckLinear(fk1=774.8853, fk2=1321.0789)
        >>> round(float(brightness_temperature_band(9.0, band10)), 3)
        295.739
    """
    if isinstance(coefficients, PlanckLinear):
        return _linear(
            radiance,
            coefficients.fk1,
            coefficients.fk2,
            coefficients.bc1,
            coefficients.bc2,
        )
    return _quadratic(
        radiance,
        coefficients.wavelength,
        coefficients.c0,
        coefficients.c1,
        coefficients.c2,
        constants=constants,
    )


@elementwise()
def _linear(radiance: Any, fk1: Any, fk2: Any, bc1: Any, bc2: Any) -> Any:
    xp, (radiance, fk1, fk2, bc1, bc2) = floating_inputs(radiance, fk1, fk2, bc1, bc2)
    positive = radiance > 0
    # the inner where keeps the unused branch finite, and so its gradient
    safe = xp.where(positive, radiance, xp.ones_like(radiance))
    # T = (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂
    temperature = (fk2 / xp.log1p(fk1 / safe) - bc1) / bc2
    return xp.where(positive, temperature, like(xp.nan, temperature, xp))


@elementwise()
def _quadratic(
    radiance: Any, wavelength: Any, c0: Any, c1: Any, c2: Any, *, constants: Constants
) -> Any:
    h, c, k = constants.planck, constants.speed_of_light, constants.boltzmann
    xp, (radiance, wavelength, c0, c1, c2) = floating_inputs(
        radiance, wavelength, c0, c1, c2, params=(h, c, k)
    )
    positive = radiance > 0
    # the inner where keeps the unused branch finite, and so its gradient
    safe = xp.where(positive, radiance, xp.ones_like(radiance))
    # hc / kλ, K
    scale = like(h * c / k, wavelength, xp) / wavelength
    # ln(2hc² / λ⁵ L); λ⁵ alone is below float32's range of normal numbers
    log_ratio = (
        xp.log(like(2.0 * h * c**2, wavelength, xp))
        - 5.0 * xp.log(wavelength)
        - xp.log(safe)
    )
    # Tₑ = (hc / kλ) / ln(2hc² / λ⁵ L + 1)
    effective = scale / xp.logaddexp(xp.zeros_like(log_ratio), log_ratio)
    # T = c₀ + c₁ Tₑ + c₂ Tₑ²
    temperature = c0 + effective * (c1 + effective * c2)
    return xp.where(positive, temperature, like(xp.nan, temperature, xp))
