"""Spectral densities per nanometre and per wavenumber."""

from __future__ import annotations

from typing import Any

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise


__all__ = [
    "per_cm_to_per_nm",
    "per_nm_to_per_cm",
]

#: λ ν = 10⁷ for λ in nm and ν in cm⁻¹.
_NM_CM = 1e7


@elementwise()
def per_nm_to_per_cm(density: Any, wavelength_nm: Any) -> Any:
    """A spectral density per nm as one per cm⁻¹.

    E_ν = E_λ |dλ/dν| = E_λ λ² / 10⁷

    Only the values change; a caller that also wants an ascending wavenumber
    axis reverses the arrays.

    Args:
        density: E_λ, per nm: irradiance, radiance or any other density.
        wavelength_nm: λ, nm, at which it is given.

    Returns:
        E_ν, per cm⁻¹.

    Examples:
        >>> from radiomax.radiometry import per_nm_to_per_cm
        >>> round(float(per_nm_to_per_cm(0.25, 1600.0)), 12)
        0.064
    """
    xp, (density, wavelength) = floating_inputs(density, wavelength_nm)
    # E_ν = E_λ λ² / 10⁷
    return density * wavelength**2 / like(_NM_CM, density, xp)


@elementwise()
def per_cm_to_per_nm(density: Any, wavenumber_cm: Any) -> Any:
    """A spectral density per cm⁻¹ as one per nm.

    E_λ = E_ν |dν/dλ| = E_ν ν² / 10⁷

    Args:
        density: E_ν, per cm⁻¹.
        wavenumber_cm: ν, cm⁻¹, at which it is given.

    Returns:
        E_λ, per nm.

    Examples:
        >>> from radiomax.radiometry import per_cm_to_per_nm
        >>> round(float(per_cm_to_per_nm(0.064, 1e7 / 1600.0)), 12)
        0.25
    """
    xp, (density, wavenumber) = floating_inputs(density, wavenumber_cm)
    # E_λ = E_ν ν² / 10⁷
    return density * wavenumber**2 / like(_NM_CM, density, xp)
