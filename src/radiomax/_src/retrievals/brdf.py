"""The kernel BRDF model, and the adjustment of a reflectance to nadir view."""

from __future__ import annotations

from typing import Any

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise
from radiomax._src.retrievals.kernels import li_sparse_reciprocal, ross_thick


__all__ = [
    "brdf_reflectance",
    "nadir_adjustment",
]


@elementwise()
def brdf_reflectance(
    f_iso: Any,
    f_vol: Any,
    f_geo: Any,
    solar_zenith: Any,
    view_zenith: Any,
    relative_azimuth: Any,
    *,
    degrees: bool = True,
) -> Any:
    """Reflectance of a surface from its kernel weights.

    R = f_iso + f_vol K_vol + f_geo K_geo

    with K_vol the RossThick kernel (`ross_thick`) and K_geo the
    LiSparse-Reciprocal kernel with MODIS's crowns (`li_sparse_reciprocal`).
    The weights are fitted elsewhere, or come from a product.

    Args:
        f_iso: Isotropic weight f_iso, the reflectance at nadir with the Sun
            overhead.
        f_vol: Volumetric weight f_vol.
        f_geo: Geometric weight f_geo.
        solar_zenith: Solar zenith θₛ, degrees (radians with
            ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        relative_azimuth: Relative azimuth ϕ, the same unit; 0 is backscatter.
        degrees: Whether the angles are in degrees.

    Returns:
        R, dimensionless; NaN where either zenith is 90° or more.

    Examples:
        >>> import numpy as np
        >>> from radiomax.retrievals import brdf_reflectance
        >>> azimuth = np.asarray([0.0, 180.0])
        >>> brdf_reflectance(0.25, 0.15, 0.03, 45.0, 30.0, azimuth).round(
        ...     4
        ... ).tolist()
        [0.2712, 0.1845]
    """
    _, (f_iso, f_vol, f_geo, solar, view, azimuth) = floating_inputs(
        f_iso, f_vol, f_geo, solar_zenith, view_zenith, relative_azimuth
    )
    volumetric = ross_thick(solar, view, azimuth, degrees=degrees)
    geometric = li_sparse_reciprocal(solar, view, azimuth, degrees=degrees)
    # R = f_iso + f_vol K_vol + f_geo K_geo
    return f_iso + f_vol * volumetric + f_geo * geometric


@elementwise()
def nadir_adjustment(
    f_iso: Any,
    f_vol: Any,
    f_geo: Any,
    solar_zenith: Any,
    view_zenith: Any,
    relative_azimuth: Any,
    *,
    reference_zenith: Any = None,
    degrees: bool = True,
) -> Any:
    """The factor that takes an observed reflectance to nadir view.

    c = R(θ_ref, 0) / R(θₛ, θᵥ, ϕ), and the adjusted reflectance is c ρ

    with R the kernel model (`brdf_reflectance`). Multiplying the observation
    keeps its detail and its noise, where R at nadir would replace it by the
    model.

    Args:
        f_iso: Isotropic weight f_iso.
        f_vol: Volumetric weight f_vol.
        f_geo: Geometric weight f_geo.
        solar_zenith: Solar zenith θₛ of the observation, degrees (radians
            with ``degrees=False``).
        view_zenith: View zenith θᵥ, the same unit.
        relative_azimuth: Relative azimuth ϕ, the same unit; 0 is backscatter.
        reference_zenith: Solar zenith θ_ref to adjust to, the same unit; the
            observation's own by default, which adjusts the view only.
        degrees: Whether the angles are in degrees.

    Returns:
        c, dimensionless; NaN where either modelled reflectance is not
        positive, or a zenith is 90° or more.

    Examples:
        A pixel seen from the Sun's side is brighter than at nadir:

        >>> from radiomax.retrievals import nadir_adjustment
        >>> round(float(nadir_adjustment(0.25, 0.15, 0.03, 45.0, 30.0, 0.0)), 4)
        0.774
        >>> float(nadir_adjustment(0.25, 0.15, 0.03, 45.0, 0.0, 0.0))
        1.0
    """
    reference = solar_zenith if reference_zenith is None else reference_zenith
    xp, (f_iso, f_vol, f_geo, solar, view, azimuth, reference) = floating_inputs(
        f_iso, f_vol, f_geo, solar_zenith, view_zenith, relative_azimuth, reference
    )
    zero = like(0.0, solar, xp)
    observed = brdf_reflectance(
        f_iso, f_vol, f_geo, solar, view, azimuth, degrees=degrees
    )
    # R(θ_ref, 0); at nadir the azimuth has no meaning
    nadir = brdf_reflectance(
        f_iso, f_vol, f_geo, reference, zero, zero, degrees=degrees
    )
    # false where either is NaN, which keeps its gradient out
    valid = (observed > 0) & (nadir > 0)
    safe = xp.where(valid, observed, xp.ones_like(observed))
    factor = nadir / safe
    return xp.where(valid, factor, like(xp.nan, factor, xp))
