"""Conversions between what a sensor counts and what it measures, per pixel."""

from __future__ import annotations

from radiomax._src.radiometry.band import (
    band_average,
    band_irradiance,
    trapezoid_weights,
)
from radiomax._src.radiometry.density import per_cm_to_per_nm, per_nm_to_per_cm
from radiomax._src.radiometry.linear import (
    counts_to_radiance,
    counts_to_reflectance,
    radiance_to_counts,
)
from radiomax._src.radiometry.reflectance import (
    radiance_from_reflectance,
    reflectance,
)
from radiomax._src.radiometry.response import (
    apply_response,
    apply_response_adjoint,
    band_centre,
    box_response,
    equivalent_width,
    gaussian_response,
    response_coverage,
    response_matrix,
    sampled_response_matrix,
    triangle_response,
)


__all__ = [
    "apply_response",
    "apply_response_adjoint",
    "band_average",
    "band_centre",
    "band_irradiance",
    "box_response",
    "counts_to_radiance",
    "counts_to_reflectance",
    "equivalent_width",
    "gaussian_response",
    "per_cm_to_per_nm",
    "per_nm_to_per_cm",
    "radiance_from_reflectance",
    "radiance_to_counts",
    "reflectance",
    "response_coverage",
    "response_matrix",
    "sampled_response_matrix",
    "trapezoid_weights",
    "triangle_response",
]
