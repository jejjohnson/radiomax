"""Ocean-colour chlorophyll from a ratio of reflectances, and published coefficients."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from radiomax._src.backend import floating_inputs, like
from radiomax._src.labelled import elementwise
from radiomax._src.retrievals.coefficients import unpack_coefficients
from radiomax.exceptions import ValidationError


__all__ = [
    "OCX_R2022",
    "BandRatioTable",
    "band_ratio_chlorophyll",
]


def band_ratio_chlorophyll(blue: Sequence[Any], green: Any, coefficients: Any) -> Any:
    """Chlorophyll-a from the largest of several blue reflectances over a green one.

    log₁₀ chl = ∑ᵢ aᵢ rⁱ, r = log₁₀(max(R_blue) / R_green)

    The formula of the OCx algorithms, with any sensor's coefficients. Those
    of NASA's 2022 reprocessing are in ``OCX_R2022``. NASA's standard product
    is this above 0.35 mg m⁻³ only: below 0.25 mg m⁻³ it is another algorithm
    (the colour index), and a blend of the two between them.

    Args:
        blue: One remote-sensing reflectance R_blue per blue band, sr⁻¹; the
            largest is used at each pixel.
        green: Remote-sensing reflectance R_green, sr⁻¹.
        coefficients: (a₀, a₁, …), of any degree: a sequence, an array with
            them on the last axis, or a DataArray with them on ``"feature"``.

    Returns:
        Chlorophyll-a concentration, mg m⁻³; NaN where the green or the
        largest blue reflectance is not positive, or any of them is NaN.

    Raises:
        ValidationError: If there is no blue reflectance or no coefficient, or
            ``blue`` is one array rather than a sequence of them.

    Examples:
        MODIS's bands at 443, 488 and 547 nm:

        >>> from radiomax.retrievals import OCX_R2022, band_ratio_chlorophyll
        >>> modis = OCX_R2022["MODIS"]
        >>> chlorophyll = band_ratio_chlorophyll(
        ...     [0.0050, 0.0046], 0.0030, modis.coefficients
        ... )
        >>> round(float(chlorophyll), 4)
        0.5578
    """
    if not isinstance(blue, Sequence) or not blue:
        msg = "'blue' must be a sequence of one or more reflectances"
        raise ValidationError(msg)
    values = unpack_coefficients(coefficients, None)
    largest = functools.reduce(_larger, blue)
    ratio, valid = _log_ratio(largest, green)
    # ∑ᵢ aᵢ rⁱ by Horner's rule, from the highest power down
    total = values[-1]
    for coefficient in reversed(values[:-1]):
        total = _horner(total, ratio, coefficient)
    return _power_of_ten(total, valid)


@elementwise()
def _larger(one: Any, other: Any) -> Any:
    xp, (one, other) = floating_inputs(one, other)
    # NaN where either is NaN, by the standard
    return xp.maximum(one, other)


@elementwise(n_outputs=2)
def _log_ratio(blue: Any, green: Any) -> tuple[Any, Any]:
    """r = log₁₀(R_blue / R_green), 0 where either is not positive, and where it is."""
    xp, (blue, green) = floating_inputs(blue, green)
    valid = (blue > 0) & (green > 0)
    # the inner where keeps the unused branch finite, and so its gradient
    one = like(1.0, blue, xp)
    safe_blue, safe_green = xp.where(valid, blue, one), xp.where(valid, green, one)
    return xp.log10(safe_blue) - xp.log10(safe_green), valid


@elementwise()
def _horner(total: Any, ratio: Any, coefficient: Any) -> Any:
    _, (total, ratio, coefficient) = floating_inputs(total, ratio, coefficient)
    return total * ratio + coefficient


@elementwise()
def _power_of_ten(exponent: Any, valid: Any) -> Any:
    """10ˣ, and NaN where not valid."""
    xp, (exponent,) = floating_inputs(exponent, params=(valid,))
    value = xp.exp(exponent * like(math.log(10.0), exponent, xp))
    return xp.where(valid, value, like(xp.nan, value, xp))


@dataclasses.dataclass(frozen=True, slots=True)
class BandRatioTable:
    """Published coefficients of a band ratio, for one sensor and one version.

    Plain data, for `band_ratio_chlorophyll`: a new version of a table is a
    new record, and the old one stays.

    Attributes:
        sensor: The sensor, as the source names it.
        algorithm: The algorithm, as the source names it.
        version: The version of the source.
        blue_nm: Nominal wavelengths of the blue bands, nm.
        green_nm: Nominal wavelength of the green band, nm.
        coefficients: a₀, a₁, … of the polynomial.
        source: Where the numbers were read.

    Examples:
        >>> from radiomax.retrievals import OCX_R2022
        >>> OCX_R2022["SeaWiFS"].blue_nm, OCX_R2022["SeaWiFS"].green_nm
        ((443.0, 489.0, 510.0), 555.0)
    """

    sensor: str
    algorithm: str
    version: str
    blue_nm: tuple[float, ...]
    green_nm: float
    coefficients: tuple[float, ...]
    source: str


def _ocx_r2022() -> Mapping[str, BandRatioTable]:
    """Table 1 of the source: the OCx component of the standard product."""
    source = (
        "Werdell, O'Reilly, Hu, Feng, Lee, Franz, Bailey and Proctor (2023), "
        "Chlorophyll a, NASA OBPG algorithm document v1.1, Table 1, "
        "doi:10.5067/JCQB8QALDOYD"
    )
    oc4 = (0.32814, -3.20725, 3.22969, -1.36769, -0.81739)
    rows = (
        ("SeaWiFS", "OC4", (443.0, 489.0, 510.0), 555.0, oc4),
        (
            "MODIS",
            "OC3M",
            (443.0, 488.0),
            547.0,
            (0.26294, -2.64669, 1.28364, 1.08209, -1.76828),
        ),
        (
            "VIIRS-SNPP",
            "OC3_VIIRS_SNPP",
            (443.0, 486.0),
            551.0,
            (0.23548, -2.63001, 1.65498, 0.16117, -1.37247),
        ),
        (
            "VIIRS-NOAA20",
            "OC3_VIIRS_NOAA20",
            (445.0, 489.0),
            556.0,
            (0.28153, -2.65472, 1.30882, 1.31521, -2.08622),
        ),
        (
            "VIIRS-NOAA21",
            "OC3_VIIRS_NOAA21",
            (445.0, 488.0),
            555.0,
            (0.24765, -2.54926, 1.55323, 0.39485, -1.54632),
        ),
        ("PACE OCI", "OC4", (442.0, 490.0, 510.0), 555.0, oc4),
        (
            "MERIS",
            "OC4E",
            (443.0, 489.0, 510.0),
            560.0,
            (0.42487, -3.20974, 2.89721, -0.75258, -0.98259),
        ),
        (
            "OCTS",
            "OC4O",
            (443.0, 489.0, 516.0),
            565.0,
            (0.54655, -3.51799, 3.39128, -0.91567, -0.97112),
        ),
        (
            "GOCI",
            "OC4",
            (412.0, 443.0, 489.0),
            555.0,
            (0.28043, -2.49033, 1.53980, -0.09926, -0.68403),
        ),
        ("HAWKEYE", "OC4", (447.0, 488.0, 510.0), 556.0, oc4),
        (
            "OLCI",
            "OC4",
            (443.0, 490.0, 510.0),
            560.0,
            (0.42540, -3.21679, 2.86907, -0.62628, -1.09333),
        ),
        (
            "CZCS",
            "OC3",
            (443.0, 520.0),
            555.0,
            (0.31841, -4.56386, 8.63979, -8.41411, 1.91532),
        ),
    )
    return MappingProxyType(
        {
            sensor: BandRatioTable(
                sensor, algorithm, "R2022", blue, green, values, source
            )
            for sensor, algorithm, blue, green, values in rows
        }
    )


#: NASA's OCx coefficients of its 2022 reprocessing, by sensor.
OCX_R2022: Mapping[str, BandRatioTable] = _ocx_r2022()
