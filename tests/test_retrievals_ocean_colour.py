"""The maximum band ratio, and the coefficient table it ships with."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _backends import KINDS, assert_backends_agree, needs_jax
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.exceptions import ValidationError
from radiomax.retrievals import OCX_R2022, BandRatioTable, band_ratio_chlorophyll


OC4 = OCX_R2022["SeaWiFS"].coefficients
REFLECTANCE = st.floats(1e-4, 0.03)

#: Table 1 of the algorithm document (doi:10.5067/JCQB8QALDOYD, v1.1), as it
#: prints the rows: the blue bands, the green band, and a₀ to a₄.
TABLE_1 = """
SeaWiFS      443>489>510/555 0.32814; -3.20725; 3.22969; -1.36769; -0.81739
MODIS        443>488/547     0.26294; -2.64669; 1.28364; 1.08209; -1.76828
VIIRS-SNPP   443>486/551     0.23548; -2.63001; 1.65498; 0.16117; -1.37247
VIIRS-NOAA20 445>489/556     0.28153; -2.65472; 1.30882; 1.31521; -2.08622
VIIRS-NOAA21 445>488/555     0.24765; -2.54926; 1.55323; 0.39485; -1.54632
PACE-OCI     442>490>510/555 0.32814; -3.20725; 3.22969; -1.36769; -0.81739
MERIS        443>489>510/560 0.42487; -3.20974; 2.89721; -0.75258; -0.98259
OCTS         443>489>516/565 0.54655; -3.51799; 3.39128; -0.91567; -0.97112
GOCI         412>443>489/555 0.28043; -2.49033; 1.53980; -0.09926; -0.68403
HAWKEYE      447>488>510/556 0.32814; -3.20725; 3.22969; -1.36769; -0.81739
OLCI         443>490>510/560 0.42540; -3.21679; 2.86907; -0.62628; -1.09333
CZCS         443>520/555     0.31841; -4.56386; 8.63979; -8.41411; 1.91532
"""


class TestBandRatio:
    def test_one_coefficient_is_a_constant(self) -> None:
        got = band_ratio_chlorophyll([np.asarray([0.004, 0.01])], 0.003, [0.3])
        assert_matches_oracle(got, np.full(2, 10.0**0.3), rtol=1e-14)

    @given(REFLECTANCE)
    def test_equal_blue_and_green(self, reflectance) -> None:
        got = band_ratio_chlorophyll([reflectance, 0.5 * reflectance], reflectance, OC4)
        assert_matches_oracle(got, 10.0 ** OC4[0], rtol=1e-14)

    @given(REFLECTANCE, REFLECTANCE, REFLECTANCE, REFLECTANCE)
    def test_the_formula(self, first, second, third, green) -> None:
        ratio = math.log10(max(first, second, third) / green)
        expected = 10.0 ** sum(a * ratio**i for i, a in enumerate(OC4))
        got = band_ratio_chlorophyll([first, second, third], green, OC4)
        assert_matches_oracle(got, expected, rtol=1e-10)

    @given(REFLECTANCE, REFLECTANCE, REFLECTANCE, st.floats(0.1, 10.0))
    def test_a_common_factor_cancels(self, first, second, green, factor) -> None:
        once = band_ratio_chlorophyll([first, second], green, OC4)
        scaled = [factor * first, factor * second]
        again = band_ratio_chlorophyll(scaled, factor * green, OC4)
        assert_matches_oracle(again, once, rtol=1e-10)

    def test_a_reflectance_that_is_not_positive_is_nan(self) -> None:
        blue = np.asarray([0.005, -0.001, 0.0, 0.005, 0.005, np.nan])
        green = np.asarray([0.003, 0.003, 0.003, 0.0, -0.001, 0.003])
        for coefficients in (OC4, [0.3]):
            got = band_ratio_chlorophyll([blue], green, coefficients)
            assert np.isnan(got).tolist() == [False] + [True] * 5

    def test_a_dark_band_does_not_hide_a_bright_one(self) -> None:
        # only the largest has to be positive
        got = band_ratio_chlorophyll([-0.001, 0.005], 0.003, OC4)
        assert_matches_oracle(
            got, band_ratio_chlorophyll([0.005], 0.003, OC4), rtol=0.0
        )

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        def total(blue, green):
            return jax.numpy.nansum(band_ratio_chlorophyll([blue], green, OC4))

        gradient = jax.grad(total, argnums=(0, 1))(
            np.asarray([0.005, -0.001, 0.0, 0.004]), np.asarray([0.003] * 3 + [0.0])
        )
        assert np.isfinite(np.stack(gradient)).all()
        # more blue is less chlorophyll
        assert gradient[0][0] < 0.0 < gradient[1][0]

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: band_ratio_chlorophyll([x[0], x[1]], x[2], x[3:]),
            np.asarray([0.5, 0.46, 0.3, *OC4]),
        )
        blue = np.linspace(0.002, 0.01, 6).reshape(2, 3)
        green = np.full((2, 3), 0.003)
        assert_backends_agree(
            lambda b, g: band_ratio_chlorophyll([b, 0.9 * b], g, OC4),
            blue,
            green,
            rtol32=1e-5,
        )
        assert_traces(
            lambda b, g: band_ratio_chlorophyll([b, 0.9 * b], g, OC4),
            blue.ravel(),
            green.ravel(),
        )

    @pytest.mark.parametrize("kind", KINDS, ids=lambda kind: kind.name)
    def test_coefficients_as_an_array_of_each_kind(self, kind) -> None:
        blue = np.linspace(0.002, 0.01, 4)
        coefficients = kind.make(np.asarray(OC4))
        if kind.name == "xarray":
            coefficients = coefficients.rename(dim1="feature")
        got = band_ratio_chlorophyll([kind.make(blue)], 0.003, coefficients)
        expected = band_ratio_chlorophyll([blue], 0.003, OC4)
        assert_matches_oracle(kind.read(got), expected, rtol=1e-13)

    def test_dataarrays(self) -> None:
        xr = pytest.importorskip("xarray")
        blue = xr.DataArray(np.full((2, 3), 0.005), dims=("y", "x"))
        got = band_ratio_chlorophyll([blue, 0.9 * blue], 0.6 * blue, OC4)
        assert got.dims == ("y", "x")
        assert band_ratio_chlorophyll([blue], 0.003, [0.3]).dims == ("y", "x")

    def test_invalid_arguments(self) -> None:
        for blue in ([], np.asarray([0.005, 0.004]), 0.005):
            with pytest.raises(ValidationError, match="'blue' must be a sequence"):
                band_ratio_chlorophyll(blue, 0.003, OC4)
        with pytest.raises(ValidationError, match="needs at least 1"):
            band_ratio_chlorophyll([0.005], 0.003, [])


class TestTable:
    def test_every_number_is_the_sources(self) -> None:
        rows = [line.split(maxsplit=2) for line in TABLE_1.strip().splitlines()]
        assert [name.replace("PACE-OCI", "PACE OCI") for name, *_ in rows] == list(
            OCX_R2022
        )
        for (_, bands, numbers), table in zip(rows, OCX_R2022.values(), strict=True):
            blue, green = bands.split("/")
            assert table.blue_nm == tuple(float(b) for b in blue.split(">"))
            assert table.green_nm == float(green)
            assert table.coefficients == tuple(float(n) for n in numbers.split(";"))

    @pytest.mark.parametrize("table", OCX_R2022.values(), ids=lambda t: t.sensor)
    def test_each_record(self, table: BandRatioTable) -> None:
        assert OCX_R2022[table.sensor] is table
        assert table.version == "R2022"
        assert "doi:10.5067/JCQB8QALDOYD" in table.source
        assert table.algorithm.startswith(f"OC{len(table.blue_nm) + 1}")

    @pytest.mark.parametrize("table", OCX_R2022.values(), ids=lambda t: t.sensor)
    def test_chlorophyll_decreases_with_the_ratio(self, table: BandRatioTable) -> None:
        # from 0.35 mg m⁻³, where the product starts to use it, to 20 mg m⁻³
        ratio = 10.0 ** np.linspace(-0.5, 0.7, 1201)
        chlorophyll = band_ratio_chlorophyll([ratio], 1.0, table.coefficients)
        used = (chlorophyll > 0.35) & (chlorophyll < 20.0)
        assert used.sum() > 100
        assert (np.diff(chlorophyll[used]) < 0).all()

    def test_it_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            OCX_R2022["MODIS"] = OCX_R2022["CZCS"]  # ty: ignore
