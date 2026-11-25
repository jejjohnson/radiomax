"""Band brightness temperatures against the readers they replace."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax, needs_xarray
from geophysix.constants import STANDARD
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.thermal import (
    PlanckLinear,
    PlanckQuadratic,
    brightness_temperature,
    brightness_temperature_band,
    from_per_um,
)


# the values in geoproducts' GOES tests (geotoolz `3d05a9c`)
GOES = PlanckLinear(fk1=10803.3, fk2=1392.74, bc1=0.0755, bc2=0.99975)
# Landsat 8 TIRS band 10, as in geoops' radiometry tests
LANDSAT = PlanckLinear(fk1=774.8853, fk2=1321.0789)
# a correction of the size AHI files carry, and not a published one
AHI = PlanckQuadratic(wavelength=1.0407e-5, c0=-0.1, c1=1.0005, c2=-1.6e-6)
# h, c and k written to the digits a file header keeps
HEADER = dataclasses.replace(
    STANDARD,
    planck=6.62606957e-34,
    speed_of_light=2.99792458e8,
    boltzmann=1.3806488e-23,
)


def _geoproducts_goes(radiance: np.ndarray, record: PlanckLinear) -> np.ndarray:
    positive = np.where(radiance > 0, radiance, np.nan)
    log_term = np.log(record.fk1 / positive + 1.0)
    return (record.fk2 / log_term - record.bc1) / record.bc2


def _geoproducts_himawari(
    radiance_um: np.ndarray, record: PlanckQuadratic, constants
) -> np.ndarray:
    h, c, k = constants.planck, constants.speed_of_light, constants.boltzmann
    spectral = radiance_um * 1e6
    positive = np.where(spectral > 0, spectral, np.nan)
    effective = (h * c / (k * record.wavelength)) / np.log(
        2.0 * h * c**2 / (record.wavelength**5 * positive) + 1.0
    )
    return record.c0 + record.c1 * effective + record.c2 * effective**2


class TestPlanckLinear:
    @pytest.mark.parametrize("record", [GOES, LANDSAT])
    def test_the_goes_reader(self, record: PlanckLinear) -> None:
        # gate G2-readers
        radiance = np.concatenate([np.geomspace(1e-2, 200.0, 50), [0.0, -0.3]])
        expected = _geoproducts_goes(radiance, record)
        temperature = brightness_temperature_band(radiance, record)
        assert_matches_oracle(temperature, expected, rtol=0.0, atol=1e-6)

    def test_the_defaults_are_landsats_formula(self) -> None:
        radiance = np.asarray([5.0, 9.0, 12.0])
        expected = 1321.0789 / np.log(774.8853 / radiance + 1.0)
        got = brightness_temperature_band(radiance, LANDSAT)
        assert_matches_oracle(got, expected, rtol=1e-14)

    @given(st.floats(1e-3, 300.0), st.floats(1.001, 5.0))
    def test_it_increases_with_radiance(self, radiance, factor) -> None:
        low = brightness_temperature_band(radiance, GOES)
        assert low < brightness_temperature_band(radiance * factor, GOES)

    @given(st.floats(150.0, 400.0))
    def test_it_inverts_its_own_planck_function(self, t) -> None:
        # L = fk₁ / (exp(fk₂ / (b₁ + b₂ T)) − 1)
        radiance = GOES.fk1 / np.expm1(GOES.fk2 / (GOES.bc1 + GOES.bc2 * t))
        assert_matches_oracle(
            brightness_temperature_band(radiance, GOES), t, rtol=1e-13
        )


class TestPlanckQuadratic:
    @pytest.mark.parametrize("constants", [STANDARD, HEADER])
    def test_the_himawari_reader(self, constants) -> None:
        # gate G2-readers; the reader's radiance is per µm
        per_um = np.concatenate([np.geomspace(1e-2, 40.0, 50), [0.0, -0.3]])
        expected = _geoproducts_himawari(per_um, AHI, constants)
        temperature = brightness_temperature_band(
            per_um * 1e6, AHI, constants=constants
        )
        assert_matches_oracle(temperature, expected, rtol=0.0, atol=1e-6)

    @given(st.floats(3e-6, 1.5e-5), st.floats(1e-3, 50.0))
    def test_without_a_correction_it_is_monochromatic(self, wavelength, per_um) -> None:
        nu = 1e-2 / wavelength
        expected = brightness_temperature(nu, from_per_um(per_um, nu))
        record = PlanckQuadratic(wavelength=wavelength)
        got = brightness_temperature_band(per_um * 1e6, record)
        assert_matches_oracle(got, expected, rtol=1e-12)

    def test_the_constants_matter(self) -> None:
        standard = brightness_temperature_band(8e6, AHI)
        header = brightness_temperature_band(8e6, AHI, constants=HEADER)
        assert 1e-5 < abs(float(standard - header)) < 1e-3


@pytest.mark.parametrize("record", [GOES, LANDSAT, AHI])
class TestBoth:
    def test_a_radiance_that_is_not_positive_is_nan(self, record) -> None:
        radiance = np.asarray([0.0, -1.0, np.nan, 10.0])
        temperature = brightness_temperature_band(radiance, record)
        assert np.isnan(temperature).tolist() == [True, True, True, False]

    def test_backends(self, record) -> None:
        unit = 1e6 if isinstance(record, PlanckQuadratic) else 1.0
        radiance = unit * np.linspace(1.0, 12.0, 6).reshape(2, 3)
        assert_backends_agree(
            lambda r: brightness_temperature_band(r, record), radiance, rtol32=5e-6
        )
        assert_traces(
            lambda r: brightness_temperature_band(r, record), radiance.ravel()
        )

    def test_gradient(self, record) -> None:
        unit = 1e6 if isinstance(record, PlanckQuadratic) else 1.0
        assert_grad_matches_fd(
            lambda r: brightness_temperature_band(unit * r, record),
            np.asarray([1.0, 5.0, 12.0]),
        )

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self, record) -> None:
        import jax

        gradient = jax.grad(
            lambda r: jax.numpy.nansum(brightness_temperature_band(r, record))
        )(np.asarray([0.0, -1.0, 10.0]))
        assert np.isfinite(gradient).all()
        assert gradient[2] > 0.0

    @needs_jax
    def test_a_record_crosses_jit_and_is_fitted(self, record) -> None:
        import jax

        radiance = np.asarray([2.0, 8.0])
        jitted = jax.jit(brightness_temperature_band)(radiance, record)
        expected = brightness_temperature_band(radiance, record)
        assert_matches_oracle(jitted, expected, rtol=1e-13)
        # the gradient is a record of the same type, with a number in each field
        gradient = jax.grad(lambda r: brightness_temperature_band(radiance, r).sum())(
            record
        )
        assert type(gradient) is type(record)
        fields = dataclasses.astuple(gradient)
        assert all(np.isfinite(float(field)) for field in fields)

    def test_coefficients_per_band_broadcast(self, record) -> None:
        scale = np.asarray([1.0, 1.01])
        fields = dataclasses.astuple(record)
        arrays = [(field * scale)[:, None] for field in fields]
        radiance = np.full((2, 3), 10.0)
        together = brightness_temperature_band(radiance, type(record)(*arrays))
        for band, factor in enumerate(scale):
            alone = type(record)(*(field * factor for field in fields))
            expected = brightness_temperature_band(radiance[band], alone)
            assert_matches_oracle(together[band], expected, rtol=1e-13)


@needs_xarray
def test_a_record_of_dataarrays_aligns_by_name() -> None:
    import xarray as xr

    bands = ["B13", "B14"]
    cube = xr.DataArray(
        np.full((3, 4, 2), 60.0), dims=("y", "x", "band"), coords={"band": bands}
    )
    per_band = {"dims": "band", "coords": {"band": bands}}
    record = PlanckLinear(
        fk1=xr.DataArray([10803.3, 8510.0], **per_band),
        fk2=xr.DataArray([1392.74, 1286.0], **per_band),
    )
    temperature = brightness_temperature_band(cube, record)
    assert temperature.dims == ("y", "x", "band")
    assert list(temperature["band"].values) == bands
    alone = brightness_temperature_band(60.0, PlanckLinear(8510.0, 1286.0))
    assert_matches_oracle(
        temperature.sel(band="B14").values, np.full((3, 4), alone), rtol=1e-14
    )
