"""Every public function stays on its input's device, and keeps torch's gradients."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import array_api_strict as strict
import pytest
from _backends import HAS_TORCH

from radiomax import altimetry, geometry, radiometry, retrievals, sar, thermal


MODULES = (geometry, radiometry, thermal, retrievals, sar, altimetry)
SECOND = strict.Device("device1")

#: Functions that return NumPy whatever they are given, or take no array.
HOST_ONLY = {
    "response_coverage",
    "response_matrix",
    "scan_angles",
    "utc_day_seconds",
}

Make = Callable[..., Any]
GOES = {"lon_0": 0.0, "height": 35_786_023.0, "sweep": "x"}
SATELLITE = (42_164_160.0, 0.0, 0.0)
GRID = (2100.0, 2200.0, 2300.0, 2400.0)


def _matrix(make: Make) -> Any:
    """(2, 4) responses."""
    return make([[0.1, 0.8, 0.4, 0.2], [0.2, 0.3, 0.9, 0.1]])


#: A call of each function, whose first array comes from ``make``. The other
#: arguments are host values wherever they can be, because those have to move.
CALLS: dict[str, Callable[[Make], Any]] = {
    "air_mass_factor": lambda m: geometry.air_mass_factor(m([20.0, 40.0]), 30.0),
    "apparent_position": lambda m: geometry.apparent_position(
        m([20.0, 40.0]), 10.0, 1e4, *SATELLITE
    ),
    "bowtie_detector_positions": lambda m: geometry.bowtie_detector_positions(
        m([1.1, 1.6]), 10
    ),
    "earth_intersection": lambda m: geometry.earth_intersection(
        m([2e7, 1e7]), 0.0, 0.0, -1.0, 0.1, 0.1
    ),
    "earth_sun_distance": lambda m: geometry.earth_sun_distance(m([100.0, 9000.0])),
    "geostationary_lonlat": lambda m: geometry.geostationary_lonlat(
        m([0.01, 0.05]), 0.02, **GOES
    ),
    "geostationary_on_earth": lambda m: geometry.geostationary_on_earth(
        m([0.01, 0.16]), 0.02, height=35_786_023.0, sweep="x"
    ),
    "geostationary_scan_angles": lambda m: geometry.geostationary_scan_angles(
        m([10.0, 30.0]), 20.0, **GOES
    ),
    "phase_angle": lambda m: geometry.phase_angle(m([20.0, 40.0]), 30.0, 50.0),
    "relative_azimuth": lambda m: geometry.relative_azimuth(m([20.0, 40.0]), 100.0),
    "scan_pixel_growth": lambda m: geometry.scan_pixel_growth(m([10.0, 50.0]), 7e5),
    "scattering_angle": lambda m: geometry.scattering_angle(
        m([20.0, 40.0]), 30.0, 50.0
    ),
    "shell_air_mass": lambda m: geometry.shell_air_mass(m([0.5, 0.9]), 0.0, 1e3),
    "solar_angles": lambda m: geometry.solar_angles(m([10.0, 30.0]), 20.0, 9000, 4e4),
    "solar_declination": lambda m: geometry.solar_declination(m([100.0, 9000.0])),
    "solar_zenith": lambda m: geometry.solar_zenith(m([10.0, 30.0]), 20.0, 9000, 4e4),
    "view_angles": lambda m: geometry.view_angles(m([10.0, 30.0]), 20.0, *SATELLITE),
    "apply_response": lambda m: radiometry.apply_response(
        _matrix(m), [1.0, 2.0, 3.0, 4.0]
    ),
    "apply_response_adjoint": lambda m: radiometry.apply_response_adjoint(
        _matrix(m), [1.0, 2.0]
    ),
    "band_average": lambda m: radiometry.band_average(
        m([1.0, 2.0, 3.0, 4.0]),
        _matrix(list),
        GRID,
        where=[True, True, False, True],
        nan_policy="omit",
    ),
    "band_centre": lambda m: radiometry.band_centre(_matrix(m), GRID),
    "band_irradiance": lambda m: radiometry.band_irradiance(
        m([1.0, 2.0, 3.0, 4.0]), _matrix(list), GRID
    ),
    "box_response": lambda m: radiometry.box_response(m(GRID), 2250.0, 200.0),
    "counts_to_radiance": lambda m: radiometry.counts_to_radiance(
        m([100.0, 200.0]), 0.1, -2.0
    ),
    "counts_to_reflectance": lambda m: radiometry.counts_to_reflectance(
        m([100.0, 200.0]), 1e-4, -0.1
    ),
    "equivalent_width": lambda m: radiometry.equivalent_width(_matrix(m), GRID),
    "gaussian_response": lambda m: radiometry.gaussian_response(m(GRID), 2250.0, 200.0),
    "per_cm_to_per_nm": lambda m: radiometry.per_cm_to_per_nm(m([0.1, 0.2]), 4e3),
    "per_nm_to_per_cm": lambda m: radiometry.per_nm_to_per_cm(m([0.1, 0.2]), 2.3e3),
    "radiance_from_reflectance": lambda m: radiometry.radiance_from_reflectance(
        m([0.1, 0.2]), 245.0, 30.0
    ),
    "radiance_to_counts": lambda m: radiometry.radiance_to_counts(
        m([10.0, 20.0]), 0.1, -2.0
    ),
    "reflectance": lambda m: radiometry.reflectance(m([10.0, 20.0]), 245.0, 30.0),
    "sampled_response_matrix": lambda m: radiometry.sampled_response_matrix(
        _matrix(m), GRID
    ),
    "trapezoid_weights": lambda m: radiometry.trapezoid_weights(m(GRID)),
    "triangle_response": lambda m: radiometry.triangle_response(m(GRID), 2250.0, 200.0),
    "brightness_temperature": lambda m: thermal.brightness_temperature(
        1e3, m([0.1, 0.2])
    ),
    "brightness_temperature_band": lambda m: thermal.brightness_temperature_band(
        m([8.0, 12.0]), thermal.PlanckLinear(774.8853, 1321.0789)
    ),
    "from_per_ghz": lambda m: thermal.from_per_ghz(m([0.1, 0.2])),
    "from_per_um": lambda m: thermal.from_per_um(m([8.0, 9.0]), 1e3),
    "log_planck_radiance": lambda m: thermal.log_planck_radiance(
        1e3, m([250.0, 300.0])
    ),
    "planck_radiance": lambda m: thermal.planck_radiance(1e3, m([250.0, 300.0])),
    "to_per_ghz": lambda m: thermal.to_per_ghz(m([0.1, 0.2])),
    "to_per_um": lambda m: thermal.to_per_um(m([0.1, 0.2]), 1e3),
    "band_ratio_chlorophyll": lambda m: retrievals.band_ratio_chlorophyll(
        [m([0.005, 0.004]), 0.0045],
        0.003,
        retrievals.OCX_R2022["MODIS"].coefficients,
    ),
    "brdf_reflectance": lambda m: retrievals.brdf_reflectance(
        m([0.2, 0.3]), 0.1, 0.02, 30.0, 20.0, 50.0
    ),
    "li_sparse_reciprocal": lambda m: retrievals.li_sparse_reciprocal(
        m([20.0, 40.0]), 30.0, 50.0
    ),
    "li_sparse_reciprocal_from_cosines": lambda m: (
        retrievals.li_sparse_reciprocal_from_cosines(m([0.9, 0.7]), 0.8, 50.0)
    ),
    "nadir_adjustment": lambda m: retrievals.nadir_adjustment(
        m([0.2, 0.3]), 0.1, 0.02, 30.0, 20.0, 50.0
    ),
    "ross_thick": lambda m: retrievals.ross_thick(m([20.0, 40.0]), 30.0, 50.0),
    "ross_thick_from_cosines": lambda m: retrievals.ross_thick_from_cosines(
        m([0.9, 0.7]), 0.8, 50.0
    ),
    "split_window_features": lambda m: retrievals.split_window_features(
        m([300.0, 290.0]), 289.0, 0.97, 0.01, 2.0, form="generalised"
    ),
    "split_window_lst": lambda m: retrievals.split_window_lst(
        m([300.0, 290.0]), 289.0, 0.97, 0.01, 2.0, [1.0] * 8, form="quadratic"
    ),
    "split_window_sst": lambda m: retrievals.split_window_sst(
        m([300.0, 290.0]), 289.0, 40.0, [1.0, 1.0, 2.0, 0.5]
    ),
    "dry_troposphere_correction": lambda m: altimetry.dry_troposphere_correction(
        m([98e3, 101e3]), 45.0
    ),
    "dual_frequency_ionosphere_correction": lambda m: (
        altimetry.dual_frequency_ionosphere_correction(
            m([0.1, 0.2]), 0.9, 13.575e9, 5.3e9
        )
    ),
    "inverse_barometer": lambda m: altimetry.inverse_barometer(
        m([98e3, 101e3]), 101.1e3
    ),
    "ionosphere_correction": lambda m: altimetry.ionosphere_correction(
        m([1e17, 5e17]), 13.575e9
    ),
    "sea_surface_height": lambda m: altimetry.sea_surface_height(
        m([1000.0, 1010.0]), 960.0, -2.3, 0.1
    ),
    "wet_troposphere_correction": lambda m: altimetry.wet_troposphere_correction(
        m([10.0, 25.0]), 275.0
    ),
    "beta0_to_sigma0": lambda m: sar.beta0_to_sigma0(m([0.1, 0.2]), 40.0),
    "dn_to_backscatter": lambda m: sar.dn_to_backscatter(m([100.0, 200.0]), 300.0),
    "gamma0_to_sigma0": lambda m: sar.gamma0_to_sigma0(m([0.1, 0.2]), 40.0),
    "sigma0_to_beta0": lambda m: sar.sigma0_to_beta0(m([0.1, 0.2]), 40.0),
    "sigma0_to_gamma0": lambda m: sar.sigma0_to_gamma0(m([0.1, 0.2]), 40.0),
}


def _functions(modules: tuple[ModuleType, ...]) -> Iterator[str]:
    for module in modules:
        for name in module.__all__:
            if inspect.isfunction(getattr(module, name)):
                yield name


def _outputs(result: Any) -> tuple[Any, ...]:
    return result if isinstance(result, tuple) else (result,)


def test_every_public_function_is_here() -> None:
    assert set(_functions(MODULES)) == set(CALLS) | HOST_ONLY


@pytest.mark.parametrize("name", CALLS)
def test_the_result_is_on_the_inputs_device(name: str) -> None:
    def make(values: Any) -> Any:
        return strict.asarray(list(values), dtype=strict.float64, device=SECOND)

    for output in _outputs(CALLS[name](make)):
        assert output.device == SECOND


@pytest.mark.parametrize("name", CALLS)
def test_an_array_of_the_library_is_never_rebuilt(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `asarray` detaches a torch tensor in torch 2.5 and 2.10
    asarray = strict.asarray
    strict_type = type(asarray(0.0))

    def recording(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, strict_type):
            rebuilt.append(value)
        return asarray(value, *args, **kwargs)

    rebuilt: list[Any] = []
    monkeypatch.setattr(strict, "asarray", recording)
    CALLS[name](lambda values: asarray(list(values), dtype=strict.float64))
    assert rebuilt == []


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
@pytest.mark.parametrize("name", CALLS)
def test_torch_keeps_the_graph(name: str) -> None:
    import torch

    leaves = []

    def make(values: Any) -> Any:
        leaf = torch.tensor(list(values), dtype=torch.float64, requires_grad=True)
        leaves.append(leaf)
        return leaf

    outputs = [o for o in _outputs(CALLS[name](make)) if o.is_floating_point()]
    (leaf,) = leaves
    assert leaf.requires_grad
    if name in {"box_response", "geostationary_on_earth"}:
        # a step and a mask: no gradient reaches the input
        return
    sum(output.sum() for output in outputs if output.requires_grad).backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
