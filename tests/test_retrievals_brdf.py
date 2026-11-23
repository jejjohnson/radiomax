"""The kernel BRDF, and the adjustment to nadir view."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import radtrax_values
from hypothesis import given, strategies as st

from radiomax.retrievals import (
    brdf_reflectance,
    li_sparse_reciprocal,
    nadir_adjustment,
    ross_thick,
)


# of the size the MODIS product has over vegetation in the near infrared
WEIGHTS = (0.25, 0.15, 0.03)
ZENITH, AZIMUTH = st.floats(0.0, 75.0), st.floats(0.0, 180.0)


class TestBrdfReflectance:
    def test_the_sum_of_radtraxs_kernels(self) -> None:
        stored = radtrax_values()
        got = brdf_reflectance(
            *WEIGHTS,
            np.arccos(stored["kernel_cos_solar"]),
            np.arccos(stored["kernel_cos_view"]),
            stored["kernel_relative_azimuth"],
            degrees=False,
        )
        expected = 0.25 + 0.15 * stored["kernel_ross"] + 0.03 * stored["kernel_li"]
        assert_matches_oracle(got, expected, rtol=0.0, atol=1e-13)

    @given(st.floats(0.0, 1.0), ZENITH, ZENITH, AZIMUTH)
    def test_a_lambertian_surface(self, f_iso, solar, view, azimuth) -> None:
        # gate G3-lambert: exact
        assert brdf_reflectance(f_iso, 0.0, 0.0, solar, view, azimuth) == f_iso

    def test_linear_in_the_weights(self) -> None:
        angles = (40.0, 25.0, 100.0)
        parts = [brdf_reflectance(*row, *angles) for row in np.eye(3)]
        assert_matches_oracle(parts[1], ross_thick(*angles), rtol=1e-15)
        assert_matches_oracle(parts[2], li_sparse_reciprocal(*angles), rtol=1e-15)
        total = sum(w * p for w, p in zip(WEIGHTS, parts, strict=True))
        assert_matches_oracle(brdf_reflectance(*WEIGHTS, *angles), total, rtol=1e-15)

    def test_gradient_and_backends(self) -> None:
        assert_grad_matches_fd(
            lambda x: brdf_reflectance(*x[:3], *(100.0 * x[3:])),
            np.asarray([*WEIGHTS, 0.35, 0.2, 0.7]),
        )
        solar = np.linspace(5.0, 70.0, 6).reshape(2, 3)
        f_iso = np.full((2, 3), 0.3)
        assert_backends_agree(
            lambda f, s: brdf_reflectance(f, 0.1, 0.02, s, 30.0, 50.0),
            f_iso,
            solar,
            rtol32=1e-5,
        )
        assert_traces(
            lambda f, s: brdf_reflectance(f, 0.1, 0.02, s, 30.0, 50.0),
            f_iso.ravel(),
            solar.ravel(),
        )


class TestNadirAdjustment:
    @given(st.floats(0.01, 1.0), ZENITH, ZENITH, AZIMUTH)
    def test_a_lambertian_surface(self, f_iso, solar, view, azimuth) -> None:
        # gate G3-lambert: exact
        assert nadir_adjustment(f_iso, 0.0, 0.0, solar, view, azimuth) == 1.0

    @given(ZENITH)
    def test_nothing_to_do_at_nadir(self, solar) -> None:
        # the azimuth is 0 in both, so the two reflectances are the same numbers
        factor = nadir_adjustment(*WEIGHTS, solar, 0.0, 0.0)
        assert abs(float(factor) - 1.0) <= 1e-15

    def test_the_ratio_of_the_model(self) -> None:
        solar, view, azimuth = 50.0, 35.0, np.linspace(0.0, 180.0, 7)
        own = nadir_adjustment(*WEIGHTS, solar, view, azimuth)
        observed = brdf_reflectance(*WEIGHTS, solar, view, azimuth)
        at_nadir = brdf_reflectance(*WEIGHTS, solar, 0.0, 0.0)
        assert_matches_oracle(own * observed, np.full(7, at_nadir), rtol=1e-14)
        fixed = nadir_adjustment(*WEIGHTS, solar, view, azimuth, reference_zenith=30.0)
        at_reference = brdf_reflectance(*WEIGHTS, 30.0, 0.0, 0.0)
        assert_matches_oracle(fixed * observed, np.full(7, at_reference), rtol=1e-14)

    def test_it_removes_the_view_from_a_model_scene(self) -> None:
        # one surface seen on many dates becomes one number
        rng = np.random.default_rng(0)
        solar, view = rng.uniform(20.0, 60.0, 50), rng.uniform(0.0, 55.0, 50)
        azimuth = rng.uniform(0.0, 180.0, 50)
        seen = brdf_reflectance(*WEIGHTS, solar, view, azimuth)
        adjusted = seen * nadir_adjustment(
            *WEIGHTS, solar, view, azimuth, reference_zenith=45.0
        )
        assert np.ptp(seen) > 0.05
        assert np.ptp(adjusted) < 1e-14

    def test_a_reflectance_that_is_not_positive_is_nan(self) -> None:
        # noisy weights: negative at the observation, then at nadir
        assert np.isnan(nadir_adjustment(0.01, 0.0, 0.5, 60.0, 60.0, 180.0))
        assert np.isnan(nadir_adjustment(-0.01, 0.0, 0.0, 30.0, 20.0, 10.0))
        assert np.isnan(nadir_adjustment(*WEIGHTS, 95.0, 20.0, 10.0))

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        f_iso = np.asarray([-0.01, 0.25, 0.25])
        solar = np.asarray([30.0, 95.0, 30.0])
        gradient = jax.grad(
            lambda f, s: jax.numpy.nansum(
                nadir_adjustment(f, 0.1, 0.02, s, 40.0, 10.0)
            ),
            argnums=(0, 1),
        )(f_iso, solar)
        assert np.isfinite(np.stack(gradient)).all()

    def test_radians_gradient_and_backends(self) -> None:
        degrees = nadir_adjustment(*WEIGHTS, 50.0, 35.0, 60.0, reference_zenith=45.0)
        angles = np.deg2rad([50.0, 35.0, 60.0])
        radians = nadir_adjustment(
            *WEIGHTS, *angles, reference_zenith=np.deg2rad(45.0), degrees=False
        )
        assert_matches_oracle(radians, degrees, rtol=1e-14)
        assert_grad_matches_fd(
            lambda x: nadir_adjustment(*x[:3], *(100.0 * x[3:])),
            np.asarray([*WEIGHTS, 0.35, 0.2, 0.7]),
        )
        solar = np.linspace(5.0, 70.0, 6).reshape(2, 3)
        assert_backends_agree(
            lambda s: nadir_adjustment(*WEIGHTS, s, 30.0, 50.0), solar, rtol32=1e-5
        )
        assert_traces(
            lambda s: nadir_adjustment(*WEIGHTS, s, 30.0, 50.0), solar.ravel()
        )

    def test_weights_on_the_band_dimension(self) -> None:
        xr = pytest.importorskip("xarray")
        per_band = {"dims": "band", "coords": {"band": ["red", "nir"]}}
        f_iso = xr.DataArray([0.05, 0.25], **per_band)
        f_vol = xr.DataArray([0.02, 0.15], **per_band)
        view = xr.DataArray(np.linspace(0.0, 50.0, 6).reshape(2, 3), dims=("y", "x"))
        factor = nadir_adjustment(f_iso, f_vol, 0.01, 40.0, view, 20.0)
        assert factor.dims == ("band", "y", "x")
        assert list(factor["band"].values) == ["red", "nir"]
