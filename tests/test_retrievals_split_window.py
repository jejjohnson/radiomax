"""Split-window surface temperatures from given coefficients."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import KINDS, assert_backends_agree, needs_jax, needs_xarray
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from radiomax.exceptions import ValidationError
from radiomax.retrievals import (
    split_window_features,
    split_window_lst,
    split_window_sst,
)


# for the tests only; not a published set
SST = (1.5, 0.98, 2.1, 0.7)
LST = {
    "quadratic": (2.0, 1.0, 1.8, 0.3, 50.0, -4.0, -120.0, 15.0),
    "generalised": (3.0, 0.99, 0.15, -0.3, 4.0, 5.0, -10.0),
}
FORMS = pytest.mark.parametrize("form", ["quadratic", "generalised"])
TEMPERATURE = st.floats(200.0, 340.0)


def _scene(n: int = 40) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    t11 = rng.uniform(260.0, 320.0, n)
    return (
        t11,
        t11 - rng.uniform(-1.0, 4.0, n),
        rng.uniform(0.92, 1.0, n),
        rng.uniform(-0.02, 0.02, n),
        rng.uniform(0.2, 6.0, n),
    )


class TestSst:
    def test_equal_bands_at_nadir(self) -> None:
        # gate G3-window: exact
        t = np.asarray([275.0, 290.0, 303.15])
        assert_matches_oracle(
            split_window_sst(t, t, 0.0, SST), 1.5 + 0.98 * t, rtol=0.0
        )

    @given(TEMPERATURE, st.floats(-2.0, 6.0), st.floats(0.0, 75.0))
    def test_the_formula(self, t11, split, zenith) -> None:
        t12 = t11 - split
        path = 1.0 / np.cos(np.deg2rad(zenith)) - 1.0
        expected = 1.5 + 0.98 * t11 + 2.1 * (t11 - t12) + 0.7 * path
        got = split_window_sst(t11, t12, zenith, SST)
        assert_matches_oracle(got, expected, rtol=1e-13)

    def test_nan_from_the_horizon_on(self) -> None:
        zenith = np.asarray([0.0, 89.0, 90.0, 120.0])
        temperature = split_window_sst(290.0, 289.0, zenith, SST)
        assert np.isnan(temperature).tolist() == [False, False, True, True]

    @needs_jax
    def test_the_gradient_beside_an_invalid_pixel_is_finite(self) -> None:
        import jax

        gradient = jax.grad(
            lambda z: jax.numpy.nansum(split_window_sst(290.0, 289.0, z, SST))
        )(np.asarray([0.0, 40.0, 90.0, 120.0]))
        assert np.isfinite(gradient).all()
        assert gradient[1] > 0.0

    def test_radians_gradient_and_backends(self) -> None:
        radians = split_window_sst(290.0, 288.0, np.deg2rad(40.0), SST, degrees=False)
        degrees = split_window_sst(290.0, 288.0, 40.0, SST)
        assert_matches_oracle(radians, degrees, rtol=1e-14)
        assert_grad_matches_fd(
            lambda x: split_window_sst(300.0 * x[0], 290.0 * x[1], 60.0 * x[2], x[3:]),
            np.asarray([0.97, 1.0, 0.6, *SST]),
        )
        t11 = np.linspace(270.0, 300.0, 6).reshape(2, 3)
        zenith = np.linspace(0.0, 60.0, 6).reshape(2, 3)
        assert_backends_agree(
            lambda a, b, z: split_window_sst(a, b, z, SST), t11, t11 - 1.5, zenith
        )
        assert_traces(
            lambda a, z: split_window_sst(a, 280.0, z, SST), t11.ravel(), zenith.ravel()
        )


@FORMS
class TestLst:
    def test_the_sum_of_the_features(self, form) -> None:
        scene = _scene()
        features = split_window_features(*scene, form=form)
        assert features.shape == (40, len(LST[form]))
        got = split_window_lst(*scene, LST[form], form=form)
        assert_matches_oracle(got, features @ np.asarray(LST[form]), rtol=1e-13)

    def test_linear_in_the_coefficients(self, form) -> None:
        scene = _scene()
        a = np.asarray(LST[form])
        b = np.random.default_rng(1).standard_normal(a.size)
        together = split_window_lst(*scene, a + 2.0 * b, form=form)
        apart = split_window_lst(*scene, a, form=form) + 2.0 * split_window_lst(
            *scene, b, form=form
        )
        assert_matches_oracle(together, apart, rtol=0.0, atol=1e-10)

    def test_a_fit_recovers_the_coefficients(self, form) -> None:
        scene = _scene(400)
        features = split_window_features(*scene, form=form)
        temperature = split_window_lst(*scene, LST[form], form=form)
        # columns scaled to one size, so the solve is well conditioned
        scale = np.linalg.norm(features, axis=0)
        fitted, *_ = np.linalg.lstsq(features / scale, temperature, rcond=None)
        assert_matches_oracle(fitted / scale, np.asarray(LST[form]), rtol=1e-6)

    def test_a_black_surface_and_equal_bands(self, form) -> None:
        # ε̄ = 1, Δε = 0 and T₁₁ = T₁₂ leave a₀ + a₁ T₁₁
        got = split_window_lst(300.0, 300.0, 1.0, 0.0, 2.0, LST[form], form=form)
        a0, a1 = LST[form][:2]
        assert_matches_oracle(got, a0 + a1 * 300.0, rtol=1e-15)

    def test_coefficients_per_pixel(self, form) -> None:
        scene = _scene(3)
        strata = np.asarray(LST[form]) * np.asarray([[1.0], [1.01], [0.99]])
        together = split_window_lst(*scene, strata, form=form)
        for pixel in range(3):
            alone = split_window_lst(
                *(v[pixel] for v in scene), strata[pixel], form=form
            )
            assert_matches_oracle(together[pixel], alone, rtol=1e-15)

    def test_the_wrong_number_of_coefficients(self, form) -> None:
        for coefficients in (LST[form][:-1], (*LST[form], 1.0), []):
            with pytest.raises(ValidationError, match="the formula needs"):
                split_window_lst(*_scene(3), coefficients, form=form)

    def test_gradient_and_backends(self, form) -> None:
        scene = _scene(6)
        assert_backends_agree(
            lambda *v: split_window_lst(*v, LST[form], form=form), *scene, rtol32=2e-5
        )
        # differences of two temperatures keep fewer digits in float32
        assert_backends_agree(
            lambda *v: split_window_features(*v, form=form), *scene, rtol32=1e-2
        )
        assert_traces(lambda *v: split_window_lst(*v, LST[form], form=form), *scene)
        assert_traces(lambda *v: split_window_features(*v, form=form), *scene)
        assert_grad_matches_fd(
            lambda x: split_window_lst(
                300.0 * x[0], 295.0 * x[1], x[2], x[3], x[4], x[5:], form=form
            ),
            np.asarray([0.98, 0.99, 0.97, 0.01, 2.0, *LST[form]]),
        )


class TestForms:
    def test_the_quadratic_features(self) -> None:
        t1, t2, mean, diff, water = _scene(5)
        expected = [
            np.ones(5),
            t1,
            t1 - t2,
            (t1 - t2) ** 2,
            1.0 - mean,
            water * (1.0 - mean),
            diff,
            water * diff,
        ]
        got = split_window_features(t1, t2, mean, diff, water, form="quadratic")
        assert_matches_oracle(got, np.stack(expected, axis=-1), rtol=1e-13)

    def test_the_generalised_features(self) -> None:
        t1, t2, mean, diff, water = _scene(5)
        vector = np.stack([np.ones(5), (1.0 - mean) / mean, diff / mean**2], axis=-1)
        parts = [0.5 * (t1 + t2)[:, None] * vector, 0.5 * (t1 - t2)[:, None] * vector]
        expected = np.concatenate([np.ones((5, 1)), *parts], axis=-1)
        got = split_window_features(t1, t2, mean, diff, water, form="generalised")
        assert_matches_oracle(got, expected, rtol=1e-13)

    def test_no_emissivity_is_nan(self) -> None:
        mean = np.asarray([0.97, 0.0, -0.1])
        got = split_window_lst(
            300.0, 299.0, mean, 0.0, 2.0, LST["generalised"], form="generalised"
        )
        assert np.isnan(got).tolist() == [False, True, True]

    def test_an_unknown_form_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="'form' must be"):
            split_window_features(1.0, 1.0, 1.0, 0.0, 0.0, form="cubic")  # ty: ignore

    def test_a_single_number_is_not_coefficients(self) -> None:
        with pytest.raises(ValidationError, match="must be a sequence or an array"):
            split_window_sst(290.0, 289.0, 0.0, 1.0)


@pytest.mark.parametrize("kind", KINDS, ids=lambda kind: kind.name)
def test_coefficients_as_an_array_of_each_kind(kind) -> None:
    t11 = np.linspace(270.0, 300.0, 4)
    coefficients = kind.make(np.asarray(SST))
    if kind.name == "xarray":
        coefficients = coefficients.rename(dim1="feature")
    got = split_window_sst(kind.make(t11), kind.make(t11 - 1.0), 20.0, coefficients)
    assert_matches_oracle(
        kind.read(got), split_window_sst(t11, t11 - 1.0, 20.0, SST), rtol=1e-14
    )


@needs_xarray
class TestDataArrays:
    def test_features_get_a_dimension(self) -> None:
        import xarray as xr

        t11 = xr.DataArray(
            np.full((2, 3), 300.0), dims=("y", "x"), coords={"x": [5, 6, 7]}
        )
        for form, count in (("quadratic", 8), ("generalised", 7)):
            features = split_window_features(t11, t11 - 1.0, 0.97, 0.01, 2.0, form=form)
            assert features.dims == ("y", "x", "feature")
            assert features.sizes["feature"] == count
            assert list(features["x"].values) == [5, 6, 7]
            coefficients = xr.DataArray(list(LST[form]), dims="feature")
            # the features and the coefficients meet by name
            expected = (features * coefficients).sum("feature")
            got = split_window_lst(
                t11, t11 - 1.0, 0.97, 0.01, 2.0, coefficients, form=form
            )
            assert got.dims == ("y", "x")
            assert_matches_oracle(got.values, expected.values, rtol=1e-13)

    def test_coefficients_by_stratum(self) -> None:
        import xarray as xr

        t11 = xr.DataArray(np.full(3, 300.0), dims="x")
        by_region = xr.DataArray(
            np.asarray([SST, SST]),
            dims=("region", "feature"),
            coords={"region": ["a", "b"], "feature": list("abcd")},
        )
        got = split_window_sst(t11, t11 - 1.0, 0.0, by_region)
        assert set(got.dims) == {"x", "region"}
        assert "feature" not in got.coords

    def test_coefficients_without_the_dimension_are_refused(self) -> None:
        import xarray as xr

        with pytest.raises(ValidationError, match="no dimension 'feature'"):
            split_window_sst(290.0, 289.0, 0.0, xr.DataArray([1.0] * 4, dims="a"))
