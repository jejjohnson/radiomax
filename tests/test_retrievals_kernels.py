"""The Ross–Li kernels against radtrax's values, and their symmetries."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import radtrax_values
from hypothesis import given, strategies as st

from radiomax.retrievals import (
    li_sparse_reciprocal,
    li_sparse_reciprocal_from_cosines,
    ross_thick,
    ross_thick_from_cosines,
)


ZENITH = st.floats(0.0, 80.0)
AZIMUTH = st.floats(-180.0, 360.0)
ANGLE_FORM = {"ross": ross_thick, "li": li_sparse_reciprocal}
KERNELS = pytest.mark.parametrize("kernel", [ross_thick, li_sparse_reciprocal])


def _stored_angles() -> tuple[np.ndarray, ...]:
    """Radtrax's cosines as zeniths, and its azimuths, in radians."""
    stored = radtrax_values()
    return (
        np.arccos(stored["kernel_cos_solar"]),
        np.arccos(stored["kernel_cos_view"]),
        stored["kernel_relative_azimuth"],
    )


@pytest.mark.parametrize(
    ("kernel", "name"), [(ross_thick, "ross"), (li_sparse_reciprocal, "li")]
)
class TestAgainstRadtrax:
    def test_values(self, kernel, name) -> None:
        # gate G3-kernels: random angles, the hotspot (0–9) and nadir (10–13)
        expected = radtrax_values()[f"kernel_{name}"]
        got = kernel(*_stored_angles(), degrees=False)
        assert_matches_oracle(got, expected, rtol=0.0, atol=1e-13)

    @needs_jax
    def test_gradients(self, kernel, name) -> None:
        # gate G3-kernels; radtrax differentiates by cos θ: ∂/∂θ = −sin θ ∂/∂μ
        import jax

        solar, view, azimuth = _stored_angles()
        by_mu = radtrax_values()[f"kernel_{name}_gradient"]
        expected = np.stack(
            [-np.sin(solar) * by_mu[0], -np.sin(view) * by_mu[1], by_mu[2]]
        )
        gradient = jax.vmap(
            jax.grad(lambda *v: kernel(*v, degrees=False), argnums=(0, 1, 2))
        )
        got = np.stack(gradient(solar, view, azimuth))
        assert_matches_oracle(got, expected, rtol=1e-12, atol=1e-13)


@KERNELS
class TestKernels:
    @given(ZENITH, ZENITH, AZIMUTH)
    def test_reciprocity(self, kernel, solar, view, azimuth) -> None:
        # gate G3-reciprocal, relative to the largest term, 2 sec θₛ sec θᵥ
        scale = 2.0 / math.cos(math.radians(solar)) / math.cos(math.radians(view))
        difference = kernel(solar, view, azimuth) - kernel(view, solar, azimuth)
        assert abs(float(difference)) <= 1e-14 * scale

    def test_zero_at_nadir(self, kernel) -> None:
        # gate G3-reciprocal; the azimuth does not matter there
        values = kernel(0.0, 0.0, np.linspace(0.0, 180.0, 7))
        assert_matches_oracle(values, np.zeros(7), rtol=0.0, atol=1e-15)

    @given(ZENITH, ZENITH, st.floats(0.0, 180.0))
    def test_the_azimuth_is_even_and_periodic(
        self, kernel, solar, view, azimuth
    ) -> None:
        value = kernel(solar, view, azimuth)
        for same in (-azimuth, azimuth + 360.0):
            assert abs(float(kernel(solar, view, same) - value)) < 1e-10

    def test_the_hotspot_is_a_maximum_over_azimuth(self, kernel) -> None:
        values = kernel(45.0, 45.0, np.linspace(0.0, 180.0, 19))
        assert np.argmax(values) == 0
        assert (np.diff(values[:10]) < 0).all()

    def test_nan_from_the_horizon_on(self, kernel) -> None:
        zenith = np.asarray([0.0, 89.0, 90.0, 91.0, 180.0])
        for values in (kernel(zenith, 30.0, 60.0), kernel(30.0, zenith, 60.0)):
            assert np.isnan(values).tolist() == [False, False, True, True, True]

    @needs_jax
    def test_gradients_are_finite_at_nadir_the_hotspot_and_the_horizon(
        self, kernel
    ) -> None:
        import jax

        solar = np.asarray([0.0, 30.0, 0.0, 30.0, 90.0, 120.0])
        view = np.asarray([0.0, 0.0, 30.0, 30.0, 20.0, 20.0])
        azimuth = np.asarray([0.0, 0.0, 0.0, 0.0, 40.0, 40.0])
        gradient = jax.grad(lambda *v: jax.numpy.nansum(kernel(*v)), argnums=(0, 1, 2))(
            solar, view, azimuth
        )
        assert np.isfinite(np.stack(gradient)).all()

    def test_radians(self, kernel) -> None:
        degrees = kernel(40.0, 25.0, 100.0)
        radians = kernel(*np.deg2rad([40.0, 25.0, 100.0]), degrees=False)
        assert_matches_oracle(radians, degrees, rtol=1e-14)

    def test_float32(self, kernel) -> None:
        rng = np.random.default_rng(0)
        angles = rng.uniform(0.0, 1.0, (3, 500)) * np.asarray([[75.0], [60.0], [180.0]])
        expected = kernel(*angles)
        got = kernel(*angles.astype(np.float32))
        assert got.dtype == np.float32
        # near the hotspot Li's kernel has a kink, where D² is all rounding
        assert_matches_oracle(got, expected, rtol=1e-5, atol=2e-5)

    def test_gradient(self, kernel) -> None:
        assert_grad_matches_fd(
            lambda x: kernel(*x), np.asarray([35.0, 20.0, 70.0]), eps=1e-4
        )

    def test_backends(self, kernel) -> None:
        solar = np.linspace(5.0, 70.0, 6).reshape(2, 3)
        view = np.linspace(0.0, 60.0, 6).reshape(2, 3)[:, ::-1]
        azimuth = np.linspace(0.0, 180.0, 6).reshape(2, 3)
        assert_backends_agree(kernel, solar, view, azimuth, rtol32=1e-4)
        batch = [v.ravel() for v in (solar, view, azimuth)]
        assert_traces(kernel, *batch)


class TestClosedForms:
    def test_ross_at_nadir_view(self) -> None:
        # ξ = θₛ
        zenith = np.linspace(0.0, 85.0, 18)
        xi = np.deg2rad(zenith)
        expected = ((math.pi / 2 - xi) * np.cos(xi) + np.sin(xi)) / (
            np.cos(xi) + 1.0
        ) - math.pi / 4
        assert_matches_oracle(ross_thick(zenith, 0.0, 33.0), expected, rtol=1e-12)

    def test_li_at_the_hotspot(self) -> None:
        # D = 0, so cos t = 0, t = π/2 and O = sec θ; ξ = 0
        zenith = np.linspace(0.0, 80.0, 17)
        secant = 1.0 / np.cos(np.deg2rad(zenith))
        expected = secant - 2.0 * secant + secant**2
        got = li_sparse_reciprocal(zenith, zenith, 0.0)
        assert_matches_oracle(got, expected, rtol=1e-12, atol=1e-14)

    def test_li_without_overlap(self) -> None:
        # far apart the shadows do not overlap, and O = 0
        sec_s, sec_v = (1.0 / math.cos(math.radians(a)) for a in (60.0, 60.0))
        cos_xi = math.cos(math.radians(120.0))
        expected = -sec_s - sec_v + 0.5 * (1.0 + cos_xi) * sec_s * sec_v
        got = li_sparse_reciprocal(60.0, 60.0, 180.0)
        assert_matches_oracle(got, expected, rtol=1e-12)

    def test_the_crown_shape_matters(self) -> None:
        values = [
            float(li_sparse_reciprocal(40.0, 20.0, 60.0, height_over_breadth=h))
            for h in (1.0, 2.0, 2.5)
        ]
        # taller crowns overlap less
        assert values[0] > values[1] > values[2]

    def test_dataarrays(self) -> None:
        xr = pytest.importorskip("xarray")
        view = xr.DataArray(np.linspace(0.0, 60.0, 6).reshape(2, 3), dims=("y", "x"))
        for kernel in (ross_thick, li_sparse_reciprocal):
            assert kernel(30.0, view, 10.0).dims == ("y", "x")


@pytest.mark.parametrize(
    ("kernel", "name"),
    [(ross_thick_from_cosines, "ross"), (li_sparse_reciprocal_from_cosines, "li")],
)
class TestFromCosines:
    def test_radtraxs_values(self, kernel, name) -> None:
        stored = radtrax_values()
        got = kernel(
            stored["kernel_cos_solar"],
            stored["kernel_cos_view"],
            stored["kernel_relative_azimuth"],
            degrees=False,
        )
        assert_matches_oracle(got, stored[f"kernel_{name}"], rtol=0.0, atol=1e-13)

    @needs_jax
    def test_radtraxs_gradients(self, kernel, name) -> None:
        # in the cosines themselves; π/8 for RossThick at exact nadir
        import jax

        stored = radtrax_values()
        gradient = jax.vmap(
            jax.grad(lambda *v: kernel(*v, degrees=False), argnums=(0, 1, 2))
        )
        got = gradient(
            stored["kernel_cos_solar"],
            stored["kernel_cos_view"],
            stored["kernel_relative_azimuth"],
        )
        expected = stored[f"kernel_{name}_gradient"]
        assert_matches_oracle(np.stack(got), expected, rtol=1e-13, atol=1e-13)
        if name == "ross":
            nadir = np.stack(got)[:2, 10]
            assert_matches_oracle(nadir, np.full(2, math.pi / 8.0), rtol=1e-14)

    @given(ZENITH, ZENITH, st.floats(0.0, 180.0))
    def test_the_angle_form(self, kernel, name, solar, view, azimuth) -> None:
        cosines = np.cos(np.deg2rad([solar, view]))
        expected = ANGLE_FORM[name](solar, view, azimuth)
        got = kernel(*cosines, azimuth)
        # 1 − μ² keeps half of the digits near nadir and the hotspot
        assert_matches_oracle(got, expected, rtol=1e-6, atol=1e-6)

    def test_a_cosine_that_rounds_past_one(self, kernel, name) -> None:
        assert abs(float(kernel(1.0 + 4e-16, 1.0, 0.0))) < 1e-14

    def test_a_cosine_that_is_not_positive_is_nan(self, kernel, name) -> None:
        cosine = np.asarray([0.5, 0.0, -0.1, np.nan])
        for values in (kernel(cosine, 0.8, 20.0), kernel(0.8, cosine, 20.0)):
            assert np.isnan(values).tolist() == [False, True, True, True]

    @needs_jax
    def test_gradients_are_finite_beside_an_invalid_pixel(self, kernel, name) -> None:
        import jax

        gradient = jax.grad(lambda *v: jax.numpy.nansum(kernel(*v)), argnums=(0, 1, 2))(
            np.asarray([1.0, 0.8, 0.0, -0.1]),
            np.asarray([1.0, 0.8, 0.5, 0.5]),
            np.asarray([0.0, 0.0, 40.0, 40.0]),
        )
        assert np.isfinite(np.stack(gradient)).all()

    def test_gradient_and_backends(self, kernel, name) -> None:
        assert_grad_matches_fd(
            lambda x: kernel(x[0], x[1], 100.0 * x[2]), np.asarray([0.8, 0.9, 0.7])
        )
        solar = np.linspace(0.3, 0.95, 6).reshape(2, 3)
        view = solar[:, ::-1]
        azimuth = np.linspace(10.0, 170.0, 6).reshape(2, 3)
        assert_backends_agree(kernel, solar, view, azimuth, rtol32=1e-4)
        assert_traces(kernel, *(v.ravel() for v in (solar, view, azimuth)))
