"""The plane-parallel and spherical-shell air masses."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import radtrax_values
from geophysix.earth import SPHERE
from hypothesis import given, strategies as st

from radiomax.geometry import air_mass_factor, shell_air_mass


HEIGHT = st.floats(0.0, 1e5)


class TestAirMassFactor:
    def test_radtraxs_values(self) -> None:
        # gate G1-shell
        stored = radtrax_values()
        solar, view = stored["air_mass_solar_zenith"], stored["air_mass_view_zenith"]
        expected = stored["air_mass_factor"]
        assert_matches_oracle(air_mass_factor(solar, view), expected, rtol=1e-14)
        radians = air_mass_factor(np.deg2rad(solar), np.deg2rad(view), degrees=False)
        assert_matches_oracle(radians, expected, rtol=1e-14)

    def test_two_at_nadir_and_nan_from_the_horizon_on(self) -> None:
        assert air_mass_factor(0.0, 0.0) == 2.0
        for pair in ((90.0, 0.0), (0.0, 90.0), (120.0, 0.0), (0.0, 180.0)):
            assert np.isnan(air_mass_factor(*pair))
        assert np.isfinite(air_mass_factor(np.float32(89.9), np.float32(0.0)))

    @given(st.floats(0.0, 89.9), st.floats(0.0, 89.9))
    def test_at_least_two_symmetric_and_increasing(self, solar, view) -> None:
        mass = air_mass_factor(solar, view)
        assert mass >= 2.0
        assert mass == air_mass_factor(view, solar)
        assert air_mass_factor(min(solar + 0.05, 89.95), view) >= mass

    @needs_jax
    def test_a_night_pixel_does_not_spoil_the_others_gradient(self) -> None:
        import jax
        import jax.numpy as jnp

        solar = jnp.asarray([30.0, 100.0, 90.0])
        gradient = jax.grad(lambda s: jnp.nansum(air_mass_factor(s, 10.0)))(solar)
        assert np.isfinite(np.asarray(gradient)).all()
        assert gradient[0] > 0.0
        assert gradient[1:].tolist() == [0.0, 0.0]

    def test_gradient_and_backends(self) -> None:
        solar = np.asarray([10.0, 45.0, 80.0])
        view = np.asarray([0.0, 20.0, 60.0])
        assert_grad_matches_fd(lambda s: air_mass_factor(s, view), solar, eps=1e-4)
        assert_backends_agree(air_mass_factor, solar, view)
        assert_traces(air_mass_factor, solar, view)


class TestShellAirMass:
    def test_radtraxs_values(self) -> None:
        # gate G1-shell, over 0–90° and 0–100 km
        stored = radtrax_values()
        mass = shell_air_mass(
            stored["shell_cosine"][:, None], stored["shell_bottom"], stored["shell_top"]
        )
        assert_matches_oracle(mass, stored["shell_air_mass"], rtol=1e-14)

    @given(st.floats(0.05, 1.0), HEIGHT, st.floats(10.0, 1e5))
    def test_the_slant_path_over_the_thickness(self, mu, bottom, thickness) -> None:
        # the form that cancels, in long double
        r, mu_, b = (np.longdouble(v) for v in (SPHERE.semi_major, mu, bottom))
        t = b + np.longdouble(thickness)
        sine = r**2 * (1 - mu_**2)
        path = np.sqrt((r + t) ** 2 - sine) - np.sqrt((r + b) ** 2 - sine)
        expected = float(path / (t - b))
        mass = shell_air_mass(mu, bottom, bottom + thickness)
        assert_matches_oracle(mass, expected, rtol=1e-9)

    @given(st.floats(0.0, 1.0), HEIGHT, HEIGHT)
    def test_between_one_and_the_plane_parallel_one(self, mu, bottom, top) -> None:
        bottom, top = min(bottom, top), max(bottom, top) + 1.0
        mass = float(shell_air_mass(mu, bottom, top))
        assert mass >= 1.0 - 1e-12
        assert mass * mu <= 1.0 + 1e-12

    def test_the_limits(self) -> None:
        at_30 = np.cos(np.deg2rad(30.0))
        assert_matches_oracle(shell_air_mass(at_30, 0.0, 10.0), 1.0 / at_30, rtol=1e-5)
        # on a huge sphere it is plane-parallel, and a thin shell is fine
        flat = shell_air_mass(0.2, 1e3, 2e3, radius=1e15)
        assert_matches_oracle(flat, 5.0, rtol=1e-9)
        assert shell_air_mass(1.0, 5e3, 5e3) == 1.0
        # the shell of zero height at the surface is plane-parallel
        assert_matches_oracle(shell_air_mass(0.5, 0.0, 0.0), 2.0, rtol=1e-15)

    def test_below_the_horizon_is_nan(self) -> None:
        mass = shell_air_mass(np.asarray([-0.1, 0.0, 0.1]), 1e3, 2e3)
        assert np.isnan(mass).tolist() == [True, False, False]

    @needs_jax
    @pytest.mark.parametrize("bottom", [0.0, 10.0])
    def test_the_gradient_is_finite_at_the_horizon(self, bottom: float) -> None:
        import jax
        import jax.numpy as jnp

        gradient = jax.grad(shell_air_mass, (0, 2))(0.0, bottom, 1e3)
        assert all(np.isfinite(float(g)) for g in gradient)
        cosines = jnp.asarray([-0.5, 0.3])
        total = jax.grad(lambda mu: jnp.nansum(shell_air_mass(mu, bottom, 1e3)))
        assert np.isfinite(np.asarray(total(cosines))).all()

    @needs_jax
    def test_the_radius_is_differentiable(self) -> None:
        import jax

        def of_radius(radius):
            return shell_air_mass(0.1, 8e3, 9e3, radius=radius)

        step = 10.0
        r = SPHERE.semi_major
        difference = (of_radius(r + step) - of_radius(r - step)) / (2 * step)
        assert_matches_oracle(jax.grad(of_radius)(r), difference, rtol=1e-6)

    def test_gradient_and_backends(self) -> None:
        cosine = np.asarray([0.05, 0.4, 1.0])
        bottom = np.asarray([0.0, 1e3, 1e4])
        top = bottom + 1e3
        assert_grad_matches_fd(
            lambda mu: shell_air_mass(mu, bottom, top), cosine, rtol=1e-5
        )
        assert_backends_agree(shell_air_mass, cosine, bottom, top)
        assert_traces(shell_air_mass, cosine, bottom, top)
