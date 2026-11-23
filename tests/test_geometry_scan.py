"""Whiskbroom scan geometry against geoops' formulas and published numbers."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import KINDS, assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from geophysix.earth import SPHERE
from hypothesis import given, strategies as st

from radiomax.exceptions import ValidationError
from radiomax.geometry import (
    bowtie_detector_positions,
    scan_angles,
    scan_pixel_growth,
)


VIIRS = ((1776, 3), (736, 2), (640, 1))
MODIS_ALTITUDE, RADIUS = 705e3, SPHERE.semi_major
#: The horizon from 705 km, degrees.
HORIZON = float(np.rad2deg(np.arcsin(RADIUS / (RADIUS + MODIS_ALTITUDE))))


def _geoops_growth(theta: np.ndarray, h: float, radius: float):
    """geoops ``scan_pixel_growth`` (geotoolz `3d05a9c`), in radians and metres."""
    k = (radius + h) / radius
    k_sin = k * np.sin(theta)
    beta = np.arcsin(k_sin) - theta
    root = np.sqrt(1.0 - k_sin**2)
    along = ((radius + h) * np.cos(theta) - radius * root) / h
    cross = radius * (k * np.cos(theta) / root - 1.0) / h
    return along, along / np.cos(beta), cross


def _geoops_bowtie(growth: np.ndarray, n: int):
    """geoops ``bowtie_detector_positions`` (geotoolz `3d05a9c`)."""
    centre = (n - 1) / 2.0
    offset = np.arange(n) - centre
    step = np.where(offset > 0, 1, -1)
    g = growth[None, :]
    own = centre + offset[:, None] / g
    return own, centre + (offset - step * n)[:, None] / g, step


class TestScanAngles:
    def test_uniform_samples(self) -> None:
        angles = scan_angles(1354, 55.0)
        assert angles.shape == (1354,)
        assert_matches_oracle(np.diff(angles), np.full(1353, 110.0 / 1354), rtol=1e-10)
        assert_matches_oracle(angles, -angles[::-1], rtol=1e-12)
        # θ_max is the outer edge of the last sample
        assert_matches_oracle(angles[-1] + 55.0 / 1354, 55.0, rtol=1e-14)

    def test_viirs_aggregation(self) -> None:
        angles = scan_angles(3200, 56.28, VIIRS)
        native = 56.28 / 3152
        steps = np.diff(angles[1600:])
        assert_matches_oracle(steps[:590], np.full(590, 3 * native), rtol=1e-9)
        assert_matches_oracle(steps[-600:], np.full(600, native), rtol=1e-9)
        assert_matches_oracle(angles, -angles[::-1], rtol=1e-12)
        assert_matches_oracle(angles[1600], 1.5 * native, rtol=1e-12)

    def test_one_zone_of_factor_1_is_no_aggregation(self) -> None:
        assert_matches_oracle(
            scan_angles(100, 55.0, ((50, 1),)), scan_angles(100, 55.0), rtol=1e-13
        )

    def test_radians(self) -> None:
        radians = scan_angles(101, np.deg2rad(55.0), degrees=False)
        expected = scan_angles(101, 55.0)
        assert_matches_oracle(np.rad2deg(radians), expected, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize(
        ("columns", "zones", "match"),
        [
            (0, None, "'n_columns' must be positive"),
            (4, ((3, 2),), "must hold a positive, whole number"),
            (4, ((0, 1),), "must hold a positive, whole number"),
            (4, ((4, 0),), "must hold a positive, whole number"),
            (6, ((4, 2),), "make 4 columns, got 6"),
        ],
    )
    def test_invalid_arguments(self, columns, zones, match) -> None:
        with pytest.raises(ValidationError, match=match):
            scan_angles(columns, 55.0, zones)


class TestScanPixelGrowth:
    @pytest.mark.parametrize("h", [4e5, 705e3, 829e3])
    def test_geoopss_formulas(self, h: float) -> None:
        # gate G2-scan; geoops loses digits as R/h, and near the horizon
        theta = np.linspace(-1.0, 1.0, 81)
        expected = _geoops_growth(np.abs(theta), h, RADIUS)
        got = scan_pixel_growth(theta, h, degrees=False)
        for actual, wanted in zip(got, expected, strict=True):
            assert_matches_oracle(actual, wanted, rtol=1e-12)

    def test_modis_at_the_edge_of_its_scan(self) -> None:
        # gate G2-scan: a pixel of about 2.0 km × 4.8 km
        along, _, across = scan_pixel_growth(55.0, MODIS_ALTITUDE)
        assert round(float(along), 1) == 2.0
        assert round(float(across), 1) == 4.8

    def test_the_flat_earth_limit(self) -> None:
        theta = np.linspace(0.0, 60.0, 13)
        along, orbit, across = scan_pixel_growth(theta, 7e5, radius=1e15)
        secant = 1.0 / np.cos(np.deg2rad(theta))
        assert_matches_oracle(along, secant, rtol=1e-8)
        assert_matches_oracle(orbit, secant, rtol=1e-8)
        assert_matches_oracle(across, secant**2, rtol=1e-8)

    def test_the_slant_range_by_the_law_of_cosines(self) -> None:
        # R² = (R + h)² + s² − 2 (R + h) s cos θ
        theta = np.linspace(0.0, 60.0, 13)
        along, _, _ = scan_pixel_growth(theta, MODIS_ALTITUDE)
        s, top = along * MODIS_ALTITUDE, RADIUS + MODIS_ALTITUDE
        cosine = np.cos(np.deg2rad(theta))
        squared = top**2 + s**2 - 2.0 * top * s * cosine
        assert_matches_oracle(np.sqrt(squared), np.full(13, RADIUS), rtol=1e-12)

    @needs_jax
    def test_across_is_the_derivative_of_the_ground_distance(self) -> None:
        import jax

        def ground(theta):
            k = (RADIUS + MODIS_ALTITUDE) / RADIUS
            return RADIUS * (jax.numpy.arcsin(k * jax.numpy.sin(theta)) - theta)

        theta = np.deg2rad(np.asarray([0.0, 20.0, 55.0]))
        _, _, across = scan_pixel_growth(theta, MODIS_ALTITUDE, degrees=False)
        slope = jax.vmap(jax.grad(ground))(theta) / MODIS_ALTITUDE
        assert_matches_oracle(across, slope, rtol=1e-12)

    @given(st.floats(0.0, 60.0), st.floats(0.01, 1.0), st.floats(3e5, 9e5))
    def test_it_grows_from_1(self, theta, more, h) -> None:
        at = scan_pixel_growth(theta, h)
        further = scan_pixel_growth(theta + more, h)
        assert all(a >= 1.0 - 1e-15 for a in at)
        assert all(b > a for a, b in zip(at, further, strict=True))
        # on the ground the pixel grows faster across than along
        assert at[0] <= at[1] * (1.0 + 1e-15)
        assert at[1] <= at[2] * (1.0 + 1e-15)

    def test_nan_from_the_horizon_on(self) -> None:
        theta = np.asarray([0.0, HORIZON - 1e-6, HORIZON + 1e-6, 90.0, 130.0, 180.0])
        for growth in scan_pixel_growth(theta, MODIS_ALTITUDE):
            assert np.isnan(growth).tolist() == [False] * 2 + [True] * 4
            assert (
                np.isnan(scan_pixel_growth(-theta, MODIS_ALTITUDE)[0]).tolist()
                == [False] * 2 + [True] * 4
            )

    @needs_jax
    def test_the_gradient_beside_an_invalid_column_is_finite(self) -> None:
        import jax

        for index in range(3):
            gradient = jax.grad(
                lambda theta, i=index: jax.numpy.nansum(
                    scan_pixel_growth(theta, MODIS_ALTITUDE)[i]
                )
            )(np.asarray([0.0, 55.0, HORIZON + 1e-9, 90.0, 180.0]))
            assert np.isfinite(gradient).all()
            assert gradient[1] > 0.0

    def test_float32(self) -> None:
        theta = np.linspace(-60.0, 60.0, 121)
        expected = scan_pixel_growth(theta, MODIS_ALTITUDE)
        got = scan_pixel_growth(theta.astype(np.float32), MODIS_ALTITUDE)
        for actual, wanted in zip(got, expected, strict=True):
            assert actual.dtype == np.float32
            assert_matches_oracle(actual, wanted, rtol=5e-6)

    def test_gradient(self) -> None:
        assert_grad_matches_fd(
            lambda x: sum(scan_pixel_growth(50.0 * x[0], 7e5 * x[1])),
            np.asarray([0.9, 1.05]),
        )

    def test_backends(self) -> None:
        theta = np.linspace(-50.0, 50.0, 6).reshape(2, 3)
        altitude = np.full((2, 3), MODIS_ALTITUDE)
        assert_backends_agree(scan_pixel_growth, theta, altitude)
        assert_traces(scan_pixel_growth, theta.ravel(), altitude.ravel())


class TestBowtie:
    @pytest.mark.parametrize("n", [1, 10, 16])
    def test_geoopss_formulas(self, n: int) -> None:
        # gate G2-scan
        growth = np.linspace(1.0, 2.2, 7)
        expected = _geoops_bowtie(growth, n)
        got = bowtie_detector_positions(growth, n)
        for actual, wanted in zip(got, expected, strict=True):
            assert_matches_oracle(actual, wanted, rtol=1e-12)

    def test_without_growth_each_row_is_its_detector(self) -> None:
        own, neighbour, _ = bowtie_detector_positions(np.ones(3), 10)
        assert_matches_oracle(own, np.tile(np.arange(10.0)[:, None], 3), rtol=0.0)
        # and the neighbour does not reach it
        assert ((neighbour < -0.5) | (neighbour > 9.5)).all()

    @given(st.floats(1.0, 4.0), st.integers(1, 40))
    def test_the_position_on_the_ground(self, growth, n) -> None:
        # detector k of scan j is at c + jN + (k − c) g
        own, neighbour, step = bowtie_detector_positions(np.asarray(growth), n)
        centre, rows = (n - 1) / 2.0, np.arange(n)
        at_own = centre + (own - centre) * growth
        at_neighbour = centre + step * n + (neighbour - centre) * growth
        assert_matches_oracle(at_own, rows, rtol=1e-12, atol=1e-11)
        assert_matches_oracle(at_neighbour, rows, rtol=1e-12, atol=1e-11)

    def test_overlap(self) -> None:
        # at g = 2 every row is seen by its own scan and by a neighbour
        own, neighbour, _ = bowtie_detector_positions(np.asarray([2.0]), 10)
        assert ((own >= -0.5) & (own <= 9.5)).all()
        assert ((neighbour >= -0.5) & (neighbour <= 9.5)).all()

    @pytest.mark.parametrize("kind", KINDS, ids=lambda kind: kind.name)
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_backends(self, kind, dtype) -> None:
        growth = np.linspace(1.0, 2.0, 6, dtype=dtype).reshape(2, 3)
        own, neighbour, step = bowtie_detector_positions(kind.make(growth), 10)
        expected = bowtie_detector_positions(growth.astype(float), 10)
        assert kind.read(own).shape == (10, 2, 3)
        assert kind.read(own).dtype == kind.read(neighbour).dtype == dtype
        assert_matches_oracle(kind.read(own), expected[0], rtol=0.0, atol=4e-6)
        assert_matches_oracle(kind.read(neighbour), expected[1], rtol=0.0, atol=4e-6)
        assert_matches_oracle(kind.read(step), expected[2], rtol=0.0)

    def test_traces_and_gradient(self) -> None:
        assert_traces(
            lambda g: bowtie_detector_positions(g, 4)[:2], np.linspace(1.0, 2.0, 5)
        )
        assert_grad_matches_fd(
            lambda g: bowtie_detector_positions(g, 4)[1], np.asarray([1.2, 1.9])
        )

    def test_dataarrays_take_a_name(self) -> None:
        xr = pytest.importorskip("xarray")
        growth = xr.DataArray(np.linspace(1.0, 2.0, 5), dims="column")
        own, neighbour, step = bowtie_detector_positions(growth, 10)
        assert own.dims == neighbour.dims == ("detector", "column")
        assert step.dims == ("detector",)
        renamed, _, _ = bowtie_detector_positions(growth, 10, detector_dim="row")
        assert renamed.dims == ("row", "column")
        with pytest.raises(ValidationError, match="already has the dimension"):
            bowtie_detector_positions(growth, 10, detector_dim="column")

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            bowtie_detector_positions(np.ones(2), 0)
