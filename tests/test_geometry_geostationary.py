"""Geostationary navigation against PROJ, and against itself."""

from __future__ import annotations

import numpy as np
import pytest
from _backends import SECOND_DEVICE, assert_backends_agree, needs_jax
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import proj_geostationary
from geophysix.earth import WGS84
from hypothesis import given, strategies as st

from radiomax.exceptions import ValidationError
from radiomax.geometry import geostationary_lonlat, geostationary_scan_angles


HEIGHT = 35_786_023.0
#: The angular radius of the disk is 0.1519 rad.
INSIDE = 0.145

SWEEP = pytest.mark.parametrize("sweep", ["x", "y"])


def _grid(half_width: float, n: int = 41) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-half_width, half_width, n)
    return np.meshgrid(axis, axis)


def _on_earth(x, y, sweep):
    """geoproducts' ``on_earth`` (geotoolz `3d05a9c`)."""
    a, b = WGS84.semi_major, WGS84.semi_minor
    if sweep == "x":
        d = np.cos(x) * np.cos(y), np.sin(x), np.cos(x) * np.sin(y)
    else:
        d = np.cos(x) * np.cos(y), np.sin(x) * np.cos(y), np.sin(y)
    distance = a + HEIGHT
    quad = (d[0] ** 2 + d[1] ** 2) / a**2 + d[2] ** 2 / b**2
    half = distance * d[0] / a**2
    const = distance**2 / a**2 - 1.0
    return half**2 - quad * const >= 0


@SWEEP
class TestGeostationary:
    @pytest.mark.parametrize("lon_0", [0.0, -75.2, 140.7])
    def test_against_proj(self, sweep: str, lon_0: float) -> None:
        # gate G2-proj
        x, y = _grid(INSIDE)
        inside = np.hypot(x, y) < INSIDE
        x, y = x[inside], y[inside]
        lon, lat, visible = geostationary_lonlat(
            x, y, lon_0=lon_0, height=HEIGHT, sweep=sweep
        )
        expected = proj_geostationary(x, y, lon_0=lon_0, height=HEIGHT, sweep=sweep)
        assert visible.all()
        turned = (lon - expected[0] + 180.0) % 360.0 - 180.0
        assert np.abs(turned).max() < 1e-9
        assert_matches_oracle(lat, expected[1], rtol=0.0, atol=1e-9)

    def test_visible_is_geoproductss_on_earth(self, sweep: str) -> None:
        x, y = _grid(0.16, 161)
        lon, lat, visible = geostationary_lonlat(
            x, y, lon_0=0.0, height=HEIGHT, sweep=sweep
        )
        assert_matches_oracle(visible, _on_earth(x, y, sweep), rtol=0.0)
        assert_matches_oracle(np.isnan(lon), ~visible, rtol=0.0)
        assert_matches_oracle(np.isnan(lat), ~visible, rtol=0.0)

    def test_round_trip(self, sweep: str) -> None:
        x, y = _grid(INSIDE)
        inside = np.hypot(x, y) < INSIDE
        x, y = x[inside], y[inside]
        options = {"lon_0": 140.7, "height": HEIGHT, "sweep": sweep}
        lon, lat, _ = geostationary_lonlat(x, y, **options)
        assert (lon >= -180.0).all()
        assert (lon < 180.0).all()
        back_x, back_y, visible = geostationary_scan_angles(lon, lat, **options)
        assert visible.all()
        assert_matches_oracle(back_x, x, rtol=0.0, atol=1e-12)
        assert_matches_oracle(back_y, y, rtol=0.0, atol=1e-12)

    @given(st.floats(-60.0, 60.0), st.floats(-55.0, 55.0), st.floats(-180.0, 180.0))
    def test_round_trip_from_the_ground(self, sweep, away, lat, lon_0) -> None:
        options = {"lon_0": lon_0, "height": HEIGHT, "sweep": sweep}
        x, y, visible = geostationary_scan_angles(lon_0 + away, lat, **options)
        assert visible
        lon, back, _ = geostationary_lonlat(x, y, **options)
        assert abs((lon - lon_0 - away + 180.0) % 360.0 - 180.0) < 1e-9
        assert abs(back - lat) < 1e-9

    def test_the_other_side_of_the_earth(self, sweep: str) -> None:
        # 81.3° away is the limb on the equator
        lon = np.asarray([0.0, 81.0, 81.5, 90.0, 180.0, -81.5])
        x, y, visible = geostationary_scan_angles(
            lon, 0.0, lon_0=0.0, height=HEIGHT, sweep=sweep
        )
        assert visible.tolist() == [True, True, False, False, False, False]
        assert np.isnan(x).tolist() == np.isnan(y).tolist() == (~visible).tolist()

    def test_north_is_up_and_east_is_right(self, sweep: str) -> None:
        options = {"lon_0": 0.0, "height": HEIGHT, "sweep": sweep}
        lon, lat, _ = geostationary_lonlat(0.05, 0.05, **options)
        assert lon > 0.0
        assert lat > 0.0
        x, y, _ = geostationary_scan_angles(-20.0, -30.0, **options)
        assert x < 0.0
        assert y < 0.0

    def test_radians(self, sweep: str) -> None:
        options = {"height": HEIGHT, "sweep": sweep}
        degrees = geostationary_lonlat(0.05, -0.08, lon_0=-75.2, **options)
        radians = geostationary_lonlat(
            0.05, -0.08, lon_0=np.deg2rad(-75.2), degrees=False, **options
        )
        for got, want in zip(radians[:2], degrees[:2], strict=True):
            assert_matches_oracle(np.rad2deg(got), want, rtol=1e-13)
        x, y, _ = geostationary_scan_angles(
            *radians[:2], lon_0=np.deg2rad(-75.2), degrees=False, **options
        )
        assert_matches_oracle(np.stack([x, y]), np.asarray([0.05, -0.08]), rtol=1e-12)

    def test_float32_is_good_to_1e_4_degrees(self, sweep: str) -> None:
        x, y = _grid(0.13)
        inside = np.hypot(x, y) < 0.13
        x, y = x[inside], y[inside]
        options = {"lon_0": -75.2, "height": HEIGHT, "sweep": sweep}
        # the reference takes the same, rounded, angles
        single = x.astype(np.float32), y.astype(np.float32)
        lon, lat, _ = geostationary_lonlat(*single, **options)
        assert lon.dtype == lat.dtype == np.float32
        expected = geostationary_lonlat(*(v.astype(float) for v in single), **options)
        assert np.abs(lon - expected[0]).max() < 1e-4
        assert np.abs(lat - expected[1]).max() < 1e-4

    def test_gradient(self, sweep: str) -> None:
        options = {"lon_0": 10.0, "height": HEIGHT, "sweep": sweep}
        assert_grad_matches_fd(
            lambda v: sum(geostationary_lonlat(v[0], v[1], **options)[:2]) / 100.0,
            np.asarray([0.06, -0.09]),
            eps=1e-7,
        )
        assert_grad_matches_fd(
            lambda v: sum(geostationary_scan_angles(v[0], v[1], **options)[:2]),
            np.asarray([35.0, -42.0]),
            eps=1e-4,
            rtol=1e-6,
        )

    @needs_jax
    def test_gradients_are_finite_at_nadir_and_beside_space(self, sweep: str) -> None:
        import jax

        options = {"lon_0": 0.0, "height": HEIGHT, "sweep": sweep}

        def position(x, y):
            lon, lat, _ = geostationary_lonlat(x, y, **options)
            return jax.numpy.nansum(lon + lat)

        x, y = np.asarray([0.0, 0.1, 0.16, 0.2]), np.asarray([0.0, 0.05, 0.0, 0.2])
        assert np.isfinite(np.stack(jax.grad(position, (0, 1))(x, y))).all()

        def angles(lon, lat):
            x, y, _ = geostationary_scan_angles(lon, lat, **options)
            return jax.numpy.nansum(x + y)

        lon, lat = (
            np.asarray([0.0, 40.0, 90.0, 180.0]),
            np.asarray([0.0, 30.0, 0.0, 9.0]),
        )
        assert np.isfinite(np.stack(jax.grad(angles, (0, 1))(lon, lat))).all()

    def test_backends(self, sweep: str) -> None:
        x = np.linspace(-0.1, 0.1, 6).reshape(2, 3)
        y = np.linspace(0.02, 0.11, 6).reshape(2, 3)
        options = {"lon_0": 10.0, "height": HEIGHT, "sweep": sweep}
        assert_backends_agree(
            lambda x, y: geostationary_lonlat(x, y, **options)[:2],
            x,
            y,
            rtol=1e-11,
            rtol32=1e-4,
            waits=SECOND_DEVICE,
        )
        assert_traces(
            lambda x, y: geostationary_lonlat(x, y, **options),
            x.ravel(),
            y.ravel(),
            rtol=1e-11,
        )
        lon, lat, _ = geostationary_lonlat(x, y, **options)
        assert_backends_agree(
            lambda lon, lat: geostationary_scan_angles(lon, lat, **options)[:2],
            lon,
            lat,
            rtol=1e-11,
            rtol32=1e-4,
            waits=SECOND_DEVICE,
        )
        assert_traces(
            lambda lon, lat: geostationary_scan_angles(lon, lat, **options),
            lon.ravel(),
            lat.ravel(),
            rtol=1e-11,
        )


def test_the_sweep_axes_differ_by_kilometres() -> None:
    options = {"lon_0": 0.0, "height": HEIGHT}
    lon_x, lat_x, _ = geostationary_lonlat(0.08, 0.08, sweep="x", **options)
    lon_y, lat_y, _ = geostationary_lonlat(0.08, 0.08, sweep="y", **options)
    apart = np.hypot(lon_x - lon_y, lat_x - lat_y) * 111.0
    assert 1.0 < apart < 30.0


def test_an_unknown_sweep_is_refused() -> None:
    for function in (geostationary_lonlat, geostationary_scan_angles):
        with pytest.raises(ValidationError, match="'sweep' must be 'x' or 'y'"):
            function(0.0, 0.0, lon_0=0.0, height=HEIGHT, sweep="z")  # ty: ignore


def test_a_fixed_grid_of_dataarrays() -> None:
    xr = pytest.importorskip("xarray")
    x = xr.DataArray(np.linspace(-0.1, 0.1, 5), dims="x")
    y = xr.DataArray(np.linspace(0.1, -0.1, 4), dims="y")
    lon, lat, visible = geostationary_lonlat(x, y, lon_0=0.0, height=HEIGHT, sweep="y")
    assert set(lon.dims) == set(lat.dims) == set(visible.dims) == {"y", "x"}
    assert visible.all()
