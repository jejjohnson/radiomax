"""DataArrays through `elementwise`: labels kept, aligned by name, never guessed."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from radiomax import BackendError, ValidationError
from radiomax._src.backend import floating_inputs
from radiomax._src.labelled import elementwise, is_dataarray


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """A record argument, as a band's coefficients are."""

    scale: float = 1.0


@elementwise()
def affine(values, slope, offset=0.0, *, record: Record | None = None) -> Any:
    """Slope · values + offset, scaled by the record."""
    record = Record() if record is None else record
    xp, (values, slope, offset) = floating_inputs(values, slope, offset)
    return xp.multiply(slope, values) * record.scale + offset


@elementwise(n_outputs=2)
def polar(east, north, *, degrees: bool = True) -> Any:
    """The length and the bearing of a vector."""
    xp, (east, north) = floating_inputs(east, north)
    bearing = xp.atan2(east, north)
    length = xp.sqrt(east**2 + north**2)
    return length, bearing * 180.0 / xp.pi if degrees else bearing


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


@pytest.fixture
def counts(xr):
    return xr.DataArray(
        np.arange(6.0, dtype=np.float32).reshape(2, 3),
        dims=("band", "x"),
        coords={
            "band": ["B6", "B7"],
            "x": [1, 2, 3],
            "wavelength": ("band", [1.6, 2.2]),
        },
        name="counts",
        attrs={"units": "1"},
    )


@pytest.fixture
def gain(xr):
    return xr.DataArray(
        np.asarray([2.0, 3.0], dtype=np.float32), {"band": ["B6", "B7"]}, ("band",)
    )


def test_plain_arrays_pass_through() -> None:
    out = affine(np.asarray([1.0, 2.0]), 2.0)
    assert type(out) is np.ndarray
    assert out.tolist() == [2.0, 4.0]
    assert not is_dataarray(out)


def test_coordinates_stay_and_the_name_and_attributes_go(counts) -> None:
    out = affine(counts, 2.0)
    assert is_dataarray(out)
    assert out.dims == counts.dims
    assert set(out.coords) == {"band", "x", "wavelength"}
    assert out.name is None
    assert out.attrs == {}
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.values, 2.0 * counts.values)


def test_coefficients_align_on_their_dimension(counts, gain) -> None:
    out = affine(counts, gain, offset=gain)
    expected = counts.values * np.float32([[2.0], [3.0]]) + np.float32([[2.0], [3.0]])
    np.testing.assert_array_equal(out.values, expected)


def test_alignment_is_by_name_and_not_by_position(xr) -> None:
    square = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("y", "x"))
    per_row = xr.DataArray([10.0, 100.0], dims="y")
    out = affine(square, per_row)
    assert out.sel(x=1).values.tolist() == [10.0, 300.0]


def test_mismatched_coordinates_raise(counts, gain) -> None:
    with pytest.raises(ValueError, match="band"):
        affine(counts, gain.assign_coords(band=["B6", "B8"]))


@pytest.mark.parametrize("plain", [np.ones(3), [1.0, 1.0, 1.0], (1.0, 1.0, 1.0)])
def test_an_array_without_names_is_not_guessed(counts, plain) -> None:
    with pytest.raises(ValidationError, match="'slope' has dimensions but no names"):
        affine(counts, plain)


def test_scalars_records_and_options_pass_through(xr, counts) -> None:
    out = affine(counts, np.float32(2.0), record=Record(scale=2.0))
    np.testing.assert_array_equal(out.values, 4.0 * counts.values)
    east = xr.DataArray([1.0, 0.0], dims="x")
    _, bearing = polar(east, 0.0, degrees=False)
    assert bearing.values.tolist() == [np.pi / 2, 0.0]


def test_several_outputs_are_a_tuple_of_dataarrays(xr) -> None:
    east = xr.DataArray([3.0, 0.0], {"x": [10, 20]}, ("x",), name="east")
    length, bearing = polar(east, east + 4.0)
    assert is_dataarray(length) and is_dataarray(bearing)
    assert length.x.values.tolist() == [10, 20]
    assert length.name is None and bearing.name is None
    assert length.values.tolist() == [np.hypot(3.0, 7.0), 4.0]


def test_several_outputs_of_plain_arrays() -> None:
    length, bearing = polar(np.asarray([0.0]), np.asarray([2.0]))
    assert (length.tolist(), bearing.tolist()) == ([2.0], [0.0])


def test_dask_stays_lazy(xr) -> None:
    da = pytest.importorskip("dask.array")
    data = da.from_array(np.arange(6.0).reshape(2, 3), chunks=1)
    cube = xr.DataArray(data, dims=("band", "x"))
    out = affine(cube, xr.DataArray([2.0, 3.0], dims="band"))
    assert isinstance(out.data, da.Array)
    expected = np.asarray([[0.0, 2.0, 4.0], [9.0, 12.0, 15.0]])
    np.testing.assert_array_equal(out.compute().values, expected)


def test_a_dataarray_of_jax_holds_jax(xr) -> None:
    jnp = pytest.importorskip("jax.numpy")
    out = affine(xr.DataArray(jnp.arange(3.0), dims="x"), 2.0)
    assert type(out.data) is type(jnp.zeros(1))


def test_transforms_go_through_the_decorator() -> None:
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    x = jnp.asarray([1.0, 2.0])
    np.testing.assert_array_equal(jax.jit(affine)(x, 2.0), affine(x, 2.0))
    np.testing.assert_array_equal(jax.vmap(affine, (0, None))(x, 2.0), affine(x, 2.0))
    assert float(jax.grad(affine)(1.0, 2.0)) == 2.0


def test_a_dataset_is_not_an_array(xr) -> None:
    with pytest.raises(BackendError, match="Dataset"):
        affine(xr.Dataset({"a": ("x", [1.0])}), 2.0)


def test_the_decorator_keeps_the_signature_and_docstring() -> None:
    import inspect

    assert affine.__name__ == "affine"
    assert affine.__doc__ == "Slope · values + offset, scaled by the record."
    assert list(inspect.signature(affine).parameters)[:2] == ["values", "slope"]


def test_a_missing_argument_is_reported_as_python_does() -> None:
    with pytest.raises(TypeError, match="slope"):
        affine(1.0)
