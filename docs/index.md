# radiomax

> Observation geometry and radiometry on any array library.

radiomax is the GeoStack's library of observation geometry and radiometry: it converts and locates what a sensor saw. It holds the Sun's and the sensor's angles, the conversions between counts, radiance and reflectance, spectral responses, Planck's law and brightness temperature, and empirical formulas applied pixel by pixel.

It is written once, in the Python array API standard: NumPy in gives NumPy out, JAX in gives JAX out, and the same for CuPy and any other array-API library. An xarray DataArray in gives a DataArray out. Neither JAX nor xarray is imported at module level, so the base install needs neither.

## Installation

radiomax is not on PyPI yet, and neither is [geophysix](https://github.com/jejjohnson/geophysix), which it builds on.

::::{tab-set}
:::{tab-item} uv

```bash
uv pip install "geophysix @ git+https://github.com/jejjohnson/geophysix@v0.0.5" "radiomax @ git+https://github.com/jejjohnson/radiomax"
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/radiomax.git
cd radiomax
uv run poe install
```
:::
::::

## Status

The public modules [`geometry`](xref:api#radiomax.geometry), [`radiometry`](xref:api#radiomax.radiometry), [`thermal`](xref:api#radiomax.thermal), [`retrievals`](xref:api#radiomax.retrievals) and [`sar`](xref:api#radiomax.sar) are built. `altimetry` is planned.

## Design rules

1. **Convert and locate.** Geometry, unit conversions and empirical formulas with given coefficients, per pixel. A model of radiation in an atmosphere belongs in radtrax.
2. **Any array library.** The result is the same kind of array as the input, and a DataArray keeps its labels.
3. **SI units, with two named exceptions.** Angles are degrees unless `degrees=False`, and spectral code takes wavenumbers in cm⁻¹, in arguments named `wavenumber_cm`.
4. **Constants enter in the input's dtype**, so float32 stays float32.
5. **Fail with a typed exception.** Everything raised derives from [`RadiomaxError`](xref:api#radiomax.RadiomaxError), which is a `ValueError`. An invalid pixel is NaN, not an error.
