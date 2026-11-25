---
name: add-sensor-function
description: Add a function to a public module of the package — where it goes in _src/<module>/, its units and record arguments, array-API code that keeps DataArray labels, a Google-style docstring, the oracle, property, gradient and backend-parity tests, the API page and the capability index. Use when asked to add, port or implement a conversion, an angle, a formula or a new module in src/radiomax.
---

# Add a function

Read "1. Public functions" and "The public API" under "The contracts" in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` for the quantity and its synonyms, and the "Reuse before you write" table in `AGENTS.md`.
- A constant or a fact about the Earth itself is in geophysix.
- A combination of existing names needs no new function.

## 2. Pick its home

Private code goes in `src/radiomax/_src/<module>/<topic>.py`; the public module (`geometry`, `radiometry`, `thermal`, `retrievals`, `sar`, `altimetry`) re-exports it. The ownership test decides: geometry, a unit conversion, or an empirical formula with given coefficients, per pixel, belongs here; a model of radiation in an atmosphere belongs in radtrax. A new public module is created by its first feature issue. A new exception goes in `exceptions.py`, derives from `RadiomaxError` and joins `tests/test_exceptions.py`.

## 3. Write it

- Open with `xp, (x, y) = floating_inputs(x, y)` from `pipekit_array.functions`, and call only functions of the array API standard on `xp`. Never `import jax` or `import xarray` at module level.
- Decorate an elementwise function with `@elementwise()` from `pipekit_array.functions` (`n_outputs=` for a tuple), so a DataArray in gives a DataArray out.
- Constants enter in the input's dtype: `c = like(C, x, xp)`; float32 in gives float32 out.
- State every unit. Angles are degrees unless `degrees=False`; a wavenumber is `wavenumber_cm`; longitude before latitude; time is `day` and `seconds`.
- Parameters that belong together are a frozen dataclass record passed as a keyword argument with a default.
- An invalid pixel is NaN through a double `where`. Use `atan2` in place of `arccos` and `arcsin`, and no `clip` on a gradient path.
- Comments above the line with the Unicode equation above each mathematical line; descriptive names; one operation per line.
- Google-style docstring in the `python-style` layout: a one-line summary (it becomes the capability-index entry), the defining equation in Unicode (never `$…$`), `Args:` with meaning, symbol and unit, `Returns:`, `Raises:` and an `Examples:` block of rounded values.

## 4. Export and document

- Add the name to the public module's `__all__` (sorted); a new module joins `SUBMODULES` in `tests/test_public_api.py`.
- API page `docs/api/<module>.md` with a `::: radiomax.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/test_<module>.py`, one class per function:

- **Oracle.** Against pvlib, georeader, pyproj, integrix, functax or a stored radtrax value, through the helpers in `tests/_oracles.py` and `assert_matches_oracle`, with the tolerance and its source in a comment; each helper skips when its package is missing.
- **Property.** A `hypothesis` property (symmetry, monotonicity, a round trip, a limit).
- **Gradient.** `assert_grad_matches_fd` of `pipekit_array.testing`, at the edges too; it skips without JAX.
- **Backends.** `assert_backends_agree` in `tests/_backends.py` (NumPy, `array_api_strict`, JAX eager and under `jit`, dask, DataArray; float64 and float32) and `assert_traces` (`jit` and `vmap` against a loop). A linear map also has `assert_adjoint`.
- The edges (overhead, nadir, terminator, limb, zero radiance) and each invalid argument (`match=`).

## 6. Verify

```bash
uv run pytest --no-cov tests/test_<module>.py src/radiomax/_src/<module> tests/test_public_api.py tests/test_capabilities.py -v
uv run poe capabilities
uv run poe test-cov
```

then the `pre-pr-check` skill.
