# AGENTS.md

Standing instructions for **every** coding agent working in this repository (Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth: `CLAUDE.md` and `.github/copilot-instructions.md` point here.

The house style (Python, maths notation, prose, pseudocode, reports) comes from the maintainer's personal skills in [jejagents](https://github.com/jejjohnson/jejagents); "House style" and "Personal skills" below say which skill covers what and carry the essentials, so an agent without those skills installed can still follow them.


## What this repo is

radiomax is the GeoStack's library of observation geometry and radiometry: it converts and locates what a sensor saw. It is a GeoNumerics library, above geophysix, written once in the Python array API standard, so it runs on NumPy, JAX, CuPy and any other array-API library, and on xarray.

People build on a package like this, so **the package is a library of primitives**: new code composes what is here, and anything genuinely new lands where the next person will find and reuse it.

The public API is the set of public modules below; private code lives in `src/radiomax/_src/<module>/`. `src/radiomax/__init__.py` exports only `__version__` and the exceptions. The modules marked planned are each created by their first feature issue ([jejjohnson/geostack#300](https://github.com/jejjohnson/geostack/issues/300) is the backlog).

| Module | Owns | Status |
|---|---|---|
| `_typing.py` | The `ArrayLike` type alias | now |
| `exceptions.py` | `RadiomaxError` (a `ValueError`) and its subclasses `BackendError`, `ValidationError` | now |
| `geometry` | Sun and view angles, relative azimuth, phase and scattering angles, air masses, the Earth–Sun distance, the two time arrays; planned: parallax, navigation, scan geometry | now |
| `radiometry` | Counts, radiance, reflectance, band averages, response matrices, spectral densities | now |
| `thermal` | Planck's law per wavenumber, its logarithm and its views per µm and per GHz, monochromatic brightness temperature, the band inverses sensors publish | now |
| `retrievals` | Empirical formulas with given coefficients: Ross–Li kernels, split-window temperatures, ocean colour | planned |
| `sar` | SAR calibration | planned |
| `altimetry` | Altimeter range corrections | planned |

### Boundaries

- **Layer rule.** radiomax imports geophysix and nothing else from the GeoStack at runtime. It never imports gridax, integrix, functax, pipekit, geotoolz, georeader or any GeoModels library; radtrax, geoops, geoproducts and xrops import it. integrix and functax are reference libraries in the tests.
- **Three runtime dependencies**: `numpy>=2.0`, `geophysix` and `array-api-compat`. JAX and xarray are extras, and any other optional one goes behind an extra and an in-function import that names the extra when it is missing.
- **Ownership test.** Does it need a model of radiation moving through an atmospheric state? Then it is radtrax's. Is it geometry, a unit conversion, or an empirical formula with given coefficients, applied pixel by pixel? Then it belongs here. A constant or a fact about the Earth itself is geophysix's.
- **No files.** radiomax reads no files and downloads nothing: coefficients, spectra and responses arrive as arrays from the readers.

When other projects build on the package, list them here and keep their imports working through the change that moves them.

## What it is built on

Each foundation brings a rule. Breaking one runs fine on your machine and fails in someone else's pipeline or in CI.

| Foundation | The package uses it for | The rule it brings |
|---|---|---|
| **NumPy ≥ 2.0** | The default namespace; host-side builders | NumPy in gives NumPy out; no function needs JAX or xarray installed. |
| **array-api-compat** | Finding the namespace of an input | Only functions of the array API standard are called on it; `array_api_strict` in the tests refuses the rest. |
| **geophysix** | Constants, `Ellipsoid`, Earth-centred coordinates | No constant or Earth radius is written here. |
| **JAX** (extra `jax`) | The same functions under `jit`, `vmap` and `grad` | Never imported at module level. |
| **xarray** (extra `xarray`) | A DataArray in, a DataArray out | Imported only when a DataArray arrives. |
| **`dataclasses`** | Records such as a band's coefficients | `frozen=True`, `slots=True`, passed as keyword arguments with defaults. |
| **pvlib, georeader, pyproj, hypothesis, array-api-strict** (the `test` group) | Reference results and property tests | Never a runtime import; an oracle test skips when its package is missing. |
| **integrix, functax** (the `oracles-jax` group) | Reference results that need JAX | Only the job with every extra installs them. |
| **pytest + `--doctest-modules`** | The suite and every docstring `Examples:` block | An example's expected output is what the code prints; a stale example is a failing test. |

## Reuse before you write

Before writing a helper, a constant, a conversion or a formula, find out whether it exists:

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public name once, at its public home, grouped by the module that defines it, with a one-line summary. It is generated (`uv run poe capabilities`) and checked by `tests/test_capabilities.py`, so it is current.
2. **Search the private helpers** in the table below: several invariants live in exactly one place each.
3. **If it is missing, add it to the module that owns the concept**, at the lowest layer every caller can reach, not inline in the one caller you have today; a helper two modules need goes into `utils.py`.
4. **One object, one name.** A name means one thing across the package and its submodules (`tests/test_capabilities.py`); a deliberate exception is listed with its reason in `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| A hand-written Earth radius, flattening, h, c or k_B | `geophysix.earth.WGS84`, `SPHERE`, `geophysix.constants.STANDARD` |
| Geodetic to Earth-centred coordinates or back | `geophysix.earth.geodetic_to_ecef`, `ecef_to_geodetic` |
| A solar zenith angle, a declination, an Earth–Sun distance, a Julian date | `radiomax.geometry.solar_angles`, `solar_zenith`, `solar_declination`, `earth_sun_distance`, `utc_day_seconds` |
| A view zenith from a satellite's position, `arccos(cos(vaa - saa))`, a phase or scattering angle | `radiomax.geometry.view_angles`, `relative_azimuth`, `phase_angle`, `scattering_angle` |
| `1 / cos(sza) + 1 / cos(vza)`, a Chapman or spherical air mass | `radiomax.geometry.air_mass_factor`, `shell_air_mass` |
| `angle * pi / 180` behind a `degrees` argument, an angle modulo 360°, a root or an `atan2` whose gradient is NaN at 0 | `radiomax._src.numerics.to_radians`, `wrap_turn`, `safe_sqrt`, `safe_atan2` |
| `gain * dn + offset`, `pi * L * d**2 / (E0 * cos(sza))` or its inverse | `radiomax.radiometry.counts_to_radiance`, `radiance_to_counts`, `counts_to_reflectance`, `reflectance`, `radiance_from_reflectance` |
| A trapezoid sum over a spectral response, a band irradiance, `E * wl**2 / 1e7` | `radiomax.radiometry.band_average`, `band_irradiance`, `trapezoid_weights`, `per_nm_to_per_cm`, `per_cm_to_per_nm` |
| A response on another grid than its table, an interpolated response times a spectrum | `radiomax.radiometry.response_matrix` (exact, host), `apply_response`, `apply_response_adjoint`, `response_coverage` |
| A Gaussian, box or triangular response, rows normalised to 1, a band centre or width | `radiomax.radiometry.gaussian_response`, `box_response`, `triangle_response`, `sampled_response_matrix`, `band_centre`, `equivalent_width` |
| Planck's law, a radiance per cm⁻¹ as one per µm or per GHz | `radiomax.thermal.planck_radiance`, `log_planck_radiance`, `to_per_um`, `from_per_um`, `to_per_ghz`, `from_per_ghz` |
| A brightness temperature from a spectral radiance | `radiomax.thermal.brightness_temperature` |
| `K2 / log(K1 / L + 1)`, or a reader's `fk1`, `fk2`, `bc1`, `bc2` | `radiomax.thermal.brightness_temperature_band` with `PlanckLinear` or `PlanckQuadratic` |
| A frozen record that must be a JAX pytree | `@register_record` from `radiomax._src.records` |
| `xp = np if isinstance(x, np.ndarray) else jnp`, or `x.astype(float)` on an input | `radiomax._src.backend.array_namespace`, `floating_inputs` |
| A constant as an array of the input's dtype | `radiomax._src.backend.like` |
| `isinstance(x, xr.DataArray)`, or a call to `xr.apply_ufunc` | `@elementwise` from `radiomax._src.labelled`, `apply_along` for a function that consumes a named dimension, `is_dataarray` |
| A new exception for a bad argument, or for an unknown array library | `ValidationError`, `BackendError` |

## The contracts

Everything composes because it keeps these contracts; the tests in "What enforces them" check most of them.

### 1. Public functions

These are the decisions of [#15](https://github.com/jejjohnson/radiomax/issues/15), where the measurements behind them are.

- **Arrays.** One implementation in functions of the Python array API standard. The namespace comes from the input: NumPy in gives NumPy out, JAX in gives JAX out, and the same for CuPy and any array-API library. JAX and xarray are never imported at module level. `xp, (x, y) = floating_inputs(x, y)` opens every function; a NumPy array joins the other library, and two other libraries in one call raise `BackendError`.
- **xarray.** A DataArray in gives a DataArray out, through `xr.apply_ufunc`, with dimensions aligned by name; `@elementwise` does it. An unlabelled array with dimensions next to a DataArray raises `ValidationError`, and the result has no name and no attributes.
- **dtype.** float32 stays float32; integer arrays become the namespace's default float; constants enter in the input's dtype.
- **Units.** SI, with two named exceptions: angles are degrees by default (`degrees=False` for radians), never guessed from the magnitude, and spectral code uses wavenumber in cm⁻¹, in arguments named `wavenumber_cm`.
- **Arguments.** Longitude before latitude; an `Ellipsoid` or `Constants` record as a keyword with a default; nothing reads a global.
- **Records.** Parameters that belong together are frozen `dataclasses`, registered as JAX pytrees only when JAX is importable.
- **Time.** Two arrays: `day`, UTC midnights since 2000-01-01, and `seconds`, the UTC second of the day, fraction included.
- **Sun.** The Astronomical Almanac formulas for the solar position and the Earth–Sun distance.
- **Invalid pixels.** NaN through a double `where`, so that the gradient of the valid pixels stays finite. Nadir, the terminator and the limb return their limit.
- **Spectral responses.** radiomax builds the response matrix; an instrument model elsewhere applies it.
- **Pure.** Return a new array and never mutate the input.
- **Documented.** Google-style docstring in the `python-style` layout: a one-line summary, the defining equation in Unicode (`eman-style-math`, no `$…$` in docstrings), `Args:` (meaning, symbol and unit, no type), `Returns:`, `Raises:` and an executable `Examples:` block.

### 2. Exceptions

- Every error derives from `RadiomaxError`, which is a `ValueError`: `ValidationError` is a bad argument (its shape, a dimension name, its value), `BackendError` an array of an unknown library or of several in one call. A new error goes into `exceptions.py`, is exported, and joins the list in `tests/test_exceptions.py`.

### The public API

- Private code lives in `src/radiomax/_src/<module>/`; the public modules re-export it. Export a public name from its module's `__all__`; both stay in plain `sorted()` order, which `tests/test_public_api.py` checks (ruff's `RUF022` is off: its isort-style order differs). `src/radiomax/__init__.py` exports `__version__` and the exceptions only.
- A new module declares a sorted `__all__`, joins `SUBMODULES` in `tests/test_public_api.py`, and gets its own page in `docs/api/` (a `::: radiomax.<module>` block), a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; a name on an existing module's page is picked up by its `:::` block.
- Run `uv run poe capabilities` after any public API change.
- Link prose to the API by the public name (``[`RadiomaxError`](xref:api#radiomax.RadiomaxError)``), see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` is exactly the set of modules; every submodule declares a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `--doctest-modules` on `src/radiomax` (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `tests/test_exceptions.py` | The hierarchy: every error is a `RadiomaxError`, which is a `ValueError` |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| `tests/conftest.py` | `jax_enable_x64` is on whenever JAX is importable |
| `tests/test_harness.py` | The harness itself: each check in `tests/_harness.py` and `tests/_backends.py` passes on a right function and fails on a wrong one, each reference in `tests/_oracles.py` agrees with a closed form, and the stored radtrax values are what their names say |
| CI job "without JAX and xarray" | The base install imports and passes the tests with neither installed |
| `[tool.coverage.report] fail_under = 90` | Coverage of `src/radiomax` in every coverage-enabled run (CI, `uv run poe test-cov`) |
| `strict_markers = true` | Only the registered tier markers (`slow`, `integration`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo), format, types on `src/radiomax` and `scripts` |

Each function adds its own four tests as `add-sensor-function` lays out: an oracle (`tests/_oracles.py`, `assert_matches_oracle`), a property, a gradient check (`assert_grad_matches_fd`) and a backend-parity test (`assert_backends_agree`, `assert_traces`). radtrax imports radiomax, so its values are stored in `tests/data/`, with the commit in the file name, by `make_radtrax_values.py`; add an array there rather than a test dependency.

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in `.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any agent can read and follow them), and two read-only review checklists in `.claude/agents/`:

| Job | Recipe |
|---|---|
| Add a function to a public module | `add-sensor-function` |
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `radiomax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `correctness-reviewer.md`, the units-and-conventions reviewer) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `uv run poe gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |


## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`, `ls` and other non-Python commands need no `uv run`.

```bash
uv run poe                  # list every task (they live in [tool.poe.tasks] in pyproject.toml)
uv run poe install          # uv sync --all-groups + pre-commit hooks
uv run poe test             # every test + doctests, no coverage (fast)
uv run poe test-cov         # the same with coverage and the 90 % gate (what CI runs)
uv run poe lint             # ruff check .   (entire repo)
uv run poe format           # ruff format . && ruff check --fix .
uv run poe typecheck        # ty check src/radiomax scripts (what CI runs)
uv run poe check            # lint + format-check + typecheck + test-cov (everything CI checks)
uv run poe capabilities     # regenerate docs/api/capabilities.md
uv run poe docs             # build both docs halves into public/ and check every link
uv run poe docs-check       # strict validation of both halves, no themed render
uv run poe docs-api         # Zensical API reference only (zensical build --strict)
uv run poe docs-serve       # build, then serve public/ at :8000
uv run poe precommit        # every pre-commit hook on every file
uv run poe build            # wheel + sdist into dist/
```

Extra arguments after a task name are passed to its command (`uv run poe test tests/test_exceptions.py -k base`).

`addopts` turns coverage on, and a partial run never reaches the 90 % gate, so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_exceptions.py -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For a parametrised test, mark only the expensive cases (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional dependencies, external tools.

CI ("Tests", `ci.yml`) runs every tier with coverage on Python 3.12 and 3.13 (`uv run pytest -v`). Select the fast tier with `uv run pytest -m "not slow and not integration"`; when the slow tiers grow, give them a CI job of their own rather than dropping them.

### Tests that assert on random draws

radiomax draws no random numbers. Its property tests use `hypothesis`; a failing example is pinned with an explicit `@example` so it keeps being run.

- **Incidental randomness** (any draw would do): seed it (`random.Random(0)`, `numpy.random.default_rng(0)`, a fixed key) so a failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical property): bound the statistic by its own sampling distribution (a number of standard errors, a test's p-value floor), not by a tolerance tuned on one draw, and say in a comment where the bound comes from.

### Before every commit

All of these must pass, from the repo root:

1. `uv run poe test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`. Never lint a subdirectory: `ruff check src/radiomax/` misses errors CI catches.
3. `uv run --group lint ruff format --check .`
4. `uv run poe typecheck` (`ty check src/radiomax scripts`, as CI runs it).
5. After changing a public API: `uv run poe capabilities`, and the `docs/api` page.
6. After changing docs, docstrings, `mkdocs.yml` or `docs/myst.yml`: `uv run poe docs` (or at least `uv run poe docs-api` when only docstrings changed).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several readings, name them instead of picking one silently; if something is unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no speculative features, no single-use abstractions, no configurability nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing style; don't refactor or add docstrings to code you didn't change; remove only what your change made unused, and mention (don't delete) unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced bug, a closed form to match) and loop until it passes; for multi-step work, state the plan with a check per step.

These are the `karpathy-coding-principles` skill in four lines. When you suggest a change, propose the code; when you fix a bug or add a feature, add the test that checks it.

## House style

### Python

`python-style` is the house style. Its general rules hold in every package built from this template:

- Python 3.12+, `from __future__ import annotations` in every module, type hints on every public function, `X | None` over `Optional`, specific exceptions with `raise … from …`, pure functions with side effects isolated and explicit.
- Every module opens with a docstring saying what it holds, in one or two sentences.
- Docstrings follow the layout in "The contracts" above; `Examples:` gives two or three cases (typical, edge, composition), each building its own small data, and `References:` names the paper when one defines the method.
- Comments go above the line, never beside it. A mathematical line has its Unicode equation above it (`# x̄ = (1/N) ∑ᵢ xᵢ`), a long function is split into stages with step headers (`# --- whiten ---`), and every other comment says why, not what.
- Names are descriptive (`posterior_precision`, not `Lam`); the symbol lives only in the equation comment.
- One operation per line, no nested calls that hide an intermediate result, 88 columns.
- One package base error; subclasses are named by what failed.
- Extract only real duplicates (code that must change together); code that merely looks alike stays separate.

Its JAX/Equinox rules apply when the package uses JAX: logic under `pkg/_src/` with public modules that only re-export, jaxtyping with the shared dim vocabulary, einx instead of `axis=` / `.T` / `reshape` / `[:, None]`, lineax for solves, arrays in `eqx.Module`s with `__check_init__` and `eqx.error_if`, `key` keyword-only and last, no silent dtype casts, `get_config()` / `from_config()`, and the four tests per public forward function (closed-form oracle, hypothesis properties, gradient check, transform invariance).

radiomax computes through plain functions and frozen `dataclasses` where `python-style` uses `eqx.Module`, jaxtyping and einx, because NumPy in must give NumPy out and JAX stays optional. Reductions use the array API's `axis=` and `xp.matmul`. Errors derive from `RadiomaxError`, which is itself a `ValueError`, so `except ValueError` keeps working. Doctests print exact floats only when the value is exactly representable; otherwise they round.

**CLIs and scripts** use `cyclopts` for arguments and `loguru` for output; never argparse, click, typer, progress `print`s or stdlib `logging`. For a repo script they are in the `dev` group already; radiomax has no CLI. `scripts/capabilities.py` still uses `argparse` and `print`; move it over when you next change it (`scripts/build_docs.py` shows the pattern: a `cyclopts.App` with `@app.default`, and `loguru` for every message).

### Maths notation

Maths follows `eman-style-math` in every format: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and every symbol is defined, with its shape, before it is used.

- **Code and docstrings:** Unicode only (x̄, σ², Σ⁻¹, ∑, 𝔼, ⊙), never `$…$`; identifiers stay ASCII.
- **MyST prose and MkDocs pages:** `$…$` and `$$…$$` with the kind-named macros (`\vect{x}`, `\gvect{\mu}`, `\mat{K}`, `\gmat{\Sigma}`). The macros are installed in `docs/myst.yml` (`math:`) and `docs/api/javascripts/mathjax.js`; regenerate both with the skill's `scripts/emit_macros.py` when the macros change.
- **GitHub issues and PRs:** expand the macros first (`scripts/expand_macros.py`), display maths in a `math` fence.
- **Everywhere:** each `$…$` expression on one source line; no spacing or sizing macros (`\,`, `\;`, `\quad`, `\qquad`, `\big`, `\tfrac`, `\boxed`, `\cdots`, `\operatorname`). Check a page with `uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/check_math.py <file> --target myst` (the skill's scripts need those three packages, which this repo does not install).

### Prose and diagrams

- Markdown prose is never hard-wrapped: one line per paragraph or list item in docs, skills and agent instructions; issue bodies follow `report` (one sentence per line); PR text, one line per paragraph. Don't rewrap untouched text in a file you edit.
- Docs, READMEs and release notes written in the maintainer's voice follow `eman-prose-tech` (the docs register): plain words, "we" for the maintainers, parentheses rather than em dashes.
- Pseudocode in design pages, issues and READMEs follows `code-sketch`: a shapes header, the equation above each line, an aligned shape column and a status marker (`today (pkg)`, `proposed (X)`, `user`) on every call into the stack.
- Never draw ASCII box-and-arrow diagrams; use Mermaid (both docs halves render it). Directory trees stay plain text.

## Personal skills

The maintainer's skills live in [jejagents](https://github.com/jejjohnson/jejagents) and install into `~/.claude/skills/` with its `install.sh`. Use them when they are installed; the sections above carry what an agent without them needs. Where a skill and this file disagree, this file wins: it records the template's deliberate departures.

| Job | Skill |
|---|---|
| Write, review or refactor Python | `python-style` |
| Maths in docstrings, docs pages, issues or PRs | `eman-style-math` |
| Pseudocode in a design page, issue or README | `code-sketch` |
| Prose in the maintainer's voice (docs, README, release notes, PR descriptions) | `eman-prose-tech` |
| Write up a research, feature, bug, design, epic or chore report | `report`, then file it here with `create-gh-issue` |
| File or link issues in another of the maintainer's repos | `gh-issue` (in this repo, `create-gh-issue` and `link-gh-issues` follow it) |
| A derivation walkthrough for the docs | `derivation`, with `derivation-myst` for a page in `docs/` |
| A learning explainer page | `explainer` |
| General coding discipline | `karpathy-coding-principles` ("Coding principles" above) |

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push to main", "merge to main"). Work on a feature branch and commit locally; never run `git push` unless asked. "Merge the branch" means push the feature branch, not merge into `main`. Confirm before any action that affects a shared branch.
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) with a lowercase subject (`feat(geometry): add …`); CI validates PR titles (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- **No AI attribution.** No `Co-Authored-By:` trailer for an AI on any commit and no "Generated with …" line in a PR description, unless the maintainer asks for one; keep human co-authors' trailers.
- **PR descriptions are for a human who did not watch the work:** why first, a small before/after table, what stays the same, one or two bullets on how it was tested. Label behaviour as "Today (bug): …" and "After this PR: …". Link review comments rather than naming reviewers.
- Write one `Closes #N` line per issue.
- Releases are cut by release-please (`release-please-config.json`): it bumps the version and writes `CHANGELOG.md` from the `feat`, `fix`, `perf` and `revert` commits; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An agent called for a small follow-up tends to write a fresh description scoped to its own work and silently discard the rest: read the existing description first and only append checklist items or update their status. Change the title only to fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

Fix a review comment in code, push the fix to the PR branch, then resolve its thread; don't post a reply comment. Don't resolve threads you didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

GraphQL is blocked in Claude sessions; there, and whenever the `gh` CLI is unavailable, use the GitHub MCP tools for the same operations (they can resolve review threads).

## Documentation

The docs are built by **two tools** and deployed as one site (`docs/README.md` has the rationale); `docs.yml` builds them on every PR and push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from `main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, example notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` (toc in `docs/myst.yml`) | `/` |
| API reference | Zensical + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`uv run poe docs` runs `scripts/build_docs.py`: it builds the API with `zensical build --strict` (Zensical reads `mkdocs.yml`), serves `site/` on port 8910 so mystmd can read its `objects.inv` (mystmd loads inventories only over http), runs `myst build --html --strict`, rewrites the localhost API URLs to `/reference/`, repairs anchors lowercased by mystmd's `$`-expansion bug, and fails if any internal link in the assembled `public/` site does not resolve. mystmd is a Node CLI (`npm install -g mystmd`), not a uv dependency; if its theme download is blocked, `docs/README.md` says how to build against a local clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`, `:::{tab-set}`, `:::{dropdown}` directives, not `!!!` / `===` / `???`.
- **Link to the API** with the `xref:` protocol and the **top-level** exported name: ``[`RadiomaxError`](xref:api#radiomax.RadiomaxError)``, not a deeper path (mystmd lowercases `$`-abbreviated anchors; see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-` (`guide/quickstart.md` → `/quickstart/`, `notebooks/deep_dive.ipynb` → `/deep-dive/`), so keep basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`; frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`): the theme prefixes root-relative ones with `BASE_URL` a second time (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs-format pages, built by Zensical (which runs only the plugins it supports and ignores the rest silently; check its plugin list before adding one): prose plus `:::` blocks, `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are Google style (`docstring_style: google`) with Unicode maths; maths in the API pages themselves goes through MathJax (`pymdownx.arithmatex`), with the `eman-style-math` macros loaded by `docs/api/javascripts/mathjax.js`.
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb` files, authored as jupytext percent `.py` and deleted after conversion; mystmd renders the stored outputs without re-executing, so each must be listed in the `docs/myst.yml` toc. Figures render inline with `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root holds scratch notebooks, outside the docs. Full standards: `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never committed); track work in GitHub issues. Draft a whole wave of issues from `docs/templates/wave-backlog.md` (see `docs/contributing.md`).
