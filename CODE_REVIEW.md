# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (a constant that promotes float32, degrees read as radians, radiance per µm read as per cm⁻¹, a lost DataArray label, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented Earth radius or physical constant (geophysix has it), solar angle, air mass, reflectance or Planck formula is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`); the library imports geophysix and `pipekit_array.functions`, and never gridax, integrix, functax, geotoolz, georeader or any model at runtime.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **Arrays.** A function computes in the namespace of its inputs, with array API functions only; NumPy in gives NumPy out; JAX and xarray are never imported at module level.
- **xarray.** A DataArray in gives a DataArray out, aligned by dimension name.
- **Constants** enter in the input's dtype (`xp.asarray(c, dtype=x.dtype)`); float32 in gives float32 out.
- **Units.** SI; degrees unless `degrees=False`, never guessed; wavenumber in cm⁻¹ where the argument is named `wavenumber_cm`. Per µm, per nm and per cm⁻¹ are never mixed.
- **Arguments.** Longitude before latitude; records as keyword arguments with defaults; time as `day` and `seconds`.
- **Invalid pixels** are NaN through a double `where`; nadir, the terminator and the limb return their limit with a finite gradient.
- **Pure.** Return a new array and never mutate the input.
- **Exceptions** derive from `RadiomaxError` (a `ValueError`), except those of the array backend (`AGENTS.md`); errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the module's `__all__` and the top-level `__all__`; a new module in `SUBMODULES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (rounded values, not 17-digit floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the top-level name (`xref:api#radiomax.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against an oracle (pvlib, georeader, pyproj, a stored value or a hand computation), a `hypothesis` property, a gradient check, and a parity test over NumPy, JAX, `array_api_strict` and DataArray; nadir, the terminator, the limb and the float32 dtype have their own tests.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line.
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

Each example below runs on NumPy alone.

### Constants take the input's dtype

```python
import numpy as np

AU_PERIHELION = 0.01671


# ❌ A float64 array constant promotes a float32 input to float64
def distance_bad(anomaly):
    return 1.0 - np.asarray(AU_PERIHELION, dtype=np.float64) * np.cos(anomaly)


# ✅ The constant enters in the input's dtype
def distance(anomaly):
    amplitude = np.asarray(AU_PERIHELION, dtype=anomaly.dtype)
    return 1.0 - amplitude * np.cos(anomaly)


anomaly = np.asarray([0.0, 0.5], dtype=np.float32)
assert distance_bad(anomaly).dtype == np.float64
assert distance(anomaly).dtype == np.float32
```

### Degrees or radians is an argument, never a guess

```python
import numpy as np


# ❌ Guesses the unit from the magnitude: 1.5 degrees is read as radians
def cos_zenith_bad(zenith):
    return np.cos(np.deg2rad(zenith) if np.max(np.abs(zenith)) > np.pi else zenith)


# ✅ The caller says which, and the default is documented
def cos_zenith(zenith, *, degrees: bool = True):
    return np.cos(np.deg2rad(zenith) if degrees else zenith)


assert np.isclose(cos_zenith(np.asarray(60.0)), 0.5)
```

### An invalid pixel is NaN, and its gradient does not spread

```python
import numpy as np


# ✅ The inner where keeps the unused branch finite, so the gradient of the
#    valid pixels is too; a single outer where would differentiate log(0)
def temperature(radiance):
    positive = radiance > 0
    safe = np.where(positive, radiance, 1.0)
    return np.where(positive, 1.0 / np.log1p(1.0 / safe), np.nan)


assert np.isnan(temperature(np.asarray(0.0)))
assert np.isfinite(temperature(np.asarray(0.1)))
```

### arccos and arcsin at the edge of their domain

```python
import numpy as np


# ❌ The derivative of arccos is infinite at ±1: the Sun overhead, nadir, the
#    hotspot. A clip does not help: it hides the value and halves the gradient
def zenith_bad(up):
    return np.arccos(np.clip(up, -1.0, 1.0))


# ✅ atan2 of the horizontal and vertical components
def zenith(horizontal, up):
    return np.arctan2(horizontal, up)


assert zenith(0.0, 1.0) == 0.0
```

### Doctest output the code really prints

```python
# ❌ Expected output copied from a calculator: the doctest fails on the
#    17th digit, or passes only on one platform
#     >>> cos_zenith(np.asarray(60.0))
#     0.5

# ✅ Round in the example, and paste what it prints
#     >>> round(float(cos_zenith(np.asarray(60.0))), 12)
#     0.5
import numpy as np

assert round(float(np.cos(np.deg2rad(60.0))), 12) == 0.5
```


---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
