---
name: radiomax-review
description: Review a change or pull request in this repository against its own rules, CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the package already provides, and units and conventions (spectral units, degrees versus radians, azimuths, dtype promotion, edges, gradients). Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note which modules, tests and docs it touches.
2. **Reuse**: run the `reuse-reviewer` subagent on the diff: a hand-written constant, solar angle, air mass or reflectance formula is the main way a package drifts from itself.
3. **Units and conventions**: run the `correctness-reviewer` subagent in parallel with step 2: spectral units, angles and azimuths, dtype promotion by constants, edges, double-`where` gradients, doctest accuracy, test tolerances.
4. **Contracts**: for each module touched, the rules under "The contracts" in `AGENTS.md`: the namespace comes from the inputs, a DataArray keeps its labels, and JAX and xarray are never imported at module level; constants enter in the input's dtype; records are frozen dataclasses passed as keyword arguments with defaults; private code sits in `_src/<module>/` and the public module only re-exports; the layer rule holds (geophysix and nothing else from the stack); exceptions derive from `RadiomaxError`; the function has its oracle, property, gradient and backend tests.
5. **Checklist**: the rest of `CODE_REVIEW.md` (public API, API page and `docs/api/index.md` row, capability index, MyST prose, tests and tiers), skipping what ruff, ty or the tests already enforce.
6. **House style**: the `python-style` review checklist if the skill is installed, otherwise "House style" in `AGENTS.md`: docstring layout with a Unicode equation, comments above the line, descriptive names, maths in the `eman-style-math` notation, `cyclopts` and `loguru` in scripts.
7. **Verify claims**: for anything you flag as a bug, run it (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with priority, file, lines and a concrete change, summary).
