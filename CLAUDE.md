# mmml — instructions for Claude sessions

## Before pushing anything that touches the CLI

If your change adds, removes, or renames CLI flags/commands (anything under
`mmml/cli` or argparse setup used by the `mmml` entry point), regenerate the
CLI reference docs and commit them with your change:

```bash
uv run python scripts/generate_cli_docs.py
git add docs/cli/commands
```

The Docs workflow runs `generate_cli_docs.py --check` and fails on any stale
page, so pushing a CLI change without the regenerated docs breaks CI on main.

## Before pushing docs changes

`mkdocs build --strict` fails on any broken link, including images. Doc pages
that reference generated figures must have those assets committed (see the
gitignore exceptions for `docs/*-assets/` directories).

## General

- CI runs `make lint` (ruff over `mmml/`, `scripts/`, and the vendored
  pycharmm package) and the full pytest suite. Run `make lint` before pushing.
- Multiple Claude sessions often work on this repo concurrently and push to
  main. `git fetch` and rebase before pushing; expect HEAD to move under you.
- Never set env vars in tests via `os.environ[...] =` — use
  `monkeypatch.setenv` so state can't leak into later tests.

## CLI reporting

Use `mmml.utils.rich_report.get_reporter()` for new or modified terminal
reports. Select the method from the information shape: `status()` for one-line
events, `summary()` for key/value metadata, and `table()` for repeated records.
These methods provide the canonical colored, borderless, copy-friendly layout
and plain-text fallback. Do not add Rich `Panel` wrappers or bordered tables for
ordinary reporting; reserve panels for genuinely interactive/live displays
where a visual boundary carries information. Migrate older direct Rich usage
incrementally when touching its call site rather than doing formatting-only
repo-wide rewrites.
Use `print_colored_json()` for JSON-shaped diagnostic output instead of
constructing a table or applying Rich markup to serialized JSON manually. It
keeps the output valid and copyable while styling paths, numbers, booleans,
empty containers, and errors consistently.

## Scientific functionality

Follow [`docs/scientific-code.md`](docs/scientific-code.md) for scientific
features, evaluations, scans, simulations, models, and data transformations.
In particular:

- Search `mmml/`, `scripts/`, `workflows/`, tests, and docs before adding a new
  tool. Promote or reuse existing package code instead of adding a parallel
  standalone implementation.
- Supported reusable behavior belongs in `mmml/`; CLIs and scripts should be
  thin callers of one canonical Python API.
- Make units, scientific conventions, resolved defaults, provenance, and input
  content hashes explicit.
- Never silently skip failures. Preserve a structured failure record and fail
  the command by default when requested work is incomplete.
- Keep numerical evaluation, artifact writing, and plotting separate. A plot
  must be reproducible from saved machine-readable results.
- Do not mutate the environment, select devices, execute calculations, or
  write files at import time. Never hard-code personal or cluster paths in
  package code.

The proposed canonical 1D scan architecture is documented in
[`docs/dimer-scan-design.md`](docs/dimer-scan-design.md).

For MD, species-aware monomer/pair ownership is defined by the versioned policy
in [`docs/md-interaction-policies.md`](docs/md-interaction-policies.md). Do not
add peptide/water positional special cases or a second interaction selector.
Every molecule and unordered molecular pair must compile to exactly one owner
(or one complementary near/far partition), and unsupported provider lowering
must fail before propagation rather than fall back to a legacy energy split.
Reusable restraints belong in `mmml/md/restraints/`; temperature schedules
belong in `mmml/md/temperature.py`; enhanced-sampling protocols such as SMD
belong in their own protocol modules.

## No magic numbers

Every scientifically meaningful numeric literal (distance, cutoff, threshold,
switch point, weight, tolerance) must be a named, documented constant or
configurable field — never a bare literal, and never a second copy of a
value that should track an existing configurable parameter. This applies
repo-wide: models, training, evaluation, and workflow code alike. See
[“No magic numbers”](docs/scientific-code.md#no-magic-numbers) for the full
rule and the electrostatics-switch bug that established it.

## PBC ML-dimer MIC wrap — do not “fix” stop_gradient

If NVE force–energy preflight fails on a liquid box, **do not** remove
`jax.lax.stop_gradient` on the MIC lattice shift that wraps monomer B in
`mmml_calculator`, and **do not** switch that wrap to smooth MIC + force VJP.
Exact MIC shifts are piecewise-constant; making them differentiable injects
huge forces near ±L/2 and breaks minimization (seen: `|F|max` → hundreds eV/Å).
Keep exact MIC + `stop_gradient`. See `.cursor/rules/pbc-dimer-mic-wrap.mdc`.
