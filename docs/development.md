# Development

## Local checks

Run tests:

```bash
uv run pytest
```

Run the same static check that CI enforces:

```bash
make lint
```

Reproduce CI's unit + functionality jobs on a machine that *has* a CHARMM build:

```bash
make test-ci
```

That target sets `KARML_DISABLE_CHARMM=1`, which makes
`charmm_paths.resolve_charmm_paths()` report no CHARMM at all. Setting
`CHARMM_LIB_DIR` to a nonexistent directory does **not** work: a lib-less
explicit override is treated as stale and silently replaced by the discovered
`setup/charmm` tree, so live-CHARMM tests keep running locally and the local
result stops matching CI.

The reusable CI install action lives at
`.github/actions/setup-karml/action.yml`. It installs uv, CPython 3.13 by
default, and then runs `uv sync` with either its default
`--extra dev --extra all-cpu` or the job-specific `sync-args` override.
Use the same extras locally when reproducing a job failure; otherwise pytest can
look green because backend-specific files skipped at import time.

CI splits the suite by dependency boundary:

| Job | What it runs |
|-----|----------------|
| Lint | `make lint`, `make lint-dupes`, and import sanity |
| Unit tests | `uv run pytest tests/unit` with coverage and `jax-pme-solver` installed |
| Functionality tests | `uv run pytest tests/functionality tests/misc tests/integration -m "not pycharmm"` |
| Data loading | Synthetic NPZ/HDF5 contract tests only; no remote datasets |
| CHARMM MPI units | `uv run pytest tests/charmm_mpi -m "charmm_mpi and not pycharmm"` |
| PyCHARMM smoke | Builds or restores `libcharmm`, builds Packmol, imports PyCHARMM, then runs live smoke tests |

The Docs workflow has a **Generated content** job that rewrites CLI pages,
package-architecture counts, and crystal-literature tables, then fails if git
still sees drift (upload `generated-docs-patch` on failure). MkDocs HTML and
PDF export run only after that job is green.

```bash
make docs-check      # same checks as Generated content, no rewrite
make docs-refresh    # rewrite generated pages
```

`make docs-check` and `make docs-refresh` both route through
`scripts/ci/refresh_generated_docs.py`. In addition to write-mode generators
(`generate_cli_docs.py`, `generate_package_architecture.py`,
`generate_crystal_lit_compare.py`), that entry point validates generated
figures, the scientific evidence registry, and hard-coded recommendation
annotations. If a documentation PR changes command registration, package
layout, crystal data, figures, or evidence metadata, run `make docs-refresh`
and commit the intentional generated drift.

## Why CI checks the *shape* of a test run

`pytest` exits 0 when every selected test skips. Every live-dependency guard in
this repo (`@pytest.mark.skipif(not can_import_pycharmm())`, `if not
CKPT.exists(): pytest.skip(...)`, and the `libcharmm`-unavailable hook in
`tests/conftest.py`) is a skip, so a broken install, a missing checkpoint, or a
`libcharmm` that exists but cannot be `dlopen`'ed converts a whole job into a
no-op that reports success.

CI therefore asserts what a run should *look* like, not just its exit code:

- `scripts/ci/check_test_report.py` reads the JUnit XML each pytest step emits
  and fails when too few tests actually passed, too many skipped, reports are
  missing, or the XML contains failures/errors.
- `scripts/ci/assert_pycharmm_live.py` imports PyCHARMM for real before the
  live-CHARMM job runs, so a present-but-unusable library fails there instead of
  turning 45 minutes of tests into skips.
- `scripts/ci/check_coverage_floor.py` checks both line percentage and absolute
  covered-line floors. The floor catches coverage slides; it is not a target for
  scientific code that can only run with CHARMM, plotting, PySCF, Torch, or GPU
  dependencies.

Locally:

```bash
make test-shape
```

When you add tests, the `--min-passed` thresholds in
`.github/workflows/ci.yml` are floors, not targets — raise them as the suite
grows so the gate keeps its teeth.

Common CI triage shortcuts:

- **Generated docs failed:** download or inspect `generated-docs.patch`, run
  `make docs-refresh`, review the `docs/` and `mkdocs.yml` diff, and commit only
  generated changes caused by your source change.
- **A shape gate failed with many skips:** compare the job's `sync-args` with
  your local extras. The unit/functionality jobs install `jax-pme-solver`
  because many long-range solver tests skip without it.
- **PyCHARMM smoke skipped or vanished:** first inspect the `PyCHARMM loads`
  step. A bad `libcharmm.so` cache, missing runtime library, or failed Packmol
  build should fail before pytest; JUnit reports decide the final test verdict.
- **Coverage floor failed:** check whether a large module stopped importing or a
  fixture tree disappeared before adding tests just to chase the percentage.

## Repository boundaries

- `karml/` is the supported distributable library and CLI. Production code must
  not import from `scripts/`, `workflows/`, or `tests/`.
- `setup/charmm/tool/pycharmm/pycharmm/` is the sole tracked PyCHARMM source.
  Do not recreate a root-level `pycharmm/` copy; changes belong in this tree.
- `scripts/` contains developer and operational entry points; `workflows/`
  contains reproducible campaign definitions. Neither is a library API.
- Large checkpoints and generated campaign output belong in external storage or
  ignored artifact paths, not in new package source commits.

Unit tests run in the **Unit tests** CI job. Functionality, misc, and
integration tests run separately. The **PyCHARMM smoke** job is the integration
boundary for a compiled CHARMM/PyCHARMM runtime.

## CLI UX conventions

Use the shared CLI helpers instead of ad hoc Rich or `print()` formatting when
adding or touching command output:

- Import `get_reporter()` from `karml.utils.rich_report` for compact reports.
  Choose `status()` for one event, `summary()` for key/value metadata, and
  `table()` for repeated records.
- Use `print_colored_json()` for JSON-shaped diagnostics. It validates through
  the standard JSON encoder first, rejects non-finite floats, and falls back to
  plain indented JSON when Rich is disabled.
- Keep ordinary reports borderless and copy-friendly. Reserve Rich panels for
  interactive/live displays or exceptional diagnostics where the visual boundary
  carries information.
- Respect the existing quiet/color controls: `quiet=True` and `KARML_QUIET=1`
  suppress helper output, `KARML_NO_RICH=1` disables Rich, and `KARML_RICH=1`
  forces terminal color.

```python
from karml.utils.rich_report import get_reporter, print_colored_json

report = get_reporter()
report.status("success", "Validated input", detail=str(config_path))
report.summary("Resolved run", {"backend": backend, "output_dir": output_dir})
print_colored_json({"output_dir": str(output_dir), "errors": {}})
```

Commands dispatched by the top-level `karml` entry point also share the
`karml.cli.help_style` argparse hook. Flat parsers are grouped automatically by
input/configuration, scientific model, execution, output/artifacts, and
diagnostics/safety. If a command genuinely needs different sections, define
explicit `add_argument_group()` blocks in its parser instead of embedding ANSI
escapes or custom color schemes.

### Configure wizards

All new or modified `karml configure` workflows must validate and preview their
generated documents before writing files:

1. build plain Python dictionaries for the documents;
2. register validation in `validate_wizard_config()` when the document type is
   new;
3. route output through `_preview_and_confirm()` or
   `_preview_documents_and_confirm()`; and
4. write files only after confirmation succeeds.

Workflow companions should reference canonical policy/configuration files. For
example, the interaction-policy wizard writes `interaction_policy.yaml` and
companion `md_system.yaml` or `dimer_scan.yaml` files that point back to that
policy instead of copying its species ownership rules.

## Docs workflow

Documentation has its own GitHub Actions workflow (`.github/workflows/docs.yml`):

- `MkDocs HTML` runs `mkdocs build --strict`.
- `PDF Export` renders `site/karml-docs.pdf`, including Mermaid diagrams, and uploads it as an artifact.
  CI installs `mmdc` from `@mermaid-js/mermaid-cli`; local PDF builds fall back to
  readable Mermaid source unless `mmdc` is on `PATH` or `KARML_DOCS_PDF_ALLOW_NPX=1` is set.
- Published site: [Read the Docs](https://karml.readthedocs.io/en/latest/) builds MkDocs via
  `.readthedocs.yaml` (same `mkdocs.yml` as local `make docs-serve`).

Build static docs:

```bash
make docs-build
```

Per-command CLI pages are generated from `karml/cli/registry.py` before each build:

```bash
make docs-refresh
make docs-check      # CI Generated content job: fail if stale
```

Build with the same strict checks used in CI:

```bash
make docs-strict
```

Export a single PDF from the MkDocs navigation:

```bash
make docs-pdf
```

Serve docs in watch mode:

```bash
make docs-serve
```

## Notes

- Keep pages focused and task-oriented.
- Add new pages under `docs/` and wire them into `mkdocs.yml` under `nav`.
- Medium PBC production: run `validate_mlpot_sparse_dimers.py` on equilibrated CRDs before long MD (see [Medium PBC](mlpot-medium-pbc.md)).
- Scientific features must follow the [scientific code and reproducibility
  policy](scientific-code.md). Use its completion checklist during review.
