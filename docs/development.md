# Development

## Local checks

Run tests:

```bash
pytest
```

Run the same static check that CI enforces:

```bash
make lint
```

Reproduce CI's `build` job on a machine that *has* a CHARMM build:

```bash
make test-ci
```

That target sets `MMML_DISABLE_CHARMM=1`, which makes
`charmm_paths.resolve_charmm_paths()` report no CHARMM at all. Setting
`CHARMM_LIB_DIR` to a nonexistent directory does **not** work: a lib-less
explicit override is treated as stale and silently replaced by the discovered
`setup/charmm` tree, so live-CHARMM tests keep running locally and the local
result stops matching CI.

## Why CI checks the *shape* of a test run

`pytest` exits 0 when every selected test skips. Every live-dependency guard in
this repo (`@pytest.mark.skipif(not can_import_pycharmm())`, `if not
CKPT.exists(): pytest.skip(...)`, and the `libcharmm`-unavailable hook in
`tests/conftest.py`) is a skip, so a broken install, a missing checkpoint, or a
`libcharmm` that exists but cannot be `dlopen`'ed converts a whole job into a
no-op that reports success.

CI therefore asserts what a run should *look* like, not just its exit code:

- `scripts/ci/check_test_report.py` reads the JUnit XML each pytest step emits
  and fails when too few tests actually passed or too many skipped.
- `scripts/ci/assert_pycharmm_live.py` imports PyCHARMM for real before the
  live-CHARMM job runs, so a present-but-unusable library fails there instead of
  turning 45 minutes of tests into skips.

Locally:

```bash
make test-shape
```

When you add tests, the `--min-passed` thresholds in
`.github/workflows/ci.yml` are floors, not targets — raise them as the suite
grows so the gate keeps its teeth.

## Repository boundaries

- `mmml/` is the supported distributable library and CLI. Production code must
  not import from `scripts/`, `workflows/`, or `tests/`.
- `setup/charmm/tool/pycharmm/pycharmm/` is the sole tracked PyCHARMM source.
  Do not recreate a root-level `pycharmm/` copy; changes belong in this tree.
- `scripts/` contains developer and operational entry points; `workflows/`
  contains reproducible campaign definitions. Neither is a library API.
- Large checkpoints and generated campaign output belong in external storage or
  ignored artifact paths, not in new package source commits.

Unit tests run in the normal CI job. The separate `charmm` CI job is the
integration boundary for a compiled CHARMM/PyCHARMM runtime.

## CLI UX conventions

Use the shared CLI helpers instead of ad hoc Rich or `print()` formatting when
adding or touching command output:

- Import `get_reporter()` from `mmml.utils.rich_report` for compact reports.
  Choose `status()` for one event, `summary()` for key/value metadata, and
  `table()` for repeated records.
- Use `print_colored_json()` for JSON-shaped diagnostics. It validates through
  the standard JSON encoder first, rejects non-finite floats, and falls back to
  plain indented JSON when Rich is disabled.
- Keep ordinary reports borderless and copy-friendly. Reserve Rich panels for
  interactive/live displays or exceptional diagnostics where the visual boundary
  carries information.
- Respect the existing quiet/color controls: `quiet=True` and `MMML_QUIET=1`
  suppress helper output, `MMML_NO_RICH=1` disables Rich, and `MMML_RICH=1`
  forces terminal color.

```python
from mmml.utils.rich_report import get_reporter, print_colored_json

report = get_reporter()
report.status("success", "Validated input", detail=str(config_path))
report.summary("Resolved run", {"backend": backend, "output_dir": output_dir})
print_colored_json({"output_dir": str(output_dir), "errors": {}})
```

Commands dispatched by the top-level `mmml` entry point also share the
`mmml.cli.help_style` argparse hook. Flat parsers are grouped automatically by
input/configuration, scientific model, execution, output/artifacts, and
diagnostics/safety. If a command genuinely needs different sections, define
explicit `add_argument_group()` blocks in its parser instead of embedding ANSI
escapes or custom color schemes.

### Configure wizards

All new or modified `mmml configure` workflows must validate and preview their
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
- `PDF Export` renders `site/mmml-docs.pdf`, including Mermaid diagrams, and uploads it as an artifact.
  CI installs `mmdc` from `@mermaid-js/mermaid-cli`; local PDF builds fall back to
  readable Mermaid source unless `mmdc` is on `PATH` or `MMML_DOCS_PDF_ALLOW_NPX=1` is set.
- Published site: [Read the Docs](https://mmml.readthedocs.io/en/latest/) builds MkDocs via
  `.readthedocs.yaml` (same `mkdocs.yml` as local `make docs-serve`).

Build static docs:

```bash
make docs-build
```

Per-command CLI pages are generated from `mmml/cli/registry.py` before each build:

```bash
uv run python scripts/generate_cli_docs.py
uv run python scripts/generate_docs_figures.py
uv run python scripts/generate_cli_docs.py --check   # CI: fail if stale
uv run python scripts/generate_docs_figures.py --check
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
