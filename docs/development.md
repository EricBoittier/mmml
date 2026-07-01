# Development

## Local checks

Run tests:

```bash
pytest
```

## Docs workflow

Documentation has its own GitHub Actions workflow (`.github/workflows/docs.yml`):

- `MkDocs HTML` runs `mkdocs build --strict`.
- `PDF Export` renders `site/mmml-docs.pdf`, including Mermaid diagrams, and uploads it as an artifact.
  CI installs `mmdc` from `@mermaid-js/mermaid-cli`; local PDF builds fall back to
  readable Mermaid source unless `mmdc` is on `PATH` or `MMML_DOCS_PDF_ALLOW_NPX=1` is set.

Build static docs:

```bash
make docs-build
```

Per-command CLI pages are generated from `mmml/cli/registry.py` before each build:

```bash
uv run python scripts/generate_cli_docs.py
uv run python scripts/generate_cli_docs.py --check   # CI: fail if stale
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
