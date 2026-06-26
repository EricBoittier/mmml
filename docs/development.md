# Development

## Local checks

Run tests:

```bash
pytest
```

## Docs workflow

Build static docs:

```bash
make docs-build
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
