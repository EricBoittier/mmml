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
