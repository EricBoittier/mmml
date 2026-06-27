# Releasing (development alpha)

MMML versions are derived from git tags by
[`versioningit`](https://versioningit.readthedocs.io/) (see `[tool.versioningit]` in
`pyproject.toml`). A clean, tagged commit produces an exact version; anything after a
tag gets a dev/local suffix.

## Fast install (what end users get)

The alpha is distributed as a Python package, not a git clone:

```bash
pip install mmml            # or: uv pip install mmml
```

The wheel/sdist are ~11 MB (they exclude large repo data such as `mmml/models/EF/data/`,
trajectories, and `setup/charmm.tar.xz`). pyCHARMM and a GPU are optional — see the README
and `AGENTS.md`.

## Cut an alpha release

1. Make sure `main` is green and the working tree is clean.
2. Choose a PEP 440 pre-release version and tag it (annotated):
   ```bash
   git tag -a v1.0.0a1 -m "mmml 1.0.0a1 (alpha)"
   git push origin v1.0.0a1
   ```
   `versioningit` turns `v1.0.0a1` into version `1.0.0a1`.
3. Build and check the artifacts:
   ```bash
   uv build                       # writes dist/*.whl and dist/*.tar.gz
   uv run --with twine twine check dist/*
   ```
4. (Optional) Publish to PyPI / TestPyPI:
   ```bash
   uv run --with twine twine upload --repository testpypi dist/*
   ```
5. Create the GitHub Release from the tag and attach `dist/*` (and, if wanted,
   `setup/charmm.tar.xz` as a release asset rather than tracking it in git).

## Keeping the git repo small

`git clone` is large because of binaries committed to history. To reduce it:

- Users (no rewrite needed): `git clone --filter=blob:none <url>` or `git clone --depth 1 <url>`.
- Maintainers (destructive, rewrites history, requires force-push and everyone re-clones):
  run `scripts/slim_repo_history.sh --analyze` then `--rewrite`. Do this only when active
  branches/PRs can be coordinated, and reclaim Git LFS storage afterwards (GitHub does not
  garbage-collect LFS objects automatically).

Do **not** commit new large/regenerable artifacts; `.gitignore` already covers
`node_modules/`, `*.dcd`, `*.traj`, `*.parquet`, `mmml/models/EF/data/`, etc.
