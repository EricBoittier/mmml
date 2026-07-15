# MMML CLI

The `mmml` command is the unified entry point for structure building, mixed MM/ML
molecular dynamics, QM data pipelines, and training workflows.

Install the optional CLI extra for shell tab completion:

```bash
uv sync --extra cli
# or: pip install 'mmml[cli]'
```

## Help layers

MMML splits help across a few commands so `mmml -h` stays short while deeper
detail stays one command away.

| Command | Purpose |
|---------|---------|
| `mmml -h` | Compact top-level summary (common subcommands + pointers) |
| `mmml commands` | All subcommands grouped by task area |
| `mmml commands --audit` | Deprecated/legacy commands and tab-completion coverage |
| `mmml examples` | Copy-paste example invocations |
| `mmml <command> --help` | Full flags for one subcommand |
| `mmml configure` | Interactive YAML / Snakemake wizard |
| `mmml env` | Resolved checkpoints, CHARMM paths, export hints |
| `mmml completion <shell>` | Print bash/zsh/fish completion script |

```bash
mmml -h
mmml commands
mmml examples
mmml md-system --help
mmml env --json
```

## Tab completion

With `argcomplete` installed (`mmml[cli]`), completion covers subcommand names
and flags (when `build_parser()` exists for that command).

```bash
eval "$(register-python-argcomplete mmml)"
# or:
eval "$(mmml completion bash)"
```

See [Tab completion](completion.md) for per-shell setup and fallbacks when
`argcomplete` is not installed.

## Command index

Browse the sidebar under **CLI** for a page per subcommand (options pulled from
each command's `argparse` help). Structure builders (`make-res`, `make-box`,
`build-crystal`) include ASE structure figures — see
[Structure building](structure-building.md).

Highlights:

**Structure & boxes** — `make-res`, `make-box`, `build-crystal`, `liquid-box`

**MD & campaigns** — `md-system` (see also [md-system YAML configs](../md-system-configs.md)),
`warmup-mlpot-jax`, `mpi-check`, `health-check`

**QM & data** — `pyscf-dft`, `pyscf-evaluate`, `fix-and-split`, `xml2npz`

**ML training** — `physnet-train`, `physnet-evaluate`, `efield-train`

**Workflow helpers** — `configure`, `env`, `commands`, `examples`

Run `mmml commands --audit` locally to see which commands are **deprecated** or
**legacy** and what to use instead.

## Typical workflows

### Condensed-phase MD (MLpot)

```bash
mmml env                                    # checkpoints + CHARMM paths
mmml configure                              # or hand-edit YAML
mmml health-check --require-gpu --live
mmml warmup-mlpot-jax --checkpoint "$MMML_CKPT" --n-monomers 20
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system --config run.yaml
```

### Train PhysNet from NPZ

```bash
mmml fix-and-split --efd data.npz --output-dir splits/
mmml physnet-train --config train.yaml
mmml physnet-evaluate --checkpoint ckpts/run --test splits/test.npz
```

## Regenerating CLI reference pages

Per-command pages under `docs/cli/commands/` are generated from
`mmml/cli/registry.py`:

```bash
uv run python scripts/generate_cli_docs.py
```

CI and `make docs-build` run this before `mkdocs build` so the sidebar stays in
sync with the registry.
