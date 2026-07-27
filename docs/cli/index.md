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

## Output conventions

CLI output is designed to stay readable in terminals and copyable in logs:

- `mmml <command> --help` uses shared argparse grouping for commands dispatched
  through `mmml`. Flat option lists are grouped by input/configuration,
  scientific model, execution, output/artifacts, and diagnostics/safety.
- Rich color is supplemental. Redirected output remains plain text, while
  `MMML_NO_RICH=1` disables Rich formatting and `MMML_RICH=1` forces it for
  terminal demos.
- JSON-shaped diagnostics use valid JSON even when color is enabled, so output
  from commands such as `mmml env --json` can still be copied into a parser.
- Long-running and setup commands should honor quiet modes where provided; the
  shared reporting helpers also respect `MMML_QUIET=1`.

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

**QM & data** — `pyscf-dft`, `pyscf-evaluate`, `fix-and-split`, `xml2npz`, `npz2traj`

**ML training & sampling** — `physnet-train`, `physnet-evaluate`, `physnet-md`,
`neb`, `dmc`, `efield-train`

**Workflow helpers** — `configure`, `env`, `commands`, `examples`

Run `mmml commands --audit` locally to see which commands are **deprecated** or
**legacy** and what to use instead.

## Configure safety model

`mmml configure` is the interactive entry point for YAML and workflow scaffolds.
For the interactive workflows (`md-single`, `md-campaign`, `physnet-train`,
`snakemake-md`, and `interaction-policy`) the wizard:

1. collects answers through numbered prompts;
2. validates the generated document before writing it;
3. prints a JSON preview of the exact configuration bundle; and
4. asks for confirmation before creating files.

The `interaction-policy` workflow can also write companion `md-system` or
`dimer-scan` configs that reference the generated `interaction_policy.yaml`
rather than duplicating ownership policy. Bundled presets (`--preset` or the
preset menu) copy maintained examples and then report the files plus the next
command to run.

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

### Diffusion Monte Carlo (PhysNetJax)

```bash
mmml dmc \
  --natm 20 --nwalker 64 --stepsize 5e-4 --nstep 200 --eqstep 50 \
  --alpha 1200.0 --max-batch 64 --seed 0 \
  --checkpoint "$MMML_CKPT" \
  --input mmml/generate/dmc/examples/acetone_dmc.extxyz \
  --output-dir runs/dmc_acetone_smoke
```

See the [DMC guide](../dmc.md) for a longer production example and output files.

## Regenerating CLI reference pages

Per-command pages under `docs/cli/commands/` are generated from
`mmml/cli/registry.py`:

```bash
uv run python scripts/generate_cli_docs.py
```

CI and `make docs-build` run this before `mkdocs build` so the sidebar stays in
sync with the registry.
