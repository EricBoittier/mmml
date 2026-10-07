# KARML CLI

The `karml` command is the unified entry point for structure building, mixed MM/ML
molecular dynamics, QM data pipelines, and training workflows.

Install the optional CLI extra for shell tab completion:

```bash
uv sync --extra cli
# or: pip install 'karml[cli]'
```

## Help layers

KARML splits help across a few commands so `karml -h` stays short while deeper
detail stays one command away.

| Command | Purpose |
|---------|---------|
| `karml -h` | Compact top-level summary (common subcommands + pointers) |
| `karml commands` | All subcommands grouped by task area |
| `karml commands --audit` | Deprecated/legacy commands and tab-completion coverage |
| `karml examples` | Copy-paste example invocations |
| `karml <command> --help` | Full flags for one subcommand |
| `karml configure` | Interactive YAML / Snakemake wizard |
| `karml env` | Resolved checkpoints, CHARMM paths, export hints |
| `karml completion <shell>` | Print bash/zsh/fish completion script |

```bash
karml -h
karml commands
karml examples
karml md-system --help
karml env --json
```

## Output conventions

CLI output is designed to stay readable in terminals and copyable in logs:

- `karml <command> --help` uses shared argparse grouping for commands dispatched
  through `karml`. Flat option lists are grouped by input/configuration,
  scientific model, execution, output/artifacts, and diagnostics/safety.
- Rich color is supplemental. Redirected output remains plain text, while
  `KARML_NO_RICH=1` disables Rich formatting and `KARML_RICH=1` forces it for
  terminal demos.
- JSON-shaped diagnostics use valid JSON even when color is enabled, so output
  from commands such as `karml env --json` can still be copied into a parser.
- Long-running and setup commands should honor quiet modes where provided; the
  shared reporting helpers also respect `KARML_QUIET=1`.

## Tab completion

With `argcomplete` installed (`karml[cli]`), completion covers subcommand names
and flags (when `build_parser()` exists for that command).

```bash
eval "$(register-python-argcomplete karml)"
# or:
eval "$(karml completion bash)"
```

See [Tab completion](completion.md) for per-shell setup and fallbacks when
`argcomplete` is not installed.

## Command index

These docs are organized the same way `karml commands` is. Each task group is a
section in the sidebar that opens with an orientation page, then its conceptual
guides, then a **Commands** subsection with one generated reference page per
subcommand (options pulled from that command's `argparse` help).

Every section follows the same shape, so the same position always means the
same thing:

| Slot | What it is |
|------|------------|
| Section index | What the area covers and where the guides are |
| Tutorial | One start-to-finish worked path |
| How-to | Task-focused pages that assume you have context |
| Commands | Generated reference, one page per subcommand |

| Section | Start here | Commands |
|---------|------------|----------|
| [Structure & boxes](../sections/structure-boxes.md) | [Structure building](structure-building.md) | `make-res`, `make-box`, `build-crystal`, `liquid-box` |
| [MD & campaigns](../sections/md-campaigns.md) | [Tri-alanine water box](../trialanine-water-box.md); [φ/ψ → umbrella teaching](../examples/tria-phi-psi-scan.md) | `md-system`, `metatomic-pbc-md`, `md-embedding`, `umbrella-sample`, `health-check` |
| [QM & data](../sections/qm-data.md) | [Training NPZ contract](../training-npz-contract.md); [Internal-coordinate scans](../ic-scan-design.md) | `pyscf-*`, `dimer-scan`, `ic-scan`, `fix-and-split` |
| [Hybrid ML/MM potentials](../sections/hybrid-potentials.md) | [Cutoffs, regions & LR solvers](../hybrid-potential-regions.md) | — |
| [Training & sampling](../sections/training-sampling.md) | [NEB](../neb.md), [DMC](../dmc.md) | `physnet-*`, `pet-physnet-distill`, `efield-*`, `kernnn-*`, `neb`, `dmc` |
| [Environment and HPCs](../sections/environment-clusters.md) | [SciCORE guide](../scicore.md) | `env`, `configure`, `doctor`, `mpi-launch`, `completion` |

**Hybrid ML/MM potentials has no command group.** Assembling a hybrid potential
is configuration, not a subcommand — which is exactly why those pages needed a
section of their own rather than being filed under training.

Structure builders (`make-res`, `make-box`, `build-crystal`) include ASE
structure figures — see [Structure building](structure-building.md).

Run `karml commands --audit` locally to see which commands are **deprecated** or
**legacy** and what to use instead.

## Configure safety model

`karml configure` is the interactive entry point for YAML and workflow scaffolds.
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
karml env                                    # checkpoints + CHARMM paths
karml configure                              # or hand-edit YAML
karml health-check --require-gpu --live
karml warmup-mlpot-jax --checkpoint "$KARML_CKPT" --n-monomers 20
KARML_MPI_NP=1 ./scripts/karml-charmm-mpirun.sh md-system --config run.yaml
```

### Train PhysNet from NPZ

```bash
karml fix-and-split --efd data.npz --output-dir splits/
karml physnet-train --config train.yaml
karml physnet-evaluate --checkpoint ckpts/run --test splits/test.npz
```

### Diffusion Monte Carlo (PhysNetJax)

```bash
karml dmc \
  --natm 20 --nwalker 64 --stepsize 5e-4 --nstep 200 --eqstep 50 \
  --alpha 1200.0 --max-batch 64 --seed 0 \
  --checkpoint "$KARML_CKPT" \
  --input karml/generate/dmc/examples/acetone_dmc.extxyz \
  --output-dir runs/dmc_acetone_smoke
```

See the [DMC guide](../dmc.md) for a longer production example and output files.

## Regenerating CLI reference pages

Per-command pages under `docs/cli/commands/` are generated from
`karml/cli/registry.py`:

```bash
uv run python scripts/generate_cli_docs.py
```

CI and `make docs-build` run this before `mkdocs build` so the sidebar stays in
sync with the registry.
