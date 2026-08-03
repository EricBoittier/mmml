# Config map — which file is for what

The repo tracks ~230 YAML configs. They are not interchangeable: some ship with
the installed package, some only work from a source checkout, and some are
campaign records that describe one machine's paths. This page says which is
which. Per-family detail stays in the READMEs next to the files.

## Families

| Location | What it is | Ships in the wheel | Edit it? |
|---|---|---|---|
| `mmml/cli/run/*.example.yaml` | Ready-to-run `mmml md-system` examples | yes | copy, then edit the copy |
| `mmml/cli/run/presets/*.yaml` | Composable fragments to `include:` | yes | no — compose instead |
| `mmml/cli/misc/*.example.yaml` | `mmml physnet-train` examples | yes | copy first |
| `mmml/mcp/examples/*.yaml` | Hybrid MD entry points per backend | yes | copy first |
| `mmml/mcp/recipes/*.yaml` | MCP tool recipes | yes | no — code reads these |
| `mmml/cli/templates/workflows/**/config.yaml` | `mmml workflow` scaffolding templates | yes | no — copied on scaffold |
| `examples/**/*.yaml` | Worked examples, usually with a README and a driver script | no | copy first |
| `workflows/**/*.yaml` | Campaign definitions, tied to specific studies | no | fork per campaign |
| `setup/environment*.yml` | Conda environments | no | pick one, see below |
| `devtools/`, `.github/`, `.readthedocs.yaml`, `mkdocs.yml`, `.codecov.yml` | Build/CI/docs tooling | no | rarely |

## Start here

```bash
# Single simulation, flat config
mmml md-system --config mmml/cli/run/md_system.example.yaml

# Dense liquid box prep
mmml md-system --config mmml/cli/run/md_system.dense_liquid_prep.example.yaml

# Full preset stack, resilient mode
mmml md-system --config mmml/cli/run/dcm_liquid_workflow.resilient.example.yaml
```

`mmml env` prints the resolved paths it will use, including the bundled
checkpoint and the resilient workflow config — run it first when a path is in
doubt.

Field-by-field reference for these: [md-system-configs.md](md-system-configs.md).
Preset composition rules:
[`mmml/cli/run/presets/README.md`](https://github.com/EricBoittier/mmml/tree/main/mmml/cli/run/presets#readme).

## The `.example.yaml` suffix is load-bearing

`pyproject.toml` `package-data` ships `cli/run/*.example.yaml`. A config in that
directory **without** the suffix is not installed, so any documented
`--config mmml/cli/run/<name>.yaml` silently works from a git checkout and fails
for anyone who `pip install`ed the package. Name new user-facing configs there
`*.example.yaml`, or extend the glob deliberately.

The same reasoning applies to `cli/misc/` and `mcp/examples/`: both are
documented as `--config` targets, so both are in `package-data`.

## Configs that need something generated first

Many configs reference `artifacts/...`, `boxes/...` or `output/...` paths that
are **not** in the repo — they are produced by an earlier step, and a config
pointing at one is a dependency, not a broken file. Roughly 38 tracked configs
reference at least one such path. Common cases:

| Pattern | Produced by |
|---|---|
| `artifacts/nh3_ch3cl/boxes/*/model.pdb` | `examples/m/08_make_boxes.sh` |
| `artifacts/md_system_from_pdb/box_*/model.{psf,crd}` | the earlier numbered configs in `examples/md_system_from_pdb/` |
| `boxes/<name>/model.{psf,crd}` | `mmml md-system` box prep, relative to the run's `output_dir` |
| `output/*.npz` | the dataset build for `mmml physnet-train` |
| `artifacts/tria_phi_psi_scan/**/*_seeds.npz` | the seed-generation step in that example |

If a config fails on a missing input, check the sibling README for the step that
produces it before treating the config as stale.

## Generated configs are not repository configuration

`mmml md-system` writes a resume bundle next to a failed run — `next_run.yaml`,
`next_run.sh`, `next_run.command`, `next_run_advice.json`
(`mmml/cli/run/md_run_advice.py`). These record one machine's job state, and are
gitignored. Do not commit them, and do not treat one found in a run directory as
an example: its `include:` paths are relative to wherever it was written and go
stale immediately.

## Environments

`setup/environment.yml` is the default. `environment-gpu.yml` and
`environment-gpu-cuda13.yml` pin CUDA builds; `environment-full.yml` adds the
optional analysis stack. `devtools/conda-envs/test_env.yaml` is CI's, not yours.

## Conventions for new configs

- Put a comment on line 1–2 giving the exact command that runs it, as the
  existing examples do.
- User-facing and inside `mmml/`? Use `*.example.yaml` and confirm a
  `package-data` glob reaches it.
- Study-specific? It belongs in `workflows/<campaign>/` with a README, not in
  `mmml/`.
- Prefer `include:` of a preset over copying its keys, so a preset fix
  propagates.
