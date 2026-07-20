# Dimer tool inventory and ownership

This inventory prevents existing dimer functionality from becoming invisible
or being mistaken for the supported API. It records the intended ownership of
the tools present on 2026-07-20.

## Classification

- **Canonical**: supported public API or CLI; new work should extend this path.
- **Supporting library**: reusable package component used by canonical or
  specialized workflows, but not itself the end-user scan interface.
- **Wrapper**: launch/configuration glue; must delegate scientific behavior.
- **Campaign**: reproducible specialized study whose scope is broader or more
  specific than the canonical 1D scan.
- **Post-processing**: consumes results and must not perform hidden evaluation.
- **Validation**: scientific or runtime check, not a production scan API.
- **Exploratory**: useful research prototype with no stability guarantee.
- **Deprecated**: retained for provenance only; do not extend or import.

## Canonical path

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.dimer_scan` | Canonical | Public configuration, evaluation, result bundle, provenance, plotting, and `run_dimer_scan`. Extend this for supported 1D calculator-neutral scans. |
| `mmml.cli.misc.dimer_scan` | Canonical | Thin `mmml dimer-scan` adapter. It must not acquire independent scientific logic. |
| `mmml.analysis.dimer_scans` | Supporting library | Deterministic geometry and low-level scan helpers. Migrate silent failure-dropping evaluators toward the canonical structured result path. |
| `mmml.analysis.dimer_molecules` | Supporting library | Named monomers and orientations. Treat orientation definitions as versioned scientific inputs. |
| `mmml.analysis.dimer_cgenff` | Supporting library | CGenFF-specific dimer analysis; keep calculator/domain-specific behavior outside the generic scan core. |
| `mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy` | Supporting library | Sparse-dimer policy for MLpot/PyCHARMM, not a general scan interface. |

## Campaigns and wrappers

| Path | Class | Ownership and next action |
|---|---|---|
| `scripts/run_dimer_scan_campaign.py` | Campaign | Multi-backend/reference campaign. Reuse canonical result/config types when migrated; do not make it the general API. |
| `scripts/run_dimer_scan_campaign_gpu.sh` | Wrapper | GPU launcher for the campaign above. Keep scientific defaults in a serializable campaign config. |
| `workflows/des_dimer_pair_scans/` | Campaign | Reproducible DES pair-scan Snakemake campaign. Preserve as a workflow; progressively delegate geometry/evaluation/result semantics to package APIs. |
| `scripts/scan_mlpot_dimer_2d_pycharmm.py` | Campaign | Specialized 1D/2D hybrid PyCHARMM and long-range scan. Retain until its decompositions and solver semantics have canonical equivalents. |
| `scripts/run_dcm_aco_dimer_lr_scans.sh` | Wrapper | Launch matrix for the specialized long-range campaign. |
| `scripts/run_mlpot_dimer_2d_scans.sh` | Wrapper | Legacy 2D/cutoff campaign launcher. |
| `scripts/run_corrected_cache_step23200_dimer_scan.slurm` | Wrapper | Checkpoint-specific historical Slurm launcher; freeze inputs and mark complete when no longer used. |
| `scripts/run_corrected_cache_step5400_dimer_scan.slurm` | Wrapper | Checkpoint-specific historical Slurm launcher; freeze inputs and mark complete when no longer used. |
| `scripts/run_hybrid_step3000_dimer_scan.slurm` | Wrapper | Checkpoint-specific historical Slurm launcher. |
| `scripts/run_mbd_checkpoint_dimer_compare.slurm` | Wrapper | MBD comparison launcher. |
| `scripts/run_recent_checkpoint_series_dimer_compare.slurm` | Wrapper | Series comparison launcher; replace “recent” inputs with explicit checkpoint manifests before reuse. |
| `scripts/run_recent_step19800_dimer_compare.slurm` | Wrapper | Historical checkpoint launcher; “recent” is not a reproducible identity. |
| `scripts/run_recent_step36800_dimer_compare.slurm` | Wrapper | Historical checkpoint launcher; “recent” is not a reproducible identity. |
| `scripts/meoh_dimer_lambda_ti.py` | Campaign | Alchemical TI study, not a rigid scan API. |
| `scripts/meoh_dimer_lambda_mbar.py` | Post-processing | MBAR analysis for the TI study. |
| `scripts/scan_meoh_dimer_2d_cutoffs.py` | Campaign | Methanol cutoff study with specialized dimensions. |
| `scripts/scan_hybrid_dimer.py` | Exploratory | Hybrid prototype. Audit conventions against the canonical API before reuse. |
| `scripts/scan_dimer_orientations.py` | Campaign | Orientation atlas/campaign. Feed successful orientation definitions into the versioned orientation registry. |
| `scripts/scan_meoh_dimer_distance.py` | Exploratory | Earlier single-purpose distance scan. Do not extend; migrate any unique calculator behavior into `mmml.dimer_scan`. |
| `scripts/run_charmm_and_plot_dimers.py` | Exploratory | Mixed execution/plotting prototype. Split responsibilities if any behavior is promoted. |
| `workflows/pbc_solvent_burst/scripts/scan_heterogeneous_dimer_boundary.py` | Validation | Workflow-specific PBC boundary diagnostic. |

## Post-processing and dataset preparation

| Path | Class | Ownership and next action |
|---|---|---|
| `scripts/clean_dimer_scan_data.py` | Post-processing | Preserve raw data; record transformation configuration and output schema. |
| `scripts/merge_dimer_scan_references.py` | Post-processing | Reference merge; retain source identity and reject ambiguous overwrites. |
| `scripts/plot_dimer_lr_scan_compare.py` | Post-processing | Plot saved long-range results only. |
| `scripts/plot_dimer_scan_components.py` | Post-processing | Plot saved decompositions only. |
| `scripts/plot_ab_initio_dimer_surfaces.py` | Post-processing | Plot saved ab-initio surfaces only. |
| `scripts/make_dimer_scan_dataset.py` | Campaign | Dataset construction; needs an explicit schema and source manifest. |
| `scripts/reorder_dimer_atoms_to_reference.py` | Post-processing | Deterministic atom-order transformation with dedicated regression tests. |
| `scripts/split_dcm_mp2_monomers_dimers.py` | Post-processing | Dataset-specific split; preserve source indices and split rules. |

## Validation tools

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.mode_check` | Canonical | Monomer **and** small-cluster local diagnostics (FD forces, X–H stretch, ASE vib, kick). `--cutoff-sweep` samples COM stations on the hybrid handoff ruler. Not a binding-energy COM scan — that remains `mmml.dimer_scan`. |
| `mmml.cli.misc.mode_check` | Canonical | Thin `mmml mode-check` adapter (`--pbc-fd` covers the former unregistered `check_fd.py` path; `--cutoff-sweep` for region ladder). |
| `mmml.cli.run.md_pbc_suite.check_fd` | Wrapper | Legacy script entry; delegates to `mmml.mode_check.pbc_fd`. Do not extend scientific logic here. |
| `scripts/validate_dimer_rays.py` | Validation | Geometry/ray validation, not scan execution. |
| `scripts/validate_mlpot_sparse_dimers.py` | Validation | Production preflight for sparse dimer selection. |
| `tests/functionality/dimer_scans/` | Validation | Environment-dependent functional scan checks. |
| `tests/functionality/long_range/01_mic_analytic_dimer.py` | Validation | Analytic minimum-image check. |
| `mmml/mcp/recipes/dimer_smoke.yaml` | Validation | Agent-facing smoke recipe; keep it pointed at supported commands. |

## Deprecated and historical paths

| Path | Class | Ownership and next action |
|---|---|---|
| `mmml.interfaces.aseInterface.dimers` | Deprecated | Machine-specific paths, environment/device mutation, and execution at import time. Retain only for provenance until unique behavior is extracted; never import from new code. |
| `mmml.generate.sample.pycharmm_dimers` | Exploratory | Audit before reuse; promote only deterministic, tested package behavior. |
| `mmml.generate.sample/dimers.sh` | Deprecated | Historical generation shell path. Replace with a documented package/CLI caller if still needed. |

Data assets such as `mmml/generate/sample/meoh_dimer.pdb` and bundled model
fixtures such as `mmml/models/physnetjax/defaults/meoh_dimer_portable.json` are
inputs, not tools. Their content identity should be recorded when used.

## Maintenance rule

Update this inventory in the same pull request that adds, removes, supersedes,
or changes the ownership of a dimer-related tool. A new supported scan path
requires an explicit explanation of why `mmml.dimer_scan` cannot be extended.
