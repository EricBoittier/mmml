# MMML validation campaign

This workflow is the campaign index for proving that supported MM, ML, and
hybrid MM/ML simulations work across liquids, gas-phase peptides, and solvated
peptides. It does not treat a submitted job as success: every task has explicit
acceptance checks and a proof directory.

## Scientific goals

| Goal | Systems | Required proof |
|---|---|---|
| `platform` | every compute environment | environment report, JAX x64/CUDA status, import and tiny-energy checks |
| `force_energy` | representative monomers/dimers | component energies, LJ/electrostatic audit, finite differences, cutoff scans |
| `backend_parity` | DCM:10 and TIP3:small | identical-state NVE/NVT traces, drift metrics, round-trip state audit |
| `pure_liquids` | BENZ, TIP3, DCM, ACO | preparation audit, stable NVT/NPT, density/RDF/temperature/energy plots |
| `peptide_gas` | alanine, trialanine | minimized structure, finite-difference forces, NVE/NVT stability and conformational plots |
| `peptide_solution` | alanine+TIP3, trialanine+TIP3 | solvation audit, stable NVT/NPT, peptide-water RDF/contact and conformation plots |
| `coverage` | all supported combinations | machine-readable method/backend support matrix with pass, fail, blocked, unsupported |

Each task writes under `artifacts/validation_campaign/<run_id>/<task_id>/`:

- `request.json`: immutable task/environment/config/checkpoint request;
- `stdout.log`, `stderr.log`, and scheduler metadata;
- `status.json`: terminal state and exact failure, if any;
- `metrics.json`: quantitative acceptance metrics;
- `proof.json`: acceptance checks and source artifact hashes;
- `plots/`: human-readable figures using the repository plot style;
- `provenance.json`: git revision, dirty diff hash, Python/JAX/CUDA/CHARMM details.

The campaign summary is generated only from `proof.json` receipts. Missing
proof is reported as `INCOMPLETE`; unsupported combinations are distinguished
from regressions.

## Environments

- `pcbach`: CPU-heavy and memory-heavy reference generation, cache building,
  classical MM, analysis, and large scans.
- `scicore`: primary GPU production and broad checkpoint/method matrices.
- `pcstudix`: fast smoke tests and focused GPU diagnostics; do not launch the
  full matrix until the handoff gates pass.
- `local_laptop`: CPU debugging, dry-runs, unit tests, plots, and collection.
- `local_computer`: CUDA debugging and single-GPU integration tests.

Environment definitions live in `environments/*.yaml`. Cluster launch uses
`scripts/submit_slurm.py`; local launch uses `scripts/run_local.py`. Both write
the same request/provenance/status layout.

## Commands

List the campaign and current proof state:

```bash
python workflows/validation_campaign/scripts/campaign.py list
python workflows/validation_campaign/scripts/campaign.py status
```

Generate (but do not submit) Slurm scripts:

```bash
python workflows/validation_campaign/scripts/campaign.py prepare \
  --environment pcstudix --tier smoke
```

Submit ready tasks after inspecting the generated scripts:

```bash
python workflows/validation_campaign/scripts/campaign.py submit \
  --environment pcstudix --tier smoke
```

Run a local tier:

```bash
python workflows/validation_campaign/scripts/campaign.py run-local \
  --environment local_laptop --tier static
```

Campaign state is summarized to `artifacts/validation_campaign/summary.{json,md}`:

```bash
python workflows/validation_campaign/scripts/campaign.py status --write
```

## Current hard blocker

`backend_parity.dcm10_pycharmm_jaxmd` is blocked at the PyCHARMM velocity
handoff boundary. Text/native restart attempts start at 0 K, while direct
`dynamics_run_kw(init_velocities=...)` currently SIGSEGVs in `dynopt` due to
the gfortran `bind(c)` assumed-shape-array ABI. This blocker must remain visible
in the summary and must gate solvent-burst production that alternates backends.

