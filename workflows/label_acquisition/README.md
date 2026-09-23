# Label acquisition: structure selection for expensive reference calculations

Compare methods for choosing which unlabeled atomistic structures to send to
an expensive reference method before transfer-learning a student potential.

The **foundation / teacher potential is a cheap surrogate, not ground truth.**
Acquisition methods never see expensive energies or forces until their
selection manifests are written and frozen.

This workflow starts from a student already trained with a foundation teacher,
an unlabeled candidate pool, and (for information-gain) the existing
student-training set as a *design prior* — not calibrated uncertainty against
the expensive method.

## Methods

| Method | Representation | Selection |
|--------|----------------|-----------|
| `stratified_random` | composition × thermodynamic-condition strata | random within strata |
| `activation_fps` | species-aware pooled invariant features before the energy readout | PCA + farthest-point sampling |
| `activation_largest_norm` | same activations | largest L2 (diagnostic ablation) |
| `output_grad_fps` | energy Jacobian w.r.t. final readout parameters | PCA + FPS |
| `output_grad_largest_norm` | same Jacobians | largest L2 |
| `force_doptimal` | weighted energy *and* force Jacobians | greedy regularized D-optimal |
| `loss_grad_fps` | student–teacher loss gradient w.r.t. readout | PCA + FPS |
| `loss_grad_largest_norm` | same loss gradients | largest L2 |
| `unmodified_student` | — | no extra labels (baseline) |

PCA is the only dimensionality reduction (no UMAP / t-SNE / random projection).
Held-out validation / test trajectories never enter PCA or selection.

For a **linear** energy readout, sum-pooled activations equal `∇_w E`.  PhysNet
uses `e3x.Dense(1)` then `nn.Dense(1)` plus optional `energy_bias`, ZBL, and
electrostatics; the report measures alignment instead of assuming equivalence.

Loss gradients use teacher labels.  Substituting the student's own predictions
would give zero gradients.  Shared student–teacher errors are invisible.
FPS here is **not** the original classification BADGE algorithm.

## Smoke (mock results — not science)

The mock Morse backend and linear student certify the DAG.  Keep their
artifacts separate from any production tree.

```bash
cd workflows/label_acquisition
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2 -n
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2
```

Python-only equivalent (no Snakemake):

```bash
uv run mmml label-acquire --config workflows/label_acquisition/config.smoke.yaml -o /tmp/acq_smoke all
```

Pass criteria: dry-run exits 0; smoke run writes `report/report.md`; mock
cache reuses labels when selections overlap.

## Production (do not invent settings)

`config.yaml` is a template.  These choices are **unresolved** until you fill them:

1. **Reference backend** — PySCF / ORCA / Molpro / … including method, basis,
   charge, multiplicity, and units.  Smoke's `mock_morse` is not a substitute.
2. **Compute budget** — unique-label cap, walltime, GPU hours.  The report
   lists each method's *nominal* budget and the *actual* unique calculations
   (overlapping selections share the label cache).
3. **Deployment conditions** — which held-out trajectories and MD settings
   define “best subset”.  Evaluation is fixed and independent of acquisition.
4. **Student / teacher checkpoints** — frozen student JSON/Orbax and the
   foundation teacher used as a surrogate.
5. **Candidate pool** — unlabeled NPZ (`R`, `Z`, `N`) with trajectory grouping
   keys (`group_seed` / `source_path`) so neighboring frames cannot leak.

Do not launch costly QC or MD from the unresolved template.

## Outputs

Under `output_root` (repo-root `artifacts/label_acquisition/<tag>/` by default):

- `pool/` — stable IDs, grouped splits, duplicate map
- `representations/` — activation / Jacobian / loss-gradient shards
- `pca/` — fit (centering, scaling, retained variance) on **candidates only**
- `selections/<method>/budget<k>/seed<s>/manifest.json` — immutable ID lists
- `labels/cache/<structure_id>/<method_hash>/` — reused expensive labels
- `eval/` — held-out E/F metrics and matched NVE stability
- `report/report.md` — learning curves, overlap, diversity, cost, stability

## Later rounds

This revision is **one-shot** (`round: 0`).  After fine-tuning, point `student`
at the new checkpoint, increment `round`, and re-run `extract` so
representations reflect the updated model.  Selection manifests stay immutable
per round.

## Tests

```bash
uv run pytest tests/unit/test_acquisition_ids.py \
  tests/unit/test_acquisition_splits.py \
  tests/unit/test_acquisition_selection.py \
  tests/unit/test_acquisition_pca.py \
  tests/unit/test_acquisition_labels.py \
  tests/unit/test_acquisition_jacobians.py \
  tests/unit/test_acquisition_workflow_smoke.py -q
```
