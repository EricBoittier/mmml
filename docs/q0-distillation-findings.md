# Q⁰ charge campaign: distillation findings, 2026-08-02

Three findings from wiring teacher distillation into the Spooky trainer and
running it against the real checkpoints. The third blocks the campaign.

## 1. Job 206072 failed at the NVE gate, as designed

The epoch-2 Q⁰ minimization/NVE job did not stall — it was refused:

```
NVE refused: post-FIRE max|F|=16.3019 eV/Å > size-scaled gate 7.0292 eV/Å
             (base 1.5000 × 4.686 for N=2196)
dominant_term (by mean|F|) ml_dimer
```

This independently reproduces the handoff's root-cause numbers (16.31 eV/Å,
dimer-ML dominated). Evidence:
`artifacts/npt_argon_water/q0_epoch2/{suite_summary_jaxmd.json,md.log,status-206072.tsv}`.

## 2. `step-00294400` cannot be used as a teacher on the SO3LR cache

The handoff selected `artifacts/spooky_so3lr/step-00294400` as the distillation
teacher. It is incompatible with the student's cache, and the architecture gate
refuses it with 8 checkpoint parameters that have nowhere to go:

```
params/Dense_13/{bias,kernel}          # learned per-atom vdW-scale head
params/{element,global}_vdw_scale      # CGenFF LJ scales
params/repulsion/…                     # ZBL (fixed separately, see below)
```

Cause — the caches differ:

| Cache | CGenFF tables | `mol_id` |
|---|---|---|
| `so3lr_train_flat_5ba739ddac63717d` (student) | ✗ | ✗ |
| `data/splits_des_ml_mm_v2/train_cache` (teacher) | ✓ | ✓ |

The entire CGenFF branch in `SpookyPhysNet` is gated on `cgenff_type_idx is not
None` (`spooky_model.py`), so on the student's cache that teacher's LJ prior and
vdW-scale head are inert and its electrostatics runs unmasked (`mol_id=None` →
`inter_monomer_mask = 1.0`). Its outputs would not be what it was trained to
produce, and the resulting distillation targets would carry a systematic,
geometry-dependent error.

`mmml/models/spookynet_calc.py` already warns about this class of mismatch at
evaluation time. **Open question:** `step-00294400`'s reported 3.50 eV/Å max
force may itself have been measured with that warning active, i.e. LJ-less.
Worth confirming before treating that number as characterizing the full model.

**Resolution:** teacher switched to `artifacts/spooky_so3lr_adam_cw2/step-02700000`
— AdamW (not Muon), trained on a CGenFF-free / `mol_id`-free SO3LR cache
(`so3lr_train_flat_2ef9214c00a78127`), same architecture family (features 128,
max_degree 2, efa off), `trainable_zbl` explicitly recorded. Chosen over the
newer `step-03600000` because it is the only adam_cw2 checkpoint with a
*completed* evaluation (`step-02900000` OOM'd and was cancelled) and with a
36-point TIP3 dimer scan — the surface that actually fails.

**Still outstanding:** this teacher has no max-force measurement on the prepared
732-water box. Gate it before spending GPU-days on a full distilled run.

A related bug was fixed while diagnosing this: checkpoints predating
`trainable_zbl` were being rebuilt with today's default (`False`), which builds
no ZBL repulsion parameters at all. The warm-start path already had this legacy
inference; the teacher path now mirrors it.

## 3. The Q⁰ experiments are not initialized from epoch-2

**This blocks the campaign.** All three Q⁰ jobs print:

```
Warm-started from …/spooky_so3lr_charges/epoch-0002:
loaded 36 parameter leaves, initialized 10 new leaves, skipped 2 incompatible leaves
```

The 10 randomly-initialized tensors include the **entire `MessagePass_2` block**.
Agreed campaign goal "initialize the charge-aware student from charge epoch 2"
is therefore not happening in any experiment.

`spooky_so3lr_charges/epoch-000{1,2,3}` each have 41 parameter leaves and **2**
MessagePass blocks. An exhaustive search over `(num_iterations, n_res, efa,
use_energy_bias)` for an exact parameter-tree match gives exactly one answer:

```
n_iter  n_res  leaves  missing  extra
     2      3      41        0      0   *** EXACT ***
```

**Epoch-2 is `num_iterations=2, n_res=3`.** Every Q⁰ sbatch passes
`--num-iterations 3 --n-res 2`.

The mismatch survives the architecture-override logic because that logic reads
the **sibling `run_config.json`**, which is workdir-level, is rewritten by each
run into that workdir, and records `num_iterations: 3, n_res: 2` — values that do
not describe these weights. `_merge_compatible_params` then merges leniently and
only prints counts.

This explains the metrics: the distilled smoke showed `E_MAE=150150` after 40
steps, and `q0-longer` still showed `E_MAE=205` at step 26,600 — both re-learning
from a partly-random model.

Distillation is exonerated independently: `TE_MAE=150155` vs `E_MAE=150150`, i.e.
the aligned teacher target and the reference agree to 0.003%; the student is
equally far from both.

### Fix

1. Set `--num-iterations 2 --n-res 3` in the Q⁰ scripts, **or**
2. better, make `--init-checkpoint` read the checkpoint's own `model_attributes`
   (all three charge checkpoints record it) instead of the sibling
   `run_config.json`, and error on leaf mismatch rather than merging leniently —
   the discipline `teacher_architecture_from_checkpoint` already applies, which
   is why the teacher path caught its mismatch and the warm-start path did not.

Any epoch-2 quantity measured through the *retraining* scripts should be
re-derived after this fix. Numbers from the evaluator may be unaffected if it
builds the architecture from `model_attributes`; that is worth checking before
concluding anything about epoch-2 itself.

## Job outcomes

| Job | Outcome | Useful |
|---|---|---|
| `206072` tip3-q0 | FAILED — NVE gate, `ml_dimer` dominant | yes — confirmed root cause |
| `206081` q0-retrain-smoke | TIMEOUT, 0 steps, no checkpoint | no |
| `206088` q0-distill-smoke | FAILED — gate rejected `step-00294400` | yes — caught finding 2 |
| `206089` q0-distill-smoke | COMPLETED, checkpoint + provenance | wiring yes, science no (finding 3) |
| `206078` q0-longer | CANCELLED at 54:42 | no — was running on finding 3 |

`artifacts/spooky_q0_longer/step-000{20000,40000}` were produced from the
wrong-architecture init and should be discarded.

`206081` timed out because `--max-structures` is read only while *building* a
cache; under `--mode train --cache-path` it silently does nothing, so the "smoke"
became a full ~924k-step epoch. Bound smokes with `--steps-per-epoch`.

## Reproducing the diagnosis

Run against `pc-studix`; see the side-checkout pattern in
[spooky-teacher-distillation.md](spooky-teacher-distillation.md) for why a clone
plus `PYTHONPATH` is required rather than editing the live tree.

Parameter-tree comparison needs no GPU: `_METADATA` lists every leaf path, so a
checkpoint's architecture can be identified on a login node by initializing
candidate models on CPU and diffing path sets.
