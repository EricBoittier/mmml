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

### Fix (implemented)

`--init-checkpoint` now applies the checkpoint's **own** `model_attributes`
before building the model, and refuses a parameter tree that does not line up
unless `--init-allow-partial` is passed. Explicit non-default architecture flags
still win, so deliberately adding a head to fine-tune still works — but it can no
longer happen silently.

Fixing the Q⁰ sbatch flags by hand would *not* have been sufficient. `--n-res 2`
and `--num-iterations 3` both equal their parser defaults, and the override logic
treats "equals the default" as "inherit" — so the values were already eligible to
be inherited; the problem was that the source being inherited from was the wrong
file. Only the checkpoint's own record carries the right answer.

Verified against the real checkpoint (job 206099):

```
Applying architecture recorded in …/spooky_so3lr_charges/epoch-0002 (model_attributes)
  Overriding cutoff: 6.0 -> 4.0 (from model_attributes)
  Overriding efa: False -> True (from model_attributes)
  Overriding n_res: 2 -> 3 (from model_attributes)
  Overriding num_iterations: 3 -> 2 (from model_attributes)
  Overriding use_energy_bias: False -> True (from model_attributes)
```

The warm-start then loads cleanly, and the distilled smoke completes (job
206104). Same script, same teacher, same 40 steps — only the init differs:

| | before fix (206089) | after fix (206104) |
|---|---:|---:|
| warm-start | 36 loaded / 10 random / 2 dropped | **41 / 0 / 0** |
| `E_MAE` (eV) | 150150 | 339 |
| `F_MAE` (eV/Å) | 2438 | 0.765 |
| `Q_MAE` (e) | 4.46 × 10⁶ | 1.48 |
| `valid_F_MAE` (eV/Å) | 1560 | 0.485 |

Forces land in a physically plausible range for the first time. The teacher also
behaves as a regularizer rather than a competing objective: student-vs-teacher
`TF_MAE` 0.829 against student-vs-reference `F_MAE` 0.765, and `TE_MAE` 343.7
against `E_MAE` 339.5.

These are 40 steps on the largest atom-count bucket and should not be read as
converged quality — but the change is unambiguous.

### A fourth discrepancy the leaf check could never catch

`cutoff: 6.0 -> 4.0`. **Epoch-2 was trained with a 4 Å cutoff**; every Q⁰ script
passes 6.0, and the sibling `run_config.json` also says 6.0.

Cutoff does not change any parameter *shape*, so all 41 leaves would have loaded
cleanly and the strict leaf check would have reported nothing — the model would
simply have computed different physics from the one whose weights it holds. This
is the argument for preferring `model_attributes` over merely validating shapes:
shape agreement is necessary, not sufficient.

Any epoch-2 quantity measured through the *retraining* scripts should be
re-derived. Numbers from the evaluator may be unaffected if it builds the
architecture from `model_attributes`; that is worth checking before concluding
anything about epoch-2 itself — including its 9.16 eV/Å static max force.

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
