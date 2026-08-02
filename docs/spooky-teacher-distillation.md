# Teacher distillation in the Spooky SO3LR trainer

`scripts/train_so3lr_spooky_extxyz.py` can distil from a frozen Spooky teacher
checkpoint. Ground-truth energy and force labels remain the primary objective;
the teacher regularizes them.

This is **not** the same thing as `mmml physnet-train --distill`, which is a
separate generic PhysNet pipeline. The two share only the blending arithmetic
(`mmml/models/physnetjax/physnetjax/training/distill.py`).

## Usage

```bash
python scripts/train_so3lr_spooky_extxyz.py \
  --mode train --cache-path "$CACHE" --workdir "$OUT" \
  --init-checkpoint "$INIT" \
  --teacher-checkpoint "$TEACHER" \
  --distill-alpha 0.75 --distill-targets energy forces \
  --distill-energy-align atomic --distill-align-batches 16
```

| Flag | Meaning |
|---|---|
| `--teacher-checkpoint` | Orbax dir or portable params JSON. Architecture is rebuilt from the teacher's **own** checkpoint, never from the student's flags. |
| `--distill-alpha` | Weight on the ground-truth term: `loss = α·gt + (1-α)·teacher`, per distilled component. `1.0` reproduces undistilled training exactly; `0.0` trains against the teacher alone. |
| `--distill-targets` | `energy`, `forces`, or both. Charges and dipoles are **rejected**, not ignored. |
| `--distill-energy-align` | `atomic` (default), `scalar`, or `none`. See below. |
| `--distill-align-batches` | Per-device batches sampled across atom-count buckets when fitting the alignment. |

## Design decisions

**The teacher is frozen reference physics**, handled exactly like the existing
MBD and multipole auxiliaries: its parameters are `stop_gradient`-mapped and its
outputs enter the loss only through `stop_gradient`. `tests/unit/test_spooky_distill.py`
proves `d(loss)/d(teacher_params) == 0` by differentiating through `make_steps`
itself, with a control asserting the same loss *is* sensitive to student
parameters.

**Charges and dipoles are never distilled.** The student's charge head is the
point of the charge-aware campaign, so it stays supervised by reference data.
Passing `--distill-targets charges` raises rather than silently dropping the
target — a run must not be able to claim charge distillation it never did.

**Energy zeros are aligned explicitly and recorded.** Teacher and student are
generally trained on different caches with different `use_energy_bias`, so their
absolute energies differ by a constant plus a per-element atomic reference.
`atomic` least-squares fits per-element offsets, falling back to `scalar` **with
the reason recorded** when the sample cannot support the fit or the fit fails to
beat it. Forces need no correction — a constant plus per-element shift
differentiates to zero — so only the energy channel is corrected.

**Architecture mismatch is a hard error.** The teacher's parameter tree must
match the rebuilt module exactly: any missing, extra, or differently-shaped leaf
aborts the run. A silently half-loaded teacher emits plausible-looking but
meaningless targets. Architecture is taken from `model_attributes` when present
(serialized from the model object itself) and from the run's `config` otherwise,
with CLI aliases (`predict_charges`→`charges`, `n_res`→`n_refinement_blocks`,
`no_zbl`→`zbl`, …). Fields a checkpoint predates fall back to the *model's*
defaults, never the current CLI defaults or the student's values.

One legacy inference is applied, mirroring the warm-start path: a checkpoint
that predates `trainable_zbl` and has ZBL on is rebuilt with
`trainable_zbl=True`, because today's default (`False`) builds no repulsion
parameters and would drop the checkpoint's four.

## Provenance

Every run writes `<workdir>/distillation.json` and embeds the same block under
`distillation` in each checkpoint:

- teacher path, SHA-256, size, parameter-leaf count
- architecture used, its source (`model_attributes` / `config`), and which
  fields fell back to defaults
- whether the CGenFF LJ term was **actually active** (not merely requested)
- teacher's training cache, global step, epoch
- alpha, targets, and the explicit `charges_distilled: false`
- the fitted energy alignment: mode, offsets, sample count, residual RMS before
  and after, and any fallback reason
- the teacher/student architecture differences

## Choosing a teacher

The teacher must be **compatible with the student's cache**, not merely good.
A teacher trained with the CGenFF LJ term and `mol_id` cannot be evaluated
faithfully on a cache lacking them: its LJ prior, learned per-atom vdW-scale head
and LJ scales all go inert, and its electrostatics runs unmasked. The
architecture gate refuses such a teacher outright — see
[q0-distillation-findings.md](q0-distillation-findings.md) for the case that
motivated this.

Check before choosing:

```python
# does the cache carry CGenFF tables and mol_id?
json.load(open(f"{cache}/_METADATA"))["tree_metadata"]  # look for cgenff_*, mol_id
```

## Scripts

- `scripts/slurm/train_spooky_q0_distill_smoke_studix.sbatch` — bounded smoke;
  asserts `distillation.json` was written before exiting 0.
- `scripts/slurm/train_spooky_q0_distill_studix.sbatch` — full run, matched to
  `train_spooky_q0_longer_studix.sbatch` in every respect except the teacher.

Bound smokes with `--steps-per-epoch`, **not** `--max-structures`: the latter is
read only while building a cache, so under `--mode train --cache-path` it
silently does nothing.
