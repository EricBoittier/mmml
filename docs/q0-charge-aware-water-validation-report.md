# Q⁰ charge-aware water validation and retraining report

Date: 2026-08-02  
System: 732 TIP3 waters, cubic 28 Å periodic box  
Scope: non-Muon Spooky/SO3LR checkpoints

## Executive summary

ML-predicted charges are now genuinely wired into the intermolecular MM
Coulomb Hamiltonian through `mm_charge_mode=q0`. Full-box chunked evaluation,
charge conservation, molecular PBC invariance, and deterministic repetition
have been verified. This corrects the earlier static-evaluation path, which
silently omitted `mm_charge_mode` and therefore continued to use fixed TIP3
charges.

The charge plumbing is correct, but none of the three available charge-head
checkpoints is currently suitable for production dynamics:

- epoch 1 has the lowest forces but chemically inverted water polarity;
- epoch 2 has physical charge signs and strict PBC invariance, but develops a
  large dimer-ML force on the prepared liquid structure;
- epoch 3 has physical charge signs but substantially larger forces.

The failure is not caused by Q⁰ electrostatics. Force decomposition assigns the
dominant contribution to the learned dimer interaction surface. A longer,
lower-learning-rate continuation from epoch 2 has therefore been submitted as
the first retraining control. Teacher distillation should follow in the actual
Spooky trainer using the stable non-charge `step-00294400` model as teacher.

## Correctness changes

### Static evaluator forwarding

`_attach_ase_mmml_calculator` now forwards:

- `mm_charge_mode`;
- `mm_charge_correction`;
- `mm_latent_charge_template`.

Without this change, requesting Q⁰ in the diagnostic script still constructed
a fixed-charge calculator.

### Chunk-safe learned charges

The Spooky liquid requires chunked model application. Previously, every live
ML charge mode raised `NotImplementedError` in that path. Chunked application
now preserves and truncates the per-atom charge auxiliary alongside energies
and forces. Q⁰ then assembles global charges from the isolated-monomer slots
and neutralizes every monomer before MM Coulomb evaluation.

### Molecular wrapping regressions

New tests distinguish valid whole-molecule image changes from invalid atom-wise
wrapping. They cover the NumPy wrapper, ASE production fallback, trajectory
writer, and JAX `PBCMapper`. Molecular translations preserve all internal water
vectors; atom-wise wrapping across a cell face is retained as an explicit
negative control.

## Checkpoint screening

| Checkpoint | O charge (e) | H charge (e) | Raw max force (eV/Å) | Decision |
|---|---:|---:|---:|---|
| charge epoch 1 | +0.0601 | −0.0301 | 4.98 | Reject: inverted polarity |
| charge epoch 2 | −0.1771 | +0.0885 | 9.16 | Best charge checkpoint; retrain |
| charge epoch 3 | about −0.102 | about +0.051 | 16.33 | Reject: excessive force |
| non-charge step 294400 | n/a | n/a | 3.50 | Preferred distillation teacher |

Epoch 1 must not be selected solely by force magnitude: positive oxygen and
negative hydrogen reverse the physical water dipole.

## Epoch-2 Q⁰ full-box proof

For the 732-water static evaluation:

- total box charge: `−7.58 × 10⁻¹⁵ e`;
- maximum absolute per-water charge: `2.78 × 10⁻¹⁷ e`;
- molecular wrapping: `ΔE = 0`, maximum `ΔF = 4.88 × 10⁻¹² eV/Å`;
- lattice translation: `ΔE = 0`, maximum `ΔF = 4.53 × 10⁻¹² eV/Å`;
- repeated base coordinates: maximum `ΔF = 1.67 × 10⁻¹³ eV/Å`;
- atom-wise split-water control: `ΔE = +8842.51 eV`, maximum force
  approximately `1882 eV/Å`.

The split-water result is a representation error by construction. Production
wrapping is molecular and passes at numerical precision.

## Static visual evidence

The validation dashboard uses the repository ICML plot style and house
colormaps. It contains a glossy POV-Ray rendering with signed Q⁰ halos and
exact-force annotations, charge distributions, force distributions, and the
per-atom force change relative to fixed TIP3 charges using the same epoch-2
checkpoint and structure.

Dashboard (generated artifact, not tracked — `artifacts/` is gitignored, so this
is a path to reproduce rather than an embedded image; embedding it breaks
`mkdocs build --strict` for anyone who has the directory locally):

    artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/
      q0_epoch2_charge_validation_dashboard.png

Machine-readable evidence:

- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/charge_data/q0_epoch2_pbc.json`
- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/charge_data/q0_epoch2_pbc.npz`
- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/charge_data/fixed_epoch2_repeat.npz`

## Dynamic safety gate

Studix job `206072` runs epoch 2 with `mm_charge_mode=q0`. It has not entered
NVE and the safety gate has not been bypassed. On the prepared structure the
force decomposition is:

| Term | Maximum force (eV/Å) | Mean force (eV/Å) |
|---|---:|---:|
| internal ML | 2.895 | 1.839 |
| dimer ML | 13.741 | 7.300 |
| MM, including Q⁰ Coulomb | 0.010 | 0.0028 |
| total | 16.308 | 8.229 |

FIRE stage 0 reduced the maximum force only from `15.0173` to `15.0131 eV/Å`
over the first 45 steps. This confirms that changing the MM charge Hamiltonian
does not repair the unstable learned dimer surface.

## Retraining started

### Longer-training control

Studix job `206078` (`q0-longer`) is submitted with separate output
`artifacts/spooky_q0_longer`:

- initialization: charge epoch 2 parameters;
- optimizer state: new AdamW state;
- learning rate: `2.5 × 10⁻⁵`;
- force weight: 75;
- dipole and charge weights: 5 each;
- 50 epochs, two GPUs;
- unchanged 8.9 GB SO3LR data source/cache;
- checkpoint every epoch plus step checkpoints.

At report time it is pending for resources, not failed.

### Retraining smoke

Studix job `206081` is a one-GPU, 512-structure, one-epoch smoke of the same
warm start and loss settings. It is intended to detect checkpoint, cache,
shape, and optimizer failures before the long job consumes substantial GPU
time. At report time it is pending for scheduler priority.

### Distillation recommendation

The current SO3LR/Spooky extxyz trainer has no teacher-loss path. The existing
`physnet-train --distill` option belongs to another training pipeline, so using
it directly would not be a faithful continuation of this experiment.

The correct next implementation is to port energy/force teacher losses into
`scripts/train_so3lr_spooky_extxyz.py` and use:

- student initialization: charge epoch 2;
- teacher: non-charge `artifacts/spooky_so3lr/step-00294400_params.json`;
- ground truth retained as the primary target;
- teacher regularization on energy and forces;
- charge, dipole, neutrality, and water-polarity constraints from the student;
- failure-cluster enrichment as a third experiment after the plain distilled
  control.

Teacher energies require an explicit offset/alignment because the charge and
non-charge models may use different energy zeros. Interaction energies and
forces should receive priority over raw absolute totals.

## Verification

The relevant local test set completed with `76 passed`. Ruff and
`git diff --check` also pass. Tests cover charge-mode semantics, chunked charge
transport, evaluator forwarding, and molecular PBC wrapping.

## Acceptance criteria for a replacement checkpoint

A checkpoint may enter production NVE only if it satisfies all of the following:

1. oxygen-negative/hydrogen-positive water polarity;
2. exact per-water neutrality within `1 × 10⁻⁶ e`;
3. molecular image invariance below `1 × 10⁻⁶ eV/Å`;
4. deterministic repeated evaluation below `1 × 10⁻⁸ eV/Å`;
5. no pathological dimer-scan discontinuities;
6. prepared-box maximum force below the existing size-scaled safety gate;
7. short NVE completion without non-finite coordinates or unsafe energy drift;
8. fixed-vs-Q⁰ force, charge, RDF, and energy plots retained as provenance.

No existing charge checkpoint currently meets the complete list.
