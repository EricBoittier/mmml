# Handoff: charge-aware hybrid ML/MM water model

Updated: 2026-08-02

## Primary goal

Produce a stable, charge-aware Spooky/SO3LR hybrid ML/MM model for condensed
phase simulation in which ML-predicted charges are genuinely used by the
intermolecular MM Coulomb Hamiltonian.

The production candidate must use `mm_charge_mode=q0`: isolated-monomer Q⁰
charges, projected to exact neutrality per molecule. Fixed TIP3 charges remain
a comparison Hamiltonian, not the final charge-aware result.

## Non-negotiable constraints

- Do not use or evaluate Muon checkpoints. The user explicitly excluded them.
- Do not replace the existing fixed-charge or non-charge workflows. Q⁰ is an
  additional demonstrator/production candidate.
- Never bypass the post-minimization force gate or NVE safety checks.
- Keep atom-wise wrapping only as a negative test. Production PBC wrapping must
  translate complete molecules and preserve internal geometry.
- Confirm a checkpoint's identity and architecture before using it.
- Reject chemically inverted water charges even if their force metrics look
  better.
- Preserve existing user changes and unrelated dirty-worktree files.
- Every claimed result needs machine-readable evidence and appropriate plots;
  charge/force results should include a glossy POV-Ray view where useful.
- Follow `docs/plotting-style-guide.md`, including house colormaps, transparent
  exports where appropriate, readable legends, and exact-frame force arrows.

## Agreed scientific goals

1. Train longer from the best physically acceptable charge checkpoint using a
   smaller learning rate and stronger force supervision.
2. Implement teacher distillation in the actual SO3LR/Spooky extxyz trainer.
3. Use the stable non-charge `step-00294400` checkpoint as the teacher.
4. Initialize the charge-aware student from charge epoch 2.
5. Retain ground-truth RIMP2 energy/force labels as the primary objective;
   teacher losses regularize the student rather than replacing reference data.
6. Distil interaction energies and forces, especially the dimer contribution,
   rather than relying only on absolute total energy.
7. Add charge, dipole, exact-neutrality, polarity, and smoothness constraints.
8. Build a third training experiment enriched with failure clusters selected
   from the high-force condensed-phase structures.
9. Compare three experiments:
   - longer supervised continuation;
   - teacher-distilled charge student;
   - distilled student plus failure-cluster/active-learning enrichment.
10. Validate every saved candidate using static full-box tests, dimer scans,
    PBC image invariance, force tails, and a short guarded NVE run.

## Current checkpoint decisions

| Checkpoint | Result | Decision |
|---|---|---|
| `artifacts/spooky_so3lr_charges/epoch-0001` | max force 4.98 eV/Å, but O positive and H negative | Reject: inverted polarity |
| `artifacts/spooky_so3lr_charges/epoch-0002` | O about −0.177 e, H about +0.0885 e; static max force 9.16 eV/Å | Student initialization / retraining base |
| `artifacts/spooky_so3lr_charges/epoch-0003` | physical signs, max force about 16.33 eV/Å | Reject: excessive force |
| `artifacts/spooky_so3lr/step-00294400_params.json` | non-charge model, max force about 3.50 eV/Å | Preferred distillation teacher |

Do not select epoch 1 based only on its lower forces. Its learned water dipole
is reversed.

## What has been implemented

### Genuine Q⁰ MM electrostatics

The static ASE evaluator now forwards `mm_charge_mode`,
`mm_charge_correction`, and `mm_latent_charge_template`. Previously the
diagnostic silently constructed a fixed-charge calculator.

Chunked model application now propagates the per-atom charge auxiliary along
with energies and forces. This is required for the 732-water box. Q⁰ charges
are assembled from isolated-monomer slots and projected to exact neutrality
before being supplied to MM Coulomb.

### Molecular PBC tests

Regression tests cover:

- molecular geometry preservation;
- atom-wise wrapping as an explicit failure control;
- ASE production wrapping;
- trajectory-output wrapping;
- JAX `PBCMapper` image invariance.

Relevant test file:
`tests/unit/test_molecular_pbc_wrapping.py`.

### Full-box epoch-2 proof

For 732 waters in a 28 Å box with Q⁰:

- total charge: `−7.58 × 10⁻¹⁵ e`;
- maximum absolute per-water charge: `2.78 × 10⁻¹⁷ e`;
- molecular-wrap maximum force difference: `4.88 × 10⁻¹² eV/Å`;
- repeated-base maximum force difference: `1.67 × 10⁻¹³ eV/Å`;
- atom-wise split-water control: `ΔE = +8842.51 eV`, maximum force about
  `1882 eV/Å`.

Evidence:

- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/charge_data/q0_epoch2_pbc.json`
- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/charge_data/q0_epoch2_pbc.npz`
- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/charge_data/fixed_epoch2_repeat.npz`
- `artifacts/npt_argon_water/checkpoint_pes_compare_20260802/povray_snapshots/q0_epoch2_charge_validation_dashboard.png`

## Root cause of the present dynamics failure

The Q⁰ electrostatics implementation is not the source of the unsafe force.
On the prepared epoch-2 structure, the decomposition was:

| Term | Maximum force (eV/Å) | Mean force (eV/Å) |
|---|---:|---:|
| internal ML | 2.895 | 1.839 |
| dimer ML | 13.741 | 7.300 |
| MM including Q⁰ Coulomb | 0.010 | 0.0028 |
| total | 16.308 | 8.229 |

The learned dimer interaction surface dominates. FIRE minimization reduced the
maximum force only very slowly, from roughly 15.0173 to 15.0131 eV/Å over the
first 45 steps. Do not relax the safety threshold to make this checkpoint run.

## Active Studix jobs at handoff

Use `ssh -F /dev/null boittier@pc-studix.chemie.unibas.ch` because the local SSH
config points at a missing RSA key. Repository path is
`/mmhome/boittier/home/mmml`.

| Job | Name | Purpose | State at last check |
|---|---|---|---|
| `206072` | `tip3-q0` | Epoch-2 Q⁰ minimization and guarded NVE | Running in minimization; had not entered NVE |
| `206078` | `q0-longer` | 50-epoch, two-GPU longer-training control | Pending for resources |
| `206081` | `q0-retrain-smoke` | One-GPU, 512-structure, one-epoch retraining smoke | Pending for priority |

Do not interpret queue disappearance as success. Confirm with `sacct`, the
status TSV, logs, and non-empty checkpoint/trajectory artifacts.

## Longer-training configuration

Script: `scripts/slurm/train_spooky_q0_longer_studix.sbatch`

- initialization: charge epoch 2 parameters;
- fresh AdamW optimizer;
- learning rate `2.5 × 10⁻⁵`;
- force weight 75;
- dipole weight 5;
- charge weight 5;
- 50 epochs on two GPUs;
- existing 8.9 GB SO3LR dataset/cache;
- separate output: `artifacts/spooky_q0_longer`.

Smoke script: `scripts/slurm/train_spooky_q0_retrain_smoke_studix.sbatch`.

The smoke must complete and write a real epoch checkpoint before the full run
is considered validated. Watch for fast failures from cache restore,
architecture matching, device count, or auto-batching.

## Distillation implementation required

The current SO3LR training entry point is
`scripts/train_so3lr_spooky_extxyz.py`. It does not yet implement teacher
distillation. The existing `mmml physnet-train --distill` path is a separate
generic PhysNet pipeline and must not be presented as equivalent.

Port the tested loss helpers from
`mmml/models/physnetjax/physnetjax/training/distill.py` into the Spooky trainer.
Required behavior:

1. accept a separate `--teacher-checkpoint`;
2. load the non-charge teacher without overwriting student architecture;
3. run the teacher without gradients;
4. support energy and force teacher targets independently;
5. align teacher/student energy zeros explicitly and record the offset;
6. blend ground truth and teacher terms with a recorded alpha;
7. leave charge/dipole losses supervised only by student/reference data;
8. serialize teacher provenance and all loss weights in checkpoint metadata;
9. add unit tests proving alpha endpoints and teacher-gradient blocking;
10. run a small smoke before submitting a full distilled job.

A reasonable first experiment is ground-truth alpha 0.75 for energy and force
distillation. Do not distil charges from the teacher because the selected
teacher has no charge head.

## Failure-cluster enrichment

After the plain distilled smoke succeeds:

1. identify waters/neighbor clusters associated with the largest dimer-ML
   forces in the prepared liquid;
2. extract intact local clusters using molecular PBC handling;
3. remove duplicates/diversify with cheap RDF/SOAP descriptors;
4. add RIMP2 labels where affordable and teacher labels elsewhere;
5. emphasize compressed contacts, hydrogen-bond rearrangements, distorted
   monomers, and ML/MM handoff geometries;
6. preserve provenance and source indices;
7. compare learnability and gzip compression against random selections.

## Checkpoint acceptance gate

A replacement checkpoint must satisfy all of the following before production:

1. oxygen-negative/hydrogen-positive water polarity;
2. per-water neutrality within `1 × 10⁻⁶ e`;
3. molecular PBC image invariance below `1 × 10⁻⁶ eV/Å`;
4. repeated evaluation below `1 × 10⁻⁸ eV/Å`;
5. no obvious dimer-scan discontinuities or pathological short-range wells;
6. prepared-box maximum force below the existing size-scaled gate;
7. successful short NVE without non-finite coordinates or unsafe drift;
8. charge, force, RDF, energy, and PBC diagnostic plots saved;
9. glossy charge-aware POV-Ray evidence using exact-frame forces;
10. machine-readable JSON/NPZ provenance for every reported number.

## Verification completed

The relevant local suite passed `76/76` tests. Ruff and `git diff --check` also
passed at the end of implementation.

Detailed companion report:
`docs/q0-charge-aware-water-validation-report.md`.
