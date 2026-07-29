# KerNN (JAX)

Kernel-descriptor neural network: pairwise distances → 1D kernels (**k33**) →
Softplus MLP → energy; forces via autodiff.

| `--distance-scheme` | Atoms | Features |
|---------------------|------:|---------:|
| `abcc` / `abcc_sym` | 4 (H₂CO) | 6 / 7 |
| `form` | 6 (formamide) | 15 |
| `acem` | 9 (acetamide) | 36 |

Optional **DualFFNet** (`--architecture dual`) is ABCC-only (H–C–O–H dihedral).

## Evaluate an existing PhysNet teacher

```bash
mmml physnet-evaluate \
  --checkpoint ckpts/run/params_acem1_....json \
  --data splits/acem/energies_forces_dipoles_test.npz \
  -o artifacts/physnet_acem_eval \
  --plots
```

## Train KerNN on ACEM / FORM

```bash
# ACEM from prepared splits (recommended)
mmml kernnn-train \
  --distance-scheme acem \
  --train-npz splits/acem/energies_forces_dipoles_train.npz \
  --valid-npz splits/acem/energies_forces_dipoles_valid.npz \
  --test-npz  splits/acem/energies_forces_dipoles_test.npz \
  --n-hidden 64 --batch-size 64 --epochs 500 \
  --workdir artifacts/kernnn/acem_gt

mmml kernnn-evaluate \
  --checkpoint artifacts/kernnn/acem_gt/best.json \
  --data splits/acem/energies_forces_dipoles_test.npz \
  --split all \
  --output-dir artifacts/kernnn/acem_gt/eval_test

# FORM from a single full NPZ (KerNN random-splits)
mmml kernnn-train \
  --distance-scheme form \
  --data form_mp2_aug-cc-pvtz_4000.npz \
  --ntrain 3200 --nvalid 400 --seed 42 \
  --workdir artifacts/kernnn/form_gt
```

Copy-paste workflows: [`examples/kernnn/train_acem_form.sh`](../../../examples/kernnn/train_acem_form.sh),
[`examples/kernnn/smoke_acem.sh`](../../../examples/kernnn/smoke_acem.sh).

## Distill from a PhysNet teacher

```bash
mmml kernnn-train \
  --distance-scheme acem \
  --train-npz splits/acem/energies_forces_dipoles_train.npz \
  --valid-npz splits/acem/energies_forces_dipoles_valid.npz \
  --teacher-checkpoint ckpts/run/params_acem1_....json \
  --distill-alpha 0.5 \
  --workdir artifacts/kernnn/acem_distill
```

Loss is `α · MSE(GT) + (1−α) · MSE(teacher)` on energies and forces
(`α=1` pure GT, `α=0` pure teacher). NPZs must include `Z` for the teacher.

**Energy zero:** PhysNet teachers often use atomization references, so raw teacher
energies can sit near ~0 while the NPZ mean is ~10² eV. Training auto-fits an
additive teacher energy offset (`mean(E_GT − E_teacher)`) unless you pass
`--no-align-teacher-energy` or an explicit `--teacher-energy-offset` (eV).
Forces are unchanged by that constant. The optimizer minimizes MSE; epoch logs
print **RMSE** in eV and eV/Å.

## ASE / scans / NEB / umbrella / DMC

```bash
mmml dimer-scan --calculator kernnn --checkpoint artifacts/kernnn/best.json ...
mmml neb --calculator kernnn --checkpoint ... --initial a.xyz --final b.xyz ...
mmml umbrella-sample --model kernnn --checkpoint ...
mmml dmc --model kernnn --natm 9 --checkpoint artifacts/kernnn/acem_gt/best.json ...
```

## Hybrid MLpot / md-system

Pass a KerNN JSON checkpoint (`model_type: kernnn`). Monomer size must match
`config.n_atoms` (4 / 6 / 9 for abcc / form / acem).
