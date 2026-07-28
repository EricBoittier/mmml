# KerNN (legacy PyTorch prototype)

Kernel-descriptor neural network for small molecules. Pairwise distances are
mapped through 1D kernels (default **k33**), standardized, and fed to a Softplus
MLP that predicts energy; forces come from autodiff.

This tree is the original **PyTorch** research dump (H2CO / formaldehyde). For
production use inside MMML, prefer the JAX package:

- Package: [`mmml.models.kernnn`](../../mmml/models/kernnn/)
- CLI: `mmml kernnn-train`, `mmml kernnn-evaluate`
- ASE / scans: `--calculator kernnn --checkpoint …`

## Molecule convention (H2CO / ABCC)

Atom order in coordinates:

| Index | Atom |
|------:|:-----|
| 0 | C |
| 1 | O |
| 2 | H |
| 3 | H |

Six distances (not permutationally symmetrized in the trained path):

| Index | Pair |
|------:|:-----|
| 0 | C–O |
| 1 | C–H1 |
| 2 | C–H2 |
| 3 | O–H1 |
| 4 | O–H2 |
| 5 | H1–H2 |

Forward path:

```text
pos → ABCC distances R → k33(R, min_r) → (k − mean_k)/std_k → FFNet → E
F = −∂E/∂pos
```

`FFNet`: `Linear(6→20) → Softplus` ×3 → `Linear(20→1)`.

## Legacy scripts

Run from this directory (needs local `datasets/`, Torch, optional CUDA):

```bash
# Train: argv = ntrain seed
python train_kernn_gpu.py 3200 42

# Evaluate (hardcoded checkpoint path inside script)
python eval_kernn.py 3200 42

# ASE single-point / optimize / MD / vibrations (hardcoded ckpt paths)
python predict_mol.py …
python optimize.py …
python md_run.py …
python ase_vibrations.py …
```

Torch checkpoints are raw `state_dict` files and **do not** store normalization
stats (`mean_e`, `std_e`, `min_r`, `mean_k`, `std_k`). Those were hardcoded in
`KerNNCalculator/KerNNCalculator.py`. The JAX package bundles stats in the JSON
checkpoint.

## Related files

| Path | Role |
|------|------|
| `utils/distances.py` | ABCC and other molecule distance helpers |
| `utils/kernels.py` | 1D kernel family (`k20`–`k36`); training uses `k33` |
| `utils/neuralnets/FFNet.py` | Softplus MLP used by train/eval/calculator |
| `utils/neuralnets/FFNet_Dual*.py` | Unused dihedral dual-net variants |
| `KerNNCalculator/KerNNCalculator.py` | Torch ASE-like calculator (not a full ASE `Calculator`) |

See [`mmml/models/kernnn/README.md`](../../mmml/models/kernnn/README.md) for the
JAX API, CLI, and scan calculator wiring.
