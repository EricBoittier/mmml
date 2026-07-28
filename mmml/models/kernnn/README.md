# KerNN (JAX)

Kernel-descriptor neural network: pairwise **ABCC** or **ABCC_sym** distances →
1D kernels (**k33**) → Softplus MLP → energy; forces via autodiff.

Optional **DualFFNet** adds an H–C–O–H dihedral branch.

## Train / evaluate

```bash
mmml kernnn-train \
  --data datasets/h2co_ccsdt_avtz_4001.npz \
  --ntrain 3200 --seed 42 \
  --workdir artifacts/kernnn

# Optional: permutationally symmetrized distances and/or dual architecture
mmml kernnn-train --distance-scheme abcc_sym --architecture dual ...

mmml kernnn-evaluate \
  --checkpoint artifacts/kernnn/best.json \
  --data datasets/h2co_ccsdt_avtz_4001.npz \
  --split-json artifacts/kernnn/data_split.json \
  --output-dir artifacts/kernnn/eval
```

## ASE / scans / NEB / umbrella / DMC

```bash
mmml dimer-scan --calculator kernnn --checkpoint artifacts/kernnn/best.json ...
mmml ic-scan --calculator kernnn --checkpoint artifacts/kernnn/best.json ...
mmml neb --calculator kernnn --checkpoint artifacts/kernnn/best.json \
  --initial h2co_a.xyz --final h2co_b.xyz --output-dir artifacts/neb
mmml umbrella-sample --model kernnn --checkpoint artifacts/kernnn/best.json ...
mmml dmc --model kernnn --natm 4 --nwalker 64 --stepsize 5e-4 \
  --nstep 200 --eqstep 50 --alpha 1200.0 \
  --checkpoint artifacts/kernnn/best.json --input h2co.extxyz
```

## Hybrid MLpot / md-system

Pass a KerNN JSON checkpoint to `setup_calculator` / `md-system`. Checkpoints with
`config.model_type == "kernnn"` (and bundled `stats`) are auto-detected; monomers
must be 4-atom ABCC (e.g. H₂CO liquid). Dimers are evaluated as the sum of two
independent KerNN monomers.

## Python API

```python
from mmml.models.kernnn import (
    KerNNCalculator,
    DualFFNet,
    energy_and_forces,
    load_checkpoint,
)

params, config, stats, _ = load_checkpoint("artifacts/kernnn/best.json")
energy, forces = energy_and_forces(params, positions, stats, config=config)
```
