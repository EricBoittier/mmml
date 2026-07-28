# KerNN (JAX)

Kernel-descriptor neural network: pairwise **ABCC** distances → 1D kernels
(**k33**) → Softplus MLP → energy; forces via autodiff.

This package is the JAX/Flax port of the PyTorch prototype in
[`scripts/kernn`](../../../scripts/kernn/). Prefer these APIs and CLIs for new
work.

## Molecule convention (H2CO / ABCC)

Atom order: **C, O, H, H**. Six distances: C–O, C–H1, C–H2, O–H1, O–H2, H1–H2.

## Train / evaluate

```bash
mmml kernnn-train \
  --data datasets/h2co_ccsdt_avtz_4001.npz \
  --ntrain 3200 --seed 42 \
  --workdir artifacts/kernnn

mmml kernnn-evaluate \
  --checkpoint artifacts/kernnn/best.json \
  --data datasets/h2co_ccsdt_avtz_4001.npz \
  --split-json artifacts/kernnn/data_split.json \
  --output-dir artifacts/kernnn/eval
```

Checkpoints are portable JSON with `params`, `config`, and normalization
`stats` (`mean_e`, `std_e`, `min_r`, `mean_k`, `std_k`).

## ASE calculator

```python
from ase.io import read
from mmml.models.kernnn import KerNNCalculator

atoms = read("h2co.xyz")
atoms.calc = KerNNCalculator("artifacts/kernnn/best.json")
print(atoms.get_potential_energy(), atoms.get_forces())
```

## Dimer / IC scans

```bash
mmml dimer-scan --calculator kernnn --checkpoint artifacts/kernnn/best.json ...
mmml ic-scan --calculator kernnn --checkpoint artifacts/kernnn/best.json ...
```

## Importing a Torch state_dict

```python
from mmml.models.kernnn import import_torch_state_dict, H2CO_CALCULATOR_STATS

params, config, stats = import_torch_state_dict(
    "model_ema.pt",
    stats=H2CO_CALCULATOR_STATS,
    out_path="artifacts/kernnn/from_torch.json",
)
```

## Python API

```python
from mmml.models.kernnn import (
    FFNet,
    KerNNConfig,
    energy_and_forces,
    load_checkpoint,
)

params, config, stats, _ = load_checkpoint("artifacts/kernnn/best.json")
energy, forces = energy_and_forces(params, positions, stats, config=config)
```
