# SPICE-α (Zenodo 19205036) → PhysNet NPZ

[SPICE-α](https://doi.org/10.5281/zenodo.19205036) is the MACE-MDP training
release: ~1.8 million ωB97M-D3(BJ)/def2-TZVPPD configurations with energies,
gradients, dipoles, and polarizability tensors, plus IR-R-7193 and R-3B69
spectroscopy benches. License: **CC-BY-4.0**.

Paper: Gönnheimer, Reuter, Kapil, Margraf, ChemRxiv
[10.26434/chemrxiv.15000716](https://doi.org/10.26434/chemrxiv.15000716).
Also cite Eastman et al., SPICE, *Sci. Data* 2023 / *J. Chem. Theory Comput.*
2024.

This page is the extract → convert → `physnet-train` recipe. NPZ keys and
units: [Training NPZ contract](training-npz-contract.md).

## What fits MMML

| Goal | Path |
|---|---|
| Organic E/F/(D) potential | **`physnet-train`** on padded NPZ (this page) |
| Hybrid ML/MM dimer PES | DES370K dimers + `prepare-mm-dataset` + `--hybrid-mm` |
| PET-MAD / metatomic | Inference or `pet-physnet-distill` **teacher labels** — not training on this DFT |
| MACE-MDP dipole/polar clone | ACEsuit `AtomicDielectricMACE` ([upstream `train.sh`](https://github.com/Nilsgoe/MACE-MDP/blob/main/reproducibility/train.sh)) — not an MMML trainer |
| IR/Raman CSVs in the zip | Evaluation only |

Best MMML-shaped subset: **DES370K monomers + dimers** (3–34 atoms, isolated).
PubChem is the large isolated-molecule pool. Solvated PubChem / water clusters /
AA–ligand frames are **droplets**, not periodic boxes — do not pass `--use-pbc`.

## Download

One file, 12.46 GB (`md5:afcba5a263030b61deaeb79c660c2efd`):

```bash
curl -L -o SPICE-alpha.zip \
  https://zenodo.org/api/records/19205036/files/SPICE-alpha.zip/content
md5sum SPICE-alpha.zip
unzip SPICE-alpha.zip \
  'SPICE-alpha/README.md' \
  'IR-R-7193/datbase_IR-R-7193_wB97MD3.xyz' \
  'R-3B69/R-3B69.xyz'
unzip SPICE-alpha.zip 'SPICE-alpha/SPICE-alpha.tar.gz'
tar -xzf SPICE-alpha/SPICE-alpha.tar.gz --transform='s|^\./||' \
  DES370K_Dimers.hdf5 DES370K_Monomers.hdf5
```

Keep ~25 GB free if you also unpack PubChem. The tarball’s first member is
`./DES370K_Dimers.hdf5`. Bundled `extract_spice_alpha.py` dumps JSON — do not
use it at full scale.

HDF5 layout (from the record README): one group per molecule; `atomic_numbers`
`(N,)`; `conformations` `(M, N, 3)` **Å**; `dft_total_energy` `(M,)` **eV**;
`dft_total_gradient` `(M, N, 3)` **eV/Å**; `scf_dipole` `(M, 3)` **e·Å**;
`polarizability` `(M, 3, 3)` e Å²/V. These are already MMML **train** units
except the gradient sign.

`read_h5.py` looks for `mol_*` + `positions` / `total_forces` and will load
**nothing**. Paper Table 1 used neutrals only (1.66 M of 1.82 M) with a 90/5/5
split **per subset** — those index files are not in the zip.

## Convert

`F = −dft_total_gradient`. Then split **without** reconverting:

```bash
# after writing spice_des_mono.npz with R,Z,N,E,F,D in Å/eV/eV/Å/e·Å
mmml fix-and-split --efd spice_des_mono.npz -o splits_des_mono --preserve-units \
  --train-frac 0.9 --valid-frac 0.05 --test-frac 0.05
```

Default `fix-and-split` assumes Hartree / Hartree/Bohr / Debye and will
**destroy** this dataset.

Sketch (stream HDF5 → padded NPZ):

```python
import json
from pathlib import Path
import h5py
import numpy as np

def iter_frames(h5):
    for name in h5.keys():
        g = h5[name]
        if "conformations" not in g or "atomic_numbers" not in g:
            continue
        Z = np.asarray(g["atomic_numbers"][()], dtype=np.int32)
        R = np.asarray(g["conformations"][()], dtype=np.float64)
        n = int(R.shape[0])
        E = np.asarray(g["dft_total_energy"][()], dtype=np.float64)
        G = np.asarray(g["dft_total_gradient"][()], dtype=np.float64)
        D = np.asarray(g["scf_dipole"][()], dtype=np.float64) if "scf_dipole" in g else np.full((n, 3), np.nan)
        for i in range(n):
            yield Z, R[i], float(E[i]), -G[i], D[i]

def to_npz(paths, out: Path, pad: int | None = None, max_frames: int = 0):
    frames = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            for fr in iter_frames(h5):
                frames.append(fr)
                if max_frames and len(frames) >= max_frames:
                    break
        if max_frames and len(frames) >= max_frames:
            break
    pad = pad or max(len(z) for z, *_ in frames)
    n = len(frames)
    R = np.zeros((n, pad, 3)); F = np.zeros((n, pad, 3))
    Z = np.zeros((n, pad), np.int32); N = np.zeros((n,), np.int32)
    E = np.zeros((n,)); D = np.zeros((n, 3))
    for i, (z, r, e, f, d) in enumerate(frames):
        na = len(z)
        Z[i, :na] = z; R[i, :na] = r; F[i, :na] = f
        N[i] = na; E[i] = e; D[i] = d
    units = dict(R="angstrom", E="ev", F="ev_angstrom", D="e_angstrom",
                 force="negated dft_total_gradient", source="zenodo-19205036")
    np.savez_compressed(out, R=R, Z=Z, N=N, E=E, F=F, D=D,
                        _mmml_units=np.array(json.dumps(units)))
```

Inspect `units_map` on the first HDF5 group before converting. If you see
Bohr/Hartree, treat it as original SPICE and use default `fix-and-split`
(plus `--flip-forces`).

XYZ benches: IR-R-7193 has `dipole_eAA` and `polarizability_eAA2_per_V`,
`pbc="F F F"`. R-3B69 also has `energy_eV` and per-atom **forces** (do not
flip). Useful for dipole/spectra checks, not a hybrid PES.

## Train

Iodine is Z=53. Example yaml uses `max_atomic_number: 35` (Br). Total
energies need `--subtract-atom-energies`.

```yaml
# physnet-spice-alpha-des-mono.yaml
data: splits_des_mono/energies_forces_dipoles_train.npz
valid_data: splits_des_mono/energies_forces_dipoles_valid.npz
ckpt_dir: ./ckpts/spice_des_mono
tag: spice_des_mono
subtract_atom_energies: true
max_atomic_number: 53
use_pbc: false
dipole_weight: 27.21
conversion:
  energy: 23.060549
  forces: 23.060549
```

```bash
mmml physnet-train --config physnet-spice-alpha-des-mono.yaml
mmml physnet-evaluate --checkpoint ./ckpts/spice_des_mono \
  --data splits_des_mono/energies_forces_dipoles_valid.npz --plots
```

DES dimers as hybrid: convert dimers (pad ≥ 34),
`mmml prepare-mm-dataset -i dimers.npz -o dimers_mm.npz`, then
`physnet-train --hybrid-mm`. SPICE-α does **not** ship `E_int` or monomer
pairing — keep total E or build interaction labels yourself.

Pass: manifest still eV; force MAE finite; energy MAE not stuck at hundreds
of eV (forgot atom refs or double-converted). Fail: dipole MAE ~0.3 on e·Å
targets (Debye left in `D` — should not happen if `scf_dipole` was used).

## Blockers

| Issue | What to do |
|---|---|
| 12.5 GB zip / 1.8 M frames | Extract DES first; `max_frames` smoke |
| Gradient vs force | `F = −G` |
| No interaction labels | Total-E + atom refs, or pair DES monomers |
| No CGenFF | `prepare-mm-dataset` on dimers only |
| No PBC | `--no-pbc` |
| `N` = 3–110 | Pad per subset (22 / 34 / 50 / 110) |
| Charged systems | Filter; paper trained neutrals |
| Polarizability | Unused by PhysNet (`polar` is optional storage) |
