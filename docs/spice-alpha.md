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
  ./DES370K_Dimers.hdf5 ./DES370K_Monomers.hdf5
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

`F = −dft_total_gradient`. Use the importable converter
(`mmml.data.spice_alpha`); then split **without** reconverting:

```bash
python -m mmml.data.spice_alpha DES370K_Monomers.hdf5 -o spice_des_mono.npz
# optional: --max-frames 64 --neutral-only --pad 22
mmml fix-and-split --efd spice_des_mono.npz -o splits_des_mono --preserve-units \
  --train-frac 0.9 --valid-frac 0.05 --test-frac 0.05
```

```python
from mmml.data import convert_spice_alpha_hdf5

convert_spice_alpha_hdf5(["DES370K_Monomers.hdf5"], "spice_des_mono.npz")
```

Default `fix-and-split` assumes Hartree / Hartree/Bohr / Debye and will
**destroy** this dataset. The converter refuses a Bohr/Hartree `units_map`
unless you pass `--allow-atomic-units` (then use default `fix-and-split`
plus `--flip-forces` instead). Units metadata is resolved as:

- non-empty file-level `units_map` wins, unless the first molecule group
  also has a known map of a different kind (canonical vs atomic) — that
  contradiction is an error
- empty file-level `units_map` (published DES370K) falls through to the
  **first molecule group** only (no full-file scan)
- missing or empty at both levels: convert assumes the SPICE-α README
  units (Å / eV / eV/Å) and writes `_mmml_units` from that
- a non-empty string that is not a JSON object is malformed and is an
  error, not a silent eV/Å fallback

Synthetic contract tests (no Zenodo download): `pytest -m data_loading`
or `make test-data-loading`. CI: `.github/workflows/data-loading.yml`.

XYZ benches: IR-R-7193 has `dipole_eAA` and `polarizability_eAA2_per_V`,
`pbc="F F F"`. R-3B69 also has `energy_eV` and per-atom **forces** (do not
flip). Useful for dipole/spectra checks, not a hybrid PES.

## Efield + polarizability

The efield model already predicts `α = dμ/dEf`
(`mmml.models.efield.model_functions.dipole_derivative_field_batched`).
SPICE-α is **zero-field** DFT, so train at `Ef = 0` and add a polar
regularizer. Convert polar to Bohr³ first.

On the machine that already unzipped `SPICE-alpha.zip`:

```bash
# inner HDF5 (DES only)
tar -xzf SPICE-alpha/SPICE-alpha.tar.gz --transform='s|^\./||' -C SPICE-alpha \
  ./DES370K_Monomers.hdf5 ./DES370K_Dimers.hdf5

# smoke (256 frames) or drop --max-frames for the full monomer set
python -m mmml.data.spice_alpha SPICE-alpha/DES370K_Monomers.hdf5 \
  -o spice_des_mono.npz --efield --polar-units bohr3 --neutral-only \
  --split-dir splits_des_mono --max-frames 256

mmml efield-train \
  --train-npz splits_des_mono/energies_forces_dipoles_train.npz \
  --valid-npz splits_des_mono/energies_forces_dipoles_valid.npz \
  --output-dir ./ckpts/spice_ef_polar \
  --energy_weight 1 --forces_weight 100 --dipole_weight 0.1 \
  --polar_weight 1 --polar-at-zero-field \
  --num_epochs 100 --batch_size 64 --features 32 --max_degree 2
```

Wrappers: `scripts/spice_alpha/prepare_efield_dataset.sh` and
`train_efield_polar.sh`. Iodine is Z=53; the efield model max Z is 55.

### Cluster (login12 / SciCORE)

Do **not** `source scripts/scicore_env.sh` for this train. That prolog sets
`JAX_ENABLE_X64=1`; e3x `Embed` stays float32 and `MessagePass` promotes, so
`EFieldPhysNet.init` dies in `e3x.nn.add`. The train wrapper and sbatch force
`JAX_ENABLE_X64=0`. No CHARMM, no Zenodo pull.

```bash
# 0. This branch (until merged)
cd "$HOME/mmml"
git fetch origin cursor/spice-alpha-training-docs-f8f6
git checkout cursor/spice-alpha-training-docs-f8f6
git pull origin cursor/spice-alpha-training-docs-f8f6

# 1. Inner HDF5 + 256-frame smoke NPZ (CPU / login is fine)
scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha ~/data/spicealpha/mmml_efield 256
python scripts/spice_alpha/check_efield_npz.py \
  ~/data/spicealpha/mmml_efield/splits_des_mono/energies_forces_dipoles_{train,valid}.npz

# 2. GPU smoke (2 epochs, B=8, features=16, max_degree=1)
mkdir -p artifacts/spice_ef_polar
sbatch scripts/spice_alpha/train_efield_polar.sbatch

# 3. After smoke writes params-*.json and logs "polar mae"
# 256-frame extract has valid n=13; B=64 drop_last → 0 valid batches
sbatch --partition=rtx4090 --qos=rtx4090-6hours --time=06:00:00 \
  --export=ALL,MODE=full,EPOCHS=100,BATCH_SIZE=8 \
  scripts/spice_alpha/train_efield_polar.sbatch

# 4. Full DES370K monomers (new tree). SKIP_DIMERS=1. MODE=full uses
# POLAR_WEIGHT=100 so polar can compete with total |E| ~1e5 eV.
SKIP_DIMERS=1 scripts/spice_alpha/prepare_efield_dataset.sh \
  ~/data/spicealpha ~/data/spicealpha/mmml_efield_full 0
# or: sbatch scripts/spice_alpha/prepare_efield_dataset.sbatch
python scripts/spice_alpha/check_efield_npz.py \
  ~/data/spicealpha/mmml_efield_full/splits_des_mono/energies_forces_dipoles_train.npz \
  ~/data/spicealpha/mmml_efield_full/splits_des_mono/energies_forces_dipoles_valid.npz
# pad=22 polar JVP: B=16/64 failed XLA autotune. MODE=big default B=4.
sbatch --partition=rtx4090 --qos=rtx4090-6hours --time=06:00:00 \
  --export=ALL,MODE=big,EPOCHS=100,BATCH_SIZE=4,SPLITS=$HOME/data/spicealpha/mmml_efield_full/splits_des_mono,CKPT=$HOME/mmml/ckpts/spice_ef_polar_big \
  scripts/spice_alpha/train_efield_polar.sbatch
```

Pass: check script exits 0; smoke log has `polar mae` / `polar MSE` (finite);
`ckpts/spice_ef_polar/params-*.json` or `orbax/params-*` exists. Epoch-1 energy
MAE of tens–hundreds of eV is expected (zero-init heads, total E, no atom
refs). Fail: `e3x.nn.add` dtype error (x64 still on); `polar_weight is set
but NPZ has no 'polar'`; energy stuck after many epochs at hundreds of eV
(default `fix-and-split` double-converted eV→eV). OOM: `BATCH_SIZE=4
GRADIENT_CHECKPOINT=1`.

A missing `params-best.json` **during** the run does not mean valid loss
never improved. Audit Slurm + the ckpt dir (reads `history.jsonl` /
`best-valid-*.json` / sacct):

```bash
python scripts/spice_alpha/audit_efield_job.py \
  --ckpt $HOME/mmml/ckpts/spice_ef_polar_big \
  --log artifacts/spice_ef_polar/slurm-22826285.out \
  --job 22826285
```

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
| Polarizability | PhysNet ignores `polar`. Efield-train: `--polar_weight` + `--polar-units bohr3` + `Ef=0` |
| `JAX_ENABLE_X64=1` | Breaks `EFieldPhysNet.init`. Use the train wrapper / sbatch (`X64=0`) |
