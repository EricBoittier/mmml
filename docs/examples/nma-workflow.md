# NMA end-to-end: residue → methyl scan → dataset → train → eval → dimer → MD

Worked example for **N-methylacetamide (CGenFF `NMA`)** — a small amide with two
chemically distinct methyl rotors. This page ties the pieces together:

| Stage | Tool |
|-------|------|
| Build residue | `mmml make-res --res NMA` |
| Methyl rotation scan | `mmml ic-scan` |
| QM labels | `pyscf-dft` / `pyscf-evaluate` / `xml2npz` |
| Splits | `mmml fix-and-split` |
| Train | `mmml physnet-train`, `mmml train-joint` (PhysNet + DCMNet) |
| Evaluate | `mmml physnet-evaluate` |
| Intermolecular PES | `mmml dimer-scan` (1D) + 2D/hybrid scripts |
| Dynamics | `mmml liquid-box`, `mmml md-system` |

Related: [ic-scan design](../ic-scan-design.md), [dimer-scan design](../dimer-scan-design.md),
[hybrid MM charges](../hybrid-mm-charges.md), [DCM/ACO dimer scans](../functionality/dimer_scans/README.md),
[liquid-box workflow](../liquid-box-workflow.md), sibling [CYBZ tutorial](https://github.com/EricBoittier/mmml_tutorial).

---

## 0. Working directory

```bash
mkdir -p ~/test && cd ~/test
# Activate your mmml env (uv / micromamba) so `mmml` is on PATH.
```

Outputs from `make-res` land under `pdb/`, `psf/`, `xyz/`, `dcd/`, `res/` relative
to the current working directory.

---

## 1. Build the residue

```bash
mmml make-res --res NMA
# optional on clusters:
# mmml make-res --res NMA --skip-energy-show
```

| Artifact | Role |
|----------|------|
| `xyz/nma.xyz` / `xyz/initial.xyz` | ASE / ic-scan geometry |
| `pdb/nma.pdb` | VMD / Packmol |
| `psf/nma-1.psf` | CHARMM topology + **atom order** |

CGenFF atom names (PSF / 0-based ASE index):

| Index | Name | Role |
|------:|------|------|
| 0 | `CL` | Acetyl methyl C |
| 1–3 | `HL1`–`HL3` | Acetyl methyl H |
| 4 | `C` | Carbonyl C |
| 5 | `O` | Carbonyl O |
| 6 | `N` | Amide N |
| 7 | `H` | Amide H |
| 8 | `CR` | N-methyl C |
| 9–11 | `HR1`–`HR3` | N-methyl H |

Check in VMD:

```bash
vmd xyz/nma.xyz
# or with topology: vmd psf/nma-1.psf pdb/nma.pdb
```

---

## 2. Internal-coordinate scans (`ic-scan`)

Full mask / atom-order rules: [ic-scan design](../ic-scan-design.md).

ASE rotates about **a2–a3**; **a4 must be on the a3 side** and in `mask` (or
omit `mask` for the covalent-topology default).

| Rotor | Dihedral | `atoms` (0-based) |
|-------|----------|-------------------|
| Amide C–C–N–C | `CL–C–N–CR` | `[0, 4, 6, 8]` |
| Acetyl methyl | `N–C–CL–HL1` | `[6, 4, 0, 1]` |
| N-methyl | `C–N–CR–HR1` | `[4, 6, 8, 9]` |

**Broken pattern for ω:** `atoms: [0,4,6,8]` with `mask: [9,10,11]` (HR* only —
a4=`CR` never moves). Omit `mask` or include CR (and the N-methyl fragment).

Bundled examples:

- [`examples/ic_scan/nma_methyl.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/ic_scan/nma_methyl.yaml) — methyl 1D
- [`examples/ic_scan/nma_omega_methyl_2d.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/ic_scan/nma_omega_methyl_2d.yaml) — ω + N-methyl 1D/2D

### Prepare geometries only (for external QM)

```bash
mmml ic-scan --config examples/ic_scan/nma_omega_methyl_2d.yaml \
  --prepare-only --output artifacts/nma_omega_2d --overwrite
```

Writes `trajectory.extxyz` / `.traj`, `data.csv`, `manifest.json` with
`status=prepared` (no energies).

### Evaluate in-process (xTB smoke)

```bash
mmml ic-scan --config examples/ic_scan/nma_methyl.yaml \
  --output artifacts/nma_methyl_xtb --overwrite
```

From a local `~/test` tree after `make-res`:

```bash
mmml ic-scan --config ic_scan/nma_methyl.yaml \
  --output ic_scan/methyl_xtb --overwrite
```

Expect ~0.5 kcal/mol (acetyl) and ~1 kcal/mol (N-methyl) barriers at GFN2-xTB —
a smoke check, not a production reference.

Swap `calculator: physnet` + `checkpoint:` (or `pyscf`) for ML/QM single-points
on the same rigid grid.

---

## 3. Prepare a training dataset

Methyl scans alone are not enough for a robust potential. Typical MMML path
(same as the CYBZ tutorial):

### 3a. Sample geometries

```bash
# Hessian / normal-mode sampling around the make-res minimum
mmml pyscf-dft --mol xyz/nma.xyz --energy --gradient --hessian --harmonic
mmml normal-mode-sample -i out/results.h5 -o sampled.npz --max-samples 1000
```

Or fold in ic-scan frames (convert extxyz → NPZ with your preferred script /
ASE loop), MD snapshots, or Molpro XML via `mmml xml2npz`.

### 3b. Label with QM (E / F / dipole / ESP)

```bash
mmml pyscf-evaluate -i sampled.npz -o evaluated.npz --esp
# or Molpro → NPZ:
# mmml xml2npz molpro_outputs/ -o evaluated.npz --recursive --validate
```

### 3c. Units + train/valid/test splits

```bash
mmml fix-and-split --efd evaluated.npz --output-dir splits/
# With ESP grids for joint / DCMNet training:
# mmml fix-and-split --efd evaluated.npz --grid grids_esp.npz --output-dir splits/
```

Produces `energies_forces_dipoles_{train,valid,test}.npz` and matching
`grids_esp_*.npz` when grids are supplied.

Validate:

```bash
mmml validate splits/energies_forces_dipoles_train.npz
```

---

## 4. Training

### PhysNet (E/F, optional dipole)

```bash
mmml configure          # physnet-train workflow → train.yaml
# or:
mmml physnet-train --config train.yaml
# or positional data:
# mmml physnet-train \
#   --data splits/energies_forces_dipoles_train.npz \
#   --valid-data splits/energies_forces_dipoles_valid.npz \
#   --ckpt-dir ckpts/nma_physnet
```

Hybrid MM-charge modes (fixed / latent / fixed+latent) are documented in
[hybrid-mm-charges.md](../hybrid-mm-charges.md); example YAMLs live under
`examples/hybrid_mm_charges/`.

### Joint PhysNet + DCMNet (ESP / multipoles)

There is **no** standalone `mmml dcmnet` command. Joint training is:

```bash
mmml train-joint \
  --train-efd splits/energies_forces_dipoles_train.npz \
  --train-esp splits/grids_esp_train.npz \
  --valid-efd splits/energies_forces_dipoles_valid.npz \
  --valid-esp splits/grids_esp_valid.npz \
  --ckpt-dir ckpts/nma_joint
```

See [dcmnet_calculators.md](../dcmnet_calculators.md) for ASE inference from
joint checkpoints.

---

## 5. Evaluation

```bash
mmml physnet-evaluate \
  --checkpoint ckpts/nma_physnet \
  --data splits/energies_forces_dipoles_test.npz \
  -o eval_out/ --plots
```

Sanity-check the potential on the same methyl grid used earlier:

```yaml
# nma_methyl_ml.yaml — same DoFs as examples/ic_scan/nma_methyl.yaml
structure: xyz/nma.xyz
calculator: physnet
checkpoint: ckpts/nma_physnet/...   # portable JSON or Orbax dir
evaluate: energy
# … dofs / scans unchanged …
```

```bash
mmml ic-scan --config nma_methyl_ml.yaml --output ic_scan/methyl_ml --overwrite
```

Compare ML vs xTB / QM barriers on `data.csv`.

---

## 6. Intermolecular scans (dimers)

### 1D COM scan (first-class CLI)

```bash
mmml dimer-scan NMA \
  --calculator physnet --checkpoint "$MMML_CKPT" \
  --distance 2.5:6.0:0.1 \
  --energy-definition interaction \
  --output results/nma_dimer_1d
```

Non-campaign residues use a generic centroid–centroid orientation (see
[dimer-scan design](../dimer-scan-design.md)). For xTB reference:

```bash
mmml dimer-scan NMA --calculator xtb --distance 3.0:6.0:0.25 \
  --output results/nma_dimer_xtb
```

### 2D / hybrid LR scans (scripts)

Production **2D** hybrid COM scans (cutoffs + Ewald / jax-pme / …) follow the
DCM/ACO pattern — not the `ic-scan` CLI:

```bash
export MMML_CKPT=/path/to/nma_or_species_checkpoint
./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \
  NMA:2 --scan-1d --mlpot-pbc --lr-solver ewald …
```

Full solver matrix documentation: [dimer scans (DCM/ACO)](../functionality/dimer_scans/README.md).
Adapt composition/`MMML_CKPT` once an NMA-trained checkpoint exists.

---

## 7. Condensed-phase MD

### Box

```bash
mmml make-box --res NMA --n 50 --side_length 30.0
# or certified liquid workflow:
mmml liquid-box --help
```

### Hybrid MD

```bash
mmml env
export MMML_CKPT=/path/to/checkpoint
mmml md-system --setup pbc_nvt --composition NMA:50 \
  --temperature 300 --output-dir runs/nma_nvt
# production path with OpenMPI + libcharmm:
# MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh md-system --config run.yaml
```

Presets and YAML ownership: [md-system configs](../md-system-configs.md).
Long-range solvers: [long-range solver tutorial](../long-range-solver-tutorial.md).

---

## 8. Checklist

1. [ ] `mmml make-res --res NMA` → inspect `xyz/nma.xyz` in VMD  
2. [ ] `mmml ic-scan` methyl 1D (prepare-only and/or xTB/ML)  
3. [ ] Sample + `pyscf-evaluate` / `xml2npz` → `evaluated.npz`  
4. [ ] `mmml fix-and-split` → `splits/`  
5. [ ] `mmml physnet-train` and/or `mmml train-joint`  
6. [ ] `mmml physnet-evaluate` on test split; re-run methyl `ic-scan` with ML  
7. [ ] `mmml dimer-scan NMA …` (± 2D hybrid script when ready)  
8. [ ] `make-box` / `liquid-box` → `mmml md-system`

---

## Atom-order / mask pitfalls

- Amide **C–C–N–C** is `CL–C–N–CR = [0, 4, 6, 8]`. Omit `mask`, or include **CR
  (a4)** in it. `mask: [9,10,11]` alone does **not** rotate ω.
- Methyl rotors need methyl carbon as **a3** and a methyl H as **a4**
  (`N–C–CL–HL1`, `C–N–CR–HR1`), not `HL–CL–C–N`.
- Details and 2D notes: [ic-scan design](../ic-scan-design.md).
