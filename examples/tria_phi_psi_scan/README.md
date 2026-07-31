# Trialanine φ/ψ scan — gas then solvent-relaxed

Central ALA of CGenFF `TRIA` (ACE–ALA₃–CT3):

| Angle | Atoms (0-based) | Definition |
|-------|-----------------|------------|
| φ | 14, 16, 18, 24 | C1–N2–CA2–C2 |
| ψ | 16, 18, 24, 26 | N2–CA2–C2–N3 |

Pipeline:

1. **Gas**: set φ/ψ → CHARMM constrained MM min → optional ML FIRE (`FixInternals`)
2. **Solvent**: inject each gas peptide into a TIP3 box → water-only SD (PEPT fixed) → joint SD/ABNR with `CONS DIHE` on φ/ψ
3. **Figure**: side-by-side Ramachandran ΔE maps

## Smoke (cluster / PyCHARMM node)

```bash
cd ~/mmml
export MMML_CKPT="${MMML_CKPT:-examples/sppoky-epoch-0010_params.json}"
OUT=artifacts/tria_phi_psi_scan

# 1) Coarse gas grid (30° → 13×13; use 60° for a quicker smoke)
uv run python scripts/scan_trialanine_phi_psi_pes.py \
  --checkpoint "$MMML_CKPT" \
  --phi=-180:180:60 --psi=-180:180:60 \
  --out "$OUT/gas" \
  --mm-sd-steps 50 --mm-abnr-steps 50 \
  --relax-steps 100

# 2) Solvent relax from gas NPZ (fewer waters for smoke)
uv run python scripts/scan_trialanine_phi_psi_solvent.py \
  --gas-npz "$OUT/gas/phi_psi_pes.npz" \
  --out "$OUT/solvent" \
  --n-waters 40 --box-side-A 28 \
  --mm-sd-steps 50 --mm-abnr-steps 50 \
  --water-only-sd-steps 30

# 3) Figure
uv run python scripts/plot_tria_phi_psi_gas_solvent.py \
  --gas-csv "$OUT/gas/phi_psi_pes.csv" \
  --solvent-csv "$OUT/solvent/phi_psi_solvent.csv" \
  -o "$OUT/figures/gas_vs_solvent.png"
```

Production-ish: `--phi=-180:180:30 --psi=-180:180:30`, `--n-waters 200 --box-side-A 30`.
(Use the `=` form for negative ranges — otherwise argparse treats `-180:…` as a flag.)

## Figure without a scan

```bash
uv run python scripts/plot_tria_phi_psi_gas_solvent.py --demo \
  -o artifacts/tria_phi_psi_scan/figures/gas_vs_solvent.DEMO.png
```

## Umbrella sampling (gas φ / ψ)

After the gas scan NPZ + traj exist, run a **periodic dihedral** umbrella with
`mmml umbrella-sample` (`DihedralCV`, degrees / eV·deg⁻²). Seeds come from the
scan (`seed_mode: frames`); stretch seeding is distance-only.

```bash
# Export one seed frame per φ window (ψ held near -60°); Z from sibling .traj
uv run python scripts/export_tria_phi_umbrella_seeds.py \
  --gas-npz "$OUT/gas/phi_psi_pes.npz" \
  --cv phi --n-windows 7 \
  -o "$OUT/gas/umbrella_phi_seeds.npz"

# Smoke (~7 windows); swap yaml for production 13-window / longer nsteps
uv run mmml umbrella-sample \
  --config examples/tria_phi_psi_scan/yaml/umbrella_phi_gas_smoke.yaml

uv run mmml umbrella-mbar --run-dir "$OUT/umbrella_phi_gas_smoke"

uv run python scripts/plot_tria_dihedral_umbrella_pmf.py \
  --run-dir "$OUT/umbrella_phi_gas_smoke" \
  --xlabel 'φ (deg)' \
  -o "$OUT/figures/umbrella_phi_pmf.png"
```

ψ: `--cv psi` → `umbrella_psi_seeds.npz` + `yaml/umbrella_psi_gas.yaml`.

YAML CV block:

```yaml
cv_x:
  kind: dihedral
  atoms: [14, 16, 18, 24]   # φ; use [16, 18, 24, 26] for ψ
seed_mode: frames           # required (stretch is distance-only)
k_ev_A2: 0.05               # eV/deg² (raise if histograms drift)
```

**Solvent dihedral umbrella** (`hybrid_jaxmd`) is not wired yet — use the
constrained solvent relax above for solvent landscapes; gas `packed_ml` path
is ready.

## Notes

- Solvent *minimize* stage is **MM-constrained** (CHARMM); hybrid ML peptide + MM solvent can be layered later via `md-embedding` / mechanical policy.
- Gas NPZ `positions_A[i,j]` must stay finite (clash rejects still keep MM-min frames).
- Dihedral restraints / umbrella CVs use the same atom indices as the gas scan — do not reorder the PSF.
