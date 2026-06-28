# DES dimer–dimer 2D scans (Snakemake)

2D center-of-mass separation scans for all unordered pairs from the **DES dimers**
chemical panel (PhysNet checkpoint `examples/ckpts_json/DESdimers_params.json`) plus
CGENFF alkane stereoisomers available in `top_all36_cgenff.rtf`.

## Species panel (12 → 78 pairs)

| Group | CGENFF RESI | Notes |
|-------|-------------|--------|
| water | TIP3 | TIP3P |
| dichloromethane | DCM | |
| acetone | ACO | |
| ethanol | ETOH | |
| methanol | MEOH | |
| ethane | ETHA | |
| benzene | BENZ | |
| n-butane | BUTA | |
| iso-butane | IBUT | |
| n-pentane | PENT | |
| neopentane | NEOP | |
| n-hexane | HEXA | |

**Not in CGENFF:** linear n-heptane and n-octane (no `C7H16` / `C8H18` alkane RESI).

Homodimers use `RES:2`; heterodimers use `RES_A:1,RES_B:1`.

## Backends (per grid point)

| Backend | Method | Config key |
|---------|--------|------------|
| CHARMM | CGENFF MM `ENER` | `backends.charmm` |
| xTB | GFN2-xTB (tblite) | `backends.xtb` |
| ORCA | MP2/def2-SVP | `backends.orca_mp2` |

Default grid: 3.0–10.0 Å, 12×12 points (`config.yaml` → `scan`).

## Prerequisites

```bash
cd ~/mmml
uv sync --extra gpu          # PyCHARMM + CHARMM
uv sync --extra quantum-crosscheck   # tblite xTB (optional)
# ORCA binary on PATH or export ORCA=/path/to/orca
```

## Run

```bash
cd workflows/des_dimer_pair_scans
bash scripts/preflight.sh

# Dry-run DAG (78 pair jobs + collect)
bash scripts/snakemake_local.sh 2 -n

# Local (2 concurrent pairs)
bash scripts/snakemake_local.sh 2

# Slurm CPU farm
bash scripts/snakemake_slurm.sh 8

# Single pair
bash scripts/job_shell.sh aco__meoh
```

## Report with Matplotlib figures

After scans (complete or partial):

```bash
# Summary table + HTML report with one PNG per pair (78 total)
snakemake results/report.html --configfile config.yaml

# Or directly:
python scripts/collect_scans.py --output-csv results/summary.csv --output-md results/summary.md
python scripts/build_report.py --output-html results/report.html
```

Open `results/report.html` in a browser. Figures live in `results/figures/<pair_tag>.png`.
Pairs without `scan_2d.npz` get a grey **pending** placeholder panel.

### Smoke test (CHARMM-only, 5×5 grid)

```bash
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 4
# Rebuild report only (after scans exist):
MMML_WORKFLOW_CONFIG=config.smoke.yaml snakemake report --configfile config.smoke.yaml
```

## Outputs

```
artifacts/des_dimer_pair_scans/<pair_tag>/
  scan_2d.npz      # d01/d02 grids + energy arrays
  scan_2d.json     # metadata
  done.txt
  stdout.log       # when run via Snakemake
results/
  summary.csv
  summary.md
  report.html          # all pairs with embedded figure gallery
  report_manifest.json
  figures/             # one PNG per pair (ΔE heatmaps)
    aco__meoh.png
    ...
```

NPZ arrays (when backend enabled): `charmm_ENER_kcal`, `xtb_energy_kcal`,
`orca_mp2_energy_kcal`, plus distance metrics from the scan geometry.

## Cost note

78 pairs × 144 grid points × ORCA MP2 is expensive. Reduce `scan.steps` or disable
`backends.orca_mp2` for exploratory CHARMM/xTB passes, then enable ORCA on a subset.
