# TIP3 PhysNet + hybrid Ewald → IR

Staged campaign for the charge-less PhysNet portable checkpoint
(`test-f41c04c0-…_epoch-251_portable.json`): fixed CGenFF charges, hybrid-native
`--lr-solver ewald --ewald-omit-self`.

## One-shot on gpu09

```bash
export CKPT=/mmhome/boittier/home/mmml/mmml/models/physnetjax/defaults/hf_json/test-f41c04c0-62e3-4785-9018-351ffdc161c4_epoch-251_portable.json

# Preflight only (minutes)
STAGE=fd ./scripts/run_tip3_physnet_ewald_ir_campaign.sh

# Dimer Ewald scan + tip3_50 PyCHARMM smoke
STAGE=scan,smoke ./scripts/run_tip3_physnet_ewald_ir_campaign.sh

# Production jaxmd NVE (50 ps default) + IR
STAGE=prod,analyze PS_PROD=50 ./scripts/run_tip3_physnet_ewald_ir_campaign.sh

# Everything
./scripts/run_tip3_physnet_ewald_ir_campaign.sh
```

## Stages

| Stage | What | Pass |
|-------|------|------|
| `fd` | `mode-check --pbc-fd --residue TIP3` | `fd_force_max_abs_diff_eVA < 0.05` |
| `scan` | TIP3:2 COM scan `pbc_hybrid_ewald_omit_self` | `scan_1d.npz` written |
| `box_opt` | **Pinned:** count@30 Å → ~903 TIP3, ρ≈1.0 → live CHARMM `PRSI` MC/1D + CPT refine → `box_pressure_opt/{box.json,model.psf,model.crd}` | `status=pass`, handoff CRD present |
| `npt` | **Default NpT = PyCHARMM CPT** from certified handoff (`mini,heat,equi`, Hoover, `pref=1 atm`) | equi restart under `npt_charmm/` |
| `smoke` | Hybrid heat/NVE at **fixed L** from certified CRD (Packmol fallback if no handoff) | exit 0 |
| `prod` | TIP3:90 jaxmd `pbc_nve` (default 50 ps, `dt=0.25`, record/10) | `*.h5` under prod dir |
| `analyze` | `scripts/analyze_water_nve_h5.py` | OH power peak **~2800–3600 cm⁻¹** (not ~40) |

### Box pressure opt (`STAGE=box_opt`)

Finds a cubic `L` for a target pressure (default 1 atm) before hybrid heat.
Default NpT path is **PyCHARMM CPT** (not jaxmd). The first slice runs
`mmml liquid-box`, then live CHARMM virial `PRSI` MC + 1D refine + short CPT
(`run_box_pressure_opt_charmm_live`). Writes handoff
`box_pressure_opt/{box.json,model.psf,model.crd}` for fixed-L hybrid smoke.
Offline CI: `USE_CHARMM_PRESSURE=0` → synthetic `P∝1/L³`.

**Pinned liquid recipe (gpu09-validated):** `BOX_MODE=count` `BOX_SIZE=30`
`TARGET_DENSITY=1.0` → **N≈903**, **L=30 Å**, **ρ≈1.00 g/cm³**, MM GRMS≈0.04.
OpenMPI/PRRTE may still print exit 1; the script trusts `box.json` `status=pass`.

`TIP3:90` @ 30 Å is only ~0.1 g/cm³ (Packmol smoke fallback). Densified alt:
`BOX_MODE=density N_MOL=90` → L≈13.9 Å.

```bash
# continue from certified liquid_box without Packmol rebuild:
WIPE=0 STAGE=box_opt ./scripts/run_tip3_physnet_ewald_ir_campaign.sh
# full rebuild + live CHARMM pressure + CPT refine:
WIPE=1 STAGE=box_opt ./scripts/run_tip3_physnet_ewald_ir_campaign.sh

# CHARMM CPT NpT after box_opt (prefers box_pressure_opt handoff):
STAGE=npt ./scripts/run_tip3_physnet_ewald_ir_campaign.sh

# Hybrid smoke at fixed L from certified CRD:
STAGE=smoke ./scripts/run_tip3_physnet_ewald_ir_campaign.sh

# reuse an older box_opt dir:
BOX_OPT_OUT=./scratch/tip3_physnet_ewald_ir/tip3_90_box_opt STAGE=npt,smoke \
  ./scripts/run_tip3_physnet_ewald_ir_campaign.sh
```

## Density / packing note

TIP3:50 @ 30 Å (~0.055 g/cm³) thrashes hybrid FIRE/SD. A **lattice grid** at
TIP3:90 / 30 Å (~1 g/cm³) is also too hard for hybrid FIRE (stalls at
fmax≈5–6 eV/Å) and can spike MLpot SD to 1e5+ GRMS. Smoke uses **Packmol +
CHARMM MM pretreat**, density-prep off, and `--no-monomer-physnet-mini`.

Do **not** resume `tip3_90_smoke/next_run` / `baseline.res` after a gate fail.
Wipe and re-run:

```bash
rm -rf scratch/tip3_physnet_ewald_ir/tip3_90_smoke
git pull
STAGE=smoke ./scripts/run_tip3_physnet_ewald_ir_campaign.sh
```

In a good smoke log, confirm Packmol + `CHARMM MM pretreat` before hybrid FIRE,
and **no** `90 PhysNet group(s)`.

## Artifacts

```
scratch/tip3_physnet_ewald_ir/
  pbc_fd_tip3.json
  dimer_scan/.../scan_1d.npz
  tip3_30A_box_opt/{liquid_box/,box_pressure_opt/{box.json,model.psf,model.crd},npt_charmm/}
  tip3_90_smoke/
  tip3_90_nve/*.h5
  analysis/{ir_spectrum.png,oh_bond_power_spectra.png,summary.json}
```
