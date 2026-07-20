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
| `box_opt` | CHARMM-default prep: `liquid-box` → pressure MC + 1D `L` refine → `box_pressure_opt/box.json` | `box.json` with `final_cubic_side_A` |
| `smoke` | TIP3:90 / 30 Å Packmol + CHARMM MM pretreat → hybrid heat/NVE; `--mlpot-pbc`, density-prep off, `--no-monomer-physnet-mini` | exit 0 |
| `prod` | TIP3:90 jaxmd `pbc_nve` (default 50 ps, `dt=0.25`, record/10) | `*.h5` under prod dir |
| `analyze` | `scripts/analyze_water_nve_h5.py` | OH power peak **~2800–3600 cm⁻¹** (not ~40) |

### Box pressure opt (`STAGE=box_opt`)

Finds a cubic `L` for a target pressure (default 1 atm) before hybrid heat.
Default NpT path is **PyCHARMM CPT** (not jaxmd). The first slice runs
`mmml liquid-box`, then pressure-objective MC + golden-section refine
(`mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt`). Offline CI uses a
synthetic `P∝1/L³` model; pass `charmm_pressure_fn` for live virial `PRSI`.

**Density:** `TIP3:90` @ 30 Å is **~0.1 g/cm³**, not 1 g/cm³ (need ~903 waters
at 30 Å, or `L≈13.9` Å for 90 waters). `box_opt` defaults to
`BOX_MODE=count` (fill 30 Å at 1 g/cm³). Wipe the out dir before re-runs so a
stale fail `box.json` is not reused.

```bash
rm -rf scratch/tip3_physnet_ewald_ir/tip3_90_box_opt
STAGE=box_opt ./scripts/run_tip3_physnet_ewald_ir_campaign.sh
# or smaller densified N=90 box:
BOX_MODE=density N_MOL=90 ./scripts/run_tip3_box_pressure_opt.sh
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
  tip3_90_box_opt/{liquid_box/,box_pressure_opt/box.json}
  tip3_90_smoke/
  tip3_90_nve/*.h5
  analysis/{ir_spectrum.png,oh_bond_power_spectra.png,summary.json}
```
