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
| `smoke` | TIP3:90 / 30 Å Packmol + CHARMM MM pretreat → hybrid heat/NVE; `--mlpot-pbc`, density-prep off, `--no-monomer-physnet-mini` | exit 0 |
| `prod` | TIP3:90 jaxmd `pbc_nve` (default 50 ps, `dt=0.25`, record/10) | `*.h5` under prod dir |
| `analyze` | `scripts/analyze_water_nve_h5.py` | OH power peak **~2800–3600 cm⁻¹** (not ~40) |

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
  tip3_90_smoke/
  tip3_90_nve/*.h5
  analysis/{ir_spectrum.png,oh_bond_power_spectra.png,summary.json}
```
