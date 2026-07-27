# NH₃–CH₃Cl PhysNet results

Placeholder — run the example pipeline to populate metrics and figures:

```bash
source examples/m/_env.sh
bash examples/m/run_all.sh
```

See [`examples/m/README.md`](https://github.com/EricBoittier/mmml/blob/main/examples/m/README.md). Checkpoint and dataset:
commit `30eb7a01f7fcf1d42a795f188526a80e547110fd` (`examples/m/kl.json`,
`examples/m/nh3_ch3cl_filtered.npz`).

## ADUMB (PyCHARMM)

| Example | RCs | Config | Script |
|---------|-----|--------|--------|
| 1D bond difference | `rdif` = \(r_{\mathrm{ClC}}-r_{\mathrm{CN}}\) ∈ [-3, 3] Å | `examples/m/yaml/adumb_nc_distance.yaml` | `09_adumb_nc_distance.sh` |
| 2D Cl⋯C / C⋯N | `rcl`, `rcn` distances | `yaml/adumb_clc_cn_2d.yaml` | `10_adumb_clc_cn_2d.sh` |
| 1D + TIP3 | same `rdif` (PBC skeleton) | `yaml/adumb_nc_distance_tip3.yaml` | `SOLVATED=1` on `09_*.sh` |

Optional `USE_NPZ_PDB=1` feeds `07_export_solute_pdb.py` geometry. The 1D RC is
an RXNCOR **`combination` of two distances** (CHARMM `COMBI_TYPE`):
\(\xi = r_{\mathrm{ClC}} - r_{\mathrm{CN}}\). For 2D, define two distances,
`rxncor set nrxn 2 …`, two `umbrella rxncor … name …` cards, then one
`umbrella init`. ADUMB **NAME** tokens are ≤4 characters (`rdif`, `rcl`, `rcn`).

### CHARMM build

- Pref keys: **ADUMB** + **ADUMBRXNCOR** (`?ADUMBRXN == 1`).
  `scripts/rebuild_charmm_mlpot.sh` enables `ADUMBRXNCOR` by default.
- Without `ADUMBRXNCOR`, `umbrella rxncor` prints `Unknown umbrella specified`
  and heat often SIGSEGVs under MLpot/MPI.
- Prefer `umbrella rxncor … min 0.0 max …` when libcharmm is unpatched.
  Unpatched `UM1RXN` treated the upper edge as `(max − min)` for any nonzero
  `min` (e.g. `min 2 max 6` → abort once value \(> 4\); `min -3 max 3` →
  upper check against 6 instead of 3). mmml patches that check to `[min, max]`.
  The 1D difference example uses `min -3 max 3` and **requires** that patch
  plus a rebuild that installs into the `CHARMM_LIB_DIR` PyCHARMM actually
  loads (do not leave `CHARMM_LIB_DIR` on a stale PhysNet tree).

### Align `umbrella init` with heat length

Adaptive umbrella expects the dynamics length to match the WHAM schedule:

\[
n_{\mathrm{step}} = \mathrm{round}(ps_\mathrm{heat} \times 1000 / dt_\mathrm{fs})
\]

\[
n_{\mathrm{sim}} \times \mathrm{update} = n_{\mathrm{step}}
\]

| Mode | `dt_fs` | `ps_heat` | \(n_{\mathrm{step}}\) | Example `umbrella init` |
|------|---------|-----------|------------------------|-------------------------|
| Smoke | 1.0 | 0.2 | 200 | `nsim 4 update 50 equi 25` |
| Medium (current YAML) | 1.0 | 100 | 100 000 | `nsim 100 update 1000 equi 200` |
| Longer heat | 1.0 | 1000 | 1 000 000 | `nsim 1000 update 1000 equi 500` (tune) |
| Longer @ 0.5 fs | 0.5 | 1000 | 2 000 000 | retune so `nsim * update == nstep` |

Also keep `umbrella` `temp` near the thermostat target, and open
`ADUMB-WUNI.DAT` / `UMBCOR` / `RXNCOR_TRACE.DAT` under the job `output_dir`
(YAML uses basename `OPEN` names; lingo runs with `cwd=output_dir`).

Wipe `{output_dir}` (or at least `next_run.*` **and**
`pycharmm_pre_dynamics_lingo.inp`) before switching smoke ↔ long heat so a
leftover resume does not re-launch the wrong schedule. Stale staged lingo is a
common cause of `UM1RXN` `reaction coordinate out of range` after editing
`umbrella rxncor min`/`max` in the YAML — confirm the staged file has
`min 0.0` before trusting a long heat.

### Expected artifacts (smoke)

**1D** — `artifacts/nh3_ch3cl/adumb_nc_distance/` (`ps_heat=100`):

- `pycharmm_pre_dynamics_lingo.inp` with `combination rcl 1.0 rcn -1.0`, `name rdif`
- `ADUMB-WUNI.DAT`, `UMBCOR`, `RXNCOR_TRACE.DAT` (or lowercase)
- heat restart / DCD

**2D** — `artifacts/nh3_ch3cl/adumb_clc_cn_2d/`:

- lingo with `nrxn 2`, two `umbrella rxncor` cards (`r_cl`, `r_cn`)
- `ADUMB-WUNI.DAT`, `UMBCOR`, `RXNCOR_RCL.DAT`, `RXNCOR_RCN.DAT`
- heat restart / DCD

### Results (fill in after a successful run)

_Pending — paste pass criteria, RC trace summary, and figure paths here once
smoke or production heat completes._

| Run | Config | RCs | \(ps_\mathrm{heat}\) | Exit | Notes / figures |
|-----|--------|-----|----------------------|------|-----------------|
| | 1D / 2D | | | | |
