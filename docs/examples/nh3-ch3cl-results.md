# NH₃–CH₃Cl PhysNet results

Placeholder — run the example pipeline to populate metrics and figures:

```bash
source examples/m/_env.sh
bash examples/m/run_all.sh
```

See [`examples/m/README.md`](../../examples/m/README.md). Checkpoint and dataset:
commit `30eb7a01f7fcf1d42a795f188526a80e547110fd` (`examples/m/kl.json`,
`examples/m/nh3_ch3cl_filtered.npz`).

## ADUMB N⋯C distance (PyCHARMM)

Vacuum smoke: `examples/m/yaml/adumb_nc_distance.yaml` via
`bash examples/m/09_adumb_nc_distance.sh` (optional `USE_NPZ_PDB=1`).
Solvated TIP3 skeleton: `yaml/adumb_nc_distance_tip3.yaml`.

Reaction coordinate is the AMM1 N1⋯CH₃Cl C1 distance (`rxncor` +
`umbrella rxncor`), not a peptide dihedral.

### CHARMM build

- Pref keys: **ADUMB** + **ADUMBRXNCOR** (`?ADUMBRXN == 1`).
  `scripts/rebuild_charmm_mlpot.sh` enables `ADUMBRXNCOR` by default.
- Without `ADUMBRXNCOR`, `umbrella rxncor` prints `Unknown umbrella specified`
  and heat often SIGSEGVs under MLpot/MPI.
- Prefer `umbrella rxncor … min 0.0 max …` (CHARMM c38 `adumbrxncor.inp` style).
  Unpatched `UM1RXN` in `eadumb.F90` treated the upper edge as `(max − min)`
  when `min > 0`, so `min 2 max 6` aborted once \(r_\mathrm{NC} > 4\) Å.
  mmml patches that check to `[min, max]`; still keep `min 0` for smoke
  robustness across libcharmm ages.

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

Under `artifacts/nh3_ch3cl/adumb_nc_distance/` (or `ARTIFACTS_DIR`):

- `pycharmm_pre_dynamics_lingo.inp` containing `umbrella rxncor`
- `ADUMB-WUNI.DAT` / `adumb-wuni.dat`, `UMBCOR` / `umbcor`,
  `RXNCOR_TRACE.DAT` / `rxncor_trace.dat`
- heat restart / DCD from the short heat leg

### Results (fill in after a successful run)

_Pending — paste pass criteria, \(r_\mathrm{NC}\) trace summary, and figure
paths here once smoke or production heat completes._

| Run | Config | \(ps_\mathrm{heat}\) | Exit | Notes / figures |
|-----|--------|----------------------|------|-----------------|
| | | | | |
