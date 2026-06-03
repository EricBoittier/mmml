# CHARMM / PyCHARMM settings by workflow mode

Reference for `mmml md-system --backend pycharmm` and `tests/functionality/mlpot/*`.
CLI flags live in `cli_common.add_charmm_output_args()`; stage builders in `dynamics.py`; orchestration in `staged_workflow.py`.

## Console verbosity (global)

| Mode | CLI | `PRNLev` | `WRNLev` | `BOMBlev` | Notes |
|------|-----|----------|----------|-----------|--------|
| Default | *(none)* | 5 | 5 | -2 | Verbose Fortran core |
| Quiet | `--quiet` | 0 | 0 | -2 | Also sets mini/dyn print to once per pass/stage |
| Custom | `--prnlev N` `--warnlev N` `--bomlev N` | N | N | N | Applied at workflow start via `apply_charmm_verbosity()` |

`--show-energy` / `--skip-energy-show` control Python `energy.show()` only, not `DYNA>` lines.

## Print cadence (what you see during runs)

| Output type | CHARMM keyword | CLI flag | Default | When it fires |
|-------------|----------------|----------|---------|----------------|
| SD / mini energy line | `nprint` | `--nprint` | **50** | Every N SD steps (MLpot mini + CHARMM MM pre-min) |
| Dynamics energy summary | `nprint` | `--dyn-nprint` | **500** | `DYNA>` block every N integration steps |
| Detailed dynamics props | `iprfrq` | `--dyn-iprfrq` | **2000** | Extra `DYNA PROP>` / extended rows |
| Restart / velocity save | `isvfrq` | *(same as iprfrq)* | **2000** | Restart file timing |
| **Heating velocity rescale** | **`ihtfrq`** | **`--heat-ihtfrq`** | **0 → use `--dyn-nprint`** | **COM + “VELOCITIES ASSIGNED” banners** (often mistaken for `nprint`) |
| Quiet dynamics | — | `--quiet` | `nprint = iprfrq = isvfrq = nstep` | One summary per stage |

**Heating spam:** If you still see output every ~10 steps during `heat`, check `ihtfrq` (not only `dyn-nprint`). Legacy code used `ihtfrq=10` hardcoded. Current staged workflow sets `ihtfrq = resolve_heat_ihtfrq()` (default: match `--dyn-nprint`, e.g. 500).

Example for rare heating banners:

```bash
mmml md-system ... --dyn-nprint 2000 --dyn-iprfrq 5000 --heat-ihtfrq 2000
```

Finer temperature ramp (more rescales, more console):

```bash
mmml md-system ... --heat-ihtfrq 40
```

## Stage → integrator & thermostat

| Stage | Vacuum (`free_*`) | PBC (`pbc_*`) | Thermostat / T control | `ihtfrq` (after CLI) |
|-------|-------------------|---------------|------------------------|----------------------|
| **mini** (MLpot) | SD ×2 optional `cons_fix` | same | — | `nprint` from `--nprint` |
| **MM pre-min** | CHARMM SD/ABNR, MM only | same | — | same `nprint` |
| **heat** | Verlet + velocity scaling | Verlet + IMAGE lists | **`--heat-firstt`**, **`--heat-finalt`**, `TEMINC` | DCM:9 default **0→240 K** over **20 ps**, **`--heat-ihtfrq` 100**, `iasors=0` scale after cold start. **`--temperature`** is for later stages only unless `--heat-finalt` omitted |
| **nve** | Leap, microcanonical | Leap + IMAGE | none (`ihtfrq=0`) | print from `--dyn-nprint` |
| **equi** | Velocity scaling (restart: off) | CPT + Hoover NPT | vacuum: like heat; PBC: `hoover reft`, `cpt` | vacuum restart: **0**; vacuum cold start: ramp |
| **prod** | Hoover NVT | CPT Hoover NPT | `hoover reft`, `tmass` | 0 |

Staged workflow **always overwrites** `nprint`, `iprfrq`, `isvfrq` from `resolve_dynamics_print_kwargs()` after each builder runs.

## Stage → lists, trajectory, stability

| Setting | CLI | Default (typical) | heat | nve | equi/prod PBC |
|---------|-----|-------------------|------|-----|----------------|
| Timestep | `--dt-fs` | 0.25 fs → 0.00025 ps | all | all | all |
| Length | `--ps`, `--ps-heat`, … | setup-dependent | | | |
| DCD frames | `--dcd-nsavc` | 1 | all | all | all |
| Energy drift stop | `--echeck` / `--no-echeck` | 100 kcal/mol (scaled for large clusters) | all | all | equi/prod ≥500 if user value low |
| Nonbond rebuild | `inbfrq` | -1 (heat/PBC builders) | periodic lists | | |
| MLpot SD lists | `inbfrq=0` | mini only | avoids mlpot_update segfault | | |

## `md-system` setup presets

| `--setup` | Default `--md-stages` | Ensemble | PBC |
|-----------|----------------------|----------|-----|
| `free_nvt` | `mini,heat` | nvt (heat only; no Hoover prod in preset) | no |
| `free_nve` | `mini,nve` | nve | no |
| `pycharmm_minimize` | `mini` | — | no |
| `pycharmm_full` | `mini,heat,nve,equi,prod` | mixed | no |
| `pbc_nvt` | `mini,heat,equi` | nvt | yes |
| `pbc_nve` | `mini,heat,nve,equi,prod` | nve | yes |
| `pbc_npt` | `mini,heat,nve,equi,prod` | npt | yes |

## ML/MM pair list vs `mm_r_min` (not dropping atoms)

| What | Dropped when “out of range”? |
|------|------------------------------|
| **Atoms** | Never — all PSF atoms stay in MLpot and CHARMM |
| **Monomers** | Never — both DCM residues always evaluated |
| **ML monomer / dimer terms** | ML dimer models use their own cutoff (`mm_switch_on`, PhysNet `cutoff`); dimers within range get ML |
| **MM atom–atom pairs** (vacuum, small N) | **No spatial cull** — all cross-monomer pairs (e.g. 5×5) stay in the list |
| **MM weight `s_MM(r)`** | Goes to **0** when dimer COM is in the pure-ML zone (complementary handoff) |
| **`mm_r_min`** | Extra inner rule on **MM only**: scale MM to zero when dimer COM &lt; `mm_r_min` (~6.2 Å with default handoff). Does **not** remove atoms or monomers |

Neighbor lists for large PBC systems use an outer cutoff **`mm_switch_on + mm_switch_width` (12 Å)** plus optional jax-md skin — pairs near that cutoff stay listed so they can enter range later. That is separate from `mm_r_min`.

## Trajectory files

Before each dynamics stage, an existing stage DCD (e.g. `heat_dcm_2.dcd`) is **removed** by default so the new run does not append to old frames. Use `--rescue-old-dcd` to archive to `*.rescued.N.dcd` instead.

## Post-dynamics validation (staged workflow)

After each dynamics stage, the workflow checks the stage restart step and DCD frame count. If CHARMM stops early (common: **`echeck`** exceeded when an H stretches off or energy spikes during heating), the run **fails** instead of printing `Staged workflow OK`.

| Symptom | Likely cause |
|---------|----------------|
| 1 DCD frame, restart step ~700, asked for 40k steps | `echeck` abort (~0.18 ps of 10 ps heat) |
| H “falls off” in VMD | No SHAKE; ML USER-only dynamics; bad heat / echeck stop |

Use `--allow-incomplete-dynamics` only for debugging. For heat tests on small clusters, consider `--no-echeck` or a larger `--echeck` after confirming minimization is sound.

## Related flags (not print, but often paired)

| Flag | Purpose |
|------|---------|
| `--bonded-mm-mini` | CHARMM bonded-only recovery SD after selected stages |
| `--fix-resids` / `--constrain-resids` | `cons_fix` in mini pass 2 / MD |
| `--quiet` | Low `PRNLev` + coarse print |
| `--charmm-sd-steps` / `--charmm-abnr-steps` | MM pre-min before MLpot |

## Tests (no CHARMM)

```bash
pytest tests/unit/test_charmm_output_settings.py tests/unit/test_staged_md_cli.py -q
```

See also `THERMOSTAT_INVESTIGATION.md` for Hoover vs velocity-scaling design notes.
