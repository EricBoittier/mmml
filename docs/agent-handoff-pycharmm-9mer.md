# Agent handoff: PyCHARMM MLpot smoke → 9-mer stability

**Date context:** June 2026. User: Eric / mmml on cluster `/mmhome/boittier/home/mmml`.

## User preferences (read first)

- **Do not run** `mmml md-system` or full MD from the agent (no CHARMM/MPI/GPU env; too slow).
- Prefer **unit tests** + **documented commands** the user runs locally.
- TDD order: **CHARMM FF + `cons_fix`** → **MMML** → full `md-system`.
- See `.cursor/rules/mmml-agent-workflow.mdc`.

## What was just done (local repo, may need `git pull` on cluster)

| Change | Files |
|--------|--------|
| Pre-dynamics gate: require MLpot USER before dynamics | `cli_common.assert_dynamics_ready`, `staged_workflow.py`, `run_workflow.py`, `tests/unit/test_assert_dynamics_ready.py` |
| Heating print: `ihtfrq` defaults to `--dyn-nprint` (not hardcoded 10) | `cli_common.resolve_heat_ihtfrq`, `dynamics.py`, `staged_workflow.py` |
| DCD: **delete** old stage DCD by default; `--rescue-old-dcd` to archive | `staged_workflow._reset_stage_trajectory`, `--rescue-old-dcd` in `cli_common` / `md_system.py` |
| Docs | `mlpot/CHARMM_SETTINGS.md`, `tests/functionality/constraints/README.md` |
| Monomer constraint unit tests | `tests/unit/test_monomer_constraints.py` |

**Cluster may still show** `Pre-dynamics GRMS OK: 0.0000` until updated code is deployed.

## DCM:2 smoke command (user’s last runs)

```bash
mmml md-system \
  --setup free_nvt \
  --backend pycharmm \
  --composition DCM:2 \
  --md-stages mini,heat \
  --checkpoint /path/to/dcm1-.../ckpts/dcm1-... \
  --output-dir artifacts/pycharmm_mlpot/dcm2_smoke_v2 \
  --mini-nstep 100 \
  --ps-heat 2.0 \
  --heat-ihtfrq 100 \
  --dyn-nprint 500 \
  --skip-energy-show \
  --seed 123
```

**Re-running without `--quiet`** — user flagged for VMD / extra-potential diagnosis.

## Constraint check: none applied on this command

Verified from CLI defaults + `staged_workflow.py`:

| Mechanism | Default | DCM:2 smoke |
|-----------|---------|-------------|
| `--fix-resids` | `""` | **not set** → no SD pass-2 `cons_fix` |
| `--constrain-resids` | `""` | **not set** → `setup_cons_fix` skipped (`if dynamics_constrain:`) |
| `--no-fix` | false | only matters if `--fix-resids` non-empty |
| `--flat-bottom-radius` / `--fb-rad` | None | **not in command** → `apply_flat_bottom_from_args` no-ops |
| Log (prior non-quiet run) | | `cons_fix: no monomers constrained` on HEAT |

**Code paths:**

- `resolve_fix_resids` → `[]` when `fix_resids` empty.
- `resolve_constrain_resids` → `[]` when `constrain_resids` empty.
- `minimize_with_mlpot`: `fixed_ml_selection=None` → single free SD pass only.

To **test** constraints later: `--fix-resids 1` and/or `--constrain-resids 1` — see `tests/functionality/constraints/README.md`.

## VMD: “extra potentials” without `--quiet`

Not `cons_fix` for DCM:2 smoke. Likely visible physics / artifacts:

1. **MLpot USER** — all-ML BLOCK zeros CHARMM VDW/ELEC/bonded during dynamics; motion is ML + thermostat.
2. **Heating** — CHARMM `ihtfrq` velocity rescaling (COM/velocity banners when not quiet).
3. **Wrong topology** — use `cluster_for_vmd_*.psf` for DCD, **not** `mini_full_mlpot_*.psf` (bond-stripped PSF).
4. **Two minimization tracks** — `mini_charmm_mm_*.dcd` (CGENFF MM) vs `mini_full_mlpot_*.dcd` (ML SD); heat continues from ML coords.
5. **MMFP** — only if `--flat-bottom-radius` or `--fb-rad` set (not on smoke cmd).
6. **Overlap rescue** — if triggered during dynamics (check non-quiet log for `overlap` / bonded rescue).

**Non-quiet log lines to confirm:**

- `MLpot SD minimize` / `Post MLpot mini GRMS`
- `HEAT: ... | cons_fix: ...`
- `MLpot USER active before staged dynamics` (should appear **before** heat after gate fix)
- `Removed prior DCD` (new default) vs `Rescued existing DCD`
- `DYNA>` / `AVER` temperature after heat

## ML/MM handoff (user clarification)

- **Atoms/monomers are never dropped** from the model when COM &lt; cutoff.
- `mm_r_min` (~6.2 Å) only zeros **MM weight** for close dimers; ML still runs.
- Vacuum dimers: all cross-monomer atom pairs stay listed; `s_MM` handles ML-only zone.
- See `CHARMM_SETTINGS.md` § “ML/MM pair list vs mm_r_min”.

## Next goal: **9-mer stability**

Interpret as **DCM:9** (or similar 9-molecule cluster) — confirm stable heating / geometry, not flying apart or absurd T.

### Run script (user runs on cluster)

```bash
cd /mmhome/boittier/home/mmml
git pull   # mpirun + run_dcm9_stability prepend pip cuDNN
uv sync --extra gpu
module unload cudnn   # if cluster loads cudnn/9.4 (< 9.10.1 breaks JAX CUDA 13)

export MMML_CKPT=/path/to/dcm1-.../ckpts/dcm1-...
./scripts/run_dcm9_stability.sh
# optional: PS_HEAT=20 ENABLE_FB=1 FB_RAD=14 ./scripts/run_dcm9_stability.sh
```

**JAX CUDA / cuDNN error** (`cuDNN 90400 found` → no `cuda` backend): cluster `module load cudnn/9.4` wins over JAX unless pip `nvidia-cudnn-cu13>=9.10.1` is installed and prepended (`scripts/setup_jax_cuda_env.sh`, `mmml-charmm-mpirun.sh`). Smoke test: `python -c "import jax; print(jax.devices())"` should list `cuda:0`.

`scripts/run_dcm9_stability.sh` wraps `mmml-charmm-mpirun.sh`, sets Packmol `R ≈ 9.6 Å` from the DCM:90 scaling formula, and matches the good DCM:2 smoke flags (no `cons_fix`, no `--bonded-mm-mini`, no flat-bottom unless `ENABLE_FB=1`). Extra CLI args are passed through (`"$@"`).

Tune `PACKMOL_R` / `FB_RAD` after mini using `scripts/estimate_droff_from_crd.py` if needed.

Optional large-cluster aids (if COM drifts):

- `--flat-bottom-radius` + `--flat-bottom-selection "TYPE C"` for DCM
- `--no-scale-echeck` only if auto-loosened echeck hides problems

### Pass/fail criteria (9-mer)

| Check | OK if |
|-------|--------|
| Pre-dynamics | `USER` active, GRMS not 0.0 (after gate fix) |
| Post mini | GRMS modest (e.g. &lt; 10 kcal/mol/Å); no abort |
| Heat `AVER` T | ~300 K (not 400–500+ like bad DCM:2 bonded run) |
| VMD | No monomer–monomer overlap; COM not collapsing; use `cluster_for_vmd` PSF |
| Artifacts | Fresh `heat_*.dcd` (deleted old DCD, not appended to rescue file) |

### Tests to run locally (fast)

```bash
pytest tests/unit/test_monomer_constraints.py \
       tests/unit/test_assert_dynamics_ready.py \
       tests/unit/test_charmm_output_settings.py -q
```

### Open issues / tech debt

1. `Pre-dynamics GRMS 0.0000` false pass on cluster without `require_mlpot_user` deploy.
2. `--bonded-mm-mini` misleading on all-ML small clusters (false GRMS spikes).
3. Heating still velocity-scaling (`ihtfrq`), not Hoover — see `THERMOSTAT_INVESTIGATION.md`.
4. Constraint verification (CHARMM FF → MMML) not yet automated beyond unit tests.

### Key files

- Staged MD: `mmml/interfaces/pycharmmInterface/mlpot/staged_workflow.py`
- Constraints CLI: `mmml/interfaces/pycharmmInterface/mlpot/cli_common.py` (`add_monomer_constraint_args`)
- MD entry: `mmml/cli/run/md_system.py`
- Constraints README: `tests/functionality/constraints/README.md`
- Settings table: `mmml/interfaces/pycharmmInterface/mlpot/CHARMM_SETTINGS.md`

### User log artifact

- `DCM2LOG` in repo root (177 lines, quiet smoke v2) — reference only.
