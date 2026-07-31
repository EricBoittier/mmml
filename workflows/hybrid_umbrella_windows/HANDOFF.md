# Handoff: parallel hybrid umbrella (Snakemake / GPU)

**Date:** 2026-07-31  
**Owner context:** NH₃–CH₃Cl solvated mechanical-embedding umbrella PMF  
**CV:** `ξ = r(C–Cl) − r(C–N)` via YAML `cv_x.pairs` + `coefficients: [1.0, -1.0]`  
**Engine:** `hybrid_jaxmd` (ML solute AMM1+CH3CL, MM solvent)

Do **not** run full MD / `mmml md-system` / Packmol box builds in the agent sandbox unless the user asks. Prefer unit tests + copy-paste Slurm commands.

---

## Goal

Run production umbrella windows **in parallel** (one GPU Slurm job per window), then assemble + MBAR.

Workflow root: `workflows/hybrid_umbrella_windows/`  
Example entrypoint: `examples/m/15_umbrella_snakemake.sh`

---

## What works / was built

### Per-window resume (library + CLI)

- `mmml/umbrella/hybrid_windows.py` — `windows/wXXX.npz` checkpoints; assemble into `umbrella_snapshots.npz`
- CLI: `mmml umbrella-sample --resume --windows N` / `--no-resume-failed`
- Shell: `RESUME=1`, `WINDOWS=…` in `examples/m/14_umbrella_sample_sol_prod.sh`
- Tests: `tests/unit/test_hybrid_windows_resume.py` (run: `uv run pytest tests/unit/test_hybrid_windows_resume.py -q`)

### Assemble → MBAR (verified offline)

`tests/unit/test_hybrid_windows_assemble_mbar.py` packs synthetic `windows/wXXX.npz`
the same way `run_umbrella_hybrid_nvt` does, then runs `run_umbrella_mbar` on the
result. It covers the parts of the workflow tail that need no GPU or CHARMM:

- failed window → NaN row in the PMF, still counted in `failed_windows`, kept at
  its original ξ₀ index (`n_windows_used = K - 1`)
- antisymmetric `cv_spec` round-trip through the NPZ — with the legacy single-pair
  CV instead, MBAR fails to converge outright
- the `umbrella_summary.json` fields `scripts/run_mbar.sh` turns into `mbar/status.json`

Still unverified on real data: the `assemble` rule itself (needs CHARMM + the model
to rebuild the hybrid system before packing).

### Snakemake workflow

```text
make_box → window[{000..N-1}] (parallel GPU) → assemble → mbar
```

| File | Role |
|------|------|
| `Snakefile` | Rules; GPU jobs get `--gres=gpu:1` |
| `config.yaml` | ACN prod (default), 30 windows, `max_jobs: 8` |
| `config.tip3.yaml` | TIP3 prod |
| `config.smoke.yaml` | 3-window tip3 smoke |
| `scripts/env_shell.sh` | Env + CUDA preflight/retries + `JAX_PLATFORMS=cuda` |
| `scripts/run_one_window.sh` | `--resume --windows N` |
| `scripts/run_assemble.sh` | `--resume --no-resume-failed` (pack only) |
| `scripts/run_mbar.sh` | `umbrella-mbar --run-dir` + `mbar/status.json` |
| `scripts/snakemake_slurm.sh` | Login-node launcher |
| `profiles/slurm/config.yaml` | executor=slurm, `retries: 3` |

Artifacts (ACN prod):

```text
artifacts/nh3_ch3cl/boxes/acn/model.{psf,pdb}
artifacts/nh3_ch3cl/umbrella_nc_acn_prod/
  windows/wXXX.npz
  logs/window_wXXX.log
  umbrella_snapshots.npz   # after assemble
  umbrella_summary.json
  mbar/status.json
```

Cluster path is often `/mmhome/boittier/home/mmml` (same tree as `~/mmml`).

### Race fix (required for parallel)

**Bug:** Every parallel job with `--resume` called `bootstrap_windows_from_snapshots`, racing on shared `wXXX.tmp.npz` → `FileNotFoundError` on `os.replace`.

**Fix (in tree):**

1. Skip bootstrap when `--windows` / `only_windows` is set — `should_bootstrap_windows()` in `hybrid_windows.py`, called from `hybrid.py`.
2. Unique temp names `wXXX.<pid>.<uuid>.tmp.npz` in `save_window_checkpoint`.

Both halves are pinned by `tests/unit/test_hybrid_windows_resume.py`.

### CUDA flake mitigation (partial)

Window logs showed:

```text
cuInit(0) failed: CUDA_ERROR_UNKNOWN
Backend 'cuda' is not in the list of known backends: ['cpu', 'tpu']
```

Mitigations added (not fully validated in prod yet):

- CUDA preflight + retries in `env_shell.sh` (`MMML_CUDA_INIT_RETRIES`, default 12)
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Explicit `--gres=gpu:1` on GPU rules
- Profile `retries: 3`
- Warning if `snakemake_slurm.sh` is launched from inside a GPU allocation

**Strong advice:** run the Snakemake **controller on the login node**, not an interactive `gpu0N` shell. User was on `gpu08` when many failures appeared.

---

## Current user state (last known)

- ACN prod campaign was being submitted via Snakemake.
- Some windows failed (bootstrap race, then CUDA init).
- User cancelled **only** the snakemake-named jobs (`a2888619`), **not** long-running `run` jobs (`204929`, `204930` on gpu08/gpu09).
- Workflow dir may need `snakemake --unlock` after Ctrl+C.

**Never suggest** `scancel -u "$USER"`. User has other GPU jobs. Scope cancels by JOBID list or job name (e.g. `a2888619`).

```bash
# Safe pattern — name from `squeue` NAME column for this snakemake run
squeue -u "$USER" -h -o "%i %j" | awk '$2 == "a2888619" {print $1}' | xargs -r scancel

# Or explicit IDs only
scancel 205272 205273 …
```

Unlock:

```bash
cd ~/mmml/workflows/hybrid_umbrella_windows
uv run --with snakemake snakemake --unlock
```

Resubmit (login node):

```bash
cd ~/mmml/workflows/hybrid_umbrella_windows
nohup bash scripts/snakemake_slurm.sh 8 > snakemake_gpu.log 2>&1 &
# or: SOLVENT=acn JOBS=8 bash examples/m/15_umbrella_snakemake.sh
```

Finished `windows/wXXX.npz` are kept; only missing outputs re-run.

---

## Related science / CLI context (prior work)

- Prod YAML: `examples/m/yaml/umbrella_nc_{tip3,acn}_prod.yaml` — `dt=0.25 fs`, `nsteps=80000` (20 ps), `max_seed_force: 80`, ξ ∈ [-1.3, 1.6], `k_ev_A2: 6.505`
- Checkpoint: prefer `examples/m/model_ext.json` (not stale `kl.json`)
- Serial prod: `GPU=1 SOLVENT=acn bash examples/m/14_umbrella_sample_sol_prod.sh`
- MBAR drops failed/NaN windows; hybrid uses `energies_unbiased_ev`
- Neighbor build: Vesin / chunked NumPy in `mm_system_energy.py` (was O(N²) hang)
- PBC MIC wrap: do **not** remove `stop_gradient` on dimer MIC wrap (see `.cursor/rules/pbc-dimer-mic-wrap.mdc`)

---

## Likely next tasks for the new agent

Everything left needs the cluster. `pc-mm025` has no Slurm config (`squeue` fails
with `_establish_config_source`), so run these from the studix login node.

1. Confirm CUDA preflight + `--gres=gpu:1` actually stops `CUDA_ERROR_UNKNOWN` when controller is on **login**.
2. If CUDA still flakes: inspect Slurm plugin GPU request vs studix (`scontrol show job <id>` for GRES), consider lowering `-j` / `max_jobs`, or exclusive node flags the site supports.
3. Run the real `assemble` rule once a full set of `wXXX.npz` exists (the pure
   pack + MBAR half is already covered by unit tests — see above).
4. Optionally: tip3 prod via `config.tip3.yaml`.
5. Do **not** commit unless user asks.

### Checked 2026-07-31 (offline, no cluster)

- `uv run pytest tests/unit -k "umbrella or hybrid_windows" -q` → 77 passed
- `MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2 -n` →
  DAG resolves to `make_box → window×3 → assemble → mbar`, GPU rules carry
  `slurm_extra=--gres=gpu:1`, `mbar` correctly requests `gpu=0`
- Working tree clean at `828e57422`; all workflow files tracked

---

## Quick test commands

```bash
uv run pytest tests/unit -k "umbrella or hybrid_windows" -q
cd workflows/hybrid_umbrella_windows
MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_local.sh 2 -n
```

---

## Agent workflow rules (mmml)

- No long MD in agent sessions.
- TDD with mocked PyCHARMM in `tests/unit/`.
- CI: `.github/workflows/ci.yml` runs `pytest -v tests/`.
- See `devtools/AGENTS.md` / `.cursor/rules/mmml-agent-workflow.mdc`.
