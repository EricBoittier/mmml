# DCM heat scaling (PyCHARMM MLpot)

Snakemake workflow for **heat-only** `mmml md-system` runs on **DCM:5 … DCM:90** (step 5),
with two timesteps (**0.25 fs** and **0.125 fs**) and configurable repeats.

Sibling to [dcm_nve_scaling](../dcm_nve_scaling/) (short NVE screening).

## Do not commit run outputs

`artifacts/`, `results/`, and `.snakemake/` are gitignored.

## Prerequisites

```bash
export MMML_CKPT=/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1-c137fb42-1f65-4748-880b-8f8184a20f70
export JAX_ENABLE_X64=1
```

- GPU JAX (`uv sync --extra gpu`) or cluster conda/micromamba env with `jax` + `mmml`
- Runs use plain `mmml md-system` (`use_mpirun: false`) — do **not** nest `mmml-charmm-mpirun.sh` inside Snakemake Slurm jobsteps
- `packmol` on PATH
- `snakemake` **and** [`snakemake-executor-plugin-slurm`](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) (Snakemake 8+ no longer bundles Slurm)

```bash
# one-off check (from workflow dir)
uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake --version
```

Without the plugin, `--profile profiles/slurm` fails with:
`invalid choice: 'slurm' (choose from local, dryrun, touch)`.

## Job matrix

From [config.yaml](config.yaml) defaults:

| Axis | Values |
|------|--------|
| Cluster size | 5, 10, …, 90 (18 sizes) |
| Repeat `N` | `[1]` (extend `repeats:` for more) |
| `dt-fs` | `0.25`, `0.125` |

**36 jobs** with default config (18 × 2 dt × 1 repeat). Each job gets a unique seed:
`seed_base + N×10000 + repeat×100 + dt_offset` (0 for 0.25 fs, 1 for 0.125 fs).

Outputs under **repo root** (`../../artifacts/...` from the workflow dir):

```
../../artifacts/pycharmm_mlpot/dcm{N}_npt_x64_{repeat}/dt025/
../../artifacts/pycharmm_mlpot/dcm{N}_npt_x64_{repeat}/dt0125/
```

Equivalent to your manual command (with `X` = cluster size, `N` = repeat):

```bash
export JAX_ENABLE_X64=1
mmml md-system \
  --setup pycharmm_full --backend pycharmm \
  --composition DCM:X \
  --output-dir artifacts/pycharmm_mlpot/dcmX_npt_x64_N/dt025 \
  --md-stages mini,heat --box-size 180 \
  --ps-heat 1000 --n-heat-segments 4000 --heat-thermostat hoover \
  --dt-fs 0.25 \
  --flat-bottom-radius 55 --packmol-radius 15 --temperature 220 \
  --dynamics-overlap-action rescue \
  --checkpoint "$MMML_CKPT" --seed <unique> \
  --dcd-nsavc 500 --dynamics-intra-min-distance 0.5 \
  --ml-gpu-count 1 --ml-batch-size 2056 --no-echeck
```

## Run

```bash
cd workflows/dcm_heat_scaling
bash scripts/preflight.sh
snakemake -n
```

Local (serialize PyCHARMM via `mpi=1`):

```bash
snakemake -j2 --resources gpu=1 mpi=1 --keep-going
```

Slurm (gpu08 / gpu09, alternating per job):

```bash
# recommended wrapper (checks plugin, passes --resources)
nohup bash scripts/snakemake_slurm.sh 4 > snakemake_slurm.log 2>&1 &

# equivalent manual command
nohup uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm -j4 --resources gpu=1 mpi=1 --keep-going \
  > snakemake_slurm.log 2>&1 &
```

Slurm resources per job match your header (via Snakemake resources, not `slurm_extra`):

- `slurm_partition=gpu`, `gpu=1` (→ `--gpus=1`)
- `nodes=1`, `tasks=1`, `cpus_per_task=4`, `mem_mb_per_cpu=3000` (single process, 4 CPUs)
- `slurm_extra`: `--mail-user=...`, `--nodelist=gpu08,gpu09` (SLURM picks a free node)

Do **not** set `cpus_per_task=4` with `gpu=1` — the Slurm plugin adds `--ntasks-per-gpu=4` and the node may reject the request.

Single job:

```bash
snakemake ../../artifacts/pycharmm_mlpot/dcm25_npt_x64_1/dt025/done.txt --cores 1
```

Dry-run one size × both dt:

```bash
snakemake -n ../../artifacts/pycharmm_mlpot/dcm10_npt_x64_1/dt025/done.txt \
              ../../artifacts/pycharmm_mlpot/dcm10_npt_x64_1/dt0125/done.txt
```

## Config notes

- **`repeats`**: add `[1, 2, 3]` for multiple independent trajectories per (N, dt).
- **`slurm_runtime_min`**: default 2880 (48 h); heat with `ps_heat=1000` is long — raise if needed.
- **`slurm_gpu_nodes`**: `[gpu08, gpu09]` — edit if your cluster changes.
- Radii (`flat_bottom_radius`, `packmol_radius`) are fixed in config; consider scaling for large N.
- **`md_stages: mini,heat`**: heat-only skips `minimize_with_mlpot` and usually fails on fresh output dirs.
- Snakemake captures rule logs automatically — do not add `> {log.stdout}` in the shell block (creates empty logs).
