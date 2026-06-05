# DCM:3 NVE cutoff sweep (PyCHARMM MLpot)

Free-space **NVE** on **DCM:3** with multiple **ML/MM cutoff presets** and **initial trimer COM
geometries**. Parses CHARMM `DYNA>` lines to rank presets by energy smoothness (std dev, step
jumps, drift).

Sibling workflows: [dcm_nve_scaling](../dcm_nve_scaling/) (cluster size), [dcm5_md_benchmark](../dcm5_md_benchmark/) (multi-backend).

## Matrix

| Cutoff preset | mm_on / mm_w / ml_w (Å) | Round-1 note |
|---------------|-------------------------|--------------|
| `dcm9_stability` | 7 / 5 / 0.1 | Robust on all geoms; production default |
| `extended_handoff` | 8 / 3 / 1.5 | Best close/mid; failed far/asymmetric |
| `extended_mm5` | 8 / 5 / 1.5 | **New:** extended ML taper + dcm9 MM width |
| `handoff_7p5` | 7.5 / 5 / 0.1 | **New:** between dcm9 and extended on distance |

COM geometries (all presets × each): `close` (4.0 × 4.5 Å), `mid` (6.0 × 7.0 Å),
`far` (8.0 × 9.0 Å), `asymmetric` (5.0 × 9.0 Å, 90°).

**16 NVE jobs** (4×4) plus 4 geometry-prep jobs.

Dropped after round 1: `code_default` (5.5 / 1.5 / 0.1) and `wide_ml_taper` (7 / 5 / 1.0).
See preset comments in [config.yaml](config.yaml). Baseline matches
[docs/mlpot-settings.md](../../docs/mlpot-settings.md).

## Prerequisites

- `export MMML_CKPT=/path/to/dcm_physnet_ckpt`
- GPU JAX (`uv sync --extra gpu`) or env with `jax` + `mmml`
- OpenMPI + `libcharmm` (use `scripts/mmml-charmm-mpirun.sh`)
- `packmol`, `snakemake`

## Run

```bash
export MMML_CKPT=/path/to/your/dcm_ckpt
cd workflows/dcm3_nve_cutoff_sweep
bash scripts/preflight.sh
snakemake -n
snakemake -j4 --resources gpu=1 mpi=1 --keep-going
```

Single variant:

```bash
snakemake results/runs/dcm9_stability/mid/nve_metrics.json --cores 1
```

## Outputs

| Path | Content |
|------|---------|
| `results/geometries/{geom}/` | Pre-MLpot PSF + trimer `initial.crd` |
| `results/runs/{preset}/{geom}/` | Mini + NVE artifacts, `stdout.log` |
| `results/runs/{preset}/{geom}/nve_metrics.json` | Energy drift / smoothness metrics |
| `results/cutoff_sweep_summary.csv` | All jobs, sortable |
| `results/cutoff_sweep_report.md` | Ranked table + suggested preset |

### Smoothness metrics

From `scripts/analyze_nve_energy.py` (CHARMM `DYNA>` total energy):

- `etot_std_kcal` — fluctuation around mean total energy
- `max_abs_etot_step_delta_kcal` — largest frame-to-frame jump (switching noise)
- `etot_drift_kcal` — end − start total energy
- `smoothness_score` = std + max step Δ + 0.1×|drift| (**lower is smoother**)

## Tuning

Edit [config.yaml](config.yaml):

- `ps_nve` — default 1.0 ps screening leg (4000 steps @ 0.25 fs)
- `geometry_variants` — add COM pairs `(d01, d02, angle_02_deg)`
- `cutoff_presets` — add or adjust switch widths
- `no_echeck: true` — avoid early abort on ML USER energy spikes during comparison

## Troubleshooting

### NVE stops at restart step 2 / 2 DCD frames

Usually `dynamics_overlap_check_interval` is too small for `dcd_nsavc: 1`. With
`interval: 1`, overlap chunking uses 2-step chunks (`nsavc + 1`), and scratch
restart handoffs fail after the first chunk. Set the interval to the full NVE
step count (default **4000** for `ps_nve: 1.0`, `dt_fs: 0.25`).

```bash
grep -E 'overlap \(NVE\)|integrated |incomplete|restart step' \
  results/runs/extended_handoff/mid/stdout.log | tail -20
rm -f results/runs/extended_handoff/mid/done.txt
snakemake results/runs/extended_handoff/mid/done.txt -j1 --resources gpu=1 mpi=1
```

## Do not commit run outputs

`results/` and `.snakemake/` are gitignored.

## Unit tests (no CHARMM)

```bash
uv run pytest tests/unit/test_dcm3_nve_cutoff_analyze.py -q
```
