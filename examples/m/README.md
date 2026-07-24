# NH₃–CH₃Cl PhysNet example (`examples/m`)

Checkpoint and filtered dataset from commit
[`30eb7a01f7fcf1d42a795f188526a80e547110fd`](https://github.com/EricBoittier/mmml/commit/30eb7a01f7fcf1d42a795f188526a80e547110fd):

| File | Role |
|------|------|
| `kl.json` | Portable PhysNet params (`natoms=9`, charges + ZBL, vacuum) |
| `nh3_ch3cl_filtered.npz` | 16 000 frames (`N=9` dimers + NH₃ / CH₃Cl monomers) |
| `top_ch3cl.rtf` | Append topology for CGenFF residue `CH3CL` (used by Packmol `md-system`) |

Docs report (after running the pipeline):
[`docs/examples/nh3-ch3cl-results.md`](../../docs/examples/nh3-ch3cl-results.md).

## Environment

```bash
cd /path/to/mmml
source examples/m/_env.sh
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `MMML_CKPT` | `examples/m/kl.json` | Checkpoint |
| `MMML_DATA` | `examples/m/nh3_ch3cl_filtered.npz` | Eval NPZ |
| `MMML_CGENFF_EXTRA_RTF` | `examples/m/top_ch3cl.rtf` | Enables `CH3CL` in compositions |
| `ARTIFACTS_DIR` | `artifacts/nh3_ch3cl` | Outputs |

## Quick run (full report)

```bash
bash examples/m/run_all.sh
```

Steps:

1. `01_evaluate.sh` — `mmml physnet-evaluate --plots`
2. `run_md_smokes.sh` — free-space NVE/NVT
3. `02_figures_and_report.py` — house-style figures + MkDocs page

### Evaluate only

```bash
NUM_SAMPLES=256 bash examples/m/01_evaluate.sh
uv run python examples/m/02_figures_and_report.py
```

### MD smokes

**ML-only Python** (no CHARMM; geometry from a dataset dimer frame):

```bash
uv run python examples/m/03_free_nve_ase.py --n-steps 40
uv run python examples/m/04_free_nvt_ase.py --n-steps 40
uv run python examples/m/05_free_nve_jaxmd.py --n-steps 40
uv run python examples/m/06_free_nvt_jaxmd.py --n-steps 40
```

**`md-system` Packmol** (`AMM1:1,CH3CL:1`, `--include-mm` off; needs PyCHARMM for PSF):

```bash
source examples/m/_env.sh
uv run mmml md-system --config examples/m/yaml/free_nve_ase.yaml
uv run mmml md-system --config examples/m/yaml/free_nve_jaxmd.yaml
uv run mmml md-system --config examples/m/yaml/free_nve_pycharmm.yaml
uv run mmml md-system --config examples/m/yaml/free_nvt_ase.yaml
uv run mmml md-system --config examples/m/yaml/free_nvt_jaxmd.yaml
uv run mmml md-system --config examples/m/yaml/free_nvt_pycharmm.yaml
```

Skip CHARMM-backed legs: `RUN_MD_SYSTEM=0 bash examples/m/run_md_smokes.sh`  
or `RUN_PYCHARMM=0` to keep ASE/JAX-MD `md-system` only when PyCHARMM is present.

## Pass / fail

| Check | Criterion |
|-------|-----------|
| Evaluate | `artifacts/nh3_ch3cl/evaluate/metrics.json` written; finite MAE/RMSE |
| ASE/JAX-MD smokes | `md_summary.json` with finite `E1`; NVE ΔE reported |
| Docs | `docs/examples/nh3-ch3cl-results.md` + PNGs under `docs/images/examples/nh3-ch3cl/` |
