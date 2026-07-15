# SciCORE runtime and Slurm environment

This is the supported SciCORE environment for MMML jobs that combine JAX,
PyCHARMM, OpenMPI, and the repository-built `libcharmm.so`. Run the acceptance
smokes below after changing the compiler, MPI, PyCHARMM, CHARMM, or CUDA stack.

## Runtime bootstrap

Batch shells must source the repository prolog before importing MMML or
PyCHARMM:

```bash
cd "$HOME/mmml"
source scripts/scicore_env.sh
source .venv/bin/activate
```

The prolog initializes Lmod in non-login shells and loads this matched stack:

- `foss/2023b`
- `CMake/3.27.6-GCCcore-13.2.0`
- GCC 13.2.0, OpenMPI 4.1.6, and FFTW 3.3.10 from the EasyBuild modules
- `JAX_ENABLE_X64=1`

SciCORE's EasyBuild OpenMPI installation is complete and Slurm-aware. The
generic MMML launcher contains fallback workarounds for incomplete local MPI
installations; those must remain disabled on SciCORE:

```bash
export MMML_NO_MPI_MCA_PREFIX=1
export MMML_NO_MPI_MPI_PRELOAD=1
export MMML_NO_MPI_OPAL_PRELOAD=1
export MMML_NO_MPI_PMIX_PRELOAD=1
```

`scripts/scicore_env.sh` exports these values. Without
`MMML_NO_MPI_MCA_PREFIX=1`, the launcher forces a fallback MCA component path
and the `sgd` rank segfaults before Python executes its first instruction.
Preloading MPI, OPAL, or PMIx DSOs is likewise unnecessary with this module
stack. Keep the module-provided `LD_LIBRARY_PATH`; the launcher forwards it to
the rank.

## CHARMM build and verification

The PBC workflow selects a compiled MLpot pair-capacity tier automatically.
For a 204-atom water/methanol smoke it selects the default
`max_Npr=8,000,000` no-DOMDEC tier under:

```text
~/.cache/mmml-charmm-build/tier_8000000_nodomdec/lib/libcharmm.so
```

Verify dependencies only after sourcing the SciCORE prolog:

```bash
source scripts/scicore_env.sh
ldd "$HOME/.cache/mmml-charmm-build/tier_8000000_nodomdec/lib/libcharmm.so"
```

No line may contain `not found`. A login-shell `ldd` without the prolog will
incorrectly report missing `libmpi.so.40` and `GLIBCXX_3.4.32`.

Run the rank-zero import gate before molecular dynamics:

```bash
export CHARMM_LIB_DIR="$HOME/.cache/mmml-charmm-build/tier_8000000_nodomdec/lib"
scripts/mmml-charmm-mpirun.sh python -c '
print("BEFORE", flush=True)
import pycharmm
import pycharmm.lingo
print("PYCHARMM_OK", flush=True)
'
```

The output must contain both markers and CHARMM `NORMAL TERMINATION`.

## Slurm resources

Use a GPU partition and its matching QoS. Start with a small mixed-system gate
on an idle RTX4090 node:

```bash
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-30min
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
```

Longer GPU cells use the QoS matching their selected partition. The Snakemake
driver itself is a small CPU job on `scicore`. SciCORE node names must not be
replaced by the pc-studix `gpu08`/`gpu09` lists. Use:

```bash
export MMML_SLURM_NO_NODELIST=1
export MMML_SLURM_EXTRA="--qos=rtx4090-6hours"
```

The workflow's `Snakefile` honors these scheduler overrides while retaining
the same scientific configuration.

## Water/methanol acceptance and gated campaign

The first acceptance cell is the 50:50, 0.1-density, 18 Å, 300 K system:

```text
tag:          meohx50_tip3x50_12_t300_l18
composition:  MEOH:6,TIP3:6
atoms:        6×6 + 6×3 = 54
```

Set the validated portable checkpoint and use float64 throughout:

```bash
export MMML_CKPT="$HOME/mmml/artifacts/checkpoints/step-00002000_params.json"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
```

Keep `ml_batch_size: 32` for the portable configuration. Validate 54- and
72-atom mixed cells first, then increase system size only after checking actual
GPU memory use. A 204-atom float64 force compile was not a useful first gate on
a shared 24 GB RTX4090. Larger sizes and batch values are separate,
site/GPU-specific escalation tests.

Submit the full campaign behind a successful smoke, never merely behind its
submission:

```bash
smoke=$(sbatch --parsable artifacts/scicore_submission/smoke.sbatch)
sbatch --dependency="afterok:${smoke}" \
  artifacts/scicore_submission/small_array.sbatch
```

The smoke must complete mixed topology construction, PyCHARMM minimization and
heat, the PyCHARMM continuation, a finite handoff, and the JAX-MD burst. A
failed smoke leaves the full driver cancelled by dependency and submits no
campaign cells.

## Diagnosing startup failures

| Symptom | Cause and action |
|---|---|
| `cmake: command not found` | The batch shell did not source `scripts/scicore_env.sh`, or the CMake module failed to load. |
| Segfault before a Python `BEFORE` marker | Fallback MCA/preload workarounds were applied. Confirm the four `MMML_NO_MPI_*` variables above. |
| `libmpi.so.40 => not found` or missing `GLIBCXX_3.4.32` | The EasyBuild module environment was not loaded in that shell. |
| Import appears to hang on the login node | Do not use a login import as the acceptance test. Submit the rank-zero Slurm import gate; shared-home metadata and first-time builds can be slow. |
| JAX autotuning reports multi-GiB `CUDA_ERROR_OUT_OF_MEMORY` | Confirm `ml_batch_size: 32`, inspect `nvidia-smi` for shared/orphan ranks, and reduce the acceptance system before requesting a larger-memory partition. Do not reuse historical 512/2048 settings on a 24 GB RTX4090. |
| Production jobs appear after a failed smoke | The full driver was not submitted with `afterok:<smoke_job_id>`; cancel it and repair the dependency chain. |
