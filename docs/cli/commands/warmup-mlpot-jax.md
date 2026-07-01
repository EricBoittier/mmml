# `mmml warmup-mlpot-jax`

Serial JAX JIT warmup for MLpot.


## Usage

```bash
mmml warmup-mlpot-jax --help
```

## Options

```text
usage: mmml warmup-mlpot-jax [-h] [--checkpoint CHECKPOINT]
                             [--n-monomers N_MONOMERS]
                             [--atoms-per-monomer ATOMS_PER_MONOMER]
                             [--box-side BOX_SIDE] [--spacing SPACING]
                             [--ml-batch-size ML_BATCH_SIZE]
                             [--ml-gpu-count ML_GPU_COUNT]
                             [--ml-max-active-dimers N]
                             [--ml-switch-width ML_SWITCH_WIDTH]
                             [--mm-switch-on MM_SWITCH_ON]
                             [--mm-switch-width MM_SWITCH_WIDTH]
                             [--no-complementary-handoff] [--do-mm]
                             [--compile-threads COMPILE_THREADS]
                             [--allow-under-mpirun] [--dry-run] [--quiet]
                             [--verbose]

Warm up MLpot PhysNet JAX compilation in serial Python (multithreaded XLA).
Populates JAX_COMPILATION_CACHE_DIR for faster later runs under mpirun. Does not
import PyCHARMM or call MPI.

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        PhysNet checkpoint (default: MMML_CKPT or
                        MMML_CHECKPOINT)
  --n-monomers N_MONOMERS
                        Monomer count (default 20)
  --atoms-per-monomer ATOMS_PER_MONOMER
                        Atoms per monomer for synthetic lattice (default 10)
  --box-side BOX_SIDE   PBC box side Å (0 = vacuum)
  --spacing SPACING     Lattice spacing Å
  --ml-batch-size ML_BATCH_SIZE
                        MLpot batch size
  --ml-gpu-count ML_GPU_COUNT
                        JAX pmap GPU count
  --ml-max-active-dimers N
                        Sparse ML dimer slot cap per step (PBC default all
                        unique dimers when n≤4005; same as md-system --ml-max-
                        active-dimers).
  --ml-switch-width, --ml-cutoff ML_SWITCH_WIDTH
                        COM-distance width (Å) of the ML→MM handoff. ML is fully
                        on below mm_switch_on - width and tapers to zero at
                        mm_switch_on (default: 1.5).
  --mm-switch-on MM_SWITCH_ON
                        COM distance (Å) where the complementary handoff ends:
                        ML scale reaches 0 and MM scale reaches 1 (default: 8).
  --mm-switch-width, --mm-cutoff MM_SWITCH_WIDTH
                        COM-distance width (Å) of the MM outer tail after
                        mm_switch_on. Switched MM reaches zero at mm_switch_on +
                        width (default: 5).
  --no-complementary-handoff
                        Legacy MM window: MM starts at mm_switch_on instead of
                        filling the ML taper handoff.
  --do-mm               Include MM pair path in warmup (closer to production
                        hybrid)
  --compile-threads COMPILE_THREADS
                        Override MMML_JAX_COMPILE_THREADS (default: min(16,
                        ncpu) when unset)
  --allow-under-mpirun  Allow running under mpirun (not recommended; compile
                        threads usually off)
  --dry-run             Print planned warmup settings and exit
  --quiet
  --verbose

Examples: export MMML_CKPT=/path/to/DESdimers_params.json mmml warmup-mlpot-jax
--n-monomers 20 --ml-batch-size 128 # Match DCM:60 liquid workflow (resilient
preset cutoffs + sparse dimer cap): mmml warmup-mlpot-jax --checkpoint
"$MMML_CKPT" --n-monomers 60 \ --atoms-per-monomer 5 --box-side 32 --ml-batch-
size 64 --ml-gpu-count 1 \ --ml-max-active-dimers 1770 --mm-switch-on 6.0 --mm-
switch-width 4.0 \ --ml-switch-width 1.0 --do-mm # Then under MPI: MMML_MPI_NP=2
MMML_MLPOT_SPATIAL_MPI=1 ./scripts/mmml-charmm-mpirun.sh md-system ... Do
**not** run under mpirun (compile threads are disabled there by design). Clear
stale launcher env if needed: unset OMPI_COMM_WORLD_SIZE PMI_SIZE PMIX_SIZE
```


## Related docs

- [MLpot settings](../../mlpot-settings.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
