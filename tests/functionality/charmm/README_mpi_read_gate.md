# MPI PyCHARMM READ gate (Tier 3 bisect)

Minimal harness to isolate **cooperative** `eval_charmm_script` topology I/O at `np>1`.
No JAX, ASE, Rich, or MLpot.

## Prerequisites

Prebuilt DCM:20 artifacts (once at np=1):

```bash
MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \
  tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \
  --prepare-prebuilt-only --residue DCM --n-molecules 20 --box-side 40
```

## PyCHARMM matrix (node09)

```bash
cd ~/mmml

# Baseline — must pass
MMML_MPI_NP=1 ./scripts/run_mpi_pycharmm_read_gate.sh

# Bisect modes at np=4 (record last log line before hang)
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode restart

# Optional: crystal after READ
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd --with-crystal
```

**Pass:** `PASS read_gate: mode=... np=N n_atoms=100`

At ``np>1``, bootstrap uses **shared artifact paths** by default (cooperative MPI
READ). Optional per-rank UUID copies: ``MMML_MPI_BOOTSTRAP_RANK_LOCAL=1`` (each
rank reads independently — usually leaves ``n_atoms=0`` on DOMDEC builds).

After staging (if any), all ranks **barrier** before ``read_rtf``. Expect:
``[bootstrap_sync rank */N] barrier done`` then ``[read_rtf ...] done ... n_atoms=100``.

**Hang:** last line like `[read_psf rank 2/4] begin ...` — that sub-command is the stall point.

**node09 bisect (June 2026):** `np=1` passes; `np=2` hangs on the first `read_rtf` step in
`psf-crd` mode. Root cause: per-rank SCRATCH units inside `eval_charmm_script`
(`setup/api/api_eval.F90`) desync cooperative MPI READ. Fix: restore direct
`maincomx` evaluation + line-splitting in vendored `pycharmm/lingo.py`, then
**rebuild** `libcharmm.so` (rebuild script must sync `api_eval.F90` into the tree):

```bash
grep -q "call maincomx" setup/charmm/source/api/api_eval.F90 || echo "MISSING api_eval patch"
bash scripts/rebuild_charmm_mlpot.sh --clean
```

After rebuild, re-run the matrix above. Also try `stream-inp` if rebuilding is delayed:

```bash
MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp
```

## Native CHARMM control

Proves Fortran READ works on the same PSF/CRD under the same `mpirun` wrapper:

```bash
PSF="$PWD/artifacts/domdec_spatial_smoke/dcm_20mer.psf"
CRD="$PWD/artifacts/domdec_spatial_smoke/dcm_20mer.crd"
RTF="$PWD/mmml/data/charmm/top_all36_cgenff.rtf"
PRM="$PWD/mmml/data/charmm/par_all36_cgenff.prm"

cat > /tmp/read_gate.inp <<EOF
* read gate control
*
bomlev -2
read rtf card name $RTF
read param card name $PRM
read psf card name $PSF
read coor card name $CRD
energy
stop
EOF

MMML_MPI_NP=4 ./scripts/mmml-charmm-mpirun.sh \
  "$CHARMM_EXE" -i /tmp/read_gate.inp -o /tmp/read_gate.out
```

If native passes and PyCHARMM hangs → bug is in Python `eval_charmm_script` entry, not cluster fabric.

## Implementation

Bootstrap API: [`mmml/interfaces/pycharmmInterface/charmm_mpi.py`](../../../mmml/interfaces/pycharmmInterface/charmm_mpi.py)
(`bootstrap_topology_mpi`, `bootstrap_charmm_step`).

Used by [`10_domdec_spatial_mpi_smoke.py`](../mlpot/10_domdec_spatial_mpi_smoke.py) for `np>1` live ENER load.
