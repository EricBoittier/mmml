# pycharmm

a python library for molecular dynamics with CHARMM

## Running under MPI (mpi4py)

pycharmm is normally driven with mpi4py in SPMD style: every rank runs
the same script and hosts its own **independent, serial** CHARMM (each
rank has the whole system; there is no domain decomposition across
ranks). Cross-rank coordination is done in the script via mpi4py. This
suits ensemble methods — replica exchange, free-energy windows (BLaDE is
single-rank by design), string-method replicas, etc.

- Get the rank/size from **mpi4py**, not CHARMM. Under pycharmm,
  `?numnode` is always `1` and `?mynode` always `0`:
  ```python
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank, size = comm.Get_rank(), comm.Get_size()
  ```
- Name per-rank output/scratch files by `rank` so ranks don't clobber
  each other.
- Import order doesn't matter (importing pycharmm before mpi4py used to
  hang; that's fixed).

To instead run one **genuine multi-node** CHARMM (`MPI_COMM_WORLD`,
`?numnode = N`), set `CHARMM_MULTINODE=1` before launching — advanced,
and it requires every rank to issue *identical* CHARMM commands in
lockstep (rank-dependent CHARMM calls will deadlock). See the "Parallel"
section of `doc/pycharmm.info` for details.
