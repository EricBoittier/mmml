"""Per-rank worker for the MPI usability tests (test_mpi_usable.py).

This is NOT a pytest module -- it is launched under
``mpirun -np N python _mpi_worker.py`` by the tests, which inspect its
stdout.  It exercises the way real pyCHARMM+mpi4py scripts (the Brooks
group workshop: replica exchange, string method, lambda windows) use
MPI, and prints one ``MPITEST ...`` line per rank plus a final
``MPITEST_ALL_OK`` from rank 0 when every collective agreed.

Behaviour is selected by environment variables:
  PYCHARMM_MPI_IMPORT_ORDER = "mpi_first" (default) | "pycharmm_first"
      Which of mpi4py / pycharmm is imported first.  Importing pycharmm
      first used to leave the non-master ranks idle/hung; both orders
      must now come up as an independent serial CHARMM per rank.
  PYCHARMM_MPI_SET_COMM = "" (default) | "self" | "world"
      Select MPI.COMM_SELF or MPI.COMM_WORLD before CHARMM initializes.
  PYCHARMM_MPI_REPD_LIFECYCLE = "" (default) | "1"
      Configure, run, and reset one two-replica native REPDSTR session.
"""

import os
import sys

_order = os.environ.get("PYCHARMM_MPI_IMPORT_ORDER", "mpi_first")
_set_comm = os.environ.get("PYCHARMM_MPI_SET_COMM", "")
_repd_lifecycle = os.environ.get("PYCHARMM_MPI_REPD_LIFECYCLE", "") == "1"

if _order == "pycharmm_first":
    import pycharmm
    from pycharmm import lingo

    from mpi4py import MPI
else:
    from mpi4py import MPI

    import pycharmm
    from pycharmm import lingo

import numpy as np


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if _set_comm == "self":
        pycharmm.set_mpi_comm(MPI.COMM_SELF)
    elif _set_comm == "world":
        pycharmm.set_mpi_comm(comm)

    # First CHARMM call: triggers library init and runs one command.
    # Every rank in the selected communicator issues identical commands.
    lingo.charmm_script("set mpitest 1")
    assert lingo.get_charmm_variable("MPITEST") == 1
    builtins = lingo.get_charmm_builtins()
    numnode = builtins.get("NUMNODE")
    mynode = builtins.get("MYNODE")

    # mpi4py collectives interleaved with CHARMM -- proves CHARMM did not
    # hijack or finalize MPI_COMM_WORLD out from under the script.
    gathered = comm.gather(rank, root=0)  # pickle gather
    comm.barrier()
    token = comm.bcast("HELLO" if rank == 0 else None, root=0)  # pickle bcast

    # Buffer-based collective (what the replica-exchange example uses).
    sendbuf = np.array([rank], dtype="i")
    recvbuf = np.zeros(size, dtype="i") if rank == 0 else None
    comm.Gather(sendbuf, recvbuf, root=0)

    # Point-to-point (what the simple-MPI example uses).
    peers = None
    if size > 1:
        if rank == 0:
            peers = [comm.recv(source=r, tag=r) for r in range(1, size)]
        else:
            comm.send(rank, dest=0, tag=rank)

    # A second CHARMM command after the MPI traffic -- CHARMM must still work.
    lingo.charmm_script("set mpitest 2")
    assert lingo.get_charmm_variable("MPITEST") == 2

    if _repd_lifecycle:
        from pathlib import Path

        data = Path(__file__).resolve().parents[3] / "test" / "data"
        lingo.charmm_script(
            f"""
            open unit 1 read card name {data}/toph19.rtf
            read rtf card unit 1
            close unit 1
            open unit 1 read card name {data}/param19.prm
            read param card unit 1
            close unit 1
            read sequence card
            *
            3
            AMN ALA CBX
            generate ala2 setup warn
            ic para
            ic seed 1 C 2 N 2 CA
            ic build
            """
        )

        class OneStepDynamics:
            _engine = "standard"

            def run(self):
                return lingo.charmm_script(
                    """
                    dyna verlet strt nstep 1 timestep 0.001 -
                      nprint 1 nsavc 0 nsavv 0 inbfrq 10 ihbfrq 0 -
                      firstt 300. finalt 300. tstruc 300. -
                      iasors 0 iasvel 1 iscvel 0 -
                      iseed 98123 55012 102978 101300
                    """,
                    raise_on_error=True,
                )

        lingo.charmm_script(
            "REPD NREP 2 EXCH -\n  FREQ 1 STEMP 300 DTEMP 30 UNIT 6",
            raise_on_error=True,
        )
        OneStepDynamics().run()
        lingo.charmm_script("REPDSTR RESET", raise_on_error=True)
        lingo.charmm_script("set postrepd 1", raise_on_error=True)
        assert lingo.get_charmm_variable("POSTREPD") == 1

    print(
        f"MPITEST rank={rank} size={size} numnode={numnode} mynode={mynode} bcast={token}",
        flush=True,
    )

    if rank == 0:
        assert sorted(gathered) == list(range(size)), f"gather: {gathered}"
        assert token == "HELLO", f"bcast: {token}"
        assert list(recvbuf) == list(range(size)), f"Gather: {list(recvbuf)}"
        if size > 1:
            assert sorted(peers) == list(range(1, size)), f"p2p: {peers}"
        print("MPITEST_ALL_OK", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
