"""MPI usability tests: is pyCHARMM actually usable under mpi4py?

These launch pyCHARMM under ``mpirun`` (via _mpi_worker.py) and assert
it is *usable*, not merely importable:

  * no hang       -- every run finishes within a timeout (a timeout is a
                     hard failure, not an inconclusive skip); guards the
                     mpi4py "child ranks sit idle" deadlock.
  * no crash      -- mpirun exits 0 on every rank.
  * all ranks live -- every rank prints its marker (no silently-idle child).
  * expected communicator -- ordinary runs use independent serial CHARMM
                     instances; explicit ``set_mpi_comm(MPI.COMM_WORLD)``
                     runs one parallel CHARMM across all ranks.
  * mpi4py + CHARMM coexist -- gather/bcast/Gather/send/recv/barrier run
                     interleaved with CHARMM commands and all agree, and
                     CHARMM still runs a command afterwards (proves MPI
                     was neither hijacked nor finalized underneath).

Skips cleanly when mpirun/mpiexec or mpi4py are unavailable (e.g. a
serial build). Mirrors the real Brooks-group workshop usage (replica
exchange, string method) which imports pycharmm and mpi4py in both
orders and uses these exact collectives.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "_mpi_worker.py")
_MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")
_HAVE_MPI4PY = importlib.util.find_spec("mpi4py") is not None


def _is_openmpi():
    """--oversubscribe is Open MPI-only; MPICH's mpiexec rejects it."""
    if _MPIRUN is None:
        return False
    try:
        out = subprocess.run([_MPIRUN, "--version"], capture_output=True, text=True, timeout=15)
        return "open mpi" in (out.stdout + out.stderr).lower()
    except Exception:
        return False


_OVERSUBSCRIBE = ["--oversubscribe"] if _is_openmpi() else []


pytestmark = pytest.mark.skipif(
    _MPIRUN is None or not _HAVE_MPI4PY,
    reason="requires mpirun/mpiexec and mpi4py (serial build?)",
)


def _run(nproc, env_extra=None, timeout=180):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    # --oversubscribe (Open MPI only) lets -np exceed physical slots on small
    # CI hosts; harmless when slots are plentiful, omitted on non-Open-MPI.
    cmd = [_MPIRUN, *_OVERSUBSCRIBE, "-np", str(nproc), sys.executable, "-u", _WORKER]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        pytest.skip("mpirun launch failed (no MPI runtime)")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        pytest.fail(
            f"mpirun -np {nproc} HUNG (>{timeout}s) -- likely a child-idle / "
            f"finalize deadlock.\n--- partial output ---\n{out}"
        )
    return proc


def _assert_all_ranks_ok(proc, nproc, expected_numnode=1):
    assert proc.returncode == 0, (
        f"mpirun exited {proc.returncode} (crash).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    markers = re.findall(
        r"MPITEST rank=(\d+)\s+size=\d+\s+numnode=(\d+)",
        proc.stdout,
    )
    reported = sorted(int(rank) for rank, _ in markers)
    assert reported == list(range(nproc)), (
        f"expected all {nproc} ranks to report; got {reported} "
        f"(a missing rank = silently idle/hung child).\n"
        f"--- stdout ---\n{proc.stdout}"
    )

    for rank, numnode in markers:
        assert int(numnode) == expected_numnode, (
            f"unexpected CHARMM communicator size on rank {rank}: {numnode}"
        )

    assert "MPITEST_ALL_OK" in proc.stdout, (
        f"rank-0 collectives did not all agree.\n--- stdout ---\n{proc.stdout}"
    )


@pytest.mark.parametrize("nproc", [2, 4])
def test_mpi_usable_import_mpi_first(nproc):
    """mpi4py imported before pycharmm."""
    proc = _run(nproc, {"PYCHARMM_MPI_IMPORT_ORDER": "mpi_first"})
    _assert_all_ranks_ok(proc, nproc)


@pytest.mark.parametrize("nproc", [2, 4])
def test_mpi_usable_import_pycharmm_first(nproc):
    """pycharmm imported before mpi4py -- the order that used to leave
    non-master ranks idle/hung (regression guard)."""
    proc = _run(nproc, {"PYCHARMM_MPI_IMPORT_ORDER": "pycharmm_first"})
    _assert_all_ranks_ok(proc, nproc)


def test_mpi_usable_set_mpi_comm_self():
    """Explicit set_mpi_comm(MPI.COMM_SELF) keeps each rank serial."""
    proc = _run(4, {"PYCHARMM_MPI_SET_COMM": "self"})
    _assert_all_ranks_ok(proc, 4)


def test_mpi_usable_set_mpi_comm_world():
    """Repeated CHARMM calls return on every rank of one parallel CHARMM."""
    proc = _run(2, {"PYCHARMM_MPI_SET_COMM": "world"})
    _assert_all_ranks_ok(proc, 2, expected_numnode=2)


@pytest.mark.requires_feature("REPDSTR")
def test_mpi_usable_native_repdstr_lifecycle():
    """Two two-rank replicas reset to the four-rank Python command loop.

    Needs REPDSTR compiled in: the worker issues a ``REPD`` command, and
    a build without the keyword answers "REPlica DiSTRibuted code not
    compiled", trips BOMLEV and aborts every rank -- which the harness
    would otherwise report as a genuine MPI lifecycle crash.
    """
    proc = _run(
        4,
        {
            "PYCHARMM_MPI_SET_COMM": "world",
            "PYCHARMM_MPI_REPD_LIFECYCLE": "1",
        },
    )
    _assert_all_ranks_ok(proc, 4, expected_numnode=4)
