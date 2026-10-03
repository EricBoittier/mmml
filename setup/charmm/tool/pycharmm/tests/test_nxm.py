"""N x M layout tests: N replicas, each an M-node parallel CHARMM.

These launch pyCHARMM under ``mpirun`` (via _nxm_worker.py) and assert
the layout comes up in the requested shape and, above all, that it always
finishes.  A hang is the failure mode this whole path exists to prevent,
so every run has a hard timeout and a timeout is a failure, never a skip.

Covered:
  * shape        -- a P-rank job split into groups of M really does give
                    CHARMM ?NUMNODE == M in every group, for a range of
                    N x M shapes including the 1 x P and P x 1 extremes.
  * teardown     -- a master that raises still releases its workers; the
                    job exits instead of leaving ranks blocked in bcast.
  * non-SPMD     -- master code whose CHARMM command count depends on
                    data only the master can see.  That desynchronizes a
                    group under the raw API; driven mode must absorb it,
                    checked by every rank agreeing on the resulting CHARMM
                    state, not merely by the job finishing.
  * raw API      -- set_mpi_comm() on its own still builds the layout, for
                    scripts that really are SPMD.
  * uneven split -- rejected up front with a clear error rather than
                    silently building a malformed layout.

Skips cleanly when mpirun/mpiexec or mpi4py are unavailable.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "_nxm_worker.py")
_MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")
_HAVE_MPI4PY = importlib.util.find_spec("mpi4py") is not None


def _is_openmpi():
    """Detect Open MPI, whose --oversubscribe MPICH's mpiexec rejects.

    Returns
    -------
    bool
        True if the discovered launcher is Open MPI.
    """
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


def _run(nproc, env_extra=None, timeout=240):
    """Launch the worker under mpirun and capture its output.

    Parameters
    ----------
    nproc : int
        Number of MPI ranks to launch.
    env_extra : dict, optional
        Extra environment variables selecting the worker's behaviour.
    timeout : int, optional
        Seconds to allow before treating the run as hung. A timeout fails
        the test; it is never reported as a skip.

    Returns
    -------
    subprocess.CompletedProcess
        The finished mpirun invocation.
    """
    env = dict(os.environ)
    # Deliberately do NOT put this source tree on the ranks' PYTHONPATH.
    # The whole suite runs against the pip-installed pycharmm (see
    # conftest.py): only the *installed* loader.py has the CHARMM library
    # directory baked in, by configure_file().  The copy in this tree is
    # still the CMake template, whose lib dir is the literal, unexpanded
    # "${CMAKE_INSTALL_PREFIX}" -- so a rank that imports it falls back to
    # a bare "libchmm.so" and dies in dlopen before it reaches any MPI code.
    # Inheriting the parent environment is what makes the ranks find the
    # same installed package the parent pytest process imported.
    # Keep the in-CHARMM startup deadline well under the test timeout, so a
    # rank that never reaches init aborts with CHARMM's own explanation
    # instead of us reporting a bare timeout.
    env.setdefault("CHARMM_MPI_INIT_TIMEOUT", "60")
    if env_extra:
        env.update(env_extra)
    cmd = [_MPIRUN, *_OVERSUBSCRIBE, "-np", str(nproc), sys.executable, "-u", _WORKER]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        pytest.skip("mpirun launch failed (no MPI runtime)")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        pytest.fail(
            f"mpirun -np {nproc} HUNG (>{timeout}s) with {env_extra} -- a "
            f"rank is blocked (worker never released, or a mismatched "
            f"collective).\n--- partial output ---\n{out}"
        )
    return proc


def _groups_reported(proc):
    """Extract the per-group layout each master reported.

    Parameters
    ----------
    proc : subprocess.CompletedProcess
        A finished run from :func:`_run`.

    Returns
    -------
    list of tuple
        ``(group index, numnode)`` for every group master that reported.
    """
    return [
        (int(g), int(n))
        for g, n in re.findall(r"NXM group=(\d+)\s+groups=\d+\s+numnode=(\d+)", proc.stdout)
    ]


# (nproc, group_size) -- N x M shapes, N = nproc // group_size.
_SHAPES = [
    (4, 1),  # 4 x 1: four independent serial CHARMMs
    (4, 2),  # 2 x 2: two replicas, each 2-node parallel
    (4, 4),  # 1 x 4: one replica across every rank
    (6, 3),  # 2 x 3: odd group size
    (2, 1),  # 2 x 1: smallest useful job
]


@pytest.mark.parametrize("nproc,group_size", _SHAPES)
def test_nxm_shape_driven(nproc, group_size):
    """Driven mode builds exactly the requested N x M shape."""
    proc = _run(nproc, {"PYCHARMM_NXM_GROUP_SIZE": str(group_size),
                        "PYCHARMM_NXM_MODE": "driven"})
    assert proc.returncode == 0, (
        f"mpirun exited {proc.returncode}.\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    reported = _groups_reported(proc)
    expected_groups = nproc // group_size
    assert sorted(g for g, _ in reported) == list(range(expected_groups)), (
        f"expected {expected_groups} group masters to report; got {reported} "
        f"(a missing group = a master that never finished).\n"
        f"--- stdout ---\n{proc.stdout}"
    )
    for group, numnode in reported:
        assert numnode == group_size, (
            f"group {group} came up as a {numnode}-node CHARMM, expected "
            f"{group_size}. The base communicator was not adopted as asked.\n"
            f"--- stdout ---\n{proc.stdout}"
        )

    assert "NXM_ALL_OK" in proc.stdout, (
        f"not every rank reached the final barrier -- one is still blocked.\n"
        f"--- stdout ---\n{proc.stdout}"
    )


@pytest.mark.parametrize("nproc,group_size", [(4, 2), (4, 4)])
def test_nxm_master_failure_releases_workers(nproc, group_size):
    """A master that raises must not strand its workers in bcast.

    This is the teardown deadlock the N x M path kept hitting: the run has
    to end, and every rank has to reach the final barrier, even though the
    master abandoned the command stream partway through.
    """
    proc = _run(nproc, {"PYCHARMM_NXM_GROUP_SIZE": str(group_size),
                        "PYCHARMM_NXM_MODE": "driven",
                        "PYCHARMM_NXM_FAIL_MASTER": "1"})
    assert proc.returncode == 0, (
        f"mpirun exited {proc.returncode} after a deliberate master "
        f"failure; the run should still shut down cleanly.\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "raised=deliberate master-side failure" in proc.stdout, (
        f"the master's exception did not propagate.\n--- stdout ---\n{proc.stdout}"
    )
    assert "NXM_ALL_OK" in proc.stdout, (
        f"a worker was left blocked after the master failed.\n"
        f"--- stdout ---\n{proc.stdout}"
    )


@pytest.mark.parametrize("nproc,group_size", [(4, 2), (6, 3)])
def test_nxm_driven_absorbs_non_spmd_script(nproc, group_size):
    """Driven mode keeps a group in step through non-SPMD master code.

    The master issues a *data-dependent* number of CHARMM commands -- a
    count derived from state only it can see. Under the raw set_mpi_comm
    model that desynchronizes the group immediately, because the other
    ranks have no way to know how many commands to issue. Driven mode has
    to absorb it: the workers replay exactly what the master broadcasts.

    Checked by having every rank report the CHARMM variable it ends up
    holding. A worker that dropped, duplicated or reordered a broadcast
    command disagrees with its master, and the group's values differ.
    """
    proc = _run(nproc, {"PYCHARMM_NXM_GROUP_SIZE": str(group_size),
                        "PYCHARMM_NXM_MODE": "driven",
                        "PYCHARMM_NXM_DIVERGE": "1"})
    assert proc.returncode == 0, (
        f"mpirun exited {proc.returncode}.\n--- stderr ---\n{proc.stderr}"
    )

    states = re.findall(
        r"NXMSTATE world=(\d+)\s+group=(\d+)\s+value=(\S+)", proc.stdout)
    assert len(states) == nproc, (
        f"expected all {nproc} ranks to report their CHARMM state; got "
        f"{len(states)}.\n--- stdout ---\n{proc.stdout}"
    )

    by_group = {}
    for world_rank, group, value in states:
        by_group.setdefault(int(group), {})[int(world_rank)] = float(value)

    for group, members in sorted(by_group.items()):
        values = set(members.values())
        assert len(values) == 1, (
            f"group {group} disagrees about the CHARMM variable: {members}. "
            f"A worker did not replay the master's command stream.\n"
            f"--- stdout ---\n{proc.stdout}"
        )
        # The master's command count is 1 + (its world rank % 3); the final
        # `set nxmtest <i>` leaves exactly that value behind.
        master_rank = min(members)
        expected = 1 + (master_rank % 3)
        assert values.pop() == float(expected), (
            f"group {group} settled on {members}, expected {expected} "
            f"from master world rank {master_rank}.\n"
            f"--- stdout ---\n{proc.stdout}"
        )

    assert "NXM_ALL_OK" in proc.stdout, (
        f"not every rank reached the final barrier.\n"
        f"--- stdout ---\n{proc.stdout}"
    )


@pytest.mark.parametrize("nproc,group_size", [(4, 2), (4, 1)])
def test_nxm_shape_spmd(nproc, group_size):
    """The raw set_mpi_comm route still builds the layout correctly.

    Driven mode is the recommended API, but set_mpi_comm() on its own
    remains supported for genuinely SPMD scripts; this pins that it still
    produces ?NUMNODE == M per group and terminates.
    """
    proc = _run(nproc, {"PYCHARMM_NXM_GROUP_SIZE": str(group_size),
                        "PYCHARMM_NXM_MODE": "spmd"})
    assert proc.returncode == 0, (
        f"mpirun exited {proc.returncode}.\n--- stderr ---\n{proc.stderr}"
    )
    reported = _groups_reported(proc)
    assert sorted(g for g, _ in reported) == list(range(nproc // group_size)), (
        f"got {reported}.\n--- stdout ---\n{proc.stdout}"
    )
    for group, numnode in reported:
        assert numnode == group_size, (
            f"group {group} came up as {numnode} nodes, expected {group_size}."
        )
    assert "NXM_ALL_OK" in proc.stdout


def test_nxm_split_rejects_uneven_shape():
    """An uneven split is refused with an explanation, not a bad layout."""
    from pycharmm import nxm

    class _FakeComm:
        """Minimal stand-in exposing only what nxm.split() queries."""

        def Get_size(self):
            """Return a rank count that 4 does not divide.

            Returns
            -------
            int
            """
            return 6

        def Get_rank(self):
            """Return this rank's position.

            Returns
            -------
            int
            """
            return 0

    with pytest.raises(ValueError, match="not a multiple of"):
        nxm.split(_FakeComm(), 4)

    with pytest.raises(ValueError, match="must be >= 1"):
        nxm.split(_FakeComm(), 0)
