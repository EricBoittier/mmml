"""Per-rank worker for the N x M layout tests (test_nxm.py).

This is NOT a pytest module -- it is launched under
``mpirun -np N python _nxm_worker.py`` by the tests, which inspect its
stdout.  It builds an N x M layout (N replicas, each an M-node parallel
CHARMM) and drives it through :mod:`pycharmm.nxm`, then prints one
``NXM ...`` line per group master and a final ``NXM_ALL_OK`` from world
rank 0.

Behaviour is selected by environment variables:
  PYCHARMM_NXM_GROUP_SIZE = "M" (required)
      Ranks per CHARMM group.  World size must be a multiple of it.
  PYCHARMM_NXM_MODE = "driven" (default) | "spmd"
      "driven" uses nxm.run(); "spmd" uses set_mpi_comm() directly with
      every rank issuing identical commands (the older, sharper API).
  PYCHARMM_NXM_FAIL_MASTER = "" (default) | "1"
      Raise inside the master function to prove teardown still releases
      every worker instead of leaving one blocked in bcast.
  PYCHARMM_NXM_DIVERGE = "" (default) | "1"
      In "driven" mode, have the master issue a data-dependent number of
      extra CHARMM commands -- code that would desynchronize an SPMD
      group, and which the worker loop must absorb.  Every rank then
      reports the resulting CHARMM variable so the test can check the
      workers really replayed all of them.
"""

import os
import sys

from mpi4py import MPI

import pycharmm
from pycharmm import lingo, nxm

_GROUP_SIZE = int(os.environ.get("PYCHARMM_NXM_GROUP_SIZE", "1"))
_MODE = os.environ.get("PYCHARMM_NXM_MODE", "driven")
_FAIL_MASTER = os.environ.get("PYCHARMM_NXM_FAIL_MASTER", "") == "1"
_DIVERGE = os.environ.get("PYCHARMM_NXM_DIVERGE", "") == "1"


class MasterFailure(RuntimeError):
    """Raised on purpose to exercise the shutdown path."""


def _replica(group):
    """Run on each group master; workers execute what it broadcasts.

    Parameters
    ----------
    group : pycharmm.nxm.NxMGroup
        The calling group.

    Returns
    -------
    tuple of int
        ``(?NUMNODE, ?MYNODE)`` as CHARMM reports them on the master.

    Raises
    ------
    MasterFailure
        When PYCHARMM_NXM_FAIL_MASTER is set, to exercise teardown.
    """
    group.script("set nxmtest 1")
    if _FAIL_MASTER:
        # The workers are blocked in bcast right now.  nxm.run's finally
        # must still release them, or the job hangs here forever.
        raise MasterFailure("deliberate master-side failure")
    group.script("set nxmtest 2")
    numnode = lingo.get_charmm_builtins().get("NUMNODE")
    mynode = lingo.get_charmm_builtins().get("MYNODE")
    return numnode, mynode


def _diverging_replica(group):
    """Master-only code whose command count depends on master-only data.

    Under the raw SPMD model this desynchronizes the group instantly: the
    other ranks have no way to know how many commands were issued.  Driven
    mode has to absorb it, because the workers simply replay whatever the
    master broadcasts.

    Parameters
    ----------
    group : pycharmm.nxm.NxMGroup
        The calling group.

    Returns
    -------
    int
        How many extra commands were issued, which is also the value the
        CHARMM variable NXMTEST is left holding on every rank.
    """
    group.script("set nxmtest 0")
    # A count only this rank can know.  Deliberately derived from Python
    # state the workers never see.
    extra = 1 + (MPI.COMM_WORLD.Get_rank() % 3)
    for i in range(1, extra + 1):
        group.script(f"set nxmtest {i}")
    return extra


def _run_driven(world, group_comm):
    """Drive this group through pycharmm.nxm.

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        The world communicator, used only for labelling output.
    group_comm : mpi4py.MPI.Comm
        This rank's group communicator.

    Returns
    -------
    tuple of int or None
        ``(?NUMNODE, ?MYNODE)`` on a group master in the ordinary case;
        None on worker ranks, and on every rank in the diverging case
        (which reports through its own NXMSTATE line instead).
    """
    if _DIVERGE:
        nxm.run(group_comm, _diverging_replica)
        # Every rank -- master and worker alike -- reports the CHARMM
        # variable it now holds.  If the worker loop dropped or duplicated
        # a broadcast command, the workers disagree with their master.
        value = lingo.get_charmm_variable("NXMTEST")
        print(f"NXMSTATE world={world.Get_rank()} "
              f"group={world.Get_rank() // _GROUP_SIZE} value={value}",
              flush=True)
        return None

    result = nxm.run(group_comm, _replica)
    if group_comm.Get_rank() != 0:
        # Worker ranks return None from nxm.run once released.
        return None
    numnode, mynode = result
    return numnode, mynode


def _run_spmd(world, group_comm):
    """The un-driven API: every rank issues identical commands itself.

    This is the raw set_mpi_comm route.  It is correct only because every
    rank runs exactly the same commands in the same order; the driven mode
    above exists so callers do not have to guarantee that.

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        The world communicator (unused; kept for signature symmetry with
        :func:`_run_driven`).
    group_comm : mpi4py.MPI.Comm
        This rank's group communicator, handed straight to CHARMM.

    Returns
    -------
    tuple of int or None
        ``(?NUMNODE, ?MYNODE)`` on a group master, None on other ranks.
    """
    pycharmm.set_mpi_comm(group_comm)
    lingo.charmm_script("set nxmtest 1")
    lingo.charmm_script("set nxmtest 2")
    numnode = lingo.get_charmm_builtins().get("NUMNODE")
    mynode = lingo.get_charmm_builtins().get("MYNODE")
    if group_comm.Get_rank() != 0:
        return None
    return numnode, mynode


def main():
    """Build the layout, run it in the selected mode, and report.

    Returns
    -------
    None
    """
    world = MPI.COMM_WORLD
    group_comm = nxm.split(world, _GROUP_SIZE)
    ngroups = world.Get_size() // _GROUP_SIZE

    failed = None
    try:
        if _MODE == "driven":
            reported = _run_driven(world, group_comm)
        else:
            reported = _run_spmd(world, group_comm)
    except MasterFailure as exc:
        # Expected in the FAIL_MASTER case: what matters is that we got
        # here at all (workers released) rather than hanging.
        reported = None
        failed = str(exc)

    if group_comm.Get_rank() == 0 and not _DIVERGE:
        if failed is not None:
            print(
                f"NXM group={world.Get_rank() // _GROUP_SIZE} "
                f"groups={ngroups} raised={failed}",
                flush=True,
            )
        else:
            numnode, mynode = reported
            print(
                f"NXM group={world.Get_rank() // _GROUP_SIZE} "
                f"groups={ngroups} numnode={numnode} mynode={mynode}",
                flush=True,
            )

    # Every rank must reach this barrier.  A rank still stuck in a worker
    # loop, or one that fell out of the group, never does -- and the test
    # sees a timeout rather than a false pass.
    world.barrier()
    if world.Get_rank() == 0:
        print("NXM_ALL_OK", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
