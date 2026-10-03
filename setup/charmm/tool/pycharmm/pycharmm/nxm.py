# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Driven N x M (replica/worker) runs for pyCHARMM under mpi4py.

An N x M run is ``N`` replicas, each of which is an ``M``-node parallel
CHARMM.  :func:`pycharmm.set_mpi_comm` is enough to *build* that layout,
but it leaves a sharp edge: within a group, every one of the ``M`` ranks
must issue exactly the same CHARMM commands in the same order, because
CHARMM's own collectives run across the group.  A script that is not
perfectly SPMD -- one rank taking a data-dependent branch, only the group
master writing a file, an exception on a single rank -- desynchronizes
those collectives and the group hangs.

This module removes that requirement.  Instead of every rank running the
user's script, the group master runs it and the other ranks sit in a loop
executing commands the master broadcasts::

    from mpi4py import MPI
    import pycharmm
    from pycharmm import nxm

    def replica(group):                     # runs on the group master only
        group.script('open unit 1 read card name top.rtf')
        group.script('read rtf card unit 1')
        ...
        return group.script('energy')

    groups = nxm.split(MPI.COMM_WORLD, 4)   # N groups of M = 4 ranks
    result = nxm.run(groups, replica)       # non-masters never return here

Teardown is the part that historically hung, so it is explicit: the
master broadcasts a shutdown sentinel from a ``finally`` block, which
means the workers are released even when the master's own code raises.
The only failure this cannot cover is a master that dies without
unwinding (a segfault, or ``SIGKILL``); there the MPI runtime's own
job teardown is what stops the workers.

Every rank of ``comm`` must call :func:`run`.  Adopting a communicator is
collective inside CHARMM, so a rank that skips it leaves its peers
waiting (they will abort with an explanation after
``CHARMM_MPI_INIT_TIMEOUT`` seconds rather than hang forever).
"""

import sys

from . import lingo, loader

__all__ = ['NxMGroup', 'run', 'split']

# Broadcast message kinds.  Values are arbitrary but must stay stable
# between a master and its workers within one run.
_CMD = 'cmd'
_STOP = 'stop'


def split(comm, group_size):
    """Split ``comm`` into equal groups of ``group_size`` ranks.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        The communicator to divide, usually ``MPI.COMM_WORLD``.
    group_size : int
        ``M``: how many ranks make up each parallel CHARMM.

    Returns
    -------
    mpi4py.MPI.Comm
        This rank's group communicator, ready to hand to :func:`run`.

    Raises
    ------
    ValueError
        If ``group_size`` is not positive, or does not divide the size of
        ``comm`` exactly.  An uneven split is rejected rather than
        silently producing one undersized group: CHARMM would come up
        with a different node count in that group, which shows up much
        later as mismatched results or a hang.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    if group_size < 1:
        raise ValueError(f'group_size must be >= 1, got {group_size}')
    if size % group_size:
        raise ValueError(
            f'cannot split {size} ranks into groups of {group_size}: '
            f'{size} is not a multiple of {group_size}. Launch a multiple '
            f'of {group_size} ranks, or choose a group size that divides '
            f'{size}.')
    return comm.Split(color=rank // group_size, key=rank)


class NxMGroup:
    """One replica: a group of ranks running a single parallel CHARMM.

    Instances are created by :func:`run` and passed to the master
    function; do not construct one directly.

    Attributes
    ----------
    comm : mpi4py.MPI.Comm
        The group's communicator.  This is the *host's* communicator --
        CHARMM runs on a private duplicate of it, so collectives issued
        here can never be confused with CHARMM's own traffic.
    rank : int
        This rank's position within the group (the master is 0).
    size : int
        ``M``, the number of ranks in the group.
    is_master : bool
        True on the one rank that runs the user's code.
    """

    def __init__(self, comm):
        """Wrap a group communicator.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            The group's communicator, as produced by :func:`split`.
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.is_master = self.rank == 0
        self._worker_errors = []

    def script(self, text, raise_on_error=None):
        """Run one CHARMM script on every rank of the group.

        Called on the master; the command is broadcast to the workers and
        executed everywhere, keeping CHARMM's internal collectives in
        step without the caller having to write SPMD code.

        Parameters
        ----------
        text : str
            One or more lines of CHARMM script.
        raise_on_error : bool, optional
            Passed through to :func:`pycharmm.lingo.charmm_script`.

        Returns
        -------
        int
            The master's status code, as ``charmm_script`` returns.

        Raises
        ------
        RuntimeError
            If called on a worker rank. Workers execute the commands the
            master broadcasts; issuing one directly from a worker would
            desynchronize the group.
        """
        if not self.is_master:
            raise RuntimeError(
                'NxMGroup.script() may only be called on the group master '
                '(rank 0 of the group). Worker ranks execute commands the '
                'master broadcasts; calling script() on a worker would '
                'desynchronize the group.')
        self.comm.bcast((_CMD, text, raise_on_error), root=0)
        return lingo.charmm_script(text, raise_on_error=raise_on_error)

    def barrier(self):
        """Synchronize the group's ranks.

        Uses the host communicator, not CHARMM's private duplicate, so it
        orders the caller's own MPI traffic rather than CHARMM's.

        Returns
        -------
        None
        """
        self.comm.barrier()

    def _serve(self):
        """Worker rank: execute broadcast commands until told to stop.

        A worker never leaves this loop on its own.  Breaking out early --
        on an error, say -- would strand the master in its next broadcast
        with no partner, which is precisely the deadlock this module
        exists to prevent.  Errors are recorded and reported back to the
        master during shutdown instead.
        """
        deferred = None
        while True:
            message = self.comm.bcast(None, root=0)
            if message[0] == _STOP:
                break
            _, text, raise_on_error = message
            try:
                lingo.charmm_script(text, raise_on_error=raise_on_error)
            except BaseException as exc:  # noqa: BLE001 - must not escape
                # BaseException, not Exception: KeyboardInterrupt and
                # SystemExit are exactly the ones that would otherwise
                # unwind out of this loop and strand the master in its
                # next broadcast.  Anything that is not a plain Exception
                # is re-raised once the master has released us.
                self._worker_errors.append(f'rank {self.rank}: {exc!r}')
                if deferred is None and not isinstance(exc, Exception):
                    deferred = exc
        self.comm.gather(self._worker_errors, root=0)
        if deferred is not None:
            raise deferred

    def _shutdown(self, master_failed):
        """Master rank: release every worker, then collect their errors.

        Parameters
        ----------
        master_failed : bool
            True when the master is unwinding because its own code
            raised.  Worker errors are then reported as a warning rather
            than raised, so they cannot mask the original exception.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If a worker reported an error and ``master_failed`` is False.
        """
        self.comm.bcast((_STOP, None, None), root=0)
        gathered = self.comm.gather([], root=0) or []
        errors = [err for errs in gathered for err in errs]
        if not errors:
            return
        summary = '; '.join(errors)
        if master_failed:
            print(f'[pycharmm.nxm] WARNING: worker errors during a failed '
                  f'run: {summary}', file=sys.stderr)
        else:
            raise RuntimeError(
                f'CHARMM commands failed on worker ranks of this group, '
                f'while the master rank succeeded. The group\'s replica is '
                f'not trustworthy: {summary}')


def run(comm, master, *args, **kwargs):
    """Run ``master`` on the group master and drive the other ranks.

    Every rank of ``comm`` must call this. On the group master it calls
    ``master(group, *args, **kwargs)`` and returns its result; on the
    other ranks it services broadcast commands until the master finishes
    and returns ``None``.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        This rank's group communicator -- ``M`` ranks that together make
        one parallel CHARMM. :func:`split` builds one from
        ``MPI.COMM_WORLD``.
    master : callable
        Called as ``master(group, *args, **kwargs)`` on the group master
        with an :class:`NxMGroup`. Issue CHARMM commands through
        ``group.script()``.
    *args, **kwargs
        Extra arguments forwarded to ``master``.

    Returns
    -------
    object or None
        Whatever ``master`` returned, on the group master; ``None`` on
        every other rank.

    Raises
    ------
    RuntimeError
        If CHARMM is already initialized (the communicator has to be
        chosen first), or if a worker rank hit an error the master did
        not.
    """
    if loader.is_initialized():
        raise RuntimeError(
            'pycharmm.nxm.run() must be called before CHARMM is '
            'initialized, because the base communicator can only be '
            'chosen at initialization. Call it before any other '
            'pycharmm/CHARMM operation.')

    from . import set_mpi_comm

    set_mpi_comm(comm)
    group = NxMGroup(comm)

    # Initialize CHARMM on every rank now, together.  Adopting the
    # communicator duplicates it, which is collective over the group, so
    # deferring it to the first command would put a collective in the
    # middle of the master/worker message flow where the ordering is much
    # harder to reason about.
    loader.initialize()

    if not group.is_master:
        group._serve()
        return None

    master_failed = True
    try:
        result = master(group, *args, **kwargs)
        master_failed = False
        return result
    finally:
        # Unconditional: a master that raises must still release its
        # workers, or they stay blocked in bcast forever.
        group._shutdown(master_failed)
