"""pyCHARMM under MPI: an SPMD ensemble with mpi4py.

Run with, e.g.::

    mpirun -np 4 python mpi_ensemble.py

Every rank runs this same script.  In the default (and fully supported)
model, each rank hosts its own independent *serial* CHARMM on
``MPI_COMM_SELF`` (``?numnode == 1`` everywhere), so the ranks share
nothing inside CHARMM -- you coordinate them explicitly with mpi4py.
That is the natural fit for ensemble methods: replica exchange,
free-energy windows, multi-start minimization, temperature ladders, etc.

This example runs a temperature ladder: rank ``r`` runs a short Langevin
MD of alanine dipeptide at its own temperature, and rank 0 gathers the
mean potential energy from every rank via mpi4py and prints the
``T`` vs ``<E>`` profile.  No CHARMM broadcast/reduction is involved --
all cross-rank communication is done with mpi4py.

Requires the CHARMM topology/parameter files; point ``CHARMM_DATA_DIR``
at a directory containing ``top_all36_prot.rtf`` and
``par_all36_prot.prm`` (e.g. the ``test/data`` directory of the source
tree).  See doc/pycharmm.info, node "Parallel", for the full model.
"""

import os
import sys

from mpi4py import MPI

# ---------------------------------------------------------------------------
# 1. Rank/size come from mpi4py, NEVER from CHARMM.  Under pyCHARMM,
#    CHARMM's ?numnode / ?mynode are 1 / 0 on every rank and do not
#    describe the MPI world.
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# A temperature ladder spanning 280..380 K, one rung per rank.
if size == 1:
    temperatures = [300.0]
else:
    temperatures = [280.0 + 100.0 * i / (size - 1) for i in range(size)]
my_temp = temperatures[rank]

data_dir = os.environ.get("CHARMM_DATA_DIR")
if not data_dir:
    if rank == 0:
        print("Set CHARMM_DATA_DIR to a directory with "
              "top_all36_prot.rtf / par_all36_prot.prm.", file=sys.stderr)
    sys.exit(0)

# ---------------------------------------------------------------------------
# 2. Import pyCHARMM and build the system.  Every rank does this
#    independently on its own serial CHARMM.  Name any output/scratch
#    files by the mpi4py rank so ranks never collide.
# ---------------------------------------------------------------------------
from pycharmm import (                             # noqa: E402
    read, gen, ic, energy, minimize, lingo,
    NonBondedScript,
)

read.rtf(os.path.join(data_dir, "top_all36_prot.rtf"))
read.prm(os.path.join(data_dir, "par_all36_prot.prm"), flex=True)

read.sequence_string("ALA")
gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
ic.prm_fill(False)
ic.seed(1, "CAY", 1, "CY", 1, "N")
ic.build()

NonBondedScript(
    cutnb=18.0, ctonnb=15.0, ctofnb=13.0, eps=1.0, cdie=True,
    atom=True, vatom=True, fswitch=True, vfswitch=True,
).run()

minimize.run_abnr(nstep=200, tolenr=1e-3, tolgrd=1e-3)

# ---------------------------------------------------------------------------
# 3. The per-rank "useful work": a short Langevin MD at this rank's
#    temperature.  Seed the RNG by rank so the trajectories differ.
#    (Driven through lingo here for brevity; the pycharmm.dyn module
#    exposes the same controls.)
# ---------------------------------------------------------------------------
lingo.charmm_script(f"""
scalar fbeta set 5.0 select all end
dynamics langevin leap start -
   nstep 2000 timestep 0.001 -
   firstt {my_temp} finalt {my_temp} tbath {my_temp} -
   iseed {12345 + rank} -
   inbfrq -1 ihbfrq 0 -
   iprfrq 1000 nprint 1000 -
   iasors 1 iasvel 1 iscvel 0 ichecw 0
""")
mean_e = energy.get_total()

# ---------------------------------------------------------------------------
# 4. Coordinate across ranks with mpi4py (gather), not with CHARMM.
# ---------------------------------------------------------------------------
results = comm.gather((my_temp, mean_e), root=0)
if rank == 0:
    print("\n# temperature ladder (pyCHARMM x mpi4py ensemble)")
    print("#   T (K)     final E (kcal/mol)")
    for temp, e in sorted(results):
        print(f"  {temp:8.1f}   {e:14.4f}")

# ---------------------------------------------------------------------------
# N x M: groups of parallel CHARMMs.
# ---------------------------------------------------------------------------
# The block above is N x 1 -- N independent serial CHARMMs.  To make each
# ensemble member itself an M-node *parallel* CHARMM (an N x M layout),
# use pycharmm.nxm, which splits the world into groups and drives each
# group from its master rank:
#
#     from pycharmm import nxm
#
#     def member(group):                # runs on the group master only
#         group.script('...')           # broadcast to the whole group
#         return group.script('energy')
#
#     groups = nxm.split(comm, 2)       # M = 2 ranks per CHARMM
#     result = nxm.run(groups, member)  # None on non-master ranks
#
# Driven that way, only the master runs your Python and the other ranks
# replay the CHARMM commands it broadcasts, so the group cannot fall out
# of step no matter what your script does, and a failure releases the
# workers instead of hanging.  See examples/nxm_umbrella_pmf.py for a
# complete calculation, and doc/pycharmm.info.
#
# set_mpi_comm() on its own also establishes the layout (each group
# reports ?numnode == M), but then every rank of a group must issue
# identical CHARMM commands in the same order -- CHARMM's collectives run
# across the group, so any divergence hangs it.  Prefer nxm.run() unless
# you are deliberately writing SPMD code.
