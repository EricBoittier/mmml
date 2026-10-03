"""N x M umbrella sampling: the phi free-energy profile of alanine dipeptide.

Run with, e.g.::

    export CHARMM_DATA_DIR=/path/to/charmm/test/data
    mpirun -np 8 python nxm_umbrella_pmf.py                      # 4 x 2
    mpirun -np 8 python nxm_umbrella_pmf.py --ranks-per-group 4  # 2 x 4

The chemistry
-------------
Alanine dipeptide (ACE-ALA-NME) is the standard test case for backbone
conformational free energy.  Rotating the phi backbone dihedral carries
the molecule between the extended C5 and C7eq basins at negative phi and
the alpha_L / C7ax region at positive phi, over a barrier near phi = 0
where the carbonyls eclipse.  This script computes the potential of mean
force A(phi) for that rotation by umbrella sampling: each window
restrains phi to its own phi0 with a harmonic bias, samples Langevin
dynamics, and reports the mean phi it settled at.

What to expect, and what not to trust
-------------------------------------
The barrier near phi = 0 is robust: the defaults put it around 8 kcal/mol
above the best-sampled basin, with a second, shallower region near
phi = +75.  The *relative depths of the negative-phi basins* are not
robust, and that is worth understanding rather than papering over.  phi
and psi are coupled: C5 sits near (-157, 165) and C7eq near (-83, 78), so
a one-dimensional profile along phi is only meaningful if psi reaches
equilibrium at every phi.  A few picoseconds per window does not
guarantee that -- psi tends to stay in whichever basin the previous
window left it in -- so the negative-phi part of the profile carries some
hysteresis and the run may report either C5 or C7eq as the minimum.
Converging it means far longer windows, or a two-dimensional phi/psi PMF.
Treat this script as a demonstration of the layout that happens to do
real chemistry, not as a converged free-energy calculation.

The free energy comes from umbrella integration.  For a bias
U = 1/2 k (phi - phi0)^2, the average total force in a window vanishes at
equilibrium, so the gradient of the unbiased free energy at <phi> is

    dA/dphi = -k (<phi> - phi0)

which is then integrated across the windows.  This is the first-order
estimator (Kaestner and Thiel); it needs no external WHAM code, which
keeps this example self-contained.  For production work use WHAM or MBAR
over the full sampled distributions, and far longer sampling than the few
picoseconds per window used here.

Why N x M
---------
Umbrella sampling is embarrassingly parallel across windows, and each
window is an ordinary MD run that CHARMM can itself parallelize.  That is
exactly an N x M layout: N groups, each an M-node parallel CHARMM.
``pycharmm.nxm`` runs it in driven mode -- each group's master rank
executes ``run_windows`` below, and the other M-1 ranks of that group
replay the CHARMM commands the master broadcasts.  You therefore write
ordinary serial-looking Python (loops, branches, file I/O, an early
``return``) without having to keep every rank of the group issuing
identical commands, and a failure releases the group's workers instead
of hanging the job.

A caution on M for this particular system: alanine dipeptide in vacuum is
22 atoms, far too small to divide usefully, so ``--ranks-per-group 2``
here spends more time communicating than computing and is *slower* than
``--ranks-per-group 1``.  (CHARMM's own timers say so: a default 8-rank
run reports about 63 s in "Comm coords" against 46 s in "dynamc".)  It is
run that way anyway because the point is to demonstrate the layout.
M > 1 earns its keep when each replica is a real system -- a solvated
protein, PME, thousands of atoms -- where one replica genuinely needs
more than one core.  Choose M by the size of a single replica, and N by
how many replicas you have.

The defaults sample 24 windows for 7 ps each, which takes a few minutes.
For a quick look, try ``--windows 6 --equil-steps 500 --segments 5
--steps-per-segment 100``.

Note that the window count is set by the chemistry, not by the job size:
phi needs windows close enough to overlap whatever hardware is to hand.
The windows are laid out first and then split into contiguous blocks, one
per group, so a group with several windows simply walks along them.  Ask
for more ranks and the same calculation finishes sooner; ask for
``--ranks-per-group 1`` and it degenerates to the N x 1 ensemble model,
so the same script covers both layouts.
"""

import argparse
import math
import os
import sys

import numpy as np
from mpi4py import MPI

# Backbone dihedrals of the ACE-ALA-CT3 patched dipeptide, as atom names.
PHI_NAMES = ("CY", "N", "CA", "C")
PSI_NAMES = ("N", "CA", "C", "NT")

# The same phi atoms as a CHARMM selection, for the restraint.
PHI_SELECTION = "ADP 1 CY  ADP 1 N   ADP 1 CA  ADP 1 C"

# Harmonic bias on phi, kcal/mol/rad^2.  Stiff enough to hold a window
# against the ~8 kcal/mol barrier, soft enough that neighbouring windows
# still overlap.
FORCE_CONSTANT = 100.0

TEMPERATURE = 300.0
EQUIL_STEPS = 2000        # 2 ps
SAMPLE_SEGMENTS = 20      # phi is read once per segment
STEPS_PER_SEGMENT = 250   # 0.25 ps

# Umbrella windows across the phi range. 24 windows over 360 degrees is a
# 15 degree spacing; with k = 100 kcal/mol/rad^2 each window has a spread
# of about 4 degrees, so neighbours overlap comfortably.
WINDOWS = 24

# Timestep, ps. 1 fs is safe for this molecule without SHAKE.
TIMESTEP = 0.001


def wrap_degrees(angle):
    """Fold an angle into the principal range.

    Parameters
    ----------
    angle : float
        An angle or angle difference, in degrees.

    Returns
    -------
    float
        The equivalent angle in (-180, 180].
    """
    return (angle + 180.0) % 360.0 - 180.0


def dihedral_degrees(p0, p1, p2, p3):
    """Signed dihedral angle through four points.

    Parameters
    ----------
    p0, p1, p2, p3 : numpy.ndarray
        Cartesian coordinates, shape (3,), of the four atoms in bonded
        order.

    Returns
    -------
    float
        The dihedral in degrees, signed by the IUPAC convention and in
        the range (-180, 180].
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Project b0 and b2 onto the plane perpendicular to the b1 axis, then
    # read off the angle between the projections.
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def atom_indices(names):
    """Row indices in the coordinate table for the given atom names.

    Names are unique here because the example builds a single residue; a
    multi-residue system would also need to match segid/resid.

    Parameters
    ----------
    names : sequence of str
        CHARMM atom names to locate.

    Returns
    -------
    list of int
        Zero-based row indices into the coordinate table, in the same
        order as ``names``.
    """
    from pycharmm import psf

    atypes = list(psf.get_atype())
    return [atypes.index(name) for name in names]


def measure_dihedral(indices):
    """Current value of the dihedral spanning ``indices``, in degrees.

    Reads coordinates straight out of CHARMM into numpy rather than going
    back through the script interface -- the group's ranks all hold the
    same coordinates, so the master can measure without any extra
    communication.

    Parameters
    ----------
    indices : sequence of int
        Four coordinate-table row indices, as returned by
        :func:`atom_indices`.

    Returns
    -------
    float
        The dihedral in degrees.
    """
    from pycharmm import coor

    positions = coor.get_positions().to_numpy()
    return dihedral_degrees(*(positions[i] for i in indices))


def build_system(group, data_dir):
    """Read the force field and build alanine dipeptide in this group.

    Parameters
    ----------
    group : pycharmm.nxm.NxMGroup
        The calling group; commands are broadcast to all of its ranks.
    data_dir : str
        Directory holding ``top_all36_prot.rtf`` and
        ``par_all36_prot.prm``.

    Returns
    -------
    None
    """
    group.script(f"""
        open unit 1 read card name {os.path.join(data_dir, 'top_all36_prot.rtf')}
        read rtf card unit 1
        close unit 1
        open unit 1 read card name {os.path.join(data_dir, 'par_all36_prot.prm')}
        read param card flex unit 1
        close unit 1

        read sequence card
        * alanine dipeptide
        *
        1
        ALA
        generate ADP first ACE last CT3 setup warn
        ic param
        ic seed 1 CAY 1 CY 1 N
        ic build
    """)
    # Vacuum, no cutoffs worth speaking of: this is a 22-atom molecule and
    # the point of the example is the layout, not the solvent model.
    group.script("""
        nbonds atom fswitch vswitch cutnb 990.0 ctofnb 950.0 ctonnb 900.0
    """)
    group.script("mini abnr nstep 200 tolgrd 0.001")


def run_windows(group, data_dir, phi0_list, sampling):
    """Sample every umbrella window assigned to this group.

    Runs on the group master only. The other ranks of the group are inside
    ``nxm``'s worker loop replaying every ``group.script`` issued here, so
    each window really is an M-node parallel CHARMM.

    The molecule is built once and then re-restrained per window, walking
    phi across the assigned targets.

    Parameters
    ----------
    group : pycharmm.nxm.NxMGroup
        The calling group; commands are broadcast to all of its ranks.
    data_dir : str
        Directory holding the topology and parameter files.
    phi0_list : list of tuple
        The windows assigned to this group, as ``(window index, phi0)``
        pairs with phi0 in degrees.
    sampling : tuple of int
        ``(equilibration steps, production segments, steps per segment)``.

    Returns
    -------
    list of tuple
        One ``(phi0, mean phi, mean psi)`` triple per window, all in
        degrees.
    """
    equil_steps, segments, steps_per_segment = sampling

    build_system(group, data_dir)
    phi_idx = atom_indices(PHI_NAMES)
    psi_idx = atom_indices(PSI_NAMES)
    group.script("scalar fbeta set 5.0 select all end")

    def dynamics(nstep, seed, nprint):
        """One Langevin run continuing from the current coordinates.

        Each call assigns fresh Maxwell velocities at TEMPERATURE and
        carries the coordinates over from the previous call. Repeated
        velocity randomization is itself a valid canonical sampling move,
        so segmenting this way costs nothing statistically and keeps the
        example free of restart-file bookkeeping.

        Parameters
        ----------
        nstep : int
            Dynamics steps to run.
        seed : int
            First of the four RNG seeds CHARMM expects.
        nprint : int
            Step interval for CHARMM's energy output.

        Returns
        -------
        None
        """
        group.script(f"""
            dynamics langevin leap start -
               nstep {nstep} timestep {TIMESTEP} -
               firstt {TEMPERATURE} finalt {TEMPERATURE} tbath {TEMPERATURE} -
               iseed {seed} {seed + 1} {seed + 2} {seed + 3} -
               inbfrq -1 ihbfrq 0 ilbfrq 10 -
               iprfrq {nprint} nprint {nprint} nsavc 0 nsavv 0 -
               iasors 1 iasvel 1 iscvel 0 ichecw 0
        """)

    results = []
    for window_index, phi0 in phi0_list:
        # Replace the previous window's bias with this one, then minimize:
        # the restraint drags phi from wherever the last window left it
        # into this window before any dynamics start. Windows are handed
        # out in order, so that is a short pull between neighbours.
        group.script("cons cldh")
        group.script(
            f"cons dihe {PHI_SELECTION} force {FORCE_CONSTANT} min {phi0}")
        group.script("mini abnr nstep 500 tolgrd 0.001")

        seed = 314159 + 10000 * window_index
        dynamics(equil_steps, seed, equil_steps)

        # Production: sample phi/psi at the end of each short segment.
        phi_samples = []
        psi_samples = []
        for segment in range(segments):
            dynamics(steps_per_segment, seed + 4 * (segment + 1),
                     steps_per_segment)
            phi_samples.append(measure_dihedral(phi_idx))
            psi_samples.append(measure_dihedral(psi_idx))

        # Average as displacements from a reference so the mean is not
        # corrupted by samples that wrapped across +/-180, then fold the
        # result back into the principal range.
        mean_phi = wrap_degrees(
            phi0 + sum(wrap_degrees(p - phi0) for p in phi_samples)
            / len(phi_samples))
        psi_ref = psi_samples[0]
        mean_psi = wrap_degrees(
            psi_ref + sum(wrap_degrees(p - psi_ref) for p in psi_samples)
            / len(psi_samples))
        results.append((phi0, mean_phi, mean_psi))

    return results


def integrate_pmf(windows):
    """Umbrella integration: mean forces -> A(phi), zeroed at its minimum.

    Parameters
    ----------
    windows : list of tuple
        ``(phi0, mean phi, mean psi)`` triples in degrees, in any order.

    Returns
    -------
    list of tuple
        ``(mean phi, mean psi, free energy)`` sorted by phi, with the free
        energy in kcal/mol and zeroed at its lowest window.
    """
    windows = sorted(windows, key=lambda w: w[1])
    phis = [math.radians(mean_phi) for _, mean_phi, _ in windows]
    # dA/dphi = -k (<phi> - phi0), with the displacement taken the short
    # way round the circle.
    grads = [
        -FORCE_CONSTANT * math.radians(wrap_degrees(mean_phi - phi0))
        for phi0, mean_phi, _ in windows
    ]

    pmf = [0.0]
    for i in range(1, len(phis)):
        dphi = phis[i] - phis[i - 1]
        pmf.append(pmf[-1] + 0.5 * (grads[i] + grads[i - 1]) * dphi)

    floor = min(pmf)
    return [
        (windows[i][1], windows[i][2], pmf[i] - floor)
        for i in range(len(windows))
    ]


def main():
    """Lay out the windows, run them, and print the profile.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 if the job was launched with
        a rank count or window count the calculation cannot use.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks-per-group", type=int, default=2,
                        help="M: MPI ranks forming each parallel CHARMM "
                             "(default 2; 1 gives the N x 1 ensemble)")
    parser.add_argument("--windows", type=int, default=WINDOWS,
                        help=f"umbrella windows across the phi range, shared "
                             f"out over the groups (default {WINDOWS})")
    parser.add_argument("--phi-min", type=float, default=-180.0)
    parser.add_argument("--phi-max", type=float, default=180.0)
    parser.add_argument("--equil-steps", type=int, default=EQUIL_STEPS,
                        help=f"equilibration steps per window "
                             f"(default {EQUIL_STEPS})")
    parser.add_argument("--segments", type=int, default=SAMPLE_SEGMENTS,
                        help=f"production segments per window, one phi "
                             f"sample each (default {SAMPLE_SEGMENTS})")
    parser.add_argument("--steps-per-segment", type=int,
                        default=STEPS_PER_SEGMENT,
                        help=f"dynamics steps per segment "
                             f"(default {STEPS_PER_SEGMENT})")
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    ranks_per_group = args.ranks_per_group

    if world.Get_size() % ranks_per_group:
        if world.Get_rank() == 0:
            print(f"Launch a multiple of {ranks_per_group} ranks "
                  f"(got {world.Get_size()}).", file=sys.stderr)
        return 1

    data_dir = os.environ.get("CHARMM_DATA_DIR")
    if not data_dir:
        if world.Get_rank() == 0:
            print("Set CHARMM_DATA_DIR to the CHARMM test/data directory.",
                  file=sys.stderr)
        return 1

    n_groups = world.Get_size() // ranks_per_group
    if args.windows < 3:
        if world.Get_rank() == 0:
            print(f"Need at least 3 windows to integrate a profile "
                  f"(got {args.windows}).", file=sys.stderr)
        return 1

    # Windows are a property of the chemistry, not of the job size: the
    # phi range needs fine enough spacing that neighbouring windows
    # overlap, however many ranks happen to be available.  So lay out the
    # windows first and hand each group a share of them.  phi is periodic,
    # so the last window is one spacing short of phi_max.
    spacing = (args.phi_max - args.phi_min) / args.windows
    all_windows = [(i, args.phi_min + spacing * i) for i in range(args.windows)]

    # Each group gets a *contiguous* block, not a round-robin stride, so
    # that consecutive windows within a group are neighbours in phi: every
    # window then starts from the previous window's final structure one
    # spacing away, rather than being dragged across the whole range.  The
    # blocks differ in length by at most one window.
    group_index = world.Get_rank() // ranks_per_group
    base, extra = divmod(args.windows, n_groups)
    start = group_index * base + min(group_index, extra)
    count = base + (1 if group_index < extra else 0)
    my_windows = all_windows[start:start + count]

    from pycharmm import nxm

    # A communicator of just the group masters, to collect the windows.
    # Built before CHARMM starts, and never handed to CHARMM -- CHARMM
    # gets the group communicator, on a private duplicate of its own.
    is_master = world.Get_rank() % ranks_per_group == 0
    masters = world.Split(
        color=0 if is_master else MPI.UNDEFINED, key=group_index)

    group_comm = nxm.split(world, ranks_per_group)

    if world.Get_rank() == 0:
        print(f"# alanine dipeptide phi PMF: {args.windows} windows over "
              f"{n_groups} groups x {ranks_per_group} ranks, "
              f"k = {FORCE_CONSTANT} kcal/mol/rad^2")
        print(f"# phi0 from {args.phi_min:.1f} to "
              f"{args.phi_max - spacing:.1f} deg, spacing {spacing:.1f} deg")
        sys.stdout.flush()

    # Every rank calls run(); only the group masters execute run_windows.
    sampling = (args.equil_steps, args.segments, args.steps_per_segment)
    result = nxm.run(group_comm, run_windows, data_dir, my_windows, sampling)

    if is_master:
        gathered = masters.gather(result, root=0)
        if masters.Get_rank() == 0:
            windows = [w for group_result in gathered for w in group_result]
            print(f"\n# {'phi':>8} {'psi':>8} {'A(phi)':>10}")
            print(f"# {'(deg)':>8} {'(deg)':>8} {'kcal/mol':>10}")
            for mean_phi, mean_psi, free_energy in integrate_pmf(windows):
                print(f"  {mean_phi:8.1f} {mean_psi:8.1f} {free_energy:10.2f}")
            print("\n# A(phi) is zeroed at its lowest window. The barrier "
                  "near phi = 0 is the\n# robust feature; the relative "
                  "depths of the negative-phi basins depend on\n# psi "
                  "equilibrating, which needs far longer windows than the "
                  "defaults.")
            sys.stdout.flush()

    world.barrier()
    return 0


if __name__ == "__main__":
    # Any uncaught exception on one rank must take the whole job down.
    # Otherwise the ranks that are still healthy sit in the next
    # world-level collective (the barrier at the end of main) waiting for
    # a rank that has already gone -- turning a plain Python error into a
    # hang.  nxm.run() covers the ranks inside a group; this covers the
    # job.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
