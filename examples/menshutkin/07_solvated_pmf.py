#!/usr/bin/env python3
"""Solvated 1D PMF along xi = r(C-Cl) - r(C-N), from reactants to the SSIP.

Sequential umbrella sampling: the box is packed once around a transition-state
solute, then windows are walked *outward* from xi = 0 in both directions, each
seeded from the final frame of its neighbour. This is what Turan et al. did
("restarting from previous window structures"), and it matters more here than in
the gas phase -- the solvent shell has to follow the solute continuously. Packing
a fresh box per window would leave every window's shell unequilibrated, and
teleporting the solute inside a fixed box at liquid density just produces
overlaps (there is nowhere for the solvent to go).

The window range is capped from the checkpoint's own radial cutoff, not by a
constant here: beyond it the chloride leaves the message-passing graph, charge
conservation fails (q(rest) jumps to +8.6 e), the energy jumps 600 kcal/mol and
then stops responding to geometry entirely. For model_ext.json (cutoff 8 A) that
gives xi <= +6, which already covers the chemistry -- the contact ion pair sits
at r(C-Cl) ~ 3 A (xi ~ +1.3) and the solvent-separated pair at 5-7 A (xi ~ +3.5
to +5.5). See the README, "How far can the model separate the ions?".

Three restraints, not one. xi = r(C-Cl) - r(C-N) fixes a single number and
leaves the rest of the geometry free, and two of those directions matter:
the methyl can drift away from both partners (walled by min(r) <= 2.25 A) and
the chloride can swing round to hydrogen-bond with the ammonium (walled by
angle(N-C-Cl) >= 130 deg). Both were observed producing finite, smooth, and
completely wrong sampling before the walls existed.

Only xi(t) is written per window, not full trajectories. For umbrella sampling
where every window shares one unbiased Hamiltonian and differs only by the bias,
MBAR needs nothing else: the reduced potentials enter only through differences
u_l - u_k = beta*(W_l - W_k), which depend on the collective variable alone.

    source examples/menshutkin/_env.sh
    python examples/menshutkin/07_solvated_pmf.py --solvent water
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from gpu_pairs import make_static_pair_fn  # noqa: E402
from jaxmd_box import build_jaxmd_solvated_system  # noqa: E402
from solute import (  # noqa: E402
    IDX_C,
    IDX_CL,
    IDX_N,
    SOLUTE_N_ATOMS,
    SOLVENTS,
    atomic_mass,
    load_model,
    solute_geometry_at_xi,
)
from solvent_models import get_solvent_model  # noqa: E402

EV_TO_KCAL = 23.060547830619027
# Largest window centre. This used to be set by the 8.0 A cutoff of
# model_ext.json, past which the chloride left the graph and the energy froze.
# The longrange checkpoint (cutoff 14, electrostatics to 20 A) has no such
# cliff -- it varies smoothly to 13 A with charge conserved to 0.07 e -- so the
# limit is now the smaller of the training data (max r(C-Cl) = 13.08 A) and the
# minimum-image convention (r < L/2, i.e. 15 A in a 30 A box). At xi = +8 the
# chloride sits near 9.5 A, inside both. Raise only with a box to match.
#
# Note this range already covers the chemistry of interest: the CIP is at
# r(C-Cl) ~ 3 A (xi ~ +1.3) and the solvent-separated ion pair at r ~ 5-7 A
# (xi ~ +3.5 to +5.5).
XI_MAX_SAFE = 8.0   # fallback only; the real cap is derived from the model
# Lowest energy anywhere in the training set; a window that dips below it has
# left the fitted surface and its samples are not physical.
TRAIN_MIN_EV = -30.158


def window_ladder(xi_min: float, xi_max: float, fine_to: float,
                  fine: float, coarse: float) -> np.ndarray:
    """Fine spacing through the barrier and CIP, coarser through the tail."""
    a = np.arange(xi_min, min(fine_to, xi_max) + 1e-9, fine)
    b = np.arange(min(fine_to, xi_max) + coarse, xi_max + 1e-9, coarse)
    return np.round(np.concatenate([a, b]), 4)


def n_solvent_for_density(resi: str, density: float, box_size: float) -> int:
    """Molecules needed to fill the *cube* at the experimental density."""
    from mmml.analysis.residue_geometry import load_residue_monomer_atoms

    molar_mass = float(sum(load_residue_monomer_atoms(resi, generate=True).get_masses()))
    return int(round(density * 1e-3 * box_size**3 * 6.02214076e23 / (1e24 * molar_mass)))


def main() -> int:
    # Python buffers stdout in 8 KB blocks when it is not a tty, and its own
    # io.TextIOWrapper ignores `stdbuf -oL` -- so a long run appears to hang
    # with nothing in the log for tens of minutes. Reconfigure here rather than
    # relying on the launcher remembering PYTHONUNBUFFERED.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover - very old Python
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solvent", choices=sorted(SOLVENTS), default="water")
    p.add_argument("--box-size", type=float, default=None)
    p.add_argument("--n-solvent", type=int, default=None,
                   help="Override the density-derived solvent count")
    p.add_argument("--xi-min", type=float, default=-1.3)
    p.add_argument("--xi-max", type=float, default=XI_MAX_SAFE)
    p.add_argument("--fine-to", type=float, default=2.0,
                   help="Fine spacing up to this xi, coarser beyond")
    p.add_argument("--fine", type=float, default=0.1)
    p.add_argument("--coarse", type=float, default=0.25)
    p.add_argument("--k-ev", type=float, default=6.505)
    # Flat-bottom wall on r(C-Cl)+r(C-N), the direction the xi bias cannot
    # see. Defaults bracket the training envelope over xi in [-1.3, +2.5]
    # (p1 3.98, p99 5.52, max 5.65 A) with a small margin.
    p.add_argument("--sum-min", type=float, default=3.8)
    # Just above the training maximum of 2.18 A. A first attempt at 2.35 A
    # with k = 10 eV/A^2 was violated rather than respected: the trajectory
    # sat at min(r) = 2.45-2.51 A, because the penalty there is only
    # 0.5*10*0.15^2 = 0.11 eV, nothing against the slope of the fitted
    # surface. A flat-bottom wall costs nothing inside its bottom, so it may
    # as well be stiff enough to hold.
    p.add_argument("--bond-r-max", type=float, default=2.25,
                   help="Upper bound on min(r(C-Cl), r(C-N)); training max 2.18 A")
    p.add_argument("--angle-min-deg", type=float, default=130.0,
                   help="Lower bound on the N-C-Cl angle; healthy windows "
                        "sit at 165-173 deg, reoriented ones at ~70")
    p.add_argument("--wall-k-ev", type=float, default=100.0,
                   help="Wall stiffness. At 100 eV/A^2 and 12 amu the wall "
                        "oscillation period is ~22 fs, still ~90 steps at "
                        "dt = 0.25 fs, so it does not strain the integrator.")
    p.add_argument(
        "--embedding", choices=("electrostatic", "mechanical"), default="electrostatic",
        help="electrostatic: solute-solvent Coulomb uses the ML model's "
             "fluctuating charges q_i(R), which is what lets the solvent respond "
             "to the ion pair forming. mechanical: fixed CGenFF charges on the "
             "solute (the A/B comparison; expect much less catalysis).",
    )
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--equil-ps", type=float, default=1.0, help="Per window, discarded")
    p.add_argument("--prod-ps", type=float, default=2.0, help="Per window, kept")
    # 300 was far too few. From a lattice-packed box at +296 eV, 300 FIRE steps
    # reach about -270 eV and 8000 reach -376 eV, and the difference is not
    # cosmetic: with 300 the residual strain overwhelmed the umbrella restraint,
    # so the window centred at xi = 0.00 actually sampled <xi> = +0.94.
    p.add_argument("--freeze-charge-forces", action="store_true",
                   help="Drop dq/dR from the force. Charges are still "
                        "recomputed every step and still enter the energy, "
                        "but the feedback loop that destabilises full "
                        "coupling is broken. An approximation: forces are "
                        "then not the exact gradient of the energy.")
    p.add_argument("--ramp-stages", type=int, default=5,
                   help="Stages over which to switch the ML/MM "
                        "electrostatics on during equilibration; 0 disables")
    p.add_argument("--minimize-steps", type=int, default=8000)
    # Minimisation gets its own step size rather than inheriting --dt-fs. The
    # driver hands FIRE dt_max = the MD timestep, which is sized for dynamics on
    # an equilibrated box; at 0.25 fs the minimiser diverged on the 30 A water
    # box even after the strained contacts had been cleared, while the same
    # settings were fine at 26 A. The first eighth runs at a fifth of this.
    p.add_argument("--minimize-dt-fs", type=float, default=0.05,
                   help="FIRE step size for minimisation, independent of --dt-fs")
    p.add_argument("--record-every", type=int, default=10)
    # Trajectories are not needed by MBAR -- it reconstructs the profile from
    # xi(t) alone -- but they are the only way to see what a window is actually
    # doing. The solute is 9 atoms, so keeping every recorded frame of it costs
    # nothing; the solvent is 2700 and is strided.
    p.add_argument("--save-traj", choices=("none", "solute", "full"),
                   default="solute",
                   help="Per-window extxyz trajectories written to <out>/traj/")
    p.add_argument("--traj-stride", type=int, default=20,
                   help="Stride over recorded frames for --save-traj full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    # The cap belongs to the checkpoint, not to this file. Past the model's
    # radial cutoff the chloride leaves the message-passing graph entirely:
    # charge conservation fails and the energy stops responding to geometry.
    # r(C-N) is about 1.5 A at the product end, so xi = r(C-Cl) - r(C-N) reaches
    # the cutoff when xi ~ cutoff - 1.5; half an Angstrom of margin on top.
    from solute import load_model as _lm

    _model, _ = _lm()
    _cutoff = float(getattr(_model, "cutoff", 8.0))
    xi_cap = min(XI_MAX_SAFE, _cutoff - 2.0)
    print(f"model cutoff {_cutoff:.1f} A -> xi capped at {xi_cap:+.1f} A")
    if args.xi_max > xi_cap:
        raise SystemExit(
            f"--xi-max {args.xi_max} exceeds {xi_cap:.1f} A for a model with "
            f"cutoff {_cutoff:.1f} A: past that the chloride leaves the graph, "
            "charge conservation fails and the energy freezes (see the README, "
            "'How far can the model separate the ions?')."
        )

    resi, density, default_side = SOLVENTS[args.solvent]
    box_size = args.box_size if args.box_size is not None else default_side
    artifacts = Path(os.environ.get("MENSH_ARTIFACTS", REPO_ROOT / "artifacts/menshutkin"))
    out = args.output_dir or artifacts / "pmf" / args.solvent
    out.mkdir(parents=True, exist_ok=True)

    centres = window_ladder(args.xi_min, args.xi_max, args.fine_to,
                            args.fine, args.coarse)
    print(f"solvent      {args.solvent} ({resi})   box {box_size} A")
    print(f"windows      {len(centres)}: {centres[0]:+.2f} .. {centres[-1]:+.2f} A "
          f"({args.fine} A to {args.fine_to}, then {args.coarse} A)")
    print(f"per window   {args.equil_ps} ps equil + {args.prod_ps} ps production")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    # CHARMM-free construction. The CHARMM-backed builders were abandoned after
    # repeated opaque failures (one build per process; a Packmol stage whose
    # output scored -9.5e5 eV; overlap diagnostics that contradicted each
    # other). Here the lattice builder produces a box whose contacts are
    # asserted before it is handed to the integrator, using solvent parameters
    # written out explicitly in solvent_models.py.
    xi_seed = float(centres[int(np.argmin(np.abs(centres)))])
    model_solv = get_solvent_model(args.solvent)
    system, bonded, contacts = build_jaxmd_solvated_system(
        model_solv,
        solute_geometry_at_xi(xi_seed),
        box_size,
        n_solvent=args.n_solvent,
        seed=args.seed,
        solute_charges="ml" if args.embedding == "electrostatic" else "cgenff",
    )
    solute = list(range(SOLUTE_N_ATOMS))
    print(f"atoms        {system.n_atoms} total, {SOLUTE_N_ATOMS} ML solute, "
          f"{system.metadata['n_solvent']} {model_solv.residue}")

    from mmml.md.config import EnsembleSpec, RunConfig
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.restraints import (
        AngleWall,
        BondRetentionWall,
        FlatBottomWall,
        LinearDistanceCV,
    )

    model, params = load_model()
    cv = LinearDistanceCV.difference(minuend=(IDX_C, IDX_CL), subtrahend=(IDX_C, IDX_N))
    # The bias fixes xi = r(C-Cl) - r(C-N) and nothing else. The orthogonal
    # direction, the sum r(C-Cl) + r(C-N), feels no force from it at all, and a
    # methyl that drifts away from BOTH partners satisfies any xi you like while
    # being a species the model has never seen. Measured directly: a window at
    # xi = +0.35 held xi to within 0.15 A while the sum sat at 6.25 A, where the
    # training set never exceeds 4.44 A at that xi -- then the ML energy dropped
    # through its training floor and the run diverged.
    #
    # Bounds come from the training data itself (diag/sum_envelope.py), over the
    # window range being sampled, not from a guess. Flat-bottomed, so sampling
    # inside the allowed band is unbiased and MBAR is unaffected.
    # A bound on the SUM would have to depend on xi -- the allowed sum is large
    # where one bond is long and small near the transition state -- and a single
    # global value is wrong at one end or useless at the other. A first attempt
    # at [3.8, 5.8] A pinned the system at 5.85 A while the training data at
    # xi = +0.35 never exceeds 4.44 A, and the run still diverged.
    #
    # min(r(C-Cl), r(C-N)) has no such xi dependence: across the whole training
    # set its median is 1.75 A, p99 2.03 A and max 2.18 A, in every xi bin. That
    # is the chemistry -- the methyl is always bonded to one partner or the
    # other -- and it is what to restrain. The frame that diverged sat at 2.57 A.
    bond_wall = BondRetentionWall(
        pairs=((IDX_C, IDX_CL), (IDX_C, IDX_N)),
        r_max=args.bond_r_max,
        k=args.wall_k_ev,
    )
    # Kept only as a lower guard against both bonds compressing at once, which
    # has never been observed but costs nothing to exclude.
    sum_cv = LinearDistanceCV.from_spec(
        {"pairs": [(IDX_C, IDX_CL), (IDX_C, IDX_N)], "coefficients": [1.0, 1.0]}
    )
    # xi constrains neither the sum nor the N-C-Cl angle. The angle matters
    # just as much: gas-phase windows past xi = +1.3 reoriented the chloride to
    # hydrogen-bond with the ammonium and sampled a mean angle of 70 deg instead
    # of the 165-173 deg of the reaction channel -- without crashing, so the
    # corruption would have reached the PMF silently.
    angle_wall = AngleWall(atoms=(IDX_N, IDX_C, IDX_CL),
                           theta_min_deg=args.angle_min_deg, k=args.wall_k_ev)
    walls = [bond_wall, angle_wall,
             FlatBottomWall(cv=sum_cv, lower=args.sum_min, k=args.wall_k_ev)]
    print(f"wall         min(r(C-Cl), r(C-N)) <= {args.bond_r_max:.2f} A "
          f"(k = {args.wall_k_ev} eV/A^2); training set max is 2.18 A")
    print(f"             plus angle(N-C-Cl) >= {args.angle_min_deg:.0f} deg"
          f" and r(C-Cl)+r(C-N) >= {args.sum_min:.2f} A")
    ctx = EnergyContext(model=model, params=params, options={"ml_atoms": solute})
    terms = ["ml_intra", "mm_bonded", "mm_nonbonded"]
    if args.embedding == "electrostatic":
        terms.append("ml_mm_elec")
    terms.append("rxncoor")
    terms = tuple(terms)
    print(f"embedding    {args.embedding}  ({' + '.join(terms)})")
    masses = np.asarray([atomic_mass(z) for z in np.asarray(system.Z)], dtype=float)

    # Static on-device pair list: the switched force field makes distant pairs
    # contribute exactly zero, so no neighbour list is needed and nothing is
    # rebuilt on the host. This is the difference between 2.5 and 28 steps/s.
    pair_fn = make_static_pair_fn(system, with_lambda=True)

    import dataclasses

    from mmml.md.assemble import assemble_and_run

    # Built once: the window centre arrives per step as the traced `lambda_t`
    # scalar, so the compiled graph is reused across every window and leg.
    term_kwargs_static = {
        "ml_intra": {"monomer_indices": [np.asarray(solute, dtype=np.int32)]},
        "mm_bonded": {"ml_atoms": solute, **bonded},
        "rxncoor": {"cv": cv, "target": 0.0, "k_ev_per_A2": args.k_ev,
                    "walls": walls},
    }
    if "ml_mm_elec" in terms:
        term_kwargs_static["ml_mm_elec"] = {
            "ml_atoms": solute,
            "charge_mode": "q0",
            "charge_gradient": not args.freeze_charge_forces,
        }

    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.drivers import JaxmdDriver

    # Build the energy ONCE. Rebuilding it per leg creates fresh Python closures,
    # which XLA sees as new computations and recompiles from scratch (~25 s each
    # for this system) -- 56 legs of that is ~23 minutes of pure compilation
    # before any sampling happens. Nothing in the energy depends on the current
    # positions: the umbrella centre arrives per step as the traced `lambda_t`
    # scalar, and every other term is defined by topology and parameters.
    energy = build_hybrid_energy(system, terms, ctx, term_kwargs_static)

    def run_leg(sys_in, xi0, ensemble, n_steps, record_every, tag, dt_fs=None,
                elec_scale=1.0):
        # `dt_fs` overrides the MD timestep for this leg. It exists for
        # minimisation: the driver hands FIRE dt_max = the MD timestep, which is
        # sized for dynamics on an equilibrated box, not for the first steps down
        # a lattice-packed one. Each extra distinct value costs one XLA
        # compilation, so use few of them.
        dt_fs = args.dt_fs if dt_fs is None else float(dt_fs)
        pair_fn.set_lambda(float(xi0))
        pair_fn.set_elec_scale(float(elec_scale))
        cfg = RunConfig(
            system=None,
            terms=terms,
            ensemble=EnsembleSpec(
                ensemble=ensemble, space="pbc",
                temperature_K=args.temperature, dt_fs=dt_fs, n_steps=n_steps,
                params={"masses": masses, "seed": args.seed, "float64": True},
            ),
            backend="jaxmd", output_dir=None, seed=args.seed,
        )
        driver = JaxmdDriver(
            record_every=record_every,
            neighbor_fn=pair_fn,
            output_path=None,
        )
        import time as _time

        _t0 = _time.time()
        traj = driver.run(sys_in, energy, cfg.ensemble)
        _dt = _time.time() - _t0
        run_leg.last_seconds = _dt
        run_leg.last_rate = n_steps / _dt if _dt > 0 else float("nan")
        pos = np.asarray(traj.metadata["positions"])
        energies = np.asarray(traj.metadata["energies"])
        # Stop at the first non-finite leg. Windows are seeded from each other,
        # so a single blow-up otherwise propagates NaN through every remaining
        # window at full speed -- the run looks healthy in the rate column and
        # produces nothing.
        bad = np.flatnonzero(~np.isfinite(energies))
        if bad.size:
            i = int(bad[0])
            # Dump the last frame that was still finite. Which pair of atoms has
            # collapsed says immediately whether this is an integration problem
            # or a missing short-range repulsion.
            dump = out / f"blowup_{tag}.npz"
            if i:
                # A trailing window, not just the last frame: a single frame
                # taken after the divergence has begun shows its consequences,
                # not its cause. The run-up is what identifies the culprit.
                lo = max(0, i - 20)
                np.savez(dump, R=pos[lo:i], energies=energies[lo:i],
                         first_bad=i - lo, step0=lo * record_every,
                         record_every=record_every, dt_fs=dt_fs,
                         Z=np.asarray(system.Z),
                         box=np.asarray(system.box), n_solute=len(solute))
                print(f"  last {i - lo} finite frames written to {dump}")
            raise SystemExit(
                f"\nleg '{tag}' (xi0 = {float(xi0):+.2f}) went non-finite at "
                f"recorded frame {i}/{len(energies)} "
                f"(~step {i * record_every} of {n_steps}, "
                f"{i * record_every * dt_fs:.1f} fs in).\n"
                f"last finite energy: "
                f"{energies[i - 1] if i else float('nan'):.3f} eV\n"
                + (
                    f"This is the minimiser, so the timestep is not the "
                    f"suspect. FIRE takes dt_max from --dt-fs ({dt_fs}); if the "
                    f"packed box has a strained contact it can diverge on the "
                    f"first steps. Lower --minimize-dt-fs."
                    if ensemble == "min"
                    else
                    f"Most likely --dt-fs {dt_fs} is too large: electrostatic "
                    f"embedding propagates dq/dR forces that respond to fast "
                    f"solvent motion. Check inspect_blowup.py on the dump above "
                    f"first -- if one pair sits far inside its contact distance "
                    f"this is a missing repulsion, not an integration problem."
                )
            )
        return dataclasses.replace(sys_in, R=pos[-1]), pos, energies

    # Solute-only ML energy, for the out-of-distribution check. Nine atoms, so
    # evaluating it on every recorded frame is free next to the trajectory that
    # produced them.
    import e3x

    _dst, _src = e3x.ops.sparse_pairwise_indices(SOLUTE_N_ATOMS)
    _z_ml = jnp.asarray(np.asarray(system.Z)[:SOLUTE_N_ATOMS], jnp.int32)

    _lengths = np.diag(np.asarray(system.box))

    def _unfold(frame):
        """Solute made contiguous again after the integrator wrapped it.

        Recorded coordinates are wrapped into the primary cell, so a solute
        sitting on a box face comes back split across it. Anything that reads
        solute geometry off these frames -- the reaction coordinate, the ML
        energy -- has to undo that first, or it reads distances of order the box
        length. Same correction the energy terms apply internally.
        """
        d = frame[:SOLUTE_N_ATOMS] - frame[0]
        return frame[0] + (d - _lengths * jnp.round(d / _lengths))

    @jax.jit
    def _ml_energy(frame):
        out = model.apply(
            params, atomic_numbers=_z_ml, positions=_unfold(frame),
            dst_idx=jnp.asarray(_dst, jnp.int32),
            src_idx=jnp.asarray(_src, jnp.int32), compute_forces=False,
        )
        return jnp.reshape(out["energy"], ())

    def ml_energy_of(pos):
        return np.asarray(jax.vmap(_ml_energy)(jnp.asarray(pos)))

    def min_bond_of(pos):
        """min(r(C-Cl), r(C-N)) per frame, for the wall-contact diagnostic."""
        sol = np.asarray(jax.vmap(_unfold)(jnp.asarray(pos)))
        r1 = np.linalg.norm(sol[:, IDX_C] - sol[:, IDX_CL], axis=-1)
        r2 = np.linalg.norm(sol[:, IDX_C] - sol[:, IDX_N], axis=-1)
        return np.minimum(r1, r2)

    traj_dir = out / "traj"
    if args.save_traj != "none":
        traj_dir.mkdir(parents=True, exist_ok=True)

    def write_traj(w, xi0, pos, xi, mb):
        """One extxyz per window, readable by VMD / OVITO / ASE.

        The solute is written unfolded, so the molecule is contiguous rather
        than split across a periodic face -- otherwise a viewer draws bonds
        across the box. Solvent keeps its wrapped coordinates, which is what a
        viewer expects for a periodic box.
        """
        from ase import Atoms
        from ase.io import write as ase_write

        z_all = np.asarray(system.Z)
        keep = slice(None) if args.save_traj == "full" else slice(0, SOLUTE_N_ATOMS)
        stride = args.traj_stride if args.save_traj == "full" else 1
        frames = []
        for k in range(0, len(pos), max(1, stride)):
            r = np.asarray(pos[k]).copy()
            r[:SOLUTE_N_ATOMS] = np.asarray(_unfold(jnp.asarray(r)))
            at = Atoms(numbers=z_all[keep], positions=r[keep],
                       cell=np.asarray(system.box), pbc=True)
            at.info["xi"] = float(xi[k])
            at.info["xi0"] = float(xi0)
            at.info["min_bond_A"] = float(mb[k])
            frames.append(at)
        path = traj_dir / f"w{w:03d}_xi{xi0:+.2f}.xyz"
        ase_write(str(path), frames, format="extxyz")
        return path

    def xi_of(pos):
        sol = np.asarray(jax.vmap(_unfold)(jnp.asarray(pos)))
        return np.linalg.norm(sol[:, IDX_C] - sol[:, IDX_CL], axis=-1) - np.linalg.norm(
            sol[:, IDX_C] - sol[:, IDX_N], axis=-1
        )

    # Relax the packed box once, restrained at the window nearest xi = 0.
    start_idx = int(np.argmin(np.abs(centres)))
    print(f"\nrelaxing packed box at xi0 = {centres[start_idx]:+.2f}")
    if args.minimize_steps > 0:
        # Two passes. The lattice builder places molecules on a grid at the
        # target density, which is a sound starting density but leaves individual
        # contacts strained, and FIRE at the full MD step size can diverge on
        # those first steps -- it did, on the 30 A water box, while the same
        # settings were fine at 26 A. The gentle pass exists only to clear them;
        # once it has, the main pass converges as before.
        # Minimisation runs with the solute-solvent electrostatics switched
        # off (elec_scale = 0). Turned on against a freshly packed box they win:
        # the minimiser drove xi from 0.00 to +0.94 against a 150 kcal/mol/A^2
        # restraint, i.e. the coupling exceeded 6 eV/A before dynamics even
        # started, and the run then diverged within 90 fs. With them off, the
        # minimiser does what it is for -- removing packing strain.
        n_soft = max(1, args.minimize_steps // 8)
        system, _, e = run_leg(system, centres[start_idx], "min", n_soft, n_soft,
                               "min_soft", dt_fs=args.minimize_dt_fs / 5.0,
                               elec_scale=0.0)
        print(f"  minimise/soft E {e[0]:12.3f} -> {e[-1]:12.3f} eV   "
              f"{run_leg.last_seconds:.1f}s "
              f"(dt {args.minimize_dt_fs / 5.0:g} fs, includes XLA compile)")
        system, _, e = run_leg(system, centres[start_idx], "min",
                               args.minimize_steps, args.minimize_steps, "min",
                               dt_fs=args.minimize_dt_fs, elec_scale=0.0)
        print(f"  minimise     E {e[0]:12.3f} -> {e[-1]:12.3f} eV   "
              f"{run_leg.last_seconds:.1f}s (dt {args.minimize_dt_fs:g} fs, "
              f"includes XLA compile)")
    n_eq = int(round(args.equil_ps * 1000.0 / args.dt_fs))
    # Ramp the solute-solvent electrostatics on in stages rather than switching
    # them on at full strength. The failure this avoids is a feedback loop, not
    # a singularity: the solvent pulls the chloride out, the model responds with
    # more charge transfer (q(Cl) went -0.80 -> -1.03 over 50 fs), the larger
    # charge pulls the solvent in harder, and nothing opposes it. Ramping lets
    # the first solvation shell form while the coupling is still weak.
    #
    # elec_scale is a traced scalar, so the stages cost no extra compilation.
    if args.embedding == "electrostatic" and args.ramp_stages > 0:
        n_ramp = max(1, n_eq // args.ramp_stages)
        for stage in range(1, args.ramp_stages + 1):
            scale = stage / args.ramp_stages
            system, _, e = run_leg(system, centres[start_idx], "nvt", n_ramp,
                                   args.record_every, f"ramp{stage}",
                                   elec_scale=scale)
            print(f"  ramp {scale:4.2f}     E {e[0]:12.3f} -> {e[-1]:12.3f} eV   "
                  f"{run_leg.last_seconds:5.1f}s")
    # Recorded densely rather than endpoint-only: this leg is shared by every
    # window, so if it blows up the guard in run_leg should say when.
    system, _, e = run_leg(system, centres[start_idx], "nvt", n_eq,
                           args.record_every, "equil0")
    print(f"  equilibrate  E {e[0]:12.3f} -> {e[-1]:12.3f} eV   "
          f"{run_leg.last_seconds:.1f}s ({run_leg.last_rate:.1f} steps/s)")

    n_eq = int(round(args.equil_ps * 1000.0 / args.dt_fs))
    n_prod = int(round(args.prod_ps * 1000.0 / args.dt_fs))
    results: dict[int, dict] = {}

    def walk(order, seed_system):
        """Run windows in sequence, each seeded from the previous one's end.

        One continuous leg per window; the leading ``n_eq`` steps' worth of
        frames are discarded as equilibration afterwards. Splitting it into two
        driver calls doubles the number of XLA compilations for no benefit,
        since the trajectory is continuous either way.
        """
        sys_cur = seed_system
        n_discard = max(1, n_eq // args.record_every)
        for w in order:
            xi0 = float(centres[w])
            sys_cur, pos_all, e_all = run_leg(
                sys_cur, xi0, "nvt", n_eq + n_prod, args.record_every, f"w{w}"
            )
            pos, energies = pos_all[n_discard:], e_all[n_discard:]
            if pos.shape[0] == 0:
                pos, energies = pos_all[-1:], e_all[-1:]
            xi = xi_of(pos)
            # TRAIN_MIN_EV is the floor of the *solute's* training energies, so
            # it has to be compared against the solute's own ML energy -- not
            # against `energies`, which is the whole box and is several hundred
            # eV lower simply because 586 waters are in it.
            below = int((ml_energy_of(pos) < TRAIN_MIN_EV).sum())
            # How hard is the wall working? A window that spends most of its
            # time pressed against the bound is not sampling the physical
            # ensemble -- it is sampling whatever the wall permits, and its
            # contribution to the PMF is a property of the restraint rather
            # than of the system. Reported per window so this is visible in the
            # profile rather than discovered afterwards.
            mb = min_bond_of(pos)
            wall_frac = float((mb > args.bond_r_max - 0.05).mean())
            flag = ""
            if below:
                flag = f"  !! {below}/{len(unbiased)} frames below the training floor"
            print(f"  w{w:03d} xi0={xi0:+5.2f}  <xi>={xi.mean():+6.3f} "
                  f"sd={xi.std():.3f}  E={energies.mean():10.2f} eV  "
                  f"{run_leg.last_seconds:5.1f}s "
                  f"({run_leg.last_rate:5.1f} steps/s)"
                  f"  minr={mb.mean():.2f}"
                  + (f" WALL {100 * wall_frac:.0f}%" if wall_frac > 0.05 else "")
                  + flag)
            if args.save_traj != "none":
                write_traj(w, xi0, pos, xi, mb)
            results[w] = {
                "xi0": xi0,
                "min_bond_A": mb.tolist(),
                "wall_contact_fraction": wall_frac,
                "xi": xi.tolist(),
                "energy_eV": energies.tolist(),
                "below_training_floor": below,
            }
            _write_results(results, partial=True)

    def _write_results(res, partial: bool):
        payload = {
            "solvent": args.solvent,
            "residue": resi,
            "box_size_A": box_size,
            "n_atoms": int(system.n_atoms),
            "n_solvent": int(system.metadata["n_solvent"]),
            "k_ev_A2": args.k_ev,
            "temperature_K": args.temperature,
            "dt_fs": args.dt_fs,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "cv": cv.label(),
            "embedding": args.embedding,
            "terms": list(terms),
            "xi0": centres.tolist(),
            "complete": not partial,
            "n_windows_done": len(res),
            "windows": {str(k): v for k, v in sorted(res.items())},
        }
        (out / "umbrella_windows.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\nwalking windows outward from xi = 0")
    walk(range(start_idx, len(centres)), system)
    walk(range(start_idx - 1, -1, -1), system)

    _write_results(results, partial=False)
    path = out / "umbrella_windows.json"
    print(f"\nwrote {path}")
    print(f"next: python examples/menshutkin/08_solvated_mbar.py --run-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
