"""Build a solvated ML/MM system entirely in numpy/JAX-MD -- no CHARMM.

The CHARMM-backed builders kept failing opaquely (one build per process, a
Packmol stage whose output scored -9.5e5 eV, contradictory overlap
diagnostics). Here every number is produced locally and checked before use:

- solvent molecules are placed on a jittered lattice, so the minimum separation
  is known by construction rather than hoped for;
- the solute is placed first and solvent sites overlapping it are simply not
  filled, which is the cavity that solvating it requires;
- ``FFParams`` is assembled directly from :mod:`solvent_models`, so the charges
  and LJ parameters entering the energy are the ones written in that file;
- the builder returns only after asserting the minimum intermolecular distance,
  so a bad box cannot reach the integrator.

The solute keeps a single ``mol_id`` so ``mm_nonbonded`` does not add CGenFF
Coulomb and LJ across the bond being formed, which the ML model already covers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["build_jaxmd_solvated_system", "solute_ff_params"]

# Solute in PSF/campaign order: AMM1 (N,H,H,H) then MECL (C,CL,H,H,H).
SOLUTE_Z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=np.int32)
IDX_N, IDX_C, IDX_CL = 0, 4, 5

# CGenFF nonbonded parameters for the solute atoms, by atom type:
#   NG331 (AMM1 N), HGPAM3 (AMM1 H), CG331 (MECL C), CLGA1 (MECL Cl),
#   HGA3 (MECL H). epsilon in kcal/mol (magnitude), Rmin/2 in Angstrom.
# Charges here are the CGenFF values, used ONLY for the LJ partner definition
# and for the mechanical-embedding comparison; the electrostatic-embedding path
# replaces them with the ML model's fluctuating charges.
_SOLUTE_LJ = {
    "N": (0.2000, 1.85),      # NG331
    "H_N": (0.0090, 0.875),   # HGPAM3
    "C": (0.0780, 2.050),     # CG331
    "CL": (0.1500, 2.270),    # CLA, not CLGA1 -- see below
    "H_C": (0.0240, 1.340),   # HGA3
}
# The chlorine is the one atom whose LJ type cannot simply be taken from the
# reactant. CGenFF's CLGA1 (eps 0.343, Rmin/2 1.910) describes chlorine bonded to
# carbon, and its small radius is calibrated against the -0.2 e that CGenFF puts
# there. Along this reaction the ML model drives that charge to about -0.9 e,
# i.e. the atom becomes chloride, and CLGA1's core is then far too small to hold
# a solvent hydrogen off: the collapse this caused is described in
# mmml/md/energy/terms/ml_mm_elec.py. CLA (eps 0.150, Rmin/2 2.270; Beglov &
# Roux, J. Chem. Phys. 100, 9050 (1994)) is CHARMM's chloride ion and is the
# right partner for the charge the model actually assigns over most of the PMF.
# The cost is at the reactant end, where the still-covalent Cl is modelled a
# little too large -- but there its charge is small, so its solvent
# electrostatics are weak and the error is correspondingly minor.
_SOLUTE_CHARGES = np.array([
    -1.125, 0.375, 0.375, 0.375,   # AMM1 N, H x3
    0.027, -0.204, 0.059, 0.059, 0.059,  # MECL C, Cl, H x3
])


def solute_ff_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(charges, epsilon, rmin_half)`` for the 9 solute atoms."""
    order = ["N", "H_N", "H_N", "H_N", "C", "CL", "H_C", "H_C", "H_C"]
    eps = np.array([_SOLUTE_LJ[k][0] for k in order])
    rmin = np.array([_SOLUTE_LJ[k][1] for k in order])
    return _SOLUTE_CHARGES.copy(), eps, rmin


def _lattice_sites(box: float, n_needed: int, rng) -> np.ndarray:
    """Jittered simple-cubic sites, enough to host ``n_needed`` molecules.

    A lattice guarantees a minimum spacing; the jitter breaks the symmetry so
    equilibration does not start from a crystal. Sites are generated for the
    smallest cubic grid that fits the requirement, then shuffled so the ones
    dropped for the solute cavity are not spatially biased.
    """
    # One extra shell of sites for headroom (the solute cavity removes some),
    # but no more: over-provisioning shrinks the lattice spacing, and at a
    # spacing well below the true molecular separation every accepted molecule
    # ends up on its neighbour's repulsive wall. For water at 997 kg/m3 the
    # physical spacing is (V/N)^(1/3) ~ 3.1 A; generating 2x the sites drove it
    # to 2.2 A and gave a +1220 eV nonbonded energy.
    # Match the lattice spacing to the true molecular separation, (V/N)^(1/3).
    # Two failure modes bracket this:
    #   too fine   -> every accepted molecule sits on its neighbour's repulsive
    #                 wall (2.2 A for water gave +1220 eV);
    #   too coarse -> too few sites to reach the target count.
    # A blanket "+1 shell" of headroom is wrong for large molecules: for
    # cyclohexane it forced a 4.0 A spacing for a ~5 A molecule, so most
    # insertions were rejected and the box came out at 61 % of liquid density.
    n_side = max(1, int(round(n_needed ** (1 / 3))))
    while n_side**3 < n_needed:
        n_side += 1
    spacing = box / n_side
    grid = (np.arange(n_side) + 0.5) * spacing
    sites = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    # Jitter by up to a quarter of the spacing: enough to randomise, small
    # enough that the guaranteed separation stays >= spacing/2.
    sites = sites + rng.uniform(-0.15, 0.15, sites.shape) * spacing
    rng.shuffle(sites)
    return sites, spacing


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_jaxmd_solvated_system(
    solvent_model,
    solute_geometry: np.ndarray,
    box_side: float,
    *,
    n_solvent: int | None = None,
    solute_gap: float = 2.6,
    solvent_gap: float = 1.95,
    solute_charges: str = "ml",
    nonbonded_cutoff_A: float = 12.0,
    inflate: float = 1.15,
    seed: int = 0,
    verbose: bool = True,
):
    """Solute at the box centre, solvent on a jittered lattice around it."""
    import dataclasses

    from mmml.md.system import FFParams, MolecularSystem

    rng = np.random.default_rng(seed)
    n_solute = int(solute_geometry.shape[0])
    requested_box = float(box_side)
    target = n_solvent if n_solvent is not None else solvent_model.n_for_density(requested_box)

    # Place into a deliberately inflated box, then compress back to exactly the
    # requested side. A lattice with an acceptance test cannot reach liquid
    # density directly (cyclohexane managed 63 %), and compressing an
    # under-filled box to *its own* density silently changes the cell -- which
    # once pushed it below twice the nonbonded cutoff and broke minimum image.
    # Inflating first means the caller gets the box they asked for at the
    # density they asked for.
    build_box = requested_box * inflate
    box_side = build_box

    geom = np.asarray(solute_geometry, dtype=np.float64)
    geom = geom - geom.mean(axis=0) + box_side / 2.0

    # Generate more sites than needed so the solute cavity can be carved without
    # dropping below the target count.
    sites, spacing = _lattice_sites(box_side, target, rng)
    L = np.full(3, box_side)

    # Insert molecules one at a time with an explicit atom-level acceptance
    # test. Placing them blind on a jittered lattice is not enough: with a
    # 2.75 A spacing and +/-0.25 jitter two neighbours can end up 0.7 A apart.
    # The threshold is deliberately ~1.6 A, not 2.2 A, because a water hydrogen
    # bond puts H...O at about 1.8 A and demanding more would forbid the
    # structure we are trying to build.
    # Placement happens in the inflated box but the contacts that matter are the
    # ones after compression, which scale down by 1/inflate. Demand the gap
    # pre-compression so the final box lands on the intended threshold.
    build_gap = solvent_gap * inflate
    mol_geom = solvent_model.geometry - solvent_model.geometry.mean(axis=0)
    placed_coords: list[np.ndarray] = []
    existing = geom.copy()
    n_tries_per_site = 40

    for site in sites:
        if len(placed_coords) == target:
            break
        d = site - geom
        d -= L * np.round(d / L)
        if np.linalg.norm(d, axis=-1).min() < solute_gap:
            continue  # inside the solute cavity
        for attempt in range(n_tries_per_site):
            # Retry with a fresh orientation and a small positional nudge.
            nudge = 0.0 if attempt == 0 else rng.uniform(-0.2, 0.2, 3) * spacing
            trial = site + nudge + mol_geom @ _random_rotation(rng).T
            dd = trial[:, None, :] - existing[None, :, :]
            dd -= L * np.round(dd / L)
            if np.linalg.norm(dd, axis=-1).min() >= build_gap:
                placed_coords.append(trial)
                existing = np.concatenate([existing, trial], axis=0)
                break

    n_placed = len(placed_coords)
    if n_placed < target and verbose:
        print(f"  note: placed {n_placed} of {target} requested "
              f"{solvent_model.residue} (cavity + {build_gap:.2f} A acceptance)")

    # Compress from the inflated build box back to the requested cell. Only the
    # molecular *centres* move; internal geometry is rigid, which is what a
    # density adjustment physically means.
    if n_placed:
        if n_solvent is None:
            # Honour the requested box exactly; density then follows from how
            # many molecules were placed, and is reported below.
            final_box = requested_box
        else:
            final_box = requested_box
        scale = final_box / box_side
        centre_old, centre_new = box_side / 2.0, final_box / 2.0
        placed_coords = [
            c - c.mean(axis=0) + (c.mean(axis=0) - centre_old) * scale + centre_new
            for c in placed_coords
        ]
        geom = geom - centre_old + centre_new
        box_side = final_box
        L = np.full(3, box_side)
        if verbose:
            rho = _density(solvent_model, n_placed, box_side)
            print(f"  compressed {build_box:.2f} -> {box_side:.2f} A "
                  f"({n_placed} molecules, {rho:.0f} kg/m3, "
                  f"target {solvent_model.density_kg_m3:.0f})")

    placed = [c.mean(axis=0) for c in placed_coords]
    R = np.concatenate([geom, *placed_coords], axis=0) % box_side

    n_mol_atoms = solvent_model.n_atoms
    n_atoms = n_solute + n_placed * n_mol_atoms
    Z = np.concatenate([SOLUTE_Z, np.tile(solvent_model.Z, n_placed)])

    # One mol_id for the whole solute so mm_nonbonded treats it as a single
    # molecule; the ML model already describes everything inside it.
    mol_id = np.concatenate([
        np.zeros(n_solute, dtype=np.int32),
        np.repeat(np.arange(1, n_placed + 1, dtype=np.int32), n_mol_atoms),
    ])
    monomers = [np.arange(n_solute, dtype=np.int32)] + [
        np.arange(n_solute + m * n_mol_atoms, n_solute + (m + 1) * n_mol_atoms,
                  dtype=np.int32)
        for m in range(n_placed)
    ]

    q_s, eps_s, rmin_s = solute_ff_params()
    if solute_charges == "ml":
        # Electrostatic embedding: the ml_mm_elec term supplies the solute's
        # charges from the model at every step, so the fixed values must not
        # also appear in mm_nonbonded. Zeroing them removes exactly the Coulomb
        # part and leaves the solute's Lennard-Jones untouched.
        q_s = np.zeros_like(q_s)
    elif solute_charges != "cgenff":
        raise ValueError(
            f"solute_charges must be 'ml' or 'cgenff' (got {solute_charges!r})"
        )
    charges = np.concatenate([q_s, np.tile(solvent_model.charges, n_placed)])
    epsilon = np.concatenate([eps_s, np.tile(solvent_model.epsilon, n_placed)])
    rmin_half = np.concatenate([rmin_s, np.tile(solvent_model.rmin_half, n_placed)])

    # Intramolecular exclusions: every pair inside each molecule.
    exclusions = []
    for idx in monomers:
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                exclusions.append((int(idx[a]), int(idx[b])))
    ff = FFParams(
        charges=charges,
        epsilon=epsilon,
        rmin_half=rmin_half,
        at_codes=np.zeros(n_atoms, dtype=np.int32),
        exclusions=np.asarray(sorted(exclusions), dtype=np.int32).reshape(-1, 2),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )

    system = MolecularSystem(
        R=R,
        Z=Z,
        box=np.eye(3) * box_side,
        mol_id=mol_id,
        monomer_indices=monomers,
        ff_params=ff,
        metadata={
            "builder": "jaxmd_lattice",
            "solvent": solvent_model.name,
            "residue": solvent_model.residue,
            "n_solvent": n_placed,
            "lattice_spacing_A": float(spacing),
            "solute_charges": solute_charges,
        },
    )

    # Bonded arrays for the solvent, offset into the full system.
    bonds, bond_k, bond_r0, angles, angle_k, angle_t0 = [], [], [], [], [], []
    for m in range(n_placed):
        off = n_solute + m * n_mol_atoms
        bonds.append(solvent_model.bonds + off)
        bond_k.append(solvent_model.bond_k)
        bond_r0.append(solvent_model.bond_r0)
        angles.append(solvent_model.angles + off)
        angle_k.append(solvent_model.angle_k)
        angle_t0.append(solvent_model.angle_theta0)
    bonded = {
        "bonds": np.concatenate(bonds) if bonds else np.zeros((0, 2), np.int32),
        "bond_k": np.concatenate(bond_k) if bond_k else np.zeros(0),
        "bond_r0": np.concatenate(bond_r0) if bond_r0 else np.zeros(0),
        "angles": np.concatenate(angles) if angles else np.zeros((0, 3), np.int32),
        "angle_k": np.concatenate(angle_k) if angle_k else np.zeros(0),
        "angle_theta0": np.concatenate(angle_t0) if angle_t0 else np.zeros(0),
    }

    min_box_for_cutoff = 2.0 * nonbonded_cutoff_A
    if box_side < min_box_for_cutoff:
        raise SystemExit(
            f"box side {box_side:.1f} A is below 2x the {nonbonded_cutoff_A:.1f} A "
            "nonbonded cutoff, so the minimum-image convention is violated and "
            "pairs would be counted through periodic images. Use a larger box "
            "or a shorter cutoff."
        )

    report = verify_contacts(system, n_solute)
    if verbose:
        print(f"  {n_atoms} atoms: {n_solute} ML solute + {n_placed} "
              f"{solvent_model.residue} ({solvent_model.n_atoms} atoms each)")
        print(f"  lattice spacing {spacing:.2f} A, density "
              f"{_density(solvent_model, n_placed, box_side):.0f} kg/m3")
        print(f"  closest solute-solvent  {report['solute_solvent']:.3f} A")
        print(f"  closest solvent-solvent {report['solvent_solvent']:.3f} A")
    if report["solvent_solvent"] < 1.5 or report["solute_solvent"] < 1.5:
        raise SystemExit(
            f"built box has a {min(report.values()):.3f} A intermolecular contact; "
            "refusing to hand it to the integrator"
        )
    return system, bonded, report


def _density(model, n_placed: int, box_side: float) -> float:
    return n_placed * model.molar_mass / 6.02214076e23 / (box_side**3 * 1e-24) * 1e3


def verify_contacts(system, n_solute: int) -> dict:
    """Minimum-image closest contacts, excluding intramolecular pairs."""
    R = np.asarray(system.R)
    L = np.diag(np.asarray(system.box))
    mol = np.asarray(system.mol_id)
    d = R[:, None, :] - R[None, :, :]
    d -= L * np.round(d / L)
    dist = np.linalg.norm(d, axis=-1)
    dist[mol[:, None] == mol[None, :]] = np.inf
    solute = np.arange(n_solute)
    solvent = np.arange(n_solute, R.shape[0])
    return {
        "solute_solvent": float(dist[np.ix_(solute, solvent)].min()) if solvent.size else np.inf,
        "solvent_solvent": float(dist[np.ix_(solvent, solvent)].min()) if solvent.size > 3 else np.inf,
        "overall": float(dist.min()),
    }
