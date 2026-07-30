#!/usr/bin/env python3
"""Solvated reactive ML/MM dynamics for NH3 + CH3Cl with JAX-MD.

Why this is not just ``mmml md-system --backend jaxmd``
------------------------------------------------------
The stock unified path composes ``ml_intra + mm_nonbonded``, which is wrong for
this system in three separate ways. Running it as-is on a water box gives
E0 = -1.5e6 eV, because:

1. ``ml_intra`` evaluates *every* molecule with the ML model, including the
   solvent. ``model_ext.json`` is a 9-atom NH3+CH3Cl model; asked for a water
   molecule it returns nonsense.
2. ``ml_intra`` evaluates each molecule *separately*, so NH3 and CH3Cl would be
   scored in isolation and no reaction could ever occur. The solute has to be a
   single 9-atom ML group.
3. Nothing supplies intramolecular energy for the solvent, so the solvent
   molecules simply come apart.

This script fixes all three and adds the umbrella bias:

    E = ml_intra(solute as ONE 9-atom group)
      + mm_bonded(solvent only; ML-region rows deleted)
      + mm_nonbonded(intermolecular, with the solute treated as one molecule)
      + rxncoor(xi = r(C-Cl) - r(C-N))

The ``mol_id`` merge in point 4 below matters as much as the rest: the builder
sees AMM1 and MECL as two molecules, so ``mm_nonbonded`` would add CGenFF
Coulomb and LJ between them -- an interaction the ML model already describes in
full. Merging them into one molecule removes that double count.

Usage
-----
    source examples/menshutkin/_env.sh
    python examples/menshutkin/06_solvated_md.py --solvent water --n-solvent 20 \\
        --ps 0.5 --xi0 0.0
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

EV_TO_KCAL = 23.060547830619027

# Solvent name -> (CGenFF residue, experimental density kg/m3, Turan box side A)
SOLVENTS = {
    "water": ("TIP3", 997.0, 30.0),
    "methanol": ("MEOH", 792.0, 25.0),
    "acetonitrile": ("ACN", 786.0, 28.0),
    "benzene": ("BENZ", 874.0, 27.0),
    "cyclohexane": ("CHEX", 774.0, 30.0),
}

# Composition order is AMM1 (N,H,H,H) then MECL (C,CL,H,H,H), so within the
# 9-atom solute the CV atoms sit at these indices. Matches the PDB/PSF ordering
# documented in 05_export_solute.py.
SOLUTE_N_ATOMS = 9
IDX_N, IDX_C, IDX_CL = 0, 4, 5
SOLUTE_Z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=np.int32)


def build_system(composition: str, box_size: float, seed: int, psf_path: Path):
    """Packmol composition -> MolecularSystem with CGenFF FFParams.

    Mirrors ``mmml.cli.run.md_system_unified.build_packmol_system_with_ffparams``
    but writes the PSF somewhere persistent: that helper uses a
    ``TemporaryDirectory`` which is removed before it returns, so the resulting
    ``system.psf_path`` points at a deleted file and any term needing bonded
    topology (``mm_bonded``) cannot be built.
    """
    import pycharmm.write as write

    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster
    from mmml.md.builders.placement import (
        _box,
        _composition,
        _lower_optional_psf,
        _placement_system,
    )
    from mmml.md.system import SystemSpec

    spec = SystemSpec(
        builder="packmol",
        composition=composition,
        box_size=box_size,
        seed=seed,
        params={"cube_side": box_size},
    )
    params = dict(spec.params)
    box = _box(spec, params)
    params.setdefault("center", tuple(np.diag(box) / 2.0))
    params.setdefault("cube_side", spec.box_size)

    z, positions, sizes, residues = build_packmol_composition_cluster(
        composition=_composition(spec), seed=spec.seed, **params
    )
    psf_path.parent.mkdir(parents=True, exist_ok=True)
    write.psf_card(str(psf_path))

    system = _placement_system(
        name="packmol",
        spec=spec,
        z=z,
        positions=positions,
        atoms_per_molecule=list(sizes),
        residue_names=list(residues),
        box=box,
    )
    return _lower_optional_psf(system, psf_path=psf_path, prm_paths=())


def solute_geometry_at_xi(xi_target: float) -> np.ndarray:
    """Scan frame nearest ``xi_target``, reordered to AMM1-then-MECL (PSF order).

    The scan is stored in canonical order (Cl, N, C, H(N)x3, H(C)x3); the
    builder lays atoms out residue by residue, so the two disagree. The
    permutation is the same one 05_export_solute.py writes out.
    """
    sys.path.insert(0, str(EXAMPLE_DIR))
    seed_mod = __import__("01_seed_windows")
    scan = Path(os.environ.get("MENSH_SCAN", REPO_ROOT / "examples/m/scan_nh3_ch3cl.npz"))
    _z, r_all, xi = seed_mod.load_scan(scan)
    idx = int(np.argmin(np.abs(xi - float(xi_target))))
    # canonical -> PSF order: N(1), H(3,4,5), C(2), Cl(0), H(6,7,8)
    perm = [1, 3, 4, 5, 2, 0, 6, 7, 8]
    print(f"solute seed  scan frame {idx}, xi = {xi[idx]:+.3f} A "
          f"(target {float(xi_target):+.3f})")
    return r_all[idx][perm]


def build_solvated_system(resi: str, n_solvent: int, box_size: float, seed: int,
                          psf_path: Path, solute_geometry: np.ndarray,
                          n_solute: int, min_gap: float = 2.4):
    """Solute at the box centre with a solvent cavity carved around it.

    Packmol places AMM1 and MECL as two independent species and scatters them
    (they came out 14 A apart), so the reacting complex has to be imposed
    afterwards. Every attempt to make room by *moving* solvent failed: at liquid
    density there is nowhere to move to, and both random relocation and radial
    pushing simply trade a solute clash for a solvent-solvent one (the radial
    push ended with two atoms exactly coincident and E = -9e5 eV).

    So the solvent that overlaps the solute is **deleted** instead, which is what
    every solvation tool does. The trick that makes it cheap: a CHARMM PSF
    depends only on residue *counts*, not coordinates, and all solvent molecules
    are identical. So we pack once, decide which molecules survive, then rebuild
    the topology for the surviving count and drop the coordinates in.
    """
    import dataclasses
    import subprocess

    # Pass 1 runs in a subprocess: CHARMM keeps global state, and building a
    # second composition in the same process leaves it inconsistent
    # ("SOME COORDINATES NOT BUILT", "CCNBA not allocated") and then hard-exits.
    # So the packing pass is isolated and only its coordinates come back.
    packed_npz = psf_path.with_suffix(".packed.npz")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--dump-packed",
         "--composition", f"AMM1:1,MECL:1,{resi}:{n_solvent}",
         "--box-size", str(box_size), "--seed", str(seed),
         "--dump-to", str(packed_npz)],
        check=True, cwd=REPO_ROOT,
    )
    if not packed_npz.is_file():
        raise SystemExit(
            f"the packing subprocess did not write {packed_npz}; rerun it "
            "directly to see the CHARMM output"
        )
    packed = np.load(packed_npz)
    R = np.asarray(packed["R"]).copy()
    sizes = np.asarray(packed["sizes"], dtype=int)
    lengths = np.full(3, float(box_size))
    centre = lengths / 2.0

    geom = np.asarray(solute_geometry, dtype=np.float64)
    geom = geom - geom.mean(axis=0) + centre

    bounds = np.concatenate([[0], np.cumsum(sizes)])
    solvent = [
        np.arange(bounds[i], bounds[i + 1])
        for i in range(len(sizes))
        if bounds[i] >= n_solute
    ]
    keep = []
    for idx in solvent:
        d = R[idx][:, None, :] - geom[None, :, :]
        d -= lengths * np.round(d / lengths)  # minimum image
        if np.linalg.norm(d, axis=-1).min() >= min_gap:
            keep.append(idx)
    n_removed = len(solvent) - len(keep)
    print(f"             carved cavity: removed {n_removed} of {len(solvent)} "
          f"{resi} molecules (< {min_gap} A from the solute)")

    # Pass 2: topology for the surviving count, then drop coordinates in.
    system = build_system(f"AMM1:1,MECL:1,{resi}:{len(keep)}", box_size, seed, psf_path)
    coords = np.concatenate([geom] + [R[idx] for idx in keep], axis=0)
    if coords.shape[0] != system.n_atoms:
        raise SystemExit(
            f"carved {coords.shape[0]} atoms but the rebuilt topology has "
            f"{system.n_atoms}; solvent molecule sizes must be uniform"
        )
    system = dataclasses.replace(system, R=coords)

    d = coords[:n_solute][:, None, :] - coords[n_solute:][None, :, :]
    d -= lengths * np.round(d / lengths)
    worst = float(np.linalg.norm(d, axis=-1).min())
    print(f"             closest solute-solvent contact {worst:.2f} A")
    return system


def merge_solute_molecule(system, n_solute: int):
    """Give every solute atom the same ``mol_id`` and one merged monomer entry.

    ``mm_nonbonded`` filters pairs by ``mol_id``; leaving AMM1 and MECL as
    separate molecules would let it add CGenFF Coulomb + LJ across the very bond
    being formed, on top of the ML description of the same interaction.
    """
    import dataclasses

    mol_id = np.asarray(system.mol_id).copy()
    solute_ids = set(int(m) for m in mol_id[:n_solute])
    merged = int(mol_id[0])
    for i, m in enumerate(mol_id):
        if int(m) in solute_ids:
            mol_id[i] = merged

    monomers = [np.asarray(m, dtype=np.int32) for m in system.monomer_indices]
    solvent_monomers = [m for m in monomers if int(m.min()) >= n_solute]
    merged_monomers = [np.arange(n_solute, dtype=np.int32), *solvent_monomers]

    residues = list(system.metadata.get("residue_names", ()))
    if residues:
        residues = ["SOLU", *residues[2:]]
    metadata = {**dict(system.metadata), "residue_names": tuple(residues)}
    return dataclasses.replace(
        system, mol_id=mol_id, monomer_indices=merged_monomers, metadata=metadata
    )


def check_solute_layout(system, n_solute: int) -> None:
    """Fail loudly if the builder did not put the solute first, in the order we assume."""
    z = np.asarray(system.Z)[:n_solute]
    if not np.array_equal(z, SOLUTE_Z):
        raise SystemExit(
            f"solute atoms are {z.tolist()}, expected {SOLUTE_Z.tolist()} "
            "(AMM1 N,H,H,H then MECL C,CL,H,H,H). The composition string must "
            "list AMM1 and MECL first, and the CV indices below depend on it."
        )


def dump_packed(composition: str, box_size: float, seed: int, out: Path) -> int:
    """Build one Packmol composition and save its coordinates, then exit.

    Runs as its own process so CHARMM's global state is never reused (see
    build_solvated_system).
    """
    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.md.builders.placement import _composition
    from mmml.md.system import SystemSpec

    if not ensure_pycharmm_loaded():
        raise SystemExit("PyCHARMM not available")
    # The builder wants the parsed composition, not the raw "A:1,B:2" string.
    spec = SystemSpec(builder="packmol", composition=composition,
                      box_size=box_size, seed=seed)
    centre = tuple(np.full(3, box_size / 2.0))
    z, positions, sizes, residues = build_packmol_composition_cluster(
        composition=_composition(spec), seed=seed, center=centre, cube_side=box_size
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, R=np.asarray(positions, dtype=np.float64),
             Z=np.asarray(z, dtype=np.int32),
             sizes=np.asarray(sizes, dtype=np.int32))
    print(f"dumped {len(z)} atoms to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-packed", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--composition", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dump-to", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--solvent", choices=sorted(SOLVENTS), default="water")
    parser.add_argument(
        "--n-solvent",
        type=int,
        default=None,
        help="Solvent molecules (default: from experimental density and box size)",
    )
    parser.add_argument("--box-size", type=float, default=None, help="Cubic side (A)")
    parser.add_argument("--ps", type=float, default=0.5)
    parser.add_argument("--dt-fs", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--xi0",
        type=float,
        default=None,
        help="Umbrella window centre for xi = r(C-Cl) - r(C-N) (A). "
        "Omit to run unbiased.",
    )
    parser.add_argument(
        "--k-ev", type=float, default=6.505, help="Bias force constant (eV/A^2; 6.505 = 150 kcal/mol/A^2)"
    )
    parser.add_argument("--ensemble", choices=("nvt", "nve"), default="nvt")
    parser.add_argument(
        "--minimize-steps",
        type=int,
        default=500,
        help="FIRE minimisation steps before dynamics. A raw Packmol box has "
        "residual close contacts (E0 ~ +375 eV for a 24 A water box), so "
        "starting dynamics straight from it wastes the first picosecond "
        "dumping that into heat. 0 disables.",
    )
    parser.add_argument(
        "--equilibrate-ps",
        type=float,
        default=2.0,
        help="NVT equilibration before the production leg. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.dump_packed:
        return dump_packed(args.composition, float(args.box_size),
                           int(args.seed), args.dump_to)

    resi, density, default_side = SOLVENTS[args.solvent]
    box_size = args.box_size if args.box_size is not None else default_side

    n_solvent = args.n_solvent
    if n_solvent is None:
        from mmml.analysis.residue_geometry import load_residue_monomer_atoms

        mono = load_residue_monomer_atoms(resi, generate=True)
        molar_mass = float(sum(mono.get_masses()))  # g/mol
        volume_A3 = box_size**3
        # rho [kg/m3] = 1e-3 g/cm3 ; 1 cm3 = 1e24 A3 ; N_A = 6.02214076e23
        n_solvent = int(
            round(density * 1e-3 * volume_A3 * 6.02214076e23 / (1e24 * molar_mass))
        )

    artifacts = Path(os.environ.get("MENSH_ARTIFACTS", REPO_ROOT / "artifacts/menshutkin"))
    tag = f"{args.solvent}_xi{args.xi0:+.2f}" if args.xi0 is not None else f"{args.solvent}_free"
    out = args.output_dir or artifacts / "solvated" / tag
    out.mkdir(parents=True, exist_ok=True)

    composition = f"AMM1:1,MECL:1,{resi}:{n_solvent}"
    print(f"solvent      {args.solvent} ({resi}), rho={density} kg/m3")
    print(f"composition  {composition}")
    print(f"box          {box_size} A cube")

    import jax

    jax.config.update("jax_enable_x64", True)

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    if not ensure_pycharmm_loaded():
        raise SystemExit("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

    system = build_system(composition, box_size, args.seed, out / "system.psf")
    check_solute_layout(system, SOLUTE_N_ATOMS)

    # Packmol scatters AMM1 and MECL independently; seed the reacting pair from
    # the scan at the requested xi instead.
    seed_geom = solute_geometry_at_xi(args.xi0 if args.xi0 is not None else 0.0)
    system = place_solute(system, seed_geom, SOLUTE_N_ATOMS)
    system = merge_solute_molecule(system, SOLUTE_N_ATOMS)
    solute = list(range(SOLUTE_N_ATOMS))

    print(f"atoms        {system.n_atoms} total, {SOLUTE_N_ATOMS} ML solute")
    r = np.asarray(system.R)
    xi_built = float(
        np.linalg.norm(r[IDX_C] - r[IDX_CL]) - np.linalg.norm(r[IDX_C] - r[IDX_N])
    )
    print(f"built xi     {xi_built:+.3f} A")

    # --- energy terms -------------------------------------------------------
    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.config import EnsembleSpec, RunConfig
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.restraints import LinearDistanceCV

    calc_model, calc_params = _load_model(Path(os.environ.get("MENSH_CKPT", REPO_ROOT / "model_ext.json")))

    cv = LinearDistanceCV.difference(minuend=(IDX_C, IDX_CL), subtrahend=(IDX_C, IDX_N))
    terms = ["ml_intra", "mm_bonded", "mm_nonbonded"]
    term_kwargs = {
        # One ML group covering the whole solute: this is what makes the
        # reaction possible at all.
        "ml_intra": {"monomer_indices": [np.asarray(solute, dtype=np.int32)]},
        # Solvent bonded terms only; every row touching the ML solute is dropped
        # so the CGenFF C-Cl bond cannot pin the leaving group.
        "mm_bonded": {"ml_atoms": solute},
    }
    if args.xi0 is not None:
        terms.append("rxncoor")
        term_kwargs["rxncoor"] = {"cv": cv, "target": args.xi0, "k_ev_per_A2": args.k_ev}

    ctx = EnergyContext(model=calc_model, params=calc_params, options={"ml_atoms": solute})
    energy = build_hybrid_energy(system, terms, ctx, term_kwargs)
    print(f"terms        {', '.join(terms)}")
    for fns in energy.term_fns:
        report = getattr(fns.jax_energy_fn, "bonded_report", None)
        if report:
            print(
                f"  mm_bonded: {report['n_bonds']} bonds, {report['n_angles']} angles "
                f"kept; dropped {report['dropped']}"
            )

    # --- sanity check before spending GPU time ------------------------------
    import jax.numpy as jnp

    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    nbr_fn = make_intermolecular_neighbor_fn(system, 12.0, None)
    kw = {k: jnp.asarray(v) for k, v in nbr_fn(r, np.asarray(system.box)).items()}
    e_fn = energy.as_jax_energy_fn()
    e0 = float(e_fn(jnp.asarray(r), **kw))
    f0 = -np.asarray(jax.grad(lambda x: e_fn(x, **kw))(jnp.asarray(r)))
    print(f"E0           {e0:.4f} eV   max|F| {np.abs(f0).max():.2f} eV/A")
    # Per-term breakdown: when the total is wrong this says which term to blame.
    for name, fns in zip(terms, energy.term_fns, strict=True):
        if fns.jax_energy_fn is None:
            continue
        try:
            ei = float(fns.jax_energy_fn(jnp.asarray(r), **kw))
        except TypeError:
            ei = float(fns.jax_energy_fn(jnp.asarray(r)))
        print(f"  {name:14s} {ei:16.4f} eV")
    if not np.isfinite(e0) or abs(e0) > 1e4:
        raise SystemExit(
            f"initial energy {e0} is not physical for {system.n_atoms} atoms; "
            "check that ml_intra is restricted to the solute"
        )

    # --- run ----------------------------------------------------------------
    import dataclasses

    from mmml.md.assemble import assemble_and_run

    masses = np.asarray([_mass(z) for z in np.asarray(system.Z)], dtype=float)

    def leg(sys_in, ensemble: str, n_steps: int, label: str, out_dir: Path):
        cfg = RunConfig(
            system=None,  # pre-built system is passed explicitly
            terms=tuple(terms),
            ensemble=EnsembleSpec(
                ensemble=ensemble,
                space="pbc",
                temperature_K=args.temperature,
                dt_fs=args.dt_fs,
                n_steps=n_steps,
                params={"masses": masses, "seed": args.seed, "float64": True},
            ),
            backend="jaxmd",
            output_dir=out_dir,
            seed=args.seed,
        )
        t = assemble_and_run(cfg, system=sys_in, ctx=ctx, term_kwargs=term_kwargs)
        e = np.asarray(t.metadata["energies"])
        print(f"  {label:12s} {n_steps:7d} steps   E {e[0]:12.3f} -> {e[-1]:12.3f} eV")
        final = np.asarray(t.metadata["positions"])[-1]
        return dataclasses.replace(sys_in, R=final), t

    print("running")
    if args.minimize_steps > 0:
        system, _ = leg(system, "min", args.minimize_steps, "minimise", out / "min")
    if args.equilibrate_ps > 0:
        n_eq = int(round(args.equilibrate_ps * 1000.0 / args.dt_fs))
        system, _ = leg(system, args.ensemble, n_eq, "equilibrate", out / "equil")

    n_steps = int(round(args.ps * 1000.0 / args.dt_fs))
    system, traj = leg(system, args.ensemble, n_steps, "production", out)

    energies = np.asarray(traj.metadata["energies"])
    positions = np.asarray(traj.metadata["positions"])
    xi_t = np.linalg.norm(positions[:, IDX_C] - positions[:, IDX_CL], axis=-1) - (
        np.linalg.norm(positions[:, IDX_C] - positions[:, IDX_N], axis=-1)
    )
    summary = {
        "solvent": args.solvent,
        "residue": resi,
        "composition": composition,
        "box_size_A": box_size,
        "n_atoms": int(system.n_atoms),
        "n_solvent": int(n_solvent),
        "terms": terms,
        "xi0": args.xi0,
        "k_ev_A2": args.k_ev,
        "ensemble": args.ensemble,
        "dt_fs": args.dt_fs,
        "ps": args.ps,
        "temperature_K": args.temperature,
        "E_first_eV": float(energies[0]),
        "E_last_eV": float(energies[-1]),
        "xi_mean_A": float(xi_t.mean()),
        "xi_std_A": float(xi_t.std()),
        "xi_trace_A": xi_t.tolist(),
        "energies_eV": energies.tolist(),
    }
    (out / "solvated_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"E            {energies[0]:.4f} -> {energies[-1]:.4f} eV")
    print(f"xi           {xi_t.mean():+.3f} +/- {xi_t.std():.3f} A")
    print(f"wrote        {out/'solvated_summary.json'}")
    if not np.all(np.isfinite(energies)):
        print("FAIL: non-finite energies", file=sys.stderr)
        return 1
    print("PASS")
    return 0


def _mass(z: int) -> float:
    from ase.data import atomic_masses

    return float(atomic_masses[int(z)])


def _load_model(checkpoint: Path):
    from mmml.interfaces.calculators.simple_inference import (
        create_calculator_from_checkpoint,
    )

    calc = create_calculator_from_checkpoint(str(checkpoint))
    model = getattr(calc, "model", None) or calc._mmml_physnet_model
    params = getattr(calc, "params", None) or calc._mmml_physnet_params
    return model, params


if __name__ == "__main__":
    sys.exit(main())
