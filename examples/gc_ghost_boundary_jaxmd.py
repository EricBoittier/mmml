#!/usr/bin/env python
"""Ghost-boundary ML/MM proof of concept in JAX.

This example implements the simplest ML/MM ownership policy:

    E_total = E_ML(ML region)
            + E_MM_bonded(not fully inside ML region)
            + E_MM_nonbonded(all atoms)

The "ML" term here is intentionally a small reference-distance surrogate over
the selected atoms.  It is a stand-in for a real neural model while the topology
and force ownership are wired up and testable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.fire import FIRE
from jax_md import space

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
    bonded_energy_and_forces,
)
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    CgenffBondedSystem,
    filter_bonded_topology_excluding_ml_interior,
)
from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import (
    _nbond_settings_from_cutoffs,
)
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_PRM,
    ensure_pycharmm_loaded,
    pycharmm_loud,
)
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    load_bonded_system_from_psf,
    load_nonbonded_system_from_charmm,
    nonbonded_energy_and_forces,
)
from mmml.interfaces.pycharmmInterface.nbonds_config import PbcNbondCutoffs
from mmml.interfaces.pycharmmInterface.peptide_builder import (
    build_peptide_in_charmm,
)
from mmml.interfaces.pycharmmInterface.protein_charmm_build import (
    protein_toppar_paths,
)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True, slots=True)
class PsfAtomRecord:
    index: int
    segid: str
    resid: str
    resname: str
    atom_name: str


def parse_psf_atom_records(psf_path: Path | str) -> list[PsfAtomRecord]:
    records: list[PsfAtomRecord] = []
    lines = Path(psf_path).read_text(encoding="utf-8", errors="replace").splitlines()
    for row, line in enumerate(lines):
        if "!NATOM" not in line:
            continue
        n_atoms = int(line.split()[0])
        for atom_line in lines[row + 1 : row + 1 + n_atoms]:
            parts = atom_line.split()
            records.append(
                PsfAtomRecord(
                    index=int(parts[0]) - 1,
                    segid=parts[1],
                    resid=parts[2],
                    resname=parts[3],
                    atom_name=parts[4],
                )
            )
        break
    if not records:
        raise ValueError(f"No !NATOM section found in {psf_path}")
    return records


def parse_residue_range(spec: str) -> tuple[int, int]:
    if ":" in spec:
        start, stop = spec.split(":", 1)
        return int(start), int(stop)
    value = int(spec)
    return value, value


def select_pept_residue_atoms(psf_path: Path | str, residue_range: str) -> tuple[int, ...]:
    first, last = parse_residue_range(residue_range)
    selected = [
        rec.index
        for rec in parse_psf_atom_records(psf_path)
        if rec.segid == "PEPT" and first <= int(rec.resid) <= last
    ]
    if not selected:
        raise ValueError(f"No PEPT atoms selected by residue range {residue_range!r}")
    return tuple(selected)


def build_reference_pair_surrogate(
    positions: np.ndarray,
    ml_atom_indices: tuple[int, ...],
    *,
    cutoff_A: float = 2.2,
    k_ev_A2: float = 2.0,
):
    ml_idx = np.asarray(ml_atom_indices, dtype=np.int32)
    ref = np.asarray(positions[ml_idx], dtype=np.float64)
    pair_i: list[int] = []
    pair_j: list[int] = []
    r0: list[float] = []
    for i in range(ref.shape[0]):
        for j in range(i + 1, ref.shape[0]):
            dist = float(np.linalg.norm(ref[j] - ref[i]))
            if 0.4 < dist <= cutoff_A:
                pair_i.append(i)
                pair_j.append(j)
                r0.append(dist)
    if not pair_i:
        raise ValueError("No reference ML pairs found; increase --ml-cutoff-A")

    pair_i_j = jnp.asarray(pair_i, dtype=jnp.int32)
    pair_j_j = jnp.asarray(pair_j, dtype=jnp.int32)
    r0_j = jnp.asarray(r0, dtype=jnp.float64)
    k = float(k_ev_A2)

    def energy_and_forces(local_positions):
        pos = jnp.asarray(local_positions, dtype=jnp.float64)

        def energy_fn(p):
            dr = p[pair_j_j] - p[pair_i_j]
            dist = jnp.linalg.norm(dr, axis=-1)
            return jnp.sum(0.5 * k * jnp.square(dist - r0_j))

        energy = energy_fn(pos)
        forces = -jax.grad(energy_fn)(pos)
        return energy, forces

    return energy_and_forces, len(pair_i)


def make_boundary_bonded_system(
    system: CgenffBondedSystem,
    ml_atom_indices: tuple[int, ...],
) -> CgenffBondedSystem:
    topology, bonded, urey_k, urey_r0 = filter_bonded_topology_excluding_ml_interior(
        system.topology,
        system.bonded,
        ml_atom_indices,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
    )
    return CgenffBondedSystem(
        positions=system.positions,
        topology=topology,
        bonded=bonded,
        atom_types=system.atom_types,
        charges=system.charges,
        urey_k=urey_k,
        urey_r0=urey_r0,
    )


class GhostBoundaryCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        ml_atom_indices: tuple[int, ...],
        ml_energy_fn,
        bonded_system: CgenffBondedSystem,
        nbond_data,
        nbond_settings,
        cell: np.ndarray,
    ):
        super().__init__()
        self.ml_atom_indices = np.asarray(ml_atom_indices, dtype=np.int32)
        self.ml_energy_fn = ml_energy_fn
        self.bonded_system = bonded_system
        self.nbond_data = nbond_data
        self.nbond_settings = nbond_settings
        self.cell = np.asarray(cell, dtype=np.float64)
        self.displacement_fn, _ = space.free()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = jnp.asarray(self.atoms.get_positions(), dtype=jnp.float64)

        ml_energy, ml_forces_local = self.ml_energy_fn(positions[self.ml_atom_indices])
        ml_forces = jnp.zeros_like(positions)
        ml_forces = ml_forces.at[self.ml_atom_indices].set(ml_forces_local)

        bonded_components, bonded_forces = bonded_energy_and_forces(
            positions,
            self.bonded_system.topology,
            self.bonded_system.bonded,
            self.displacement_fn,
            urey_k=self.bonded_system.urey_k,
            urey_r0=self.bonded_system.urey_r0,
            energy_unit="eV",
            include_cmap=False,
        )
        nb_components, nb_forces = nonbonded_energy_and_forces(
            positions,
            self.nbond_data,
            self.cell,
            self.nbond_settings,
        )
        nb_energy = nb_components["total"] * KCAL_MOL_TO_EV
        nb_forces_ev = nb_forces * KCAL_MOL_TO_EV

        total_energy = ml_energy + bonded_components["total"] + nb_energy
        total_forces = ml_forces + bonded_forces + nb_forces_ev

        self.results["energy"] = float(total_energy)
        self.results["forces"] = np.asarray(total_forces, dtype=np.float64)
        self.results["components"] = {
            "ml_surrogate_eV": float(ml_energy),
            "mm_bonded_boundary_eV": float(bonded_components["total"]),
            "mm_nonbonded_all_eV": float(nb_energy),
            "total_eV": float(total_energy),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="ALA ALA ALA ALA ALA")
    parser.add_argument("--ml-residues", default="2:4")
    parser.add_argument("--box-size", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/gc_ghost_boundary_jaxmd"))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--ml-cutoff-A", type=float, default=2.2)
    parser.add_argument("--ml-k-ev-A2", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM is not available.")
    pycharmm_loud()

    peptide = build_peptide_in_charmm(
        args.sequence,
        seed=args.seed,
        workdir=args.workdir,
        minimize=True,
    )
    positions = np.asarray(peptide.positions, dtype=np.float64).copy()
    positions -= positions.mean(axis=0, keepdims=True)
    positions += float(args.box_size) / 2.0

    ml_atom_indices = select_pept_residue_atoms(peptide.psf_path, args.ml_residues)
    protein_prm = protein_toppar_paths().prm
    full_bonded = load_bonded_system_from_psf(
        peptide.psf_path,
        positions,
        prm_file=protein_prm,
        extra_prm_files=(CGENFF_PRM,),
    )
    boundary_bonded = make_boundary_bonded_system(full_bonded, ml_atom_indices)

    nbond_data = load_nonbonded_system_from_charmm(peptide.psf_path, CGENFF_PRM, protein_prm)
    cutoffs = PbcNbondCutoffs(
        cubic_box_side_A=float(args.box_size),
        cutnb=15.0,
        cutim=15.0,
        ctonnb=10.0,
        ctofnb=12.0,
        ctexnb=999.0,
    )
    nbond_settings = _nbond_settings_from_cutoffs(cutoffs)
    cell = np.diag([float(args.box_size)] * 3)
    ml_energy_fn, n_ml_pairs = build_reference_pair_surrogate(
        positions,
        ml_atom_indices,
        cutoff_A=args.ml_cutoff_A,
        k_ev_A2=args.ml_k_ev_A2,
    )

    atoms = Atoms(get_Z_from_psf(), positions=positions, cell=cell, pbc=True)
    atoms.calc = GhostBoundaryCalculator(
        ml_atom_indices=ml_atom_indices,
        ml_energy_fn=ml_energy_fn,
        bonded_system=boundary_bonded,
        nbond_data=nbond_data,
        nbond_settings=nbond_settings,
        cell=cell,
    )

    initial_energy = atoms.get_potential_energy()
    print("--- Ghost-boundary ML/MM POC ---")
    print(f"Sequence: {args.sequence}")
    print(f"ML residues: {args.ml_residues} -> {len(ml_atom_indices)} atoms")
    print(f"ML surrogate reference pairs: {n_ml_pairs}")
    print(f"Boundary MM bonds kept: {boundary_bonded.topology.bonds.shape[0]} / {full_bonded.topology.bonds.shape[0]}")
    print(f"Boundary MM angles kept: {boundary_bonded.topology.angles.shape[0]} / {full_bonded.topology.angles.shape[0]}")
    print(f"Initial components: {atoms.calc.results['components']}")

    if args.steps > 0:
        traj_path = args.workdir / "ghost_boundary_fire.traj"
        log_path = args.workdir / "ghost_boundary_fire.log"
        opt = FIRE(atoms, trajectory=str(traj_path), logfile=str(log_path), maxstep=0.03)
        opt.run(fmax=0.05, steps=args.steps)
        print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
        print(f"Wrote FIRE log: {log_path}")
        print(f"Wrote FIRE trajectory: {traj_path}")
    else:
        print(f"Initial energy: {initial_energy:.6f} eV")


if __name__ == "__main__":
    main()
