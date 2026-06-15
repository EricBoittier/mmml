"""Batched MMML inference for ORCA external-tool requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import e3x
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mmml.cli.misc.fix_and_split import (
    convert_energy_ev_to_hartree,
    convert_forces_ev_angstrom_to_hartree_bohr,
)
from mmml.interfaces.calculators.simple_inference import SimpleInferenceCalculator


def _mmml_forces_to_orca_gradient(forces_ev_angstrom: np.ndarray) -> np.ndarray:
    gradient_ev_angstrom = -np.asarray(forces_ev_angstrom, dtype=float)
    return convert_forces_ev_angstrom_to_hartree_bohr(gradient_ev_angstrom).reshape(-1)


def _evaluate_structure_single(
    atoms: Atoms,
    calculator: Calculator,
    *,
    do_gradient: bool,
) -> tuple[float, list[float]]:
    atoms.calc = calculator
    properties = ["energy", "forces"] if do_gradient else ["energy"]
    calculator.calculate(atoms, properties=properties)
    energy_hartree = convert_energy_ev_to_hartree(
        np.asarray(atoms.get_potential_energy(), dtype=float)
    )
    gradient: list[float] = []
    if do_gradient:
        gradient = _mmml_forces_to_orca_gradient(atoms.get_forces()).tolist()
    return float(energy_hartree), gradient


@dataclass(frozen=True)
class OrcaStructureJob:
    """One ORCA external-tool structure evaluation."""

    atoms: Atoms
    do_gradient: bool


def can_batch_calculator(calculator: Calculator) -> bool:
    """Return True when ``calculator`` supports multi-structure GPU batching."""
    return isinstance(calculator, SimpleInferenceCalculator) or hasattr(
        calculator, "_mmml_physnet_model"
    )


def evaluate_structures_batched(
    calculator: Calculator,
    jobs: list[OrcaStructureJob],
) -> list[tuple[float, list[float]]]:
    """Evaluate one or more structures, batching on GPU when supported."""
    if not jobs:
        return []

    if len(jobs) == 1 or not can_batch_calculator(calculator):
        return [
            _evaluate_structure_single(job.atoms, calculator, do_gradient=job.do_gradient)
            for job in jobs
        ]

    if isinstance(calculator, SimpleInferenceCalculator):
        return _evaluate_simple_inference_batch(calculator, jobs)

    return _evaluate_physnet_ef_batch(calculator, jobs)


def _validate_atom_counts(jobs: list[OrcaStructureJob], natoms: int) -> list[int]:
    counts: list[int] = []
    for job in jobs:
        n_atoms = len(job.atoms)
        if n_atoms > natoms:
            raise ValueError(
                f"Structure has {n_atoms} atoms but model was trained with natoms={natoms}."
            )
        counts.append(n_atoms)
    return counts


def _build_padded_batch(
    jobs: list[OrcaStructureJob],
    *,
    natoms: int,
    cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    batch_size = len(jobs)
    atom_counts = _validate_atom_counts(jobs, natoms)

    padded_z = np.zeros((batch_size, natoms), dtype=np.int32)
    padded_r = np.zeros((batch_size, natoms, 3), dtype=np.float32)
    atom_mask = np.zeros((batch_size, natoms), dtype=np.float32)

    for batch_idx, job in enumerate(jobs):
        positions = job.atoms.get_positions()
        numbers = job.atoms.get_atomic_numbers()
        n_atoms = atom_counts[batch_idx]
        padded_z[batch_idx, :n_atoms] = numbers
        padded_r[batch_idx, :n_atoms] = positions
        atom_mask[batch_idx, :n_atoms] = 1.0

    if cutoff is None:
        template_dst, template_src = e3x.ops.sparse_pairwise_indices(natoms)
        offsets = np.arange(batch_size, dtype=np.int32) * natoms
        dst_idx = template_dst[None, :] + offsets[:, None]
        src_idx = template_src[None, :] + offsets[:, None]
        n_per_graph = np.array(atom_counts, dtype=np.int32)
        expanded_n = n_per_graph[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        batch_mask = (valid_dst & valid_src).astype(np.float32).reshape(-1)
        dst_idx = dst_idx.reshape(-1)
        src_idx = src_idx.reshape(-1)
    else:
        dst_list: list[int] = []
        src_list: list[int] = []
        for batch_idx, job in enumerate(jobs):
            positions = job.atoms.get_positions()
            n_atoms = atom_counts[batch_idx]
            base = batch_idx * natoms
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i == j:
                        continue
                    if np.linalg.norm(positions[i] - positions[j]) < cutoff:
                        dst_list.append(base + i)
                        src_list.append(base + j)
        dst_idx = np.asarray(dst_list, dtype=np.int32)
        src_idx = np.asarray(src_list, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32)

    batch_segments = np.repeat(np.arange(batch_size, dtype=np.int32), natoms)
    flat_z = padded_z.reshape(batch_size * natoms)
    flat_r = padded_r.reshape(batch_size * natoms, 3)
    flat_atom_mask = atom_mask.reshape(batch_size * natoms)
    return flat_z, flat_r, dst_idx, src_idx, batch_segments, batch_mask, flat_atom_mask, batch_size


def _results_from_batch_output(
    output: dict[str, Any],
    jobs: list[OrcaStructureJob],
    atom_counts: list[int],
    *,
    natoms: int,
) -> list[tuple[float, list[float]]]:
    energies = np.asarray(output["energy"], dtype=float).reshape(-1)
    forces = np.asarray(output["forces"], dtype=float)
    results: list[tuple[float, list[float]]] = []

    for batch_idx, job in enumerate(jobs):
        n_atoms = atom_counts[batch_idx]
        energy_hartree = convert_energy_ev_to_hartree(float(energies[batch_idx]))
        gradient: list[float] = []
        if job.do_gradient:
            base = batch_idx * natoms
            mol_forces = forces[base : base + n_atoms]
            gradient = _mmml_forces_to_orca_gradient(mol_forces).tolist()
        results.append((energy_hartree, gradient))

    return results


def _evaluate_simple_inference_batch(
    calculator: SimpleInferenceCalculator,
    jobs: list[OrcaStructureJob],
) -> list[tuple[float, list[float]]]:
    natoms = int(calculator.natoms)
    atom_counts = _validate_atom_counts(jobs, natoms)
    flat_z, flat_r, dst_idx, src_idx, batch_segments, batch_mask, flat_atom_mask, batch_size = (
        _build_padded_batch(jobs, natoms=natoms, cutoff=float(calculator.cutoff))
    )

    output = calculator.model.apply(
        calculator.params,
        atomic_numbers=jnp.array(flat_z),
        positions=jnp.array(flat_r),
        dst_idx=jnp.array(dst_idx),
        src_idx=jnp.array(src_idx),
        batch_segments=jnp.array(batch_segments),
        batch_size=batch_size,
        batch_mask=jnp.array(batch_mask),
        atom_mask=jnp.array(flat_atom_mask),
    )
    return _results_from_batch_output(output, jobs, atom_counts, natoms=natoms)


def _evaluate_physnet_ef_batch(
    calculator: Calculator,
    jobs: list[OrcaStructureJob],
) -> list[tuple[float, list[float]]]:
    model = calculator._mmml_physnet_model
    params = calculator._mmml_physnet_params
    natoms = int(model.natoms)
    atom_counts = _validate_atom_counts(jobs, natoms)
    flat_z, flat_r, dst_idx, src_idx, batch_segments, batch_mask, flat_atom_mask, batch_size = (
        _build_padded_batch(jobs, natoms=natoms, cutoff=None)
    )

    is_spooky = bool(getattr(calculator, "_mmml_physnet_is_spooky", False))
    apply_kwargs: dict[str, Any] = {
        "atomic_numbers": jnp.array(flat_z),
        "positions": jnp.array(flat_r),
        "dst_idx": jnp.array(dst_idx),
        "src_idx": jnp.array(src_idx),
        "batch_segments": jnp.array(batch_segments),
        "batch_size": batch_size,
        "batch_mask": jnp.array(batch_mask),
        "atom_mask": jnp.array(flat_atom_mask),
    }
    if is_spooky:
        apply_kwargs["charges"] = jnp.full((batch_size * natoms, 1), calculator._mmml_spooky_charge)
        apply_kwargs["spins"] = jnp.full((batch_size * natoms, 1), calculator._mmml_spooky_multiplicity)

    output = model.apply(params, **apply_kwargs)
    return _results_from_batch_output(output, jobs, atom_counts, natoms=natoms)
