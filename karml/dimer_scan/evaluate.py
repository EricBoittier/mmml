"""Calculator-neutral dimer scan evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from mmml.analysis.dimer_scans import DimerGeometry, min_fragment_contact_distance

from .config import DimerScanConfig
from .result import Provenance, ScanRecord, ScanResult

EV_TO_KCAL_MOL = 23.060548867
CalculatorFactory = Callable[[], Calculator]


def _evaluate_atoms(atoms, factory: CalculatorFactory) -> tuple[float, np.ndarray]:
    evaluated = atoms.copy()
    evaluated.calc = factory()
    energy = float(evaluated.get_potential_energy())
    forces = np.asarray(evaluated.get_forces(), dtype=float)
    if forces.shape != (len(evaluated), 3):
        raise ValueError(f"calculator returned forces with shape {forces.shape}")
    return energy, forces


def evaluate_geometries(
    config: DimerScanConfig,
    geometries: Iterable[DimerGeometry],
    calculator_factory: CalculatorFactory,
    *,
    provenance: Provenance | None = None,
) -> ScanResult:
    """Evaluate every geometry and preserve a record for every requested point."""

    geometry_list = list(geometries)
    if len(geometry_list) != len(config.distances_angstrom):
        raise ValueError("geometry count does not match configured distance count")
    records: list[ScanRecord] = []
    frames = []
    for index, geometry in enumerate(geometry_list):
        point_id = f"point-{index:06d}"
        min_contact = min_fragment_contact_distance(geometry.atoms, geometry.fragments)
        frame = geometry.atoms.copy()
        frame.info.update(
            point_id=point_id,
            scan_index=index,
            distance_angstrom=geometry.distance_angstrom,
            min_contact_angstrom=min_contact,
        )
        try:
            total_energy, total_forces = _evaluate_atoms(frame, calculator_factory)
            energy = total_energy
            forces = total_forces
            if config.energy_definition == "interaction":
                idx_a, idx_b = geometry.fragments
                energy_a, forces_a = _evaluate_atoms(geometry.atoms[idx_a], calculator_factory)
                energy_b, forces_b = _evaluate_atoms(geometry.atoms[idx_b], calculator_factory)
                energy = total_energy - energy_a - energy_b
                forces = total_forces.copy()
                forces[idx_a] -= forces_a
                forces[idx_b] -= forces_b
            frame.info.update(
                status="success",
                energy_definition=config.energy_definition,
                energy_ev=energy,
                total_energy_ev=total_energy,
            )
            frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces)
            records.append(
                ScanRecord(
                    point_id=point_id,
                    index=index,
                    distance_angstrom=geometry.distance_angstrom,
                    min_contact_angstrom=min_contact,
                    status="success",
                    energy_ev=energy,
                    energy_kcal_mol=energy * EV_TO_KCAL_MOL,
                    total_energy_ev=total_energy,
                )
            )
        except Exception as exc:
            frame.info.update(
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            records.append(
                ScanRecord(
                    point_id=point_id,
                    index=index,
                    distance_angstrom=geometry.distance_angstrom,
                    min_contact_angstrom=min_contact,
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        frames.append(frame)
    return ScanResult(
        config=config,
        records=records,
        frames=frames,
        provenance=provenance or Provenance.capture(config),
    )
