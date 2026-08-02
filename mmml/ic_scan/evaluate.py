"""Calculator-neutral internal-coordinate scan evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from .calculators import calculator_factory
from .config import IcScanConfig
from .geometry import measure_all, prepare_geometries
from .grid import ScanPoint
from .result import EV_TO_KCAL_MOL, Provenance, ScanRecord, ScanResult

CalculatorFactory = Callable[[], Calculator]


def _evaluate_atoms(atoms: Atoms, factory: CalculatorFactory) -> tuple[float, np.ndarray]:
    evaluated = atoms.copy()
    evaluated.calc = factory()
    energy = float(evaluated.get_potential_energy())
    forces = np.asarray(evaluated.get_forces(), dtype=float)
    if forces.shape != (len(evaluated), 3):
        raise ValueError(f"calculator returned forces with shape {forces.shape}")
    return energy, forces


def _actual_coords_json(atoms: Atoms, config: IcScanConfig) -> str:
    measured = measure_all(atoms, config.dofs)
    return json.dumps(measured, sort_keys=True)


def _record_from_point(
    point: ScanPoint,
    atoms: Atoms,
    config: IcScanConfig,
    *,
    status: str,
    energy_ev: float | None = None,
    max_force_ev_A: float | None = None,
    error: Exception | None = None,
) -> ScanRecord:
    return ScanRecord(
        point_id=point.point_id,
        scan_name=point.scan_name,
        global_index=point.global_index,
        local_index=point.local_index,
        active_dofs=",".join(point.active_dofs),
        coordinates_json=json.dumps(point.coordinates, sort_keys=True),
        actual_coordinates_json=_actual_coords_json(atoms, config),
        status=status,  # type: ignore[arg-type]
        energy_ev=energy_ev,
        energy_kcal_mol=None if energy_ev is None else energy_ev * EV_TO_KCAL_MOL,
        max_force_ev_A=max_force_ev_A,
        error_type=None if error is None else type(error).__name__,
        error_message=None if error is None else str(error),
    )


def run_ic_scan(
    config: IcScanConfig,
    *,
    provenance: Provenance | None = None,
    calculator: CalculatorFactory | None = None,
) -> ScanResult:
    """Prepare geometries and optionally evaluate energies/forces."""

    _, prepared = prepare_geometries(config)
    records: list[ScanRecord] = []
    frames: list[Atoms] = []
    evaluate = config.evaluate == "energy"
    factory = calculator
    if evaluate and factory is None:
        factory = calculator_factory(config)

    for point, atoms in prepared:
        frame = atoms.copy()
        frame.info.update(point.to_info())
        if not evaluate:
            frame.info["status"] = "prepared"
            records.append(
                _record_from_point(point, frame, config, status="prepared")
            )
            frames.append(frame)
            continue
        assert factory is not None
        try:
            energy, forces = _evaluate_atoms(frame, factory)
            max_f = float(np.max(np.linalg.norm(forces, axis=1)))
            frame.info.update(
                status="success",
                energy_ev=energy,
                energy_kcal_mol=energy * EV_TO_KCAL_MOL,
                max_force_ev_A=max_f,
            )
            frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces)
            records.append(
                _record_from_point(
                    point,
                    frame,
                    config,
                    status="success",
                    energy_ev=energy,
                    max_force_ev_A=max_f,
                )
            )
        except Exception as exc:
            frame.info.update(
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            records.append(
                _record_from_point(point, frame, config, status="failed", error=exc)
            )
        frames.append(frame)

    return ScanResult(
        config=config,
        records=records,
        frames=frames,
        provenance=provenance or Provenance.capture(config),
    )
