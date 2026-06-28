"""Adapters from ASE calculators and legacy QC evaluators."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mmml.interfaces.energy_forces.protocol import (
    EnergyForcesProvider,
    ProviderCapabilities,
    ProviderKind,
)
from mmml.interfaces.qc_backends.npz_output import stack_frame_results


class AseCalculatorProvider:
    """Wrap any ASE ``Calculator`` with energy/forces as an :class:`EnergyForcesProvider`."""

    name = "ase"

    def __init__(
        self,
        calculator: Calculator,
        *,
        method_label: str | None = None,
        energy_unit: str = "ev",
        force_unit: str = "ev_angstrom",
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self._calc = calculator
        self.method_label = method_label or type(calculator).__name__
        self.energy_unit = energy_unit
        self.force_unit = force_unit
        props = set(getattr(calculator, "implemented_properties", []) or [])
        self.capabilities = capabilities or ProviderCapabilities(
            kind=ProviderKind.ASE,
            supports_dipole="dipole" in props,
            supports_charges="charges" in props,
            supports_decomposed_ml=False,
        )

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        _ = context
        want_forces = "forces" in properties or "F" in properties
        want_dipole = "dipole" in properties or "D" in properties

        energies: list[float] = []
        forces: list[np.ndarray] | None = [] if want_forces else None
        dipoles: list[np.ndarray] | None = [] if want_dipole else None
        frames_z: list[np.ndarray] = []
        frames_r: list[np.ndarray] = []

        for atoms in frames:
            eval_atoms = atoms.copy()
            eval_atoms.calc = self._calc
            energies.append(float(eval_atoms.get_potential_energy()))
            zi = np.asarray(eval_atoms.get_atomic_numbers(), dtype=int)
            ri = np.asarray(eval_atoms.get_positions(), dtype=np.float64)
            frames_z.append(zi)
            frames_r.append(ri)
            if forces is not None:
                forces.append(np.asarray(eval_atoms.get_forces(), dtype=np.float64))
            if dipoles is not None:
                try:
                    dipoles.append(
                        np.asarray(eval_atoms.calc.get_dipole_moment(), dtype=np.float64)
                    )
                except Exception:
                    pass

        return stack_frame_results(
            energies=energies,
            forces=forces,
            dipoles=dipoles,
            frames_z=frames_z,
            frames_r=frames_r,
        )


class QCEvaluatorAdapter:
    """Adapt a legacy cross-check evaluator to :class:`EnergyForcesProvider`."""

    def __init__(
        self,
        evaluator: EnergyForcesProvider,
        *,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self._evaluator = evaluator
        self.name = evaluator.name
        self.method_label = evaluator.method_label
        self.energy_unit = evaluator.energy_unit
        self.force_unit = evaluator.force_unit
        self.capabilities = capabilities or getattr(
            evaluator, "capabilities", ProviderCapabilities(kind=ProviderKind.UNKNOWN)
        )

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        try:
            return self._evaluator.evaluate_batch(
                frames, properties=properties, context=context
            )
        except TypeError:
            return self._evaluator.evaluate_batch(frames, properties=properties)
