"""ML checkpoint backend for cross-check evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mmml.interfaces.qc_backends.npz_output import stack_frame_results


class MLBackend:
    """Evaluate structures with an MMML checkpoint (SimpleInferenceCalculator)."""

    name = "ml"

    def __init__(
        self,
        *,
        checkpoint: Path,
        cutoff: float | None = None,
        use_dcmnet_dipole: bool = False,
        calculator_factory: Any | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.cutoff = cutoff
        self.use_dcmnet_dipole = use_dcmnet_dipole
        self._calculator_factory = calculator_factory
        self._calc = None

    @property
    def method_label(self) -> str:
        return self.checkpoint.name

    @property
    def energy_unit(self) -> str:
        return "ev"

    @property
    def force_unit(self) -> str:
        return "ev_angstrom"

    def _get_calculator(self):
        if self._calc is not None:
            return self._calc
        if self._calculator_factory is not None:
            self._calc = self._calculator_factory(self.checkpoint)
            return self._calc
        from mmml.interfaces.calculators.simple_inference import (
            create_calculator_from_checkpoint,
        )

        self._calc = create_calculator_from_checkpoint(
            self.checkpoint,
            cutoff=self.cutoff,
            use_dcmnet_dipole=self.use_dcmnet_dipole,
        )
        return self._calc

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
    ) -> dict[str, np.ndarray]:
        calc = self._get_calculator()
        model_natoms = getattr(getattr(calc, "model", None), "natoms", None)
        want_forces = "forces" in properties or "F" in properties
        want_dipole = "dipole" in properties or "D" in properties

        energies: list[float] = []
        forces: list[np.ndarray] | None = [] if want_forces else None
        dipoles: list[np.ndarray] | None = [] if want_dipole else None
        frames_z: list[np.ndarray] = []
        frames_r: list[np.ndarray] = []

        for atoms in frames:
            zi = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            ri = np.asarray(atoms.get_positions(), dtype=np.float64)
            if model_natoms is not None:
                n_pad = int(model_natoms)
                zi = zi[:n_pad]
                ri = ri[:n_pad]
            eval_atoms = Atoms(numbers=zi, positions=ri)
            eval_atoms.calc = calc
            energies.append(float(eval_atoms.get_potential_energy()))
            frames_z.append(zi)
            frames_r.append(ri)
            if forces is not None:
                forces.append(np.asarray(eval_atoms.get_forces(), dtype=np.float64))
            if want_dipole:
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


def build_ml_backend(options: dict[str, Any]) -> MLBackend:
    checkpoint = options.get("checkpoint")
    if checkpoint is None:
        raise ValueError("ML backend requires 'checkpoint' in options.")
    return MLBackend(
        checkpoint=Path(checkpoint),
        cutoff=options.get("cutoff"),
        use_dcmnet_dipole=bool(options.get("use_dcmnet_dipole", False)),
        calculator_factory=options.get("calculator_factory"),
    )
