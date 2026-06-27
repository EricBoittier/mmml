"""xTB backend (tblite) for cross-check evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from mmml.interfaces.qc_backends.npz_output import stack_frame_results


class XTBBackend:
    """Evaluate structures with tblite GFN-xTB via ASE."""

    name = "xtb"

    def __init__(
        self,
        *,
        method: str = "GFN2-xTB",
        charge: int | None = None,
        multiplicity: int | None = None,
        calculator_factory: Any | None = None,
    ) -> None:
        self.method = method
        self.charge = charge
        self.multiplicity = multiplicity
        self._calculator_factory = calculator_factory

    @property
    def method_label(self) -> str:
        return self.method

    @property
    def energy_unit(self) -> str:
        return "ev"

    @property
    def force_unit(self) -> str:
        return "ev_angstrom"

    def _make_calculator(self, atoms: Atoms):
        if self._calculator_factory is not None:
            return self._calculator_factory(self, atoms)
        try:
            from tblite.ase import TBLite
        except ImportError as exc:
            raise ImportError(
                "xTB cross-check requires tblite. Install with: "
                "uv sync --extra quantum-crosscheck"
            ) from exc

        kwargs: dict[str, Any] = {"method": self.method}
        if self.charge is not None:
            kwargs["charge"] = self.charge
        if self.multiplicity is not None:
            kwargs["multiplicity"] = self.multiplicity
        return TBLite(**kwargs)

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
    ) -> dict[str, np.ndarray]:
        want_forces = "forces" in properties or "F" in properties
        want_dipole = "dipole" in properties or "D" in properties

        energies: list[float] = []
        forces: list[np.ndarray] | None = [] if want_forces else None
        dipoles: list[np.ndarray] | None = [] if want_dipole else None
        frames_z: list[np.ndarray] = []
        frames_r: list[np.ndarray] = []

        calc = None
        for atoms in frames:
            frames_z.append(np.asarray(atoms.get_atomic_numbers(), dtype=np.int32))
            frames_r.append(np.asarray(atoms.get_positions(), dtype=np.float64))
            if calc is None:
                calc = self._make_calculator(atoms)
            eval_atoms = atoms.copy()
            eval_atoms.calc = calc
            energies.append(float(eval_atoms.get_potential_energy()))
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


def build_xtb_backend(options: dict[str, Any]) -> XTBBackend:
    method = str(options.get("method") or "GFN2-xTB")
    charge = options.get("charge")
    mult = options.get("multiplicity")
    return XTBBackend(
        method=method,
        charge=int(charge) if charge is not None else None,
        multiplicity=int(mult) if mult is not None else None,
        calculator_factory=options.get("calculator_factory"),
    )


def tblite_available() -> bool:
    try:
        import tblite  # noqa: F401

        return True
    except ImportError:
        return False
