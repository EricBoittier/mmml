"""PySCF backend for cross-check evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from mmml.interfaces.qc_backends.npz_output import stack_frame_results


class PySCFBackend:
    """Evaluate structures with gpu4pyscf/PySCF DFT."""

    name = "pyscf"

    def __init__(
        self,
        *,
        xc: str = "PBE0",
        basis: str = "def2-SVP",
        spin: int = 0,
        charge: int = 0,
        compute_fn: Any | None = None,
    ) -> None:
        self.xc = xc
        self.basis = basis
        self.spin = spin
        self.charge = charge
        self._compute_fn = compute_fn

    @property
    def method_label(self) -> str:
        return f"{self.xc}/{self.basis}"

    @property
    def energy_unit(self) -> str:
        return "hartree"

    @property
    def force_unit(self) -> str:
        return "hartree_bohr"

    def _get_compute_fn(self):
        if self._compute_fn is not None:
            return self._compute_fn
        from mmml.interfaces.pyscf4gpuInterface.calcs import compute_dft_single

        return compute_dft_single

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
    ) -> dict[str, np.ndarray]:
        compute = self._get_compute_fn()
        want_forces = "forces" in properties or "F" in properties
        want_dipole = "dipole" in properties or "D" in properties

        energies: list[float] = []
        forces: list[np.ndarray] | None = [] if want_forces else None
        dipoles: list[np.ndarray] | None = [] if want_dipole else None
        frames_z: list[np.ndarray] = []
        frames_r: list[np.ndarray] = []

        for i, atoms in enumerate(frames):
            z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
            r = np.asarray(atoms.get_positions(), dtype=np.float64)
            frames_z.append(z)
            frames_r.append(r)
            out = compute(
                r,
                z,
                basis=self.basis,
                xc=self.xc,
                spin=self.spin,
                charge=self.charge,
                energy=True,
                gradient=want_forces,
                dipole=want_dipole,
                verbose=0,
            )
            energies.append(float(np.asarray(out["energy"]).reshape(())))
            if forces is not None and "gradient" in out:
                grad = np.asarray(out["gradient"], dtype=np.float64)
                forces.append(-grad)
            if dipoles is not None and "D" in out:
                dipoles.append(np.asarray(out["D"], dtype=np.float64).reshape(3))

        return stack_frame_results(
            energies=energies,
            forces=forces,
            dipoles=dipoles,
            frames_z=frames_z,
            frames_r=frames_r,
        )


def build_pyscf_backend(options: dict[str, Any]) -> PySCFBackend:
    return PySCFBackend(
        xc=str(options.get("functional") or options.get("xc") or "PBE0"),
        basis=str(options.get("basis") or "def2-SVP"),
        spin=int(options.get("spin", 0)),
        charge=int(options.get("charge", 0)),
        compute_fn=options.get("compute_fn"),
    )
