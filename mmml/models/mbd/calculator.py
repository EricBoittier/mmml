"""ASE calculator wrapper for learned QCML MBD models."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without ASE.
    raise ModuleNotFoundError("QCML MBD calculator requires ASE.") from exc

from mmml.models.mbd.model import E3xMBDModel, mbd_energy_and_forces


@dataclass(frozen=True)
class MBDModelConfig:
    features: int = 64
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 12.0
    max_atomic_number: int = 118

ANGSTROM_TO_BOHR = 1.0 / Bohr
HARTREE_TO_EV = Hartree
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr


def load_mbd_model(checkpoint: str | Path) -> tuple[E3xMBDModel, Any]:
    """Load a trained QCML MBD checkpoint."""
    checkpoint = Path(checkpoint).expanduser()
    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model_config.json: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(MBDModelConfig)}
    model_config = {key: value for key, value in raw_config.items() if key in valid}
    MBDModelConfig(**model_config)
    restored = ocp.PyTreeCheckpointer().restore(checkpoint)
    if "params" not in restored:
        raise KeyError(f"Checkpoint {checkpoint} does not contain params")
    return E3xMBDModel(**model_config), restored["params"]


def atoms_to_mbd_batch(
    atoms: Atoms,
    *,
    charge: float = 0.0,
    multiplicity: float = 1.0,
) -> dict[str, jax.Array]:
    """Build a batch_size=1 fully connected E3x MBD batch from ASE atoms."""
    positions_bohr = np.asarray(atoms.get_positions(), dtype=np.float32) * ANGSTROM_TO_BOHR
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    num_atoms = len(atomic_numbers)
    dst_idx, src_idx = map(np.asarray, e3x.ops.sparse_pairwise_indices(num_atoms))
    return {
        "positions": jnp.asarray(positions_bohr.reshape(-1, 3)),
        "atomic_numbers": jnp.asarray(atomic_numbers.reshape(-1)),
        "charge": jnp.asarray([charge], dtype=jnp.float32),
        "spin": jnp.asarray([multiplicity], dtype=jnp.float32),
        "dst_idx": jnp.asarray(dst_idx, dtype=jnp.int32),
        "src_idx": jnp.asarray(src_idx, dtype=jnp.int32),
        "batch_segments": jnp.zeros(num_atoms, dtype=jnp.int32),
        "batch_size": 1,
        "atom_mask": jnp.ones(num_atoms, dtype=jnp.float32),
        "edge_mask": jnp.ones(len(dst_idx), dtype=jnp.float32),
    }


def predict_mbd_from_atoms(
    model: E3xMBDModel,
    params: Any,
    atoms: Atoms,
    *,
    charge: float = 0.0,
    multiplicity: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Predict learned MBD energy/forces/properties for one ASE structure."""
    batch = atoms_to_mbd_batch(atoms, charge=charge, multiplicity=multiplicity)
    inputs = dict(batch)
    inputs.pop("batch_size")
    output, forces = mbd_energy_and_forces(
        model,
        params,
        **inputs,
        batch_size=1,
    )
    forces_hartree_bohr = np.asarray(forces, dtype=np.float64).reshape(len(atoms), 3)
    energy_hartree = float(np.asarray(output["energy"])[0])
    return {
        "energy_hartree": energy_hartree,
        "energy_ev": energy_hartree * HARTREE_TO_EV,
        "forces_hartree_bohr": forces_hartree_bohr,
        "forces_ev_angstrom": forces_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        "polarizabilities_bohr3": np.asarray(output["polarizabilities"], dtype=np.float64),
        "c6_native": np.asarray(output["c6_coefficients"], dtype=np.float64),
    }


class QCMLMBDCalculator(Calculator):
    """ASE calculator for the learned QCML MBD surrogate.

    ASE-facing `energy` is eV and `forces` are eV/Angstrom. Raw atomic-unit
    values are retained in `results`.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        charge: float = 0.0,
        multiplicity: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model, self.params = load_mbd_model(checkpoint)
        self.charge = float(charge)
        self.multiplicity = float(multiplicity)
        self._predict = jax.jit(self._predict_impl)

    def _predict_impl(self, batch: dict[str, jax.Array]) -> tuple[dict[str, jax.Array], jax.Array]:
        inputs = dict(batch)
        inputs.pop("batch_size")
        return mbd_energy_and_forces(
            self.model,
            self.params,
            **inputs,
            batch_size=1,
        )

    def predict_mbd(self, atoms: Atoms) -> dict[str, np.ndarray | float]:
        batch = atoms_to_mbd_batch(
            atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        output, forces = self._predict(batch)
        forces_hartree_bohr = np.asarray(forces, dtype=np.float64).reshape(len(atoms), 3)
        energy_hartree = float(np.asarray(output["energy"])[0])
        return {
            "energy_hartree": energy_hartree,
            "energy_ev": energy_hartree * HARTREE_TO_EV,
            "forces_hartree_bohr": forces_hartree_bohr,
            "forces_ev_angstrom": forces_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "polarizabilities_bohr3": np.asarray(output["polarizabilities"], dtype=np.float64),
            "c6_native": np.asarray(output["c6_coefficients"], dtype=np.float64),
        }

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Iterable[str] = ("energy", "forces"),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        prediction = self.predict_mbd(self.atoms)
        self.results["energy"] = prediction["energy_ev"]
        self.results["forces"] = prediction["forces_ev_angstrom"]
        self.results.update(prediction)
