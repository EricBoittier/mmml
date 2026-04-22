#!/usr/bin/env python3
"""Reusable simple ASE calculator for Joint PhysNet models."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from ase.calculators.calculator import Calculator, all_changes
from pathlib import Path
from typing import Any, Optional
import importlib.util
import pickle
import sys

from mmml.data.units import ANGSTROM_TO_BOHR


class SimpleInferenceCalculator(Calculator):
    """ASE calculator wrapper for PhysNet/DCMNet checkpoints.

    The underlying model was trained with padded batches (``natoms``), so we
    pad inputs internally while keeping the real atom count via masks.  This
    allows inference on arbitrary molecules (as long as ``natoms`` from the
    checkpoint is not exceeded).
    """

    implemented_properties = ["energy", "forces", "dipole", "charges", "multipoles"]

    def __init__(
        self,
        model: Any,
        params: Any,
        cutoff: float = 10.0,
        use_dcmnet_dipole: bool = False,
        **kwargs,
    ):  # noqa: D401
        super().__init__(**kwargs)
        self.model = model
        self.params = params
        self.cutoff = cutoff
        self.use_dcmnet_dipole = use_dcmnet_dipole
        self._last_positions: Optional[np.ndarray] = None
        self._last_atomic_numbers: Optional[np.ndarray] = None
        self._last_monopoles: Optional[np.ndarray] = None
        self._last_dipole_positions: Optional[np.ndarray] = None

        phys_cfg = getattr(model, "physnet_config", {})
        self.natoms: Optional[int] = phys_cfg.get("natoms")
        if self.natoms is None:
            raise ValueError("Model config missing 'natoms'; cannot determine padding size")

        if hasattr(model, "dcmnet_config"):
            self.n_dcm = model.dcmnet_config["n_dcm"]
        elif hasattr(model, "noneq_config"):
            self.n_dcm = model.noneq_config["n_dcm"]
        else:
            print("Model has no dcmnet_config or noneq_config")
            print(model)
            self.n_dcm = 3

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("Atoms object is required")

        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        n_atoms = len(atoms)
        self._last_positions = np.array(positions, copy=True)
        self._last_atomic_numbers = np.array(atomic_numbers, copy=True)

        if n_atoms > self.natoms:
            raise ValueError(
                f"Structure has {n_atoms} atoms but model was trained with natoms={self.natoms}. "
                "Please retrain with larger natoms or reduce the system size."
            )

        padded_positions = np.zeros((self.natoms, 3), dtype=np.float32)
        padded_positions[:n_atoms] = positions

        padded_atomic_numbers = np.zeros(self.natoms, dtype=np.int32)
        padded_atomic_numbers[:n_atoms] = atomic_numbers

        atom_mask = np.zeros(self.natoms, dtype=np.float32)
        atom_mask[:n_atoms] = 1.0

        dst_list = []
        src_list = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    if np.linalg.norm(positions[i] - positions[j]) < self.cutoff:
                        dst_list.append(i)
                        src_list.append(j)

        dst_idx = np.array(dst_list, dtype=np.int32)
        src_idx = np.array(src_list, dtype=np.int32)

        batch_segments = np.zeros(self.natoms, dtype=np.int32)
        batch_mask = np.ones(len(dst_idx), dtype=np.float32)

        output = self.model.apply(
            self.params,
            atomic_numbers=jnp.array(padded_atomic_numbers),
            positions=jnp.array(padded_positions),
            dst_idx=jnp.array(dst_idx),
            src_idx=jnp.array(src_idx),
            batch_segments=jnp.array(batch_segments),
            batch_size=1,
            batch_mask=jnp.array(batch_mask),
            atom_mask=jnp.array(atom_mask),
        )

        energy = output["energy"]
        self.results["energy"] = float(energy[0]) if energy.ndim > 0 else float(energy)
        self.results["forces"] = np.array(output["forces"])[:n_atoms]

        if "charges_as_mono" in output:
            charges = np.array(output["charges_as_mono"])[:n_atoms]
        elif "charges" in output:
            charges = np.array(output["charges"]).squeeze()[:n_atoms]
        else:
            charges = np.zeros(n_atoms)
        self.results["charges"] = charges

        if "mono_dist" in output and "dipo_dist" in output:
            mono_dist = np.array(output["mono_dist"])[:n_atoms]
            dipo_dist = np.array(output["dipo_dist"])[:n_atoms]
            self._last_monopoles = mono_dist
            self._last_dipole_positions = dipo_dist
            self.results["multipoles"] = {
                "monopoles": mono_dist,
                "dipole_positions": dipo_dist,
                "atomic_charges": mono_dist.sum(axis=-1),
            }
        else:
            self._last_monopoles = None
            self._last_dipole_positions = None
            self.results["multipoles"] = None

        if self.use_dcmnet_dipole:
            if "mono_dist" in output and "dipo_dist" in output:
                dipole = np.sum(
                    self._last_monopoles[..., None] * self._last_dipole_positions,
                    axis=(0, 1),
                )
            elif "dipole_dcm" in output:
                dipole = np.array(output["dipole_dcm"][0])
            else:
                dipole = np.array(output["dipoles"][0])
        else:
            if "dipoles" in output:
                dipole = np.array(output["dipoles"][0])
            elif "dipole_physnet" in output:
                dipole = np.array(output["dipole_physnet"][0])
            else:
                dipole = np.sum(charges[:, None] * positions, axis=0)

        self.results["dipole"] = dipole

    def get_distributed_multipoles(self) -> dict[str, np.ndarray]:
        """Return distributed multipoles from the most recent calculation."""
        if self._last_monopoles is None or self._last_dipole_positions is None:
            raise RuntimeError(
                "No distributed multipoles available. Run a calculation on a joint "
                "PhysNet+DCMNet/NonEquivariant model first."
            )
        return {
            "monopoles": self._last_monopoles.copy(),
            "dipole_positions": self._last_dipole_positions.copy(),
            "atomic_charges": self._last_monopoles.sum(axis=-1),
        }

    def get_electrostatic_potential(
        self,
        grid_points: np.ndarray,
        source: str = "dcmnet",
    ) -> np.ndarray:
        """Compute ESP on grid points from the latest prediction.

        Parameters
        ----------
        grid_points
            Grid points with shape (n_grid, 3) in Angstrom.
        source
            'dcmnet' (distributed charges) or 'physnet' (atomic point charges).
        """
        grid_points = np.asarray(grid_points, dtype=np.float32).reshape(-1, 3)
        source_norm = source.lower()

        if source_norm == "dcmnet":
            if self._last_monopoles is None or self._last_dipole_positions is None:
                raise RuntimeError(
                    "No DCMNet multipoles available. Run a calculation first."
                )
            from mmml.dcmnet.dcmnet.electrostatics import calc_esp

            charge_values = self._last_monopoles.reshape(-1)
            charge_positions = self._last_dipole_positions.reshape(-1, 3)
            esp = calc_esp(
                charge_positions=jnp.array(charge_positions),
                charge_values=jnp.array(charge_values),
                grid_positions=jnp.array(grid_points),
            )
            return np.array(esp)

        if source_norm == "physnet":
            if self._last_positions is None:
                raise RuntimeError("No cached structure available. Run a calculation first.")
            from mmml.utils.electrostatics import compute_esp_from_point_charges

            charges = np.asarray(self.results.get("charges"))
            return compute_esp_from_point_charges(
                charges=charges,
                atom_pos=self._last_positions,
                grid_positions=grid_points,
                atom_mask=None,
            )

        raise ValueError("source must be 'dcmnet' or 'physnet'")

    def write_esp_cube(
        self,
        output_path: str | Path,
        source: str = "dcmnet",
        spacing_angstrom: float = 0.25,
        padding_angstrom: float = 3.0,
        origin_angstrom: Optional[np.ndarray] = None,
        grid_shape: Optional[tuple[int, int, int]] = None,
    ) -> Path:
        """Write ESP from current structure to a Gaussian cube file.

        Parameters
        ----------
        output_path
            Destination `.cube` file path.
        source
            ESP source passed to `get_electrostatic_potential`:
            'dcmnet' or 'physnet'.
        spacing_angstrom
            Grid spacing in Angstrom for auto-generated grid.
        padding_angstrom
            Padding around the molecule in Angstrom for auto-generated grid.
        origin_angstrom
            Optional grid origin in Angstrom. If set, `grid_shape` must also be set.
        grid_shape
            Optional (nx, ny, nz). If set, `origin_angstrom` must also be set.
        """
        if self._last_positions is None or self._last_atomic_numbers is None:
            raise RuntimeError(
                "No cached structure available. Run a calculation first "
                "(e.g., atoms.get_potential_energy())."
            )

        if (origin_angstrom is None) != (grid_shape is None):
            raise ValueError(
                "origin_angstrom and grid_shape must be provided together, "
                "or both left as None for automatic grid generation."
            )

        positions = np.asarray(self._last_positions, dtype=np.float64)
        atomic_numbers = np.asarray(self._last_atomic_numbers, dtype=np.int32)

        if origin_angstrom is None:
            mins = positions.min(axis=0) - float(padding_angstrom)
            maxs = positions.max(axis=0) + float(padding_angstrom)
            extent = np.maximum(maxs - mins, 0.0)
            spacing = float(spacing_angstrom)
            nx, ny, nz = tuple(int(np.ceil(v / spacing)) + 1 for v in extent)
            nx, ny, nz = max(nx, 1), max(ny, 1), max(nz, 1)
            origin = mins
        else:
            spacing = float(spacing_angstrom)
            origin = np.asarray(origin_angstrom, dtype=np.float64).reshape(3)
            nx, ny, nz = tuple(int(v) for v in grid_shape)
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise ValueError("grid_shape must contain positive integers")

        xs = origin[0] + np.arange(nx) * spacing
        ys = origin[1] + np.arange(ny) * spacing
        zs = origin[2] + np.arange(nz) * spacing
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        esp = self.get_electrostatic_potential(grid_points, source=source)
        cube_values = np.asarray(esp, dtype=np.float64).reshape(nx, ny, nz)

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        origin_bohr = origin * ANGSTROM_TO_BOHR
        spacing_bohr = spacing * ANGSTROM_TO_BOHR
        positions_bohr = positions * ANGSTROM_TO_BOHR

        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"MMML ESP cube ({source})\n")
            f.write("Generated by SimpleInferenceCalculator.write_esp_cube\n")
            f.write(
                f"{len(atomic_numbers):5d}"
                f"{origin_bohr[0]:13.6f}{origin_bohr[1]:13.6f}{origin_bohr[2]:13.6f}\n"
            )
            f.write(f"{nx:5d}{spacing_bohr:13.6f}{0.0:13.6f}{0.0:13.6f}\n")
            f.write(f"{ny:5d}{0.0:13.6f}{spacing_bohr:13.6f}{0.0:13.6f}\n")
            f.write(f"{nz:5d}{0.0:13.6f}{0.0:13.6f}{spacing_bohr:13.6f}\n")

            for z, pos_bohr in zip(atomic_numbers, positions_bohr):
                f.write(
                    f"{int(z):5d}{0.0:13.6f}"
                    f"{pos_bohr[0]:13.6f}{pos_bohr[1]:13.6f}{pos_bohr[2]:13.6f}\n"
                )

            flat = cube_values.ravel(order="C")
            for i in range(0, flat.size, 6):
                chunk = flat[i : i + 6]
                f.write("".join(f"{val:13.5e}" for val in chunk) + "\n")

        return output_path


def create_calculator_from_checkpoint(
    checkpoint_path,
    is_noneq: bool = False,
    cutoff: Optional[float] = None,
    use_dcmnet_dipole: bool = False,
) -> SimpleInferenceCalculator:
    """Load model parameters and return a ready-to-use calculator."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Prefer canonical package path first (stable in installed/editable environments).
    try:
        from mmml.cli.misc.train_joint import (
            JointPhysNetDCMNet,
            JointPhysNetNonEquivariant,
        )
    except Exception:
        # Fallback for legacy/local workflows where only example trainer exists.
        repo_root = Path(__file__).resolve().parents[3]
        trainer_candidates = [
            repo_root / "examples" / "other" / "co2" / "dcmnet_physnet_train" / "trainer.py",
            repo_root / "examples" / "co2" / "dcmnet_physnet_train" / "trainer.py",
        ]
        trainer_path = next((p for p in trainer_candidates if p.exists()), None)
        if trainer_path is None:
            raise FileNotFoundError(
                "Could not locate trainer module. Tried package import "
                "`mmml.cli.misc.train_joint` and example trainer paths: "
                + ", ".join(str(p) for p in trainer_candidates)
            )

        spec = importlib.util.spec_from_file_location("dcmnet_trainer", trainer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load trainer module from {trainer_path}")
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)

        JointPhysNetDCMNet = trainer.JointPhysNetDCMNet  # type: ignore[attr-defined]
        JointPhysNetNonEquivariant = trainer.JointPhysNetNonEquivariant  # type: ignore[attr-defined]

    with checkpoint_path.open("rb") as f:
        checkpoint_data = pickle.load(f)

    if isinstance(checkpoint_data, dict) and "params" in checkpoint_data:
        params = checkpoint_data["params"]
    else:
        params = checkpoint_data

    config_path = checkpoint_path.parent / "model_config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open("rb") as f:
        saved_config = pickle.load(f)

    physnet_config = saved_config["physnet_config"]
    mix_coulomb_energy = saved_config.get("mix_coulomb_energy", False)

    if is_noneq:
        noneq_config = saved_config["noneq_config"]
        model = JointPhysNetNonEquivariant(
            physnet_config=physnet_config,
            noneq_config=noneq_config,
            mix_coulomb_energy=mix_coulomb_energy,
        )
    else:
        dcmnet_config = saved_config["dcmnet_config"]
        model = JointPhysNetDCMNet(
            physnet_config=physnet_config,
            dcmnet_config=dcmnet_config,
            mix_coulomb_energy=mix_coulomb_energy,
        )

    if isinstance(params, dict) and "params" not in params and (
        "physnet" in params or "noneq_model" in params
    ):
        params = {"params": params}

    if cutoff is None:
        cutoff = physnet_config.get("cutoff", 6.0)

    return SimpleInferenceCalculator(
        model=model,
        params=params,
        cutoff=cutoff,
        use_dcmnet_dipole=use_dcmnet_dipole,
    )
