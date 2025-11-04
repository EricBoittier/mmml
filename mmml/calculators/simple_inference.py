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


class SimpleInferenceCalculator(Calculator):
    """ASE calculator wrapper for PhysNet/DCMNet checkpoints.

    The underlying model was trained with padded batches (``natoms``), so we
    pad inputs internally while keeping the real atom count via masks.  This
    allows inference on arbitrary molecules (as long as ``natoms`` from the
    checkpoint is not exceeded).
    """

    implemented_properties = ["energy", "forces", "dipole", "charges"]

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

        if self.use_dcmnet_dipole:
            if "mono_dist" in output and "dipo_dist" in output:
                mono_dist = np.array(output["mono_dist"])
                dipo_dist = np.array(output["dipo_dist"])
                dipole = np.sum(mono_dist[:n_atoms, ..., None] * dipo_dist[:n_atoms], axis=(0, 1))
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

    repo_root = Path(__file__).resolve().parents[2]
    example_dir = repo_root / "examples" / "co2" / "dcmnet_physnet_train"
    trainer_path = example_dir / "trainer.py"

    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer module not found at {trainer_path}")

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
