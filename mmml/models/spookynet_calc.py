"""Standalone ASE calculator for SpookyNet-style PhysNet checkpoints (JSON format).

Loads a ``SpookyPhysNet`` model + params from a JSON checkpoint (as produced by
``mmml.utils.model_checkpoint``) and exposes it as a plain ASE ``Calculator``
so it can be dropped into the same scan pipelines as the xTB/CHARMM/learned
multipole backends (see ``mmml.analysis.dimer_scans.evaluate_scan``).

Atom counts smaller than the checkpoint's ``max_padded_atoms`` are handled by
zero-padding + masking, so one calculator instance can be reused for both the
full dimer and individual monomer fragments of varying size.
"""

from __future__ import annotations

from pathlib import Path

import ase.units
import e3x
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from mmml.interfaces.pycharmmInterface.ml_dtypes import json_tree_to_jax_params
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from mmml.utils.model_checkpoint import (
    load_model_checkpoint,
    normalize_physnet_config,
    physnet_constructor_kwargs,
)

EV_TO_KCAL_MOL = 1 / (ase.units.kcal / ase.units.mol)


class SpookyNetCalculator(Calculator):
    """ASE calculator wrapping a SpookyNet-style PhysNet model."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        charge: float = 0.0,
        spin_multiplicity: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        checkpoint = Path(checkpoint).expanduser()
        ckpt = load_model_checkpoint(
            checkpoint, use_orbax=False, load_params=True, load_config=True
        )
        params = ckpt.get("params")
        if params is None:
            raise FileNotFoundError(f"No params found in checkpoint: {checkpoint}")
        config = normalize_physnet_config(ckpt.get("config") or {})
        model_config = physnet_constructor_kwargs(config, SpookyPhysNet)
        self.model = SpookyPhysNet(**model_config)
        self.params = json_tree_to_jax_params(params)
        self.max_atoms = int(
            model_config.get("max_padded_atoms", self.model.natoms)
        )
        self.charge = float(charge)
        self.spin_multiplicity = float(spin_multiplicity)
        self._apply = jax.jit(self._make_apply_fn())

    def _make_apply_fn(self):
        model = self.model
        params = self.params

        def _fn(atomic_numbers, positions, dst_idx, src_idx, atom_mask, batch_mask, charge, spin):
            n_atoms = atomic_numbers.shape[0]
            batch_segments = jnp.zeros((n_atoms,), dtype=jnp.int32)
            q_atoms = jnp.full((n_atoms, 1), charge, dtype=jnp.float32)
            s_atoms = jnp.full((n_atoms, 1), spin, dtype=jnp.float32)
            return model.apply(
                params,
                atomic_numbers=atomic_numbers,
                charges=q_atoms,
                spins=s_atoms,
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
                batch_size=1,
                batch_mask=batch_mask,
                atom_mask=atom_mask,
            )

        return _fn

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n_real = len(atoms)
        if n_real > self.max_atoms:
            raise ValueError(
                f"SpookyNet checkpoint padded to {self.max_atoms} atoms; got {n_real}"
            )
        pad = self.max_atoms - n_real
        z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        pos = np.asarray(atoms.get_positions(), dtype=np.float32)
        if pad:
            z = np.concatenate([z, np.zeros(pad, dtype=np.int32)])
            pos = np.concatenate([pos, np.zeros((pad, 3), dtype=np.float32)], axis=0)

        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(self.max_atoms)
        atom_mask = (z > 0).astype(np.float32)
        valid_pairs = (atom_mask[dst_idx] > 0) & (atom_mask[src_idx] > 0)
        batch_mask = valid_pairs.astype(np.float32)

        output = self._apply(
            jnp.asarray(z),
            jnp.asarray(pos),
            jnp.asarray(dst_idx),
            jnp.asarray(src_idx),
            jnp.asarray(atom_mask),
            jnp.asarray(batch_mask),
            self.charge,
            self.spin_multiplicity,
        )
        self.results["energy"] = float(np.asarray(output["energy"]).squeeze())
        if "forces" in properties and "forces" in output:
            self.results["forces"] = np.asarray(output["forces"])[:n_real]
