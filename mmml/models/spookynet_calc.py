"""Standalone ASE calculator for SpookyNet-style PhysNet checkpoints (JSON format).

Loads a ``SpookyPhysNet`` model + params from a JSON checkpoint (as produced by
``mmml.utils.model_checkpoint``) and exposes it as a plain ASE ``Calculator``
so it can be dropped into the same scan pipelines as the xTB/CHARMM/learned
multipole backends (see ``mmml.analysis.dimer_scans.evaluate_scan``).

Atom counts smaller than the checkpoint's ``max_padded_atoms`` are handled by
zero-padding + masking, so one calculator instance can be reused for both the
full dimer and individual monomer fragments of varying size.

Some checkpoints (e.g. anything trained with ``scripts/train_so3lr_spooky_extxyz.py
--mbd-checkpoint ...``) were trained as a *residual* on top of a frozen,
separately-trained MBD (many-body dispersion) correction — see that script's
``energy_pred = spooky_energy + mbd_weight * mbd_energy`` (around line 497).
Evaluating the Spooky weights alone for such a checkpoint evaluates only half
of what was actually trained, and produces energies with no reason to behave
sensibly at long range (the residual network was never trained to be a
complete, physically-plateauing potential by itself). This calculator
reproduces the training-time composite automatically when the checkpoint's
own saved config records an ``mbd_checkpoint`` (unless overridden).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ase.units
import e3x
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from mmml.interfaces.pycharmmInterface.ml_dtypes import (
    json_tree_to_jax_params,
    ml_numpy_dtype,
    resolve_ml_compute_dtype,
)
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from mmml.utils.model_checkpoint import (
    load_model_checkpoint,
    normalize_physnet_config,
    physnet_constructor_kwargs,
)

EV_TO_KCAL_MOL = 1 / (ase.units.kcal / ase.units.mol)


class SpookyNetCalculator(Calculator):
    """ASE calculator wrapping a SpookyNet-style PhysNet model.

    If the checkpoint was trained with a frozen MBD correction (its saved
    config has an ``mbd_checkpoint`` entry — the training args are recorded
    verbatim by ``orbax_to_json``), that correction is loaded and added
    automatically with the recorded ``mbd_weight``, so evaluation matches
    training exactly. Pass ``mbd_checkpoint=False`` to force Spooky-only
    evaluation even if the checkpoint's config references one (e.g. to
    isolate/debug the Spooky component alone), or pass an explicit
    ``mbd_checkpoint=`` / ``mbd_weight=`` to override what's in the config
    (e.g. because the recorded path is a cluster-local path that doesn't
    exist on this machine).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        charge: float = 0.0,
        spin_multiplicity: float = 1.0,
        mbd_checkpoint: str | Path | bool | None = None,
        mbd_weight: float | None = None,
        use_orbax: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        checkpoint = Path(checkpoint).expanduser()
        ckpt = load_model_checkpoint(
            checkpoint, use_orbax=use_orbax, load_params=True, load_config=True
        )
        params = ckpt.get("params")
        if params is None:
            raise FileNotFoundError(f"No params found in checkpoint: {checkpoint}")
        raw_config = ckpt.get("config") or {}
        config = normalize_physnet_config(raw_config)
        self.checkpoint_path = checkpoint.resolve()
        self.raw_config = dict(raw_config)
        self.normalized_config = dict(config)
        
        # Check model architecture: standard PhysNet vs SpookyPhysNet
        self.compute_dtype = resolve_ml_compute_dtype()
        self.numpy_dtype = ml_numpy_dtype(self.compute_dtype)
        jax_params = json_tree_to_jax_params(params, dtype=self.compute_dtype)
        model_type = str(config.get("model_type", "")).lower()
        if model_type == "physnet" or "charge_feature_projection" not in jax_params:
            from mmml.models.physnetjax.physnetjax.models.model import PhysNet
            model_config = physnet_constructor_kwargs(config, PhysNet)
            self.model = PhysNet(**model_config)
        else:
            model_config = physnet_constructor_kwargs(config, SpookyPhysNet)
            self.model = SpookyPhysNet(**model_config)

        self.params = jax_params
        self.max_atoms = int(
            model_config.get("max_padded_atoms", getattr(self.model, "natoms", 60))
        )
        self.charge = float(charge)
        self.spin_multiplicity = float(spin_multiplicity)
        self._apply = jax.jit(self._make_apply_fn())

        # The standalone ASE adapter currently supplies only Z/R/masks to the
        # model. Dynamic CGenFF type, sigma, epsilon and molecule-id arrays are
        # therefore not present, even if they were used during training.
        self.cgenff_lj_inputs_supplied = False

        # --- Companion MBD correction (see module docstring) -------------
        self.mbd_calc = None
        self.mbd_weight = 0.0
        resolved_mbd_checkpoint: str | Path | None
        if mbd_checkpoint is False:
            resolved_mbd_checkpoint = None
        elif mbd_checkpoint is not None and mbd_checkpoint is not True:
            resolved_mbd_checkpoint = mbd_checkpoint
        else:
            # Auto-detect from the checkpoint's own recorded training config.
            resolved_mbd_checkpoint = raw_config.get("mbd_checkpoint")

        if resolved_mbd_checkpoint:
            resolved_path = Path(resolved_mbd_checkpoint).expanduser()
            if not resolved_path.exists():
                print(
                    f"  Note: checkpoint was trained with mbd_checkpoint={resolved_path} "
                    "but that path doesn't exist here — skipping the MBD correction "
                    "(energies will be Spooky-residual-only, not matching training). "
                    "Pass mbd_checkpoint=<path that exists here> to fix this."
                )
            else:
                from mmml.models.mbd.calculator import QCMLMBDCalculator

                self.mbd_calc = QCMLMBDCalculator(
                    checkpoint=resolved_path, charge=charge, multiplicity=spin_multiplicity,
                )
                self.mbd_weight = float(
                    mbd_weight if mbd_weight is not None else raw_config.get("mbd_weight", 1.0)
                )
                print(f"  Using MBD correction from {resolved_path} (weight={self.mbd_weight:g})")

    def energy_function_report(self) -> dict[str, Any]:
        """Return a machine-readable manifest of the active energy function.

        This deliberately distinguishes terms supported/trained by the model
        from terms actually supplied by this calculator adapter.  That makes
        silent omissions such as missing dynamic CGenFF LJ inputs visible.
        """
        model = self.model
        trained_with_cgenff_lj = not bool(
            self.raw_config.get("no_cgenff_vdw", False)
        )
        charges_enabled = bool(getattr(model, "charges", False))
        zbl_enabled = bool(getattr(model, "zbl", False))
        cutoff = self.normalized_config.get(
            "cutoff", getattr(model, "cutoff", None)
        )
        warnings: list[str] = []
        if trained_with_cgenff_lj and not self.cgenff_lj_inputs_supplied:
            warnings.append(
                "Checkpoint training enabled CGenFF LJ, but this standalone "
                "calculator does not supply mol_id/cgenff_type_idx/sigma/epsilon; "
                "the fixed LJ contribution is absent from evaluated energies."
            )
        configured_mbd = self.raw_config.get("mbd_checkpoint")
        if configured_mbd and self.mbd_calc is None:
            warnings.append(
                "Checkpoint records a companion MBD model, but it is not loaded; "
                "evaluated energies contain only the residual Spooky term."
            )

        return {
            "calculator": type(self).__name__,
            "checkpoint": str(self.checkpoint_path),
            "model_class": type(model).__name__,
            "precision": {
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "compute_dtype": str(self.compute_dtype),
            },
            "energy_units": "eV",
            "force_units": "eV/angstrom",
            "short_range": {
                "neural_atomic_energy": True,
                "cutoff_angstrom": None if cutoff is None else float(cutoff),
                "zbl_repulsion": zbl_enabled,
            },
            "electrostatics": {
                "predicted_atomic_charges": charges_enabled,
                "damping_sigma": self.raw_config.get(
                    "electrostatics_damping_sigma"
                ),
                "target_total_charge": self.charge,
            },
            "cgenff_lennard_jones": {
                "enabled_during_training": trained_with_cgenff_lj,
                "inputs_supplied_at_inference": self.cgenff_lj_inputs_supplied,
                "parameter_file_radius_field": "Rmin/2 (angstrom)",
                "pair_combination": "Rmin_ij = Rmin_i/2 + Rmin_j/2; epsilon_ij = sqrt(|epsilon_i epsilon_j|)",
                "charmm_form": "epsilon_ij * [(Rmin_ij/r)^12 - 2*(Rmin_ij/r)^6]",
                "conventional_sigma_conversion": "sigma_i = 2*(Rmin_i/2)/2^(1/6)",
                "predict_atomic_vdw_scale": bool(
                    getattr(model, "predict_atomic_vdw_scale", False)
                ),
                "learn_cgenff_vdw_scale": bool(
                    getattr(model, "learn_cgenff_vdw_scale", False)
                ),
            },
            "mbd": {
                "configured_checkpoint": None
                if configured_mbd is None
                else str(configured_mbd),
                "loaded": self.mbd_calc is not None,
                "weight": float(self.mbd_weight),
            },
            "warnings": warnings,
        }

    def write_energy_function_report(self, path: str | Path) -> Path:
        """Write :meth:`energy_function_report` as formatted JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.energy_function_report(), indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def _make_apply_fn(self):
        model = self.model
        params = self.params
        compute_dtype = self.compute_dtype

        def _fn(atomic_numbers, positions, dst_idx, src_idx, atom_mask, batch_mask, charge, spin, mol_id=None, cgenff_type_idx=None, cgenff_master_sigmas=None, cgenff_master_epsilons=None):
            n_atoms = atomic_numbers.shape[0]
            batch_segments = jnp.zeros((n_atoms,), dtype=jnp.int32)
            if isinstance(model, SpookyPhysNet):
                q_atoms = jnp.full((n_atoms, 1), charge, dtype=compute_dtype)
                s_atoms = jnp.full((n_atoms, 1), spin, dtype=compute_dtype)
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
                    mol_id=mol_id,
                    cgenff_type_idx=cgenff_type_idx,
                    cgenff_master_sigmas=cgenff_master_sigmas,
                    cgenff_master_epsilons=cgenff_master_epsilons,
                )
            else:
                return model.apply(
                    params,
                    atomic_numbers=atomic_numbers,
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
        pos = np.asarray(atoms.get_positions(), dtype=self.numpy_dtype)
        if pad:
            # Scatter padded "ghost" atoms far apart from real atoms and from
            # each other. Stacking them all at the origin makes pad-pad and
            # real-pad pairwise distances exactly (or near) zero, which blows
            # up 1/r terms (e.g. ZBL repulsion) to inf/NaN *before* masking
            # is applied (0 * inf = NaN survives the mask).
            far = 1.0e4 + 100.0 * np.arange(pad, dtype=self.numpy_dtype)
            pad_pos = np.stack(
                [
                    far,
                    np.zeros(pad, dtype=self.numpy_dtype),
                    np.zeros(pad, dtype=self.numpy_dtype),
                ],
                axis=1,
            )
            z = np.concatenate([z, np.zeros(pad, dtype=np.int32)])
            pos = np.concatenate([pos, pad_pos], axis=0)

        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(self.max_atoms)
        atom_mask = (z > 0).astype(self.numpy_dtype)
        valid_pairs = (atom_mask[dst_idx] > 0) & (atom_mask[src_idx] > 0)
        batch_mask = valid_pairs.astype(self.numpy_dtype)

        metadata_names = ("mol_id", "cgenff_type_idx")
        have_metadata = all(name in atoms.arrays for name in metadata_names)
        have_tables = all(
            name in atoms.info
            for name in ("cgenff_master_sigmas", "cgenff_master_epsilons")
        )
        if have_metadata != have_tables:
            raise ValueError(
                "Incomplete CGenFF inference metadata: mol_id/type indices and "
                "master sigma/epsilon tables must be supplied together"
            )
        mol_id = cgenff_type_idx = master_sigmas = master_epsilons = None
        if have_metadata:
            mol_id = np.asarray(atoms.arrays["mol_id"], dtype=np.int32)
            cgenff_type_idx = np.asarray(
                atoms.arrays["cgenff_type_idx"], dtype=np.int32
            )
            if pad:
                mol_id = np.pad(mol_id, (0, pad), constant_values=0)
                cgenff_type_idx = np.pad(
                    cgenff_type_idx, (0, pad), constant_values=0
                )
            master_sigmas = np.asarray(
                atoms.info["cgenff_master_sigmas"], dtype=self.numpy_dtype
            )
            master_epsilons = np.asarray(
                atoms.info["cgenff_master_epsilons"], dtype=self.numpy_dtype
            )
            self.cgenff_lj_inputs_supplied = True

        output = self._apply(
            jnp.asarray(z),
            jnp.asarray(pos),
            jnp.asarray(dst_idx),
            jnp.asarray(src_idx),
            jnp.asarray(atom_mask),
            jnp.asarray(batch_mask),
            self.charge,
            self.spin_multiplicity,
            None if mol_id is None else jnp.asarray(mol_id),
            None if cgenff_type_idx is None else jnp.asarray(cgenff_type_idx),
            None if master_sigmas is None else jnp.asarray(master_sigmas),
            None if master_epsilons is None else jnp.asarray(master_epsilons),
        )
        spooky_energy = float(np.asarray(output["energy"]).squeeze())
        spooky_forces = np.asarray(output["forces"])[:n_real] if "forces" in output else None

        energy = spooky_energy
        forces = spooky_forces
        self.results["spooky_energy"] = spooky_energy
        def _component_sum(name: str) -> float:
            value = output.get(name)
            return 0.0 if value is None else float(np.asarray(value).sum())

        electrostatics_energy = _component_sum("electrostatics")
        cgenff_vdw_energy = _component_sum("cgenff_vdw")
        zbl_repulsion_energy = _component_sum("repulsion")
        self.results["electrostatics_energy"] = electrostatics_energy
        self.results["cgenff_vdw_energy"] = cgenff_vdw_energy
        self.results["zbl_repulsion_energy"] = zbl_repulsion_energy
        self.results["neural_energy"] = (
            spooky_energy
            - electrostatics_energy
            - cgenff_vdw_energy
            - zbl_repulsion_energy
        )

        if self.mbd_calc is not None:
            mbd_out = self.mbd_calc.predict_mbd(atoms)
            mbd_energy = self.mbd_weight * mbd_out["energy_ev"]
            energy = spooky_energy + mbd_energy
            self.results["mbd_energy"] = mbd_energy
            if forces is not None:
                forces = forces + self.mbd_weight * mbd_out["forces_ev_angstrom"]

        self.results["energy"] = energy
        if "forces" in properties and forces is not None:
            self.results["forces"] = forces
