#!/usr/bin/env python3
"""Evaluate trained SpookyPhysNet models on test extxyz files.

Example:
    uv run python scripts/evaluate_so3lr_spooky_extxyz.py \
        --checkpoint artifacts/spooky_so3lr/epoch-50 \
        --extxyz ~/data/so3lr_test \
        --cache-dir ~/data/so3lr_orbax_cache \
        --batch-size-per-device 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_xla_flags = os.environ.get("XLA_FLAGS", "")
_disabled_pass = "--xla_disable_hlo_passes=hoist-fused-bitcasts"
if _disabled_pass not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} {_disabled_pass}".strip()

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from ase.io import iread
from flax import jax_utils
from flax.training import orbax_utils

from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from mmml.models.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_flat_data,
)

DIPOLE_KEY_ALIASES = (
    "dipole",
    "dipoles",
    "D",
    "Dxyz",
    "D_xyz",
    "dipole_moment",
    "dipole_moments",
    "mu",
    "muxyz",
    "molecular_dipole",
)


@dataclass(frozen=True)
class CacheMeta:
    extxyz: str
    size_bytes: int
    mtime_ns: int
    energy_key: str
    forces_key: str
    dipole_key: str
    charge_key: str
    spin_key: str | None
    infer_spin: bool
    default_charge: float
    default_spin: float
    max_structures: int | None
    cache_version: int = 3


def _cache_name(meta: CacheMeta) -> str:
    payload = json.dumps(asdict(meta), sort_keys=True).encode()
    return f"{Path(meta.extxyz).name.split('.')[0]}_flat_{hashlib.sha256(payload).hexdigest()[:16]}"


def _orbax_save(path: Path, target: dict[str, Any], *, force: bool) -> None:
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(target)
    try:
        checkpointer.save(path, target, save_args=save_args, force=force)
    except TypeError:
        if force and path.exists():
            import shutil
            shutil.rmtree(path)
        checkpointer.save(path, target, save_args=save_args)


def _resolve_cache_path(args: argparse.Namespace, extxyz_file: Path) -> Path:
    stat = extxyz_file.stat()
    meta = CacheMeta(
        extxyz=str(extxyz_file),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        dipole_key=args.dipole_key,
        charge_key=args.charge_key,
        spin_key=args.spin_key,
        infer_spin=args.infer_spin,
        default_charge=args.default_charge,
        default_spin=args.default_spin,
        max_structures=args.max_structures,
    )
    return Path(args.cache_dir).resolve() / _cache_name(meta)


def _get_info_scalar(info: dict[str, Any], key: str, default: float) -> float:
    value = info.get(key, default)
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    raise ValueError(f"Expected scalar '{key}', got shape {array.shape}")


def _get_info_vector(info: dict[str, Any], key: str, size: int, default: np.ndarray | None = None) -> np.ndarray:
    if key not in info:
        if default is not None:
            return default
        raise KeyError(f"Structure lacks required info key '{key}'")
    val_raw = info[key]
    if isinstance(val_raw, str):
        try:
            value = np.fromstring(val_raw, sep=" ", dtype=np.float64)
        except Exception:
            value = np.asarray(val_raw.split(), dtype=np.float64)
    else:
        value = np.asarray(val_raw, dtype=np.float64)
    value = value.reshape(-1)
    if value.size != size:
        if default is not None:
            return default
        raise ValueError(f"Expected '{key}' with {size} values, got shape {value.shape}")
    return value


def _vector_from_raw(value_raw: Any, key: str, size: int) -> np.ndarray:
    if isinstance(value_raw, str):
        text = value_raw.replace(",", " ").replace("[", " ").replace("]", " ")
        value = np.fromstring(text, sep=" ", dtype=np.float64)
    else:
        value = np.asarray(value_raw, dtype=np.float64)
    value = value.reshape(-1)
    if value.size != size:
        raise ValueError(f"Expected '{key}' with {size} values, got shape {value.shape}")
    return value


def _candidate_vector_keys(requested_key: str) -> tuple[str, ...]:
    keys = [requested_key]
    for alias in DIPOLE_KEY_ALIASES:
        if alias not in keys:
            keys.append(alias)
    return tuple(keys)


def _find_vector_key(atoms, requested_key: str, size: int) -> str | None:
    for key in _candidate_vector_keys(requested_key):
        if key in atoms.info:
            _vector_from_raw(atoms.info[key], key, size)
            return key
        if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
            _vector_from_raw(atoms.calc.results[key], key, size)
            return key
    return None


def _get_vector_from_atoms(
    atoms,
    requested_key: str,
    size: int,
    *,
    default: np.ndarray | None = None,
) -> tuple[np.ndarray, str | None]:
    key = _find_vector_key(atoms, requested_key, size)
    if key is None:
        if default is not None:
            return default, None
        aliases = ", ".join(_candidate_vector_keys(requested_key))
        raise KeyError(
            f"Structure lacks vector key '{requested_key}'. Tried aliases: {aliases}; "
            f"info keys: {sorted(atoms.info)}; "
            f"calculator result keys: {sorted(getattr(atoms.calc, 'results', {}))}"
        )
    if key in atoms.info:
        return _vector_from_raw(atoms.info[key], key, size), key
    return _vector_from_raw(atoms.calc.results[key], key, size), key


def _get_energy(atoms, key: str, structure_index: int, default: float = 0.0) -> float:
    try:
        if key in atoms.info:
            return _get_info_scalar(atoms.info, key, default)
        if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
            return float(np.asarray(atoms.calc.results[key]).reshape(-1)[0])
        if key == "energy":
            return float(atoms.get_potential_energy())
    except Exception:
        return default
    return default


def _get_forces(atoms, key: str, structure_index: int, default: np.ndarray | None = None) -> np.ndarray:
    try:
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=np.float64)
        if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
            return np.asarray(atoms.calc.results[key], dtype=np.float64)
        if key == "forces":
            return np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception:
        pass
    if default is not None:
        return default
    raise KeyError(f"Structure {structure_index} lacks forces key '{key}'")



def _infer_spin_multiplicity(atoms, total_charge: float) -> float:
    """Infer singlet/doublet multiplicity from electron parity."""
    n_protons = int(np.sum(atoms.get_atomic_numbers()))
    n_electrons = int(round(n_protons - total_charge))
    return 1.0 if n_electrons % 2 == 0 else 2.0


def cache_extxyz_to_orbax(args: argparse.Namespace, extxyz_file: Path) -> Path:
    """Parse extxyz once into flat concatenated arrays and store them via Orbax."""
    extxyz = extxyz_file.resolve()
    if not extxyz.exists():
        raise FileNotFoundError(extxyz)

    cache_path = _resolve_cache_path(args, extxyz)
    if cache_path.exists() and not args.force_recache:
        print(f"Using existing Orbax data cache: {cache_path}")
        return cache_path

    # Check what keys are available in the first structure
    first_structure = next(iread(extxyz, index="0"))
    has_energy = args.energy_key in first_structure.info or (
        first_structure.calc is not None and args.energy_key in getattr(first_structure.calc, "results", {})
    ) or args.energy_key == "energy"
    
    has_forces = args.forces_key in first_structure.arrays or (
        first_structure.calc is not None and args.forces_key in getattr(first_structure.calc, "results", {})
    ) or args.forces_key == "forces"
    
    dipole_key = _find_vector_key(first_structure, args.dipole_key, 3)
    has_dipole = dipole_key is not None
    if has_dipole and dipole_key != args.dipole_key:
        print(
            f"Using dipole key alias '{dipole_key}' for requested key '{args.dipole_key}'",
            flush=True,
        )

    r_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    f_parts: list[np.ndarray] = []
    energies: list[float] = []
    natoms: list[int] = []
    charges: list[float] = []
    spins: list[float] = []
    dipoles: list[np.ndarray] = []
    offsets = [0]

    print(f"Reading extxyz: {extxyz}")
    for i, atoms in enumerate(iread(extxyz, index=":")):
        if args.max_structures is not None and i >= args.max_structures:
            break

        n_atoms = len(atoms)
        if n_atoms == 0:
            raise ValueError(f"Structure {i} has no atoms")

        r_parts.append(atoms.get_positions().astype(np.float64, copy=False))
        z_parts.append(atoms.get_atomic_numbers().astype(np.int32, copy=False))
        
        if has_forces:
            f_parts.append(_get_forces(atoms, args.forces_key, i, default=np.zeros((n_atoms, 3), dtype=np.float64)))
        else:
            f_parts.append(np.zeros((n_atoms, 3), dtype=np.float64))
            
        if has_energy:
            energies.append(_get_energy(atoms, args.energy_key, i, default=0.0))
        else:
            energies.append(0.0)
            
        if has_dipole:
            dipole, _ = _get_vector_from_atoms(
                atoms,
                args.dipole_key,
                3,
                default=np.zeros(3, dtype=np.float64),
            )
            dipoles.append(dipole)
        else:
            dipoles.append(np.zeros(3, dtype=np.float64))
            
        natoms.append(n_atoms)
        charge = _get_info_scalar(atoms.info, args.charge_key, args.default_charge)
        charges.append(charge)
        spin = _infer_spin_multiplicity(atoms, charge) if args.infer_spin else args.default_spin
        if args.spin_key is not None:
            spin = _get_info_scalar(atoms.info, args.spin_key, args.default_spin)
        spins.append(spin)
        offsets.append(offsets[-1] + n_atoms)

        if (i + 1) % args.log_every == 0:
            print(f"  parsed {i + 1:,} structures")

    if not energies:
        raise ValueError(f"No structures read from {extxyz}")

    data = {
        "R": np.concatenate(r_parts, axis=0),
        "Z": np.concatenate(z_parts, axis=0),
        "F": np.concatenate(f_parts, axis=0),
        "mol_offsets": np.asarray(offsets, dtype=np.int64),
        "E": np.asarray(energies, dtype=np.float64).reshape(-1, 1),
        "N": np.asarray(natoms, dtype=np.int32).reshape(-1, 1),
        "Q": np.asarray(charges, dtype=np.float64).reshape(-1, 1),
        "S": np.asarray(spins, dtype=np.float64).reshape(-1, 1),
        "D": np.asarray(dipoles, dtype=np.float64).reshape(-1, 3),
    }
    data["metadata_n_structures"] = np.asarray(len(energies), dtype=np.int64)
    data["metadata_n_atoms_total"] = np.asarray(offsets[-1], dtype=np.int64)
    data["metadata_max_atoms"] = np.asarray(max(natoms), dtype=np.int32)
    data["metadata_has_energy"] = np.asarray(has_energy, dtype=bool)
    data["metadata_has_forces"] = np.asarray(has_forces, dtype=bool)
    data["metadata_has_dipole"] = np.asarray(has_dipole, dtype=bool)
    data["metadata_dipole_key"] = np.asarray(dipole_key or "", dtype="<U32")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving Orbax data cache: {cache_path}")
    _orbax_save(cache_path, data, force=True)
    print(
        "Cached "
        f"{len(energies):,} structures, {offsets[-1]:,} atoms, "
        f"max_atoms={max(natoms)}"
    )
    return cache_path


def restore_cached_data(cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    print(f"Restoring Orbax data cache: {cache_path}", flush=True)
    start = time.time()
    restored = ocp.PyTreeCheckpointer().restore(cache_path)
    data = {
        key: np.asarray(value)
        for key, value in restored.items()
        if not key.startswith("metadata_")
    }
    metadata = {
        key: np.asarray(value)
        for key, value in restored.items()
        if key.startswith("metadata_")
    }
    print(
        "Restored cache in "
        f"{time.time() - start:.1f} s: "
        f"{int(np.asarray(metadata.get('metadata_n_structures', len(data['N'])))):,} structures, "
        f"{int(np.asarray(metadata.get('metadata_n_atoms_total', len(data['R'])))):,} atoms",
        flush=True,
    )
    return data, metadata


def limit_cached_data(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    max_structures: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Limit a restored flat cache to the first ``max_structures`` molecules."""
    if max_structures is None:
        return data, metadata
    if max_structures <= 0:
        raise ValueError("max_eval_structures must be positive")
    current = int(np.asarray(data["N"]).reshape(-1).shape[0])
    limit = min(max_structures, current)
    offsets = np.asarray(data["mol_offsets"], dtype=np.int64)
    atom_limit = int(offsets[limit])
    limited = dict(data)
    for key in ("R", "Z", "F"):
        limited[key] = data[key][:atom_limit]
    for key in ("E", "N", "Q", "S", "D"):
        limited[key] = data[key][:limit]
    limited["mol_offsets"] = offsets[: limit + 1]
    updated_metadata = dict(metadata)
    updated_metadata["metadata_n_structures"] = np.asarray(limit, dtype=np.int64)
    updated_metadata["metadata_n_atoms_total"] = np.asarray(atom_limit, dtype=np.int64)
    updated_metadata["metadata_max_atoms"] = np.asarray(
        int(np.max(limited["N"])) if limit else 0,
        dtype=np.int32,
    )
    print(
        f"Limited evaluation cache to {limit:,} structures and {atom_limit:,} atoms",
        flush=True,
    )
    return limited, updated_metadata



def bucket_indices_by_natoms(data: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    n_atoms = np.asarray(data["N"], dtype=np.int32).reshape(-1)
    buckets: dict[int, list[int]] = {}
    for idx in range(len(n_atoms)):
        buckets.setdefault(int(n_atoms[idx]), []).append(idx)
    return {n: np.asarray(vals, dtype=np.int64) for n, vals in buckets.items()}


def iter_device_batches_eval(
    buckets: dict[int, np.ndarray],
    *,
    per_device_batch_size: int,
    max_pairs_per_device: int,
    n_devices: int,
) -> Any:
    for n_atoms in sorted(buckets):
        bucket_batch_size = min(
            per_device_batch_size,
            max(1, max_pairs_per_device // (n_atoms * n_atoms)),
        )
        global_batch = bucket_batch_size * n_devices
        indices = buckets[n_atoms].copy()
        
        # Track which elements are real (True) or padding (False)
        real_mask = np.ones(len(indices), dtype=bool)
        
        remainder = len(indices) % global_batch
        if remainder != 0:
            padding_needed = global_batch - remainder
            padding_indices = np.repeat(indices[-1], padding_needed)
            indices = np.concatenate([indices, padding_indices])
            padding_mask = np.zeros(padding_needed, dtype=bool)
            real_mask = np.concatenate([real_mask, padding_mask])
            
        for start in range(0, len(indices), global_batch):
            chunk = indices[start : start + global_batch]
            is_real = real_mask[start : start + global_batch]
            yield chunk.reshape(n_devices, bucket_batch_size), is_real.reshape(n_devices, bucket_batch_size)


def stack_device_batches(
    data: dict[str, np.ndarray],
    device_indices: np.ndarray,
) -> dict[str, Any]:
    batches = [
        build_spooky_batch_from_flat_data(data, device_indices[i])
        for i in range(device_indices.shape[0])
    ]
    for i, batch in enumerate(batches):
        indices = device_indices[i]
        batch["D"] = jnp.asarray(data["D"][indices], dtype=jnp.float32)
        batch["Q_total"] = jnp.asarray(data["Q"][indices], dtype=jnp.float32)
    stacked: dict[str, Any] = {}
    for key in batches[0]:
        if key == "batch_size":
            continue
        stacked[key] = jnp.stack([batch[key] for batch in batches], axis=0)
    return stacked


def create_model_from_config(config: dict[str, Any], max_atoms: int) -> SpookyPhysNet:
    charges = config.get("predict_charges", config.get("charges", False))
    n_res = config.get("n_res", config.get("n_refinement_blocks", 2))
    zbl = config.get("zbl", not config.get("no_zbl", False))
    
    return SpookyPhysNet(
        features=config.get("features", 128),
        max_degree=config.get("max_degree", 2),
        num_iterations=config.get("num_iterations", 3),
        num_basis_functions=config.get("num_basis_functions", 32),
        cutoff=config.get("cutoff", 6.0),
        max_atomic_number=config.get("max_atomic_number", 87),
        charges=charges,
        max_padded_atoms=max_atoms,
        n_refinement_blocks=n_res,
        zbl=zbl,
        efa=config.get("efa", False),
        use_energy_bias=config.get("use_energy_bias", False),
        electrostatics_damping_sigma=config.get("electrostatics_damping_sigma", 4.0),
    )


def make_eval_steps(model: SpookyPhysNet, args: argparse.Namespace, devices: list[Any]):
    per_device_batch_size = args.batch_size_per_device
    predict_charges = args.predict_charges

    def eval_fn(params, batch):
        out = model.apply(
            params,
            atomic_numbers=batch["Z"],
            charges=batch["Q_atoms"],
            spins=batch["S_atoms"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=per_device_batch_size,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        energy_pred = out["energy"].reshape(-1, 1)
        forces_pred = out["forces"].reshape(batch["F"].shape)
        
        res = {
            "E_pred": energy_pred,
            "E_ref": batch["E"].reshape(-1, 1),
            "F_pred": forces_pred,
            "F_ref": batch["F"],
        }
        if predict_charges:
            res["D_pred"] = out["dipoles"].reshape(batch["D"].shape)
            res["D_ref"] = batch["D"]
            res["Q_pred"] = out["sum_charges"].reshape(batch["Q_total"].shape)
            res["Q_ref"] = batch["Q_total"]
        return res

    return jax.pmap(eval_fn, axis_name="device", devices=devices)


def restore_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)
    
    params = restored.get("params")
    if params is None and "model" in restored:
        params = restored["model"].get("params")
        
    if params is None:
        raise ValueError(f"Checkpoint at {checkpoint_path} has no 'params' or 'model.params'")
        
    config = restored.get("config")
    if config is None:
        config = restored.get("model_attributes") or {}
        
    return params, config


def evaluate_dataset(
    model: SpookyPhysNet,
    params: dict[str, Any],
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    devices: list[Any],
) -> dict[str, float]:
    n_atoms = np.asarray(data["N"]).reshape(-1)
    buckets = bucket_indices_by_natoms(data)
    
    # Check flags
    has_energy = bool(metadata.get("metadata_has_energy", True))
    has_forces = bool(metadata.get("metadata_has_forces", True))
    has_dipole = bool(metadata.get("metadata_has_dipole", True))
    
    # Initialize metrics
    total_structures = 0
    total_atoms = 0

    energy_sae = 0.0
    energy_sse = 0.0
    force_sae = 0.0
    force_sse = 0.0
    
    predict_charges = args.predict_charges
    dipole_sae = 0.0
    dipole_sse = 0.0
    charge_sae = 0.0
    charge_sse = 0.0

    step_functions: dict[int, Any] = {}
    def eval_step_for_batch_size(batch_size: int):
        if batch_size not in step_functions:
            eval_args = argparse.Namespace(**{**vars(args), "batch_size_per_device": batch_size})
            step_functions[batch_size] = make_eval_steps(model, eval_args, devices)
        return step_functions[batch_size]

    replicated_params = jax_utils.replicate(params, devices=devices)
    batch_plan = list(iter_device_batches_eval(
        buckets,
        per_device_batch_size=args.batch_size_per_device,
        max_pairs_per_device=args.max_pairs_per_device,
        n_devices=args.num_devices,
    ))
    if args.max_eval_batches is not None:
        batch_plan = batch_plan[: args.max_eval_batches]
    planned_structures = int(sum(np.sum(is_real) for _, is_real in batch_plan))
    print(
        "Evaluation plan: "
        f"{len(batch_plan):,} batches, {planned_structures:,} real structures, "
        f"{len(step_functions)} compiled shapes initially",
        flush=True,
    )
    
    for batch_number, (device_indices, is_real) in enumerate(batch_plan, start=1):
        batch_size = int(device_indices.shape[1])
        bucket_atoms = int(n_atoms[device_indices[0, 0]])
        eval_step = eval_step_for_batch_size(batch_size)
        if batch_number == 1 or batch_number % args.progress_every == 0:
            print(
                f"  eval batch {batch_number:,}/{len(batch_plan):,}: "
                f"n_atoms={bucket_atoms}, per_device_batch={batch_size}, "
                f"seen={total_structures:,}",
                flush=True,
            )
        
        batch = stack_device_batches(data, device_indices)
        predictions = eval_step(replicated_params, batch)
        
        # Accumulate metrics on CPU
        is_real_np = np.asarray(is_real) # shape (n_devices, batch_size)
        
        if has_energy:
            E_pred = np.asarray(predictions["E_pred"]).reshape(is_real_np.shape)
            E_ref = np.asarray(predictions["E_ref"]).reshape(is_real_np.shape)
            energy_errors = np.abs(E_pred - E_ref)[is_real_np]
            energy_sae += float(np.sum(energy_errors))
            energy_sse += float(np.sum(energy_errors**2))
            
        total_structures += int(np.sum(is_real_np))
        
        if has_forces:
            F_pred = np.asarray(predictions["F_pred"]).reshape(is_real_np.shape[0], is_real_np.shape[1], bucket_atoms, 3)
            F_ref = np.asarray(predictions["F_ref"]).reshape(is_real_np.shape[0], is_real_np.shape[1], bucket_atoms, 3)
            force_errors = np.abs(F_pred[is_real_np] - F_ref[is_real_np])
            force_sae += float(np.sum(force_errors))
            force_sse += float(np.sum(force_errors**2))
            
        total_atoms += int(is_real_np.sum()) * bucket_atoms
        
        if predict_charges:
            if has_dipole:
                D_pred = np.asarray(predictions["D_pred"]).reshape(is_real_np.shape[0], is_real_np.shape[1], 3)
                D_ref = np.asarray(predictions["D_ref"]).reshape(is_real_np.shape[0], is_real_np.shape[1], 3)
                dipole_errors = np.abs(D_pred[is_real_np] - D_ref[is_real_np])
                dipole_sae += float(np.sum(dipole_errors))
                dipole_sse += float(np.sum(dipole_errors**2))
            
            Q_pred = np.asarray(predictions["Q_pred"]).reshape(is_real_np.shape)
            Q_ref = np.asarray(predictions["Q_ref"]).reshape(is_real_np.shape)
            charge_errors = np.abs(Q_pred[is_real_np] - Q_ref[is_real_np])
            charge_sae += float(np.sum(charge_errors))
            charge_sse += float(np.sum(charge_errors**2))

    metrics = {}
    if has_energy:
        metrics["energy_mae"] = energy_sae / max(1, total_structures)
        metrics["energy_rmse"] = math.sqrt(energy_sse / max(1, total_structures))
    else:
        metrics["energy_mae"] = float('nan')
        metrics["energy_rmse"] = float('nan')
        
    if has_forces:
        metrics["forces_mae"] = force_sae / max(1, total_atoms * 3)
        metrics["forces_rmse"] = math.sqrt(force_sse / max(1, total_atoms * 3))
    else:
        metrics["forces_mae"] = float('nan')
        metrics["forces_rmse"] = float('nan')
        
    if predict_charges:
        if has_dipole:
            metrics["dipole_mae"] = dipole_sae / max(1, total_structures * 3)
            metrics["dipole_rmse"] = math.sqrt(dipole_sse / max(1, total_structures * 3))
        else:
            metrics["dipole_mae"] = float('nan')
            metrics["dipole_rmse"] = float('nan')
            
        metrics["charge_mae"] = charge_sae / max(1, total_structures)
        metrics["charge_rmse"] = math.sqrt(charge_sse / max(1, total_structures))
        
    return metrics



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SpookyPhysNet model on test extxyz files.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (e.g. artifacts/spooky_so3lr/epoch-50)")
    parser.add_argument("--extxyz", required=True, help="Path to test .extxyz file or directory containing them")
    parser.add_argument("--cache-dir", required=True, help="Directory for caching parsed test structures")
    parser.add_argument("--batch-size-per-device", type=int, default=4)
    parser.add_argument("--max-pairs-per-device", type=int, default=18000)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--output", help="Optional path to write JSON evaluation summary")
    parser.add_argument(
        "--max-eval-structures",
        type=int,
        help=(
            "Limit evaluation to the first N structures after restoring the cache. "
            "For faster cache restore, prefer --max-structures, which creates a smaller cache."
        ),
    )
    parser.add_argument("--max-eval-batches", type=int, help="Stop after this many evaluation batches")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N eval batches")
    
    # Matching cache/parsing options
    parser.add_argument("--force-recache", action="store_true", help="Overwrite existing matching data cache")
    parser.add_argument("--energy-key", default="energy", help="ASE info key for energy target")
    parser.add_argument("--forces-key", default="forces", help="ASE arrays key for force target")
    parser.add_argument("--dipole-key", default="dipole", help="ASE info key for dipole target")
    parser.add_argument("--charge-key", default="charge", help="ASE info key for total molecular charge")
    parser.add_argument("--spin-key", default=None, help="ASE info key for spin multiplicity")
    parser.add_argument("--default-charge", type=float, default=0.0)
    parser.add_argument("--default-spin", type=float, default=1.0)
    parser.add_argument("--no-infer-spin", dest="infer_spin", action="store_false")
    parser.set_defaults(infer_spin=True)
    parser.add_argument("--max-structures", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10000)
    
    args = parser.parse_args()
    
    # 1. Load checkpoint parameters and config
    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"Restoring checkpoint from: {checkpoint_path}")
    params, config = restore_checkpoint(checkpoint_path)
    
    # Extract prediction config
    predict_charges = config.get("predict_charges", config.get("charges", False))
    args.predict_charges = predict_charges
    print(f"Checkpoint predict_charges setting: {predict_charges}")
    
    # 2. Get list of extxyz files to evaluate
    extxyz_input = Path(args.extxyz).resolve()
    if extxyz_input.is_dir():
        extxyz_files = sorted(list(extxyz_input.glob("*.extxyz")))
        if not extxyz_files:
            raise FileNotFoundError(f"No .extxyz files found in directory: {extxyz_input}")
    else:
        extxyz_files = [extxyz_input]
        
    print(f"Found {len(extxyz_files)} dataset(s) to evaluate.")
    
    # 3. Restore/Create JAX devices
    devices = jax.local_devices()[: args.num_devices]
    if len(devices) != args.num_devices:
        raise RuntimeError(
            f"Requested {args.num_devices} devices, but JAX sees {len(jax.local_devices())}"
        )
    print(f"Evaluating using devices: {devices}")
    
    results: dict[str, dict[str, float]] = {}
    
    # 4. Evaluate each dataset
    for extxyz_file in extxyz_files:
        print(f"\n--- Processing dataset: {extxyz_file.name} ---")
        cache_path = cache_extxyz_to_orbax(args, extxyz_file)
        data, metadata = restore_cached_data(cache_path)
        data, metadata = limit_cached_data(data, metadata, args.max_eval_structures)
        
        max_atoms = int(np.max(np.asarray(data["N"]).reshape(-1)))
        
        # Instantiate model for this max_atoms
        model = create_model_from_config(config, max_atoms=max_atoms)
        
        t0 = time.time()
        metrics = evaluate_dataset(model, params, data, metadata, args, devices)
        elapsed = time.time() - t0
        print(f"Evaluation finished in {elapsed:.1f} seconds.")
        
        results[extxyz_file.name] = metrics
        
    # 5. Print a nice formatted table of summary results
    print("\n" + "="*80)
    print("SpookyPhysNet Evaluation Summary Results")
    print("="*80)
    
    if predict_charges:
        header = f"{'Dataset':<35} | {'E MAE':<9} | {'E RMSE':<9} | {'F MAE':<9} | {'F RMSE':<9} | {'D MAE':<9} | {'Q MAE':<9}"
        print(header)
        print("-"*len(header))
        for name, m in results.items():
            print(
                f"{name:<35} | "
                f"{m['energy_mae']:.6f} | "
                f"{m['energy_rmse']:.6f} | "
                f"{m['forces_mae']:.6f} | "
                f"{m['forces_rmse']:.6f} | "
                f"{m['dipole_mae']:.6f} | "
                f"{m['charge_mae']:.6f}"
            )
    else:
        header = f"{'Dataset':<35} | {'E MAE':<9} | {'E RMSE':<9} | {'F MAE':<9} | {'F RMSE':<9}"
        print(header)
        print("-"*len(header))
        for name, m in results.items():
            print(
                f"{name:<35} | "
                f"{m['energy_mae']:.6f} | "
                f"{m['energy_rmse']:.6f} | "
                f"{m['forces_mae']:.6f} | "
                f"{m['forces_rmse']:.6f}"
            )
            
    print("="*80)
    
    # 6. Save outputs to JSON if specified
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
