#!/usr/bin/env python3
"""Cache SO3LR-style extxyz data in Orbax and train SpookyPhysNet on it.

Example:
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train_so3lr_spooky_extxyz.py \
        --extxyz /path/to/so3lr_train.extxyz \
        --cache-dir /path/to/so3lr_orbax_cache \
        --workdir artifacts/spooky_so3lr \
        --batch-size-per-device 4 \
        --epochs 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
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
import optax
import orbax.checkpoint as ocp
from ase.io import iread
from flax import jax_utils
from flax.training import orbax_utils, train_state

from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
from mmml.models.physnetjax.physnetjax.restart.restart import save_training_checkpoint
from mmml.models.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_flat_data,
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
    cache_version: int = 2


def _cache_name(meta: CacheMeta) -> str:
    payload = json.dumps(asdict(meta), sort_keys=True).encode()
    return f"{Path(meta.extxyz).stem}_flat_{hashlib.sha256(payload).hexdigest()[:16]}"


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


def _resolve_cache_path(args: argparse.Namespace) -> Path:
    extxyz = Path(args.extxyz).resolve()
    stat = extxyz.stat()
    meta = CacheMeta(
        extxyz=str(extxyz),
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


def _get_info_vector(info: dict[str, Any], key: str, size: int) -> np.ndarray:
    if key not in info:
        raise KeyError(f"Structure lacks required info key '{key}'")
    value = np.asarray(info[key], dtype=np.float64).reshape(-1)
    if value.size != size:
        raise ValueError(f"Expected '{key}' with {size} values, got shape {value.shape}")
    return value


def _get_energy(atoms, key: str, structure_index: int) -> float:
    if key in atoms.info:
        return _get_info_scalar(atoms.info, key, 0.0)
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return float(np.asarray(atoms.calc.results[key]).reshape(-1)[0])
    if key == "energy":
        try:
            return float(atoms.get_potential_energy())
        except Exception as exc:
            raise KeyError(
                f"Structure {structure_index} lacks energy in info/calculator results"
            ) from exc
    raise KeyError(
        f"Structure {structure_index} lacks energy key '{key}'; "
        f"info keys: {sorted(atoms.info)}; "
        f"calculator result keys: {sorted(getattr(atoms.calc, 'results', {}))}"
    )


def _get_forces(atoms, key: str, structure_index: int) -> np.ndarray:
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=np.float64)
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[key], dtype=np.float64)
    if key == "forces":
        try:
            return np.asarray(atoms.get_forces(), dtype=np.float64)
        except Exception as exc:
            raise KeyError(
                f"Structure {structure_index} lacks forces in arrays/calculator results"
            ) from exc
    raise KeyError(
        f"Structure {structure_index} lacks forces key '{key}'; "
        f"array keys: {sorted(atoms.arrays)}; "
        f"calculator result keys: {sorted(getattr(atoms.calc, 'results', {}))}"
    )


def _infer_spin_multiplicity(atoms, total_charge: float) -> float:
    """Infer singlet/doublet multiplicity from electron parity."""
    n_protons = int(np.sum(atoms.get_atomic_numbers()))
    n_electrons = int(round(n_protons - total_charge))
    return 1.0 if n_electrons % 2 == 0 else 2.0


def cache_extxyz_to_orbax(args: argparse.Namespace) -> Path:
    """Parse extxyz once into flat concatenated arrays and store them via Orbax."""
    extxyz = Path(args.extxyz).resolve()
    if not extxyz.exists():
        raise FileNotFoundError(extxyz)

    cache_path = _resolve_cache_path(args)
    if cache_path.exists() and not args.force_recache:
        print(f"Using existing Orbax data cache: {cache_path}")
        return cache_path

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
        f_parts.append(_get_forces(atoms, args.forces_key, i))
        energies.append(_get_energy(atoms, args.energy_key, i))
        dipoles.append(_get_info_vector(atoms.info, args.dipole_key, 3))
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving Orbax data cache: {cache_path}")
    _orbax_save(cache_path, data, force=True)
    print(
        "Cached "
        f"{len(energies):,} structures, {offsets[-1]:,} atoms, "
        f"max_atoms={max(natoms)}"
    )
    return cache_path


def restore_cached_data(cache_path: Path) -> dict[str, np.ndarray]:
    restored = ocp.PyTreeCheckpointer().restore(cache_path)
    return {
        key: np.asarray(value)
        for key, value in restored.items()
        if not key.startswith("metadata_")
    }


def split_indices(n_items: int, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items).astype(np.int64)
    n_valid = max(1, int(round(n_items * valid_fraction))) if valid_fraction > 0 else 0
    valid = order[:n_valid]
    train = order[n_valid:]
    if len(train) == 0:
        raise ValueError("Validation split consumed the full dataset")
    return train, valid


def bucket_indices_by_natoms(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[int, np.ndarray]:
    n_atoms = np.asarray(data["N"], dtype=np.int32).reshape(-1)
    buckets: dict[int, list[int]] = {}
    for idx in np.asarray(indices, dtype=np.int64):
        buckets.setdefault(int(n_atoms[idx]), []).append(int(idx))
    return {n: np.asarray(vals, dtype=np.int64) for n, vals in buckets.items()}


def iter_device_batches(
    buckets: dict[int, np.ndarray],
    *,
    per_device_batch_size: int,
    max_pairs_per_device: int,
    n_devices: int,
    rng: np.random.Generator,
) -> Any:
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    for n_atoms in bucket_keys:
        bucket_batch_size = min(
            per_device_batch_size,
            max(1, max_pairs_per_device // (n_atoms * n_atoms)),
        )
        global_batch = bucket_batch_size * n_devices
        indices = buckets[n_atoms].copy()
        rng.shuffle(indices)
        usable = (len(indices) // global_batch) * global_batch
        for start in range(0, usable, global_batch):
            chunk = indices[start : start + global_batch]
            yield chunk.reshape(n_devices, bucket_batch_size)


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


def create_model(args: argparse.Namespace, max_atoms: int) -> SpookyPhysNet:
    return SpookyPhysNet(
        features=args.features,
        max_degree=args.max_degree,
        num_iterations=args.num_iterations,
        num_basis_functions=args.num_basis_functions,
        cutoff=args.cutoff,
        max_atomic_number=args.max_atomic_number,
        charges=args.predict_charges,
        max_padded_atoms=max_atoms,
        n_refinement_blocks=args.n_res,
        zbl=not args.no_zbl,
        efa=args.efa,
        use_energy_bias=args.use_energy_bias,
    )


def make_steps(model: SpookyPhysNet, args: argparse.Namespace, devices: list[Any]):
    per_device_batch_size = args.batch_size_per_device
    energy_weight = args.energy_weight
    forces_weight = args.forces_weight
    dipole_weight = args.dipole_weight
    charges_weight = args.charges_weight

    def loss_fn(params, batch):
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
        energy_ref = batch["E"].reshape(-1, 1)
        forces_ref = batch["F"]
        force_mask = batch["atom_mask"][:, None]

        energy_mse = jnp.mean((energy_pred - energy_ref) ** 2)
        force_mse = jnp.sum(((forces_pred - forces_ref) ** 2) * force_mask)
        force_mse /= jnp.sum(force_mask) * 3.0 + 1e-8
        energy_mae = jnp.mean(jnp.abs(energy_pred - energy_ref))
        force_mae = jnp.sum(jnp.abs(forces_pred - forces_ref) * force_mask)
        force_mae /= jnp.sum(force_mask) * 3.0 + 1e-8
        loss = energy_weight * energy_mse + forces_weight * force_mse
        dipole_mae = jnp.asarray(0.0)
        charge_mae = jnp.asarray(0.0)
        dipole_mse = jnp.asarray(0.0)
        charge_mse = jnp.asarray(0.0)
        if args.predict_charges:
            dipole_pred = out["dipoles"].reshape(batch["D"].shape)
            charge_pred = out["sum_charges"].reshape(batch["Q_total"].shape)
            dipole_mse = jnp.mean((dipole_pred - batch["D"]) ** 2)
            charge_mse = jnp.mean((charge_pred - batch["Q_total"]) ** 2)
            dipole_mae = jnp.mean(jnp.abs(dipole_pred - batch["D"]))
            charge_mae = jnp.mean(jnp.abs(charge_pred - batch["Q_total"]))
            loss += dipole_weight * dipole_mse + charges_weight * charge_mse
        metrics = {
            "loss": loss,
            "energy_mae": energy_mae,
            "forces_mae": force_mae,
            "energy_mse": energy_mse,
            "forces_mse": force_mse,
            "dipole_mae": dipole_mae,
            "charge_mae": charge_mae,
            "dipole_mse": dipole_mse,
            "charge_mse": charge_mse,
        }
        return loss, metrics

    def train_step(state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        state = state.apply_gradients(grads=grads)
        metrics = jax.lax.pmean(metrics, axis_name="device")
        metrics["loss"] = jax.lax.pmean(loss, axis_name="device")
        return state, metrics

    def eval_step(state, batch):
        _, metrics = loss_fn(state.params, batch)
        return jax.lax.pmean(metrics, axis_name="device")

    return (
        jax.pmap(train_step, axis_name="device", devices=devices),
        jax.pmap(eval_step, axis_name="device", devices=devices),
    )


def mean_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics:
        return {}
    out: dict[str, float] = {}
    for key in metrics[0]:
        values = [float(np.asarray(m[key]).reshape(-1)[0]) for m in metrics]
        out[key] = float(np.mean(values))
    return out


def init_state(
    model: SpookyPhysNet,
    data: dict[str, np.ndarray],
    train_buckets: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> train_state.TrainState:
    rng = np.random.default_rng(args.seed)
    init_indices = None
    for n_atoms in sorted(train_buckets, key=lambda n: len(train_buckets[n]), reverse=True):
        if len(train_buckets[n_atoms]) >= args.batch_size_per_device:
            init_indices = train_buckets[n_atoms][: args.batch_size_per_device]
            break
    if init_indices is None:
        raise ValueError(
            "No atom-count bucket has enough structures for one per-device batch"
        )
    rng.shuffle(init_indices)
    batch = build_spooky_batch_from_flat_data(data, init_indices)
    variables = model.init(
        jax.random.PRNGKey(args.seed),
        atomic_numbers=batch["Z"],
        charges=batch["Q_atoms"],
        spins=batch["S_atoms"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=args.batch_size_per_device,
        batch_mask=batch["batch_mask"],
        atom_mask=batch["atom_mask"],
        compute_forces=False,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.clip_global_norm),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=tx,
    )


def _epoch_number_from_path(path: Path) -> int:
    try:
        return int(path.name.split("epoch-", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint_path(workdir: Path) -> Path | None:
    checkpoints = [
        path
        for path in workdir.glob("epoch-*")
        if path.is_dir() and "tmp" not in path.name
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=_epoch_number_from_path)


def resolve_restart_path(args: argparse.Namespace) -> Path | None:
    workdir = Path(args.workdir).resolve()
    if args.restart is None:
        if not args.auto_resume:
            return None
        return latest_checkpoint_path(workdir)
    if args.restart == "latest":
        latest = latest_checkpoint_path(workdir)
        if latest is None:
            print(
                f"No epoch-* checkpoints found in {workdir}; starting from scratch"
            )
        return latest
    return Path(args.restart).expanduser().resolve()


def _restore_state_from_checkpoint(
    checkpoint_path: Path,
    state: train_state.TrainState,
) -> tuple[train_state.TrainState, int, dict[str, Any]]:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)
    restored_model = restored.get("model")
    restored_params = restored.get("params")
    if restored_model is not None:
        typed_model = checkpointer.restore(
            checkpoint_path,
            item={"model": state},
            partial_restore=True,
        )["model"]
        state = state.replace(
            step=typed_model.step,
            params=typed_model.params,
            opt_state=typed_model.opt_state,
        )
    elif restored_params is not None:
        state = state.replace(params=restored_params)
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} has no 'model' or 'params'")

    epoch = int(np.asarray(restored.get("epoch", _epoch_number_from_path(checkpoint_path))))
    metrics = restored.get("metrics", {})
    return state, epoch, metrics


def _merge_compatible_params(
    initialized: Any,
    loaded: Any,
) -> tuple[Any, int, int, int]:
    if isinstance(initialized, Mapping):
        loaded_mapping = loaded if isinstance(loaded, Mapping) else {}
        merged = {}
        loaded_count = 0
        initialized_count = 0
        skipped_count = 0
        for key, initialized_value in initialized.items():
            if key not in loaded_mapping:
                merged[key] = initialized_value
                initialized_count += len(jax.tree_util.tree_leaves(initialized_value))
                continue
            value, used, fresh, skipped = _merge_compatible_params(
                initialized_value, loaded_mapping[key]
            )
            merged[key] = value
            loaded_count += used
            initialized_count += fresh
            skipped_count += skipped
        return merged, loaded_count, initialized_count, skipped_count

    if np.shape(initialized) == np.shape(loaded):
        return loaded, 1, 0, 0
    return initialized, 0, 0, 1


def _initialize_from_checkpoint(
    checkpoint_path: Path,
    state: train_state.TrainState,
) -> train_state.TrainState:
    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path)
    loaded_params = restored.get("params")
    if loaded_params is None and isinstance(restored.get("model"), Mapping):
        loaded_params = restored["model"].get("params")
    if loaded_params is None:
        raise ValueError(f"Checkpoint {checkpoint_path} has no parameters")

    params, loaded, initialized, skipped = _merge_compatible_params(
        state.params, loaded_params
    )
    print(
        f"Warm-started from {checkpoint_path}: loaded {loaded} parameter leaves, "
        f"initialized {initialized} new leaves, skipped {skipped} incompatible leaves"
    )
    return state.replace(params=params)


def _restored_best_valid_loss(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("best_valid_loss")
    if value is None and isinstance(metrics.get("valid"), dict):
        value = metrics["valid"].get("loss")
    if value is None:
        return None
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def save_epoch_checkpoint(
    workdir: Path,
    epoch: int,
    state: train_state.TrainState,
    model: SpookyPhysNet,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    ckpt = {
        "model": state,
        "params": state.params,
        "model_attributes": {
            **model.return_attributes(),
            "model_type": "spooky",
        },
        "config": vars(args),
        "epoch": epoch,
        "metrics": metrics,
    }
    save_training_checkpoint(workdir / f"epoch-{epoch:04d}", ckpt)


def train(args: argparse.Namespace, cache_path: Path) -> None:
    data = restore_cached_data(cache_path)
    n_molecules = int(np.asarray(data["E"]).shape[0])
    max_atoms = int(np.max(np.asarray(data["N"]).reshape(-1)))
    train_idx, valid_idx = split_indices(n_molecules, args.valid_fraction, args.seed)
    train_buckets = bucket_indices_by_natoms(data, train_idx)
    valid_buckets = bucket_indices_by_natoms(data, valid_idx)

    devices = jax.local_devices()[: args.num_devices]
    if len(devices) != args.num_devices:
        raise RuntimeError(
            f"Requested {args.num_devices} devices, but JAX sees {len(jax.local_devices())}: "
            f"{jax.local_devices()}"
        )

    model = create_model(args, max_atoms=max_atoms)
    state = init_state(model, data, train_buckets, args)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "run_config.json").open("w") as fh:
        json.dump({**vars(args), "cache_path": str(cache_path)}, fh, indent=2, sort_keys=True)

    print(f"JAX devices: {devices}")
    print(f"Train structures: {len(train_idx):,}; valid structures: {len(valid_idx):,}")
    print(f"Max atoms: {max_atoms}; per-device batch: {args.batch_size_per_device}")
    print(f"Checkpoint directory: {workdir}")

    start_epoch = 1
    best_valid = math.inf
    if args.init_checkpoint is not None:
        if args.restart is not None or args.auto_resume:
            raise ValueError(
                "--init-checkpoint cannot be combined with --restart or --auto-resume"
            )
        state = _initialize_from_checkpoint(
            Path(args.init_checkpoint).expanduser().resolve(), state
        )
    restart_path = resolve_restart_path(args)
    if restart_path is not None:
        state, restored_epoch, restored_metrics = _restore_state_from_checkpoint(
            restart_path, state
        )
        start_epoch = restored_epoch + 1
        restored_best = _restored_best_valid_loss(restored_metrics)
        if restored_best is not None:
            best_valid = restored_best
        print(
            f"Restarted from {restart_path} at epoch {restored_epoch}; "
            f"continuing from epoch {start_epoch}"
        )
        if start_epoch > args.epochs:
            print(
                f"Nothing to do: restart epoch {restored_epoch} is already >= "
                f"--epochs {args.epochs}"
            )
            return

    state = jax_utils.replicate(state, devices=devices)
    rng = np.random.default_rng(args.seed)
    step_functions: dict[int, tuple[Any, Any]] = {}

    def steps_for_batch_size(batch_size: int) -> tuple[Any, Any]:
        if batch_size not in step_functions:
            step_args = argparse.Namespace(
                **{**vars(args), "batch_size_per_device": batch_size}
            )
            step_functions[batch_size] = make_steps(model, step_args, devices)
        return step_functions[batch_size]

    compiled_shapes: set[tuple[int, int]] = set()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = []
        train_batches = iter_device_batches(
            train_buckets,
            per_device_batch_size=args.batch_size_per_device,
            max_pairs_per_device=args.max_pairs_per_device,
            n_devices=args.num_devices,
            rng=rng,
        )
        for step, device_indices in enumerate(train_batches, start=1):
            batch_size = int(device_indices.shape[1])
            n_atoms = int(
                np.asarray(data["N"])[device_indices[0, 0]].reshape(-1)[0]
            )
            shape = (n_atoms, batch_size)
            if shape not in compiled_shapes:
                print(
                    f"Compiling steps for {n_atoms}-atom structures "
                    f"with per-device batch {batch_size}",
                    flush=True,
                )
                compiled_shapes.add(shape)
            train_step, _ = steps_for_batch_size(batch_size)
            batch = stack_device_batches(data, device_indices)
            state, metrics = train_step(state, batch)
            train_metrics.append(metrics)
            if step % args.log_every_steps == 0:
                m = mean_metrics([metrics])
                print(
                    f"epoch {epoch:04d} step {step:06d} "
                    f"loss={m['loss']:.6g} E_MAE={m['energy_mae']:.6g} "
                    f"F_MAE={m['forces_mae']:.6g} "
                    f"D_MAE={m['dipole_mae']:.6g} Q_MAE={m['charge_mae']:.6g}"
                )
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break

        valid_metrics = []
        valid_batches = iter_device_batches(
            valid_buckets,
            per_device_batch_size=args.batch_size_per_device,
            max_pairs_per_device=args.max_pairs_per_device,
            n_devices=args.num_devices,
            rng=np.random.default_rng(args.seed + epoch),
        )
        for step, device_indices in enumerate(valid_batches, start=1):
            _, eval_step = steps_for_batch_size(int(device_indices.shape[1]))
            batch = stack_device_batches(data, device_indices)
            valid_metrics.append(eval_step(state, batch))
            if args.valid_steps and step >= args.valid_steps:
                break

        train_mean = mean_metrics(train_metrics)
        valid_mean = mean_metrics(valid_metrics)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:04d} done in {elapsed:.1f}s "
            f"train_loss={train_mean.get('loss', float('nan')):.6g} "
            f"valid_loss={valid_mean.get('loss', float('nan')):.6g} "
            f"valid_E_MAE={valid_mean.get('energy_mae', float('nan')):.6g} "
            f"valid_F_MAE={valid_mean.get('forces_mae', float('nan')):.6g} "
            f"valid_D_MAE={valid_mean.get('dipole_mae', float('nan')):.6g} "
            f"valid_Q_MAE={valid_mean.get('charge_mae', float('nan')):.6g}"
        )

        should_save = epoch % args.save_every == 0
        valid_loss = valid_mean.get("loss", math.inf)
        if valid_loss < best_valid:
            best_valid = valid_loss
            should_save = True
        if should_save:
            unreplicated = jax_utils.unreplicate(state)
            save_epoch_checkpoint(
                workdir,
                epoch,
                unreplicated,
                model,
                args,
                {"train": train_mean, "valid": valid_mean, "best_valid_loss": best_valid},
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache extxyz in Orbax and train SpookyPhysNet with charge/spin inputs."
    )
    parser.add_argument("--extxyz", required=True, help="Input extxyz file, e.g. so3lr_train.extxyz")
    parser.add_argument("--cache-dir", required=True, help="Directory for Orbax data caches")
    parser.add_argument("--workdir", default="artifacts/spooky_so3lr", help="Training output/checkpoint directory")
    parser.add_argument("--mode", choices=("cache", "train", "cache-and-train"), default="cache-and-train")
    parser.add_argument("--force-recache", action="store_true", help="Overwrite existing matching data cache")
    parser.add_argument("--energy-key", default="energy", help="ASE info key for scalar energy target")
    parser.add_argument("--forces-key", default="forces", help="ASE arrays key for force target")
    parser.add_argument("--dipole-key", default="dipole", help="ASE info key for dipole target")
    parser.add_argument("--charge-key", default="charge", help="ASE info key for total molecular charge")
    parser.add_argument("--spin-key", default=None, help="ASE info key for spin multiplicity; defaults to singlet")
    parser.add_argument("--default-charge", type=float, default=0.0)
    parser.add_argument("--default-spin", type=float, default=1.0)
    parser.add_argument(
        "--no-infer-spin",
        dest="infer_spin",
        action="store_false",
        help="Use --default-spin when --spin-key is absent instead of electron parity",
    )
    parser.set_defaults(infer_spin=True)
    parser.add_argument("--max-structures", type=int, default=None, help="Optional smoke-test cap")
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--batch-size-per-device", type=int, default=4)
    parser.add_argument(
        "--max-pairs-per-device",
        type=int,
        default=18000,
        help=(
            "Approximate per-device batch*n_atoms^2 budget; automatically reduces "
            "the structure batch for larger molecules"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--restart",
        default=None,
        help="Path to an epoch-* checkpoint, or 'latest' to use the newest under --workdir",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Warm-start compatible parameters from a checkpoint while initializing "
            "new heads and optimizer state from scratch"
        ),
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from newest epoch-* under --workdir when present",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--valid-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-global-norm", type=float, default=10.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=52.91)
    parser.add_argument("--dipole-weight", type=float, default=1.0)
    parser.add_argument("--charges-weight", type=float, default=1.0)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--num-basis-functions", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-atomic-number", type=int, default=87)
    parser.add_argument("--n-res", type=int, default=2)
    parser.add_argument("--predict-charges", action="store_true", help="Also predict atomic charges/dipoles")
    parser.add_argument("--no-zbl", action="store_true")
    parser.add_argument("--efa", action="store_true")
    parser.add_argument("--use-energy-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10000, help="Structure interval while parsing extxyz")
    parser.add_argument("--log-every-steps", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_path = _resolve_cache_path(args)
    if args.mode in {"cache", "cache-and-train"}:
        cache_path = cache_extxyz_to_orbax(args)
    if args.mode in {"train", "cache-and-train"}:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Orbax data cache not found: {cache_path}. Run with --mode cache first."
            )
        train(args, cache_path)


if __name__ == "__main__":
    main()
