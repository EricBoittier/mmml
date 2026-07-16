#!/usr/bin/env python3
"""Cache SO3LR-style extxyz data and train a SpookyPhysNet residual model.

With ``--mbd-checkpoint``, the QCML MBD model is restored once and kept frozen.
Its geometry-dependent energy and force correction are added to Spooky's output,
while only Spooky parameters are optimized.  Spooky's latent atom-centred
charges/dipoles and configurable ZBL term remain active.

Example:
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train_so3lr_spooky_extxyz.py \
        --extxyz /path/to/so3lr_train.extxyz \
        --cache-dir /path/to/so3lr_orbax_cache \
        --workdir artifacts/spooky_so3lr \
        --batch-size-per-device 64 \
        --atom-bucket-width 4 \
        --prefetch-batches 2 \
        --epochs 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import threading
import time
from collections.abc import Iterator, Mapping
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
from mmml.utils.model_checkpoint import json_to_params
from mmml.models.mbd.calculator import (
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    load_mbd_model,
)
from mmml.models.mbd.model import mbd_energy_and_forces

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

DIPOLE_KEY_ALIASES = (
    "dipole",
    "dipoles",
    "Dipole",
    "D",
    "Dxyz",
    "D_xyz",
    "dxyz",
    "DXYZ",
    "dipole_moment",
    "dipole_moments",
    "dipole_vector",
    "molecular_dipole",
    "molecular_dipole_moment",
    "total_dipole",
    "mu",
    "Mu",
    "muxyz",
    "mu_xyz",
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


def _find_exact_or_casefold_key(mapping: Mapping[str, Any], key: str) -> str | None:
    if key in mapping:
        return key
    key_lower = key.lower()
    for actual_key in mapping:
        if actual_key.lower() == key_lower:
            return actual_key
    return None


def _find_vector_key(atoms, requested_key: str, size: int) -> str | None:
    for key in _candidate_vector_keys(requested_key):
        info_key = _find_exact_or_casefold_key(atoms.info, key)
        if info_key is not None:
            _vector_from_raw(atoms.info[info_key], info_key, size)
            return info_key
        calc_results = getattr(atoms.calc, "results", {}) if atoms.calc is not None else {}
        calc_key = _find_exact_or_casefold_key(calc_results, key)
        if calc_key is not None:
            _vector_from_raw(calc_results[calc_key], calc_key, size)
            return calc_key
        array_key = _find_exact_or_casefold_key(atoms.arrays, key)
        if array_key is not None:
            value = np.asarray(atoms.arrays[array_key])
            if value.reshape(-1).size == size:
                _vector_from_raw(value, array_key, size)
                return array_key
    return None


def _get_vector_from_atoms(atoms, requested_key: str, size: int) -> tuple[np.ndarray, str]:
    key = _find_vector_key(atoms, requested_key, size)
    if key is None:
        aliases = ", ".join(_candidate_vector_keys(requested_key))
        raise KeyError(
            f"Structure lacks vector key '{requested_key}'. Tried aliases: {aliases}; "
            f"info keys: {sorted(atoms.info)}; "
            f"calculator result keys: {sorted(getattr(atoms.calc, 'results', {}))}"
        )
    if key in atoms.info:
        return _vector_from_raw(atoms.info[key], key, size), key
    calc_results = getattr(atoms.calc, "results", {}) if atoms.calc is not None else {}
    if key in calc_results:
        return _vector_from_raw(calc_results[key], key, size), key
    return _vector_from_raw(atoms.arrays[key], key, size), key


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
    resolved_dipole_key: str | None = None

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
        dipole, used_dipole_key = _get_vector_from_atoms(atoms, args.dipole_key, 3)
        if resolved_dipole_key is None:
            resolved_dipole_key = used_dipole_key
            if resolved_dipole_key != args.dipole_key:
                print(
                    f"Using dipole key alias '{resolved_dipole_key}' "
                    f"for requested key '{args.dipole_key}'",
                    flush=True,
                )
        dipoles.append(dipole)
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
    # orbax/tensorstore can't serialize string-dtype arrays, so the resolved
    # dipole key is recorded in a plain JSON sidecar instead of the pytree.
    with (cache_path / "metadata.json").open("w") as fh:
        json.dump({"dipole_key": resolved_dipole_key or ""}, fh)
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


def pad_atoms_for_count(n_atoms: int, bucket_width: int) -> int:
    """Round atom count up to the training pad width for a bucket."""
    n = int(n_atoms)
    width = max(1, int(bucket_width))
    if width == 1:
        return n
    return ((n + width - 1) // width) * width


def pairs_budget_batch_size(
    pad_atoms: int,
    *,
    per_device_batch_size: int,
    max_pairs_per_device: int,
) -> int:
    """Legacy O(n²) pair-budget cap (used when --no-auto-batch)."""
    n = max(1, int(pad_atoms))
    return min(
        int(per_device_batch_size),
        max(1, int(max_pairs_per_device) // (n * n)),
    )


def bucket_indices_by_natoms(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    bucket_width: int = 1,
) -> dict[int, np.ndarray]:
    """Group molecule indices by pad width (exact n when bucket_width=1)."""
    n_atoms = np.asarray(data["N"], dtype=np.int32).reshape(-1)
    buckets: dict[int, list[int]] = {}
    for idx in np.asarray(indices, dtype=np.int64):
        pad = pad_atoms_for_count(int(n_atoms[idx]), bucket_width)
        buckets.setdefault(pad, []).append(int(idx))
    return {n: np.asarray(vals, dtype=np.int64) for n, vals in buckets.items()}


def iter_device_batches(
    buckets: dict[int, np.ndarray],
    *,
    batch_sizes: Mapping[int, int],
    n_devices: int,
    rng: np.random.Generator,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(pad_atoms, device_indices)`` with shape ``(n_devices, B)``."""
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    for pad_atoms in bucket_keys:
        bucket_batch_size = int(batch_sizes[pad_atoms])
        if bucket_batch_size < 1:
            continue
        global_batch = bucket_batch_size * n_devices
        indices = buckets[pad_atoms].copy()
        rng.shuffle(indices)
        usable = (len(indices) // global_batch) * global_batch
        for start in range(0, usable, global_batch):
            chunk = indices[start : start + global_batch]
            yield pad_atoms, chunk.reshape(n_devices, bucket_batch_size)


def stack_device_batches(
    data: dict[str, np.ndarray],
    device_indices: np.ndarray,
    *,
    pad_atoms: int | None = None,
) -> dict[str, Any]:
    batches = [
        build_spooky_batch_from_flat_data(
            data, device_indices[i], pad_atoms=pad_atoms
        )
        for i in range(device_indices.shape[0])
    ]
    has_mm = "cgenff_master_sigmas" in data and "cgenff_master_epsilons" in data
    for i, batch in enumerate(batches):
        indices = device_indices[i]
        batch["D"] = jnp.asarray(data["D"][indices], dtype=jnp.float32)
        batch["Q_total"] = jnp.asarray(data["Q"][indices], dtype=jnp.float32)
        batch["S_total"] = jnp.asarray(data["S"][indices], dtype=jnp.float32)
        if has_mm:
            # Master tables are constant — same for every batch, just replicated per device
            batch["cgenff_master_sigmas"]   = jnp.asarray(data["cgenff_master_sigmas"],   dtype=jnp.float32)
            batch["cgenff_master_epsilons"] = jnp.asarray(data["cgenff_master_epsilons"], dtype=jnp.float32)
    stacked: dict[str, Any] = {}
    for key in batches[0]:
        if key == "batch_size":
            continue
        stacked[key] = jnp.stack([batch[key] for batch in batches], axis=0)
    return stacked


def prefetch_stacked_batches(
    index_iter: Iterator[tuple[int, np.ndarray]],
    data: dict[str, np.ndarray],
    *,
    depth: int = 2,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Build stacked device batches on a background thread.

    Yields ``(pad_atoms, per_device_batch_size, batch)``.
    Safe to break early (``steps_per_epoch`` / ``valid_steps``): the producer
    stops and joins instead of blocking forever on a full queue.
    """
    depth = max(1, int(depth))
    out_q: queue.Queue[tuple[int, int, dict[str, Any]] | None] = queue.Queue(
        maxsize=depth
    )
    error: list[BaseException] = []
    stop = threading.Event()

    def _worker() -> None:
        try:
            for pad_atoms, device_indices in index_iter:
                if stop.is_set():
                    break
                batch = stack_device_batches(
                    data, device_indices, pad_atoms=pad_atoms
                )
                item = (pad_atoms, int(device_indices.shape[1]), batch)
                while not stop.is_set():
                    try:
                        out_q.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 — surface to consumer
            error.append(exc)
        finally:
            try:
                out_q.put(None, timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

    thread = threading.Thread(target=_worker, name="so3lr-batch-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = out_q.get()
            if item is None:
                break
            yield item
    finally:
        stop.set()
        while True:
            try:
                out_q.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=30.0)
        if error:
            raise error[0]


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).upper()
    markers = (
        "RESOURCE_EXHAUSTED",
        "OUT_OF_MEMORY",
        "OUT OF MEMORY",
        "RAN OUT OF MEMORY",
        "FAILED TO ALLOCATE",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "CNDRV_ALLOC",
        "BFC_ALLOCATOR",
        "ALLOCATOR (GPU",
    )
    return any(m in text for m in markers)


def _clear_jax_caches() -> None:
    try:
        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass


def probe_max_batch_size(
    *,
    pad_atoms: int,
    max_batch: int,
    start_batch: int = 1,
    data: dict[str, np.ndarray],
    candidate_indices: np.ndarray,
    n_devices: int,
    steps_for_batch_size: Any,
    state: Any,
) -> int:
    """Find the largest per-device batch that fits for ``pad_atoms``.

    Starts at ``start_batch`` (typically the pair-budget heuristic) and grows
    upward, so large-molecule buckets do not immediately request multi‑GiB
    allocations at B=64/128.
    """
    max_batch = max(1, int(max_batch))
    start_batch = max(1, min(max_batch, int(start_batch)))
    pool = np.asarray(candidate_indices, dtype=np.int64)
    if pool.size < n_devices:
        return 1

    def _try(batch_size: int) -> bool:
        need = batch_size * n_devices
        if pool.size < need:
            return False
        print(
            f"    probe pad_atoms={pad_atoms} B/device={batch_size} ...",
            flush=True,
        )
        device_indices = pool[:need].reshape(n_devices, batch_size)
        batch = stack_device_batches(data, device_indices, pad_atoms=pad_atoms)
        train_step, _ = steps_for_batch_size(batch_size)
        try:
            new_state, metrics = train_step(state, batch)
            jax.block_until_ready(metrics["loss"])
            del new_state
            return True
        except Exception as exc:  # noqa: BLE001 — OOM is raised as various types
            if not _is_oom_error(exc):
                raise
            print(
                f"    probe pad_atoms={pad_atoms} B/device={batch_size} OOM, backing off",
                flush=True,
            )
            _clear_jax_caches()
            return False

    def _search(lo: int, hi: int, best: int) -> int:
        while lo <= hi:
            mid = (lo + hi) // 2
            if _try(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    # Establish a feasible floor at/below start_batch.
    if _try(start_batch):
        best = start_batch
    else:
        best = _search(1, start_batch - 1, 1)
        return best

    # Grow upward from the floor toward max_batch.
    trial = min(max_batch, max(best + 1, best * 2))
    while trial <= max_batch:
        if trial == best:
            break
        if _try(trial):
            best = trial
            if best >= max_batch:
                return best
            trial = min(max_batch, max(best + 1, best * 2))
            continue
        return _search(best + 1, trial - 1, best)
    return best


def resolve_batch_sizes(
    buckets: Mapping[int, np.ndarray],
    *,
    per_device_batch_size: int,
    max_pairs_per_device: int,
    auto_batch: bool,
    data: dict[str, np.ndarray] | None = None,
    n_devices: int = 1,
    steps_for_batch_size: Any | None = None,
    state: Any | None = None,
    step_functions: dict[int, Any] | None = None,
) -> dict[int, int]:
    """Per-pad-atoms batch sizes: memory probe when enabled, else pair budget."""
    sizes: dict[int, int] = {}
    # Largest pads first: they force small B and avoid leaving huge compiled
    # small-molecule kernels resident while probing 100+ atom buckets.
    pad_order = sorted(buckets, reverse=True)
    committed_bsz: set[int] = set()
    for pad_atoms in pad_order:
        indices = buckets[pad_atoms]
        heuristic = pairs_budget_batch_size(
            pad_atoms,
            per_device_batch_size=per_device_batch_size,
            max_pairs_per_device=max_pairs_per_device,
        )
        if (
            not auto_batch
            or data is None
            or steps_for_batch_size is None
            or state is None
        ):
            sizes[pad_atoms] = heuristic
            continue
        print(
            f"  auto-batch probing pad_atoms={pad_atoms} "
            f"(start B={heuristic}, cap={per_device_batch_size})...",
            flush=True,
        )
        probed = probe_max_batch_size(
            pad_atoms=pad_atoms,
            max_batch=per_device_batch_size,
            start_batch=heuristic,
            data=data,
            candidate_indices=indices,
            n_devices=n_devices,
            steps_for_batch_size=steps_for_batch_size,
            state=state,
        )
        sizes[pad_atoms] = max(1, probed)
        committed_bsz.add(sizes[pad_atoms])
        print(
            f"  auto-batch pad_atoms={pad_atoms}: B/device={sizes[pad_atoms]} "
            f"(cap={per_device_batch_size}, pair-heuristic={heuristic})",
            flush=True,
        )
        # Drop compiled step fns for batch sizes we are not keeping so the next
        # pad width is not fighting residual executable / allocator pressure.
        if step_functions is not None:
            for key in list(step_functions):
                if key not in committed_bsz:
                    del step_functions[key]
        _clear_jax_caches()
    return sizes


def accumulate_metrics(
    total: dict[str, Any] | None, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Sum metric trees on-device (no host sync)."""
    if total is None:
        return {k: v for k, v in metrics.items()}
    return {k: total[k] + metrics[k] for k in total}


def finalize_metrics(
    total: Mapping[str, Any] | None, count: int
) -> dict[str, float]:
    """Host-sync averaged metrics once (blocks the device pipeline)."""
    if not total or count <= 0:
        return {}
    means = {k: total[k] / float(count) for k in total}
    leaves = jax.tree_util.tree_leaves(means)
    if leaves:
        jax.block_until_ready(leaves)
    out: dict[str, float] = {}
    for key, value in means.items():
        out[key] = float(np.asarray(jax.device_get(value)).reshape(-1)[0])
    return out


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
        trainable_zbl=args.trainable_zbl,
        zbl_cuton=args.zbl_cuton,
        zbl_cutoff=args.zbl_cutoff,
        efa=args.efa,
        use_energy_bias=args.use_energy_bias,
        electrostatics_damping_sigma=args.electrostatics_damping_sigma,
        # --fixed-cgenff-vdw pins the CGenFF LJ term at its published parameters, so it
        # acts as a fixed physical prior the network can only add to, never scale away.
        learn_cgenff_vdw_scale=not getattr(args, "fixed_cgenff_vdw", False),
        predict_atomic_vdw_scale=not getattr(args, "fixed_cgenff_vdw", False),
        interaction_trust_map=getattr(args, "interaction_trust_map", False),
    )


def make_steps(
    model: SpookyPhysNet,
    args: argparse.Namespace,
    devices: list[Any],
    *,
    mbd_model: Any | None = None,
    mbd_params: Any | None = None,
    multipole_model: Any | None = None,
    multipole_params: Any | None = None,
):
    per_device_batch_size = args.batch_size_per_device
    energy_weight = args.energy_weight
    forces_weight = args.forces_weight
    dipole_weight = args.dipole_weight
    charges_weight = args.charges_weight
    mbd_weight = args.mbd_weight
    multipole_consistency_weight = args.multipole_consistency_weight
    neural_interaction_l2 = args.neural_interaction_l2
    if (mbd_model is None) != (mbd_params is None):
        raise ValueError("mbd_model and mbd_params must be provided together")
    if mbd_params is not None:
        # These parameters are fixed external physics.  Positions deliberately
        # remain differentiable so the residual learns against MBD forces.
        mbd_params = jax.tree.map(jax.lax.stop_gradient, mbd_params)
    if (multipole_model is None) != (multipole_params is None):
        raise ValueError("multipole_model and multipole_params must be provided together")
    if multipole_params is not None:
        # Frozen reference dipole model — only used to compute a consistency target,
        # never updated. Gradients through it are only needed w.r.t. positions if we
        # ever want to backprop the consistency loss into geometry; for now it's used
        # purely as a stop-gradient dipole reference.
        multipole_params = jax.tree.map(jax.lax.stop_gradient, multipole_params)

    def loss_fn(params, batch, mbd_scale):
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
            mol_id=batch.get("mol_id"),
            cgenff_type_idx=None if args.no_cgenff_vdw else batch.get("cgenff_type_idx"),
            cgenff_master_sigmas=None if args.no_cgenff_vdw else batch.get("cgenff_master_sigmas"),
            cgenff_master_epsilons=None if args.no_cgenff_vdw else batch.get("cgenff_master_epsilons"),
        )
        spooky_energy = out["energy"].reshape(-1, 1)
        spooky_forces = out["forces"].reshape(batch["F"].shape)
        mbd_energy = jnp.zeros_like(spooky_energy)
        mbd_forces = jnp.zeros_like(spooky_forces)
        if mbd_model is not None:
            # E (and F) in this dataset are total complex energies (confirmed: they
            # scale with atom count, ~-5 eV/atom, not the O(0.01-2 eV) size-independent
            # magnitude an interaction energy would have). So MBD is a whole-system
            # dispersion correction here, same as for single molecules — no counterpoise
            # E(AB)-E(A)-E(B) decomposition. mol_id is still used elsewhere (electrostatics,
            # CGenFF LJ) to mask those terms to inter-monomer pairs only.
            mbd_output, mbd_forces_au = mbd_energy_and_forces(
                mbd_model,
                mbd_params,
                positions=batch["R"] * ANGSTROM_TO_BOHR,
                atomic_numbers=batch["Z"],
                charge=batch["Q_total"].reshape(-1),
                spin=batch["S_total"].reshape(-1),
                dst_idx=batch["dst_idx"],
                src_idx=batch["src_idx"],
                batch_segments=batch["batch_segments"],
                batch_size=per_device_batch_size,
                atom_mask=batch["atom_mask"],
                edge_mask=batch["batch_mask"],
            )
            mbd_energy = mbd_output["energy"].reshape(-1, 1) * HARTREE_TO_EV
            mbd_forces = mbd_forces_au.reshape(batch["F"].shape) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        composite_mbd_weight = mbd_weight * mbd_scale
        energy_pred = spooky_energy + composite_mbd_weight * mbd_energy
        forces_pred = spooky_forces + composite_mbd_weight * mbd_forces
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
        multipole_dipole_mae = jnp.asarray(0.0)
        multipole_dipole_mse = jnp.asarray(0.0)
        if args.predict_charges:
            dipole_pred = out["dipoles"].reshape(batch["D"].shape)
            charge_pred = out["sum_charges"].reshape(batch["Q_total"].shape)
            dipole_mse = jnp.mean((dipole_pred - batch["D"]) ** 2)
            charge_mse = jnp.mean((charge_pred - batch["Q_total"]) ** 2)
            dipole_mae = jnp.mean(jnp.abs(dipole_pred - batch["D"]))
            charge_mae = jnp.mean(jnp.abs(charge_pred - batch["Q_total"]))
            loss += dipole_weight * dipole_mse + charges_weight * charge_mse
            if multipole_model is not None:
                # Auxiliary signal from a frozen, separately-trained multipole model:
                # encourage the network's own learned-charge dipole to agree with an
                # independent dipole prediction, on top of (not instead of) the main
                # dipole_mse loss against the DFT reference D. Combination strategy is
                # deliberately simple (plain MSE, weighted by --multipole-consistency-
                # weight) since the "right" way to blend the two hasn't been decided yet.
                multipole_out = multipole_model.apply(
                    {"params": multipole_params},
                    positions=batch["R"],
                    atomic_numbers=batch["Z"],
                    charge=batch["Q_total"].reshape(-1),
                    spin=batch["S_total"].reshape(-1),
                    dst_idx=batch["dst_idx"],
                    src_idx=batch["src_idx"],
                    batch_segments=batch["batch_segments"],
                    batch_size=per_device_batch_size,
                    atom_mask=batch["atom_mask"],
                    edge_mask=batch["batch_mask"],
                )
                multipole_dipole = jax.lax.stop_gradient(
                    multipole_out["multipoles"][:, 1:4].reshape(batch["D"].shape)
                )
                multipole_dipole_mse = jnp.mean((dipole_pred - multipole_dipole) ** 2)
                multipole_dipole_mae = jnp.mean(jnp.abs(dipole_pred - multipole_dipole))
                loss += multipole_consistency_weight * multipole_dipole_mse
        # Shrinkage of the neural *interaction* energy toward zero.
        #
        # The MM terms (CGenFF LJ + electrostatics) are a physical prior, but nothing
        # stops the neural term from swamping them: measured on the dimer scans, the
        # neural interaction energy is 4.1x LARGER on pairs with <15 training structures
        # than on pairs with >=300 -- loudest exactly where there is no evidence for it
        # (ACE-BENZ, n=2: neural 7.76 kcal/mol vs an LJ prior of 0.007).
        #
        # Penalising (E_neural(AB) - E_neural(A) - E_neural(B))^2 is ridge shrinkage
        # toward the prior: the residual decays to zero by default, and the energy/force
        # loss overrides it wherever the data actually demands. The zero-evidence limit
        # becomes CGenFF instead of an invented surface.
        #
        # E_neural(A) + E_neural(B) is obtained exactly by masking the inter-monomer
        # edges out of the pair list: with those cut, every atom only sees its own
        # monomer, and all pairwise prior terms vanish too, so subtracting the prior
        # components isolates the neural residual. Forces are not needed for the penalty,
        # so the extra pass is forward-only.
        neural_int_mse = jnp.asarray(0.0)
        if (neural_interaction_l2 > 0.0 or args.interaction_trust_map) and batch.get("mol_id") is not None:
            mol_id = batch["mol_id"]
            # edge_mask keeps only intra-monomer message-passing edges AND zeroes the
            # inter-monomer prior pair terms, so the masked pass equals the two monomers
            # evaluated in isolation (E_neural(A) + E_neural(B)).
            intra_edge = (
                jnp.take(mol_id, batch["dst_idx"]) == jnp.take(mol_id, batch["src_idx"])
            ).astype(batch["batch_mask"].dtype)
            out_intra = model.apply(
                params,
                atomic_numbers=batch["Z"],
                charges=batch["Q_atoms"],
                spins=batch["S_atoms"],
                positions=batch["R"],
                dst_idx=batch["dst_idx"],
                src_idx=batch["src_idx"],
                batch_segments=batch["batch_segments"],
                batch_size=per_device_batch_size,
                batch_mask=batch["batch_mask"] * intra_edge,
                atom_mask=batch["atom_mask"],
                mol_id=mol_id,
                cgenff_type_idx=None if args.no_cgenff_vdw else batch.get("cgenff_type_idx"),
                cgenff_master_sigmas=None if args.no_cgenff_vdw else batch.get("cgenff_master_sigmas"),
                cgenff_master_epsilons=None if args.no_cgenff_vdw else batch.get("cgenff_master_epsilons"),
                edge_mask=intra_edge,
                compute_forces=False,
            )

            def _neural_only(o):
                total = _per_structure(o["energy"], batch["batch_segments"], per_device_batch_size)
                prior = sum(
                    _per_structure(o.get(k), batch["batch_segments"], per_device_batch_size)
                    for k in ("electrostatics", "cgenff_vdw", "repulsion")
                )
                return total - prior

            neural_interaction = _neural_only(out) - _neural_only(out_intra)
            neural_int_mse = jnp.mean(neural_interaction**2)
            if args.interaction_trust_map:
                tm_loss, _ = _interaction_trust_map_loss(
                    neural_interaction,
                    out["neural_interaction_log_lambda"],
                    Z=batch["Z"],
                    R=batch["R"],
                    dst_idx=batch["dst_idx"],
                    src_idx=batch["src_idx"],
                    mol_id=mol_id,
                    batch_segments=batch["batch_segments"],
                    batch_mask=batch["batch_mask"],
                    batch_size=per_device_batch_size,
                    cutoff=args.cutoff,
                    evidence=args.trust_map_evidence,
                    hyperprior=args.trust_map_hyperprior,
                )
                loss += neural_interaction_l2 * tm_loss
            else:
                loss += neural_interaction_l2 * neural_int_mse

        metrics = {
            "loss": loss,
            "energy_mae": energy_mae,
            "forces_mae": force_mae,
            "energy_mse": energy_mse,
            "forces_mse": force_mse,
            "neural_int_mse": neural_int_mse,
            "neural_int_rms": jnp.sqrt(neural_int_mse + 1e-12),
            "dipole_mae": dipole_mae,
            "charge_mae": charge_mae,
            "dipole_mse": dipole_mse,
            "charge_mse": charge_mse,
            "mbd_energy_abs_mean": jnp.mean(jnp.abs(mbd_energy)),
            "mbd_force_abs_mean": jnp.sum(jnp.abs(mbd_forces) * force_mask)
            / (jnp.sum(force_mask) * 3.0 + 1e-8),
            "mbd_scale": mbd_scale,
            "multipole_dipole_mae": multipole_dipole_mae,
            "multipole_dipole_mse": multipole_dipole_mse,
        }
        return loss, metrics

    def train_step(state, batch):
        if args.mbd_ramp_steps > 0:
            mbd_scale = jnp.minimum(
                1.0, state.step.astype(jnp.float32) / float(args.mbd_ramp_steps)
            )
        else:
            mbd_scale = jnp.asarray(1.0, dtype=jnp.float32)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, mbd_scale
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        state = state.apply_gradients(grads=grads)
        metrics = jax.lax.pmean(metrics, axis_name="device")
        metrics["loss"] = jax.lax.pmean(loss, axis_name="device")
        return state, metrics

    def eval_step(state, batch):
        if args.mbd_ramp_steps > 0:
            mbd_scale = jnp.minimum(
                1.0, state.step.astype(jnp.float32) / float(args.mbd_ramp_steps)
            )
        else:
            mbd_scale = jnp.asarray(1.0, dtype=jnp.float32)
        _, metrics = loss_fn(state.params, batch, mbd_scale)
        return jax.lax.pmean(metrics, axis_name="device")

    return (
        jax.pmap(train_step, axis_name="device", devices=devices),
        jax.pmap(eval_step, axis_name="device", devices=devices),
    )


def mean_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Average a list of metric dicts with a single host sync at the end."""
    total: dict[str, Any] | None = None
    for m in metrics:
        total = accumulate_metrics(total, m)
    return finalize_metrics(total, len(metrics))


def init_state(
    model: SpookyPhysNet,
    data: dict[str, np.ndarray],
    train_buckets: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> train_state.TrainState:
    rng = np.random.default_rng(args.seed)
    init_indices = None
    init_pad = None
    for pad_atoms in sorted(train_buckets, key=lambda n: len(train_buckets[n]), reverse=True):
        if len(train_buckets[pad_atoms]) >= args.batch_size_per_device:
            init_indices = train_buckets[pad_atoms][: args.batch_size_per_device]
            init_pad = pad_atoms
            break
    if init_indices is None:
        raise ValueError(
            "No atom-count bucket has enough structures for one per-device batch"
        )
    rng.shuffle(init_indices)
    batch = build_spooky_batch_from_flat_data(
        data, init_indices, pad_atoms=init_pad
    )
    if (
        not args.no_cgenff_vdw
        and "cgenff_master_sigmas" in data
        and "cgenff_master_epsilons" in data
    ):
        batch["cgenff_master_sigmas"] = jnp.asarray(data["cgenff_master_sigmas"], dtype=jnp.float32)
        batch["cgenff_master_epsilons"] = jnp.asarray(data["cgenff_master_epsilons"], dtype=jnp.float32)
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
        mol_id=batch.get("mol_id"),
        cgenff_type_idx=None if args.no_cgenff_vdw else batch.get("cgenff_type_idx"),
        cgenff_master_sigmas=None if args.no_cgenff_vdw else batch.get("cgenff_master_sigmas"),
        cgenff_master_epsilons=None if args.no_cgenff_vdw else batch.get("cgenff_master_epsilons"),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.clip_global_norm),
        build_optimizer(args),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,
        tx=tx,
    )


def build_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    learning_rate: float | Any = args.learning_rate
    if args.lr_schedule == "warmup_cosine":
        if args.lr_decay_steps <= args.lr_warmup_steps:
            raise ValueError("--lr-decay-steps must be greater than --lr-warmup-steps")
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.learning_rate,
            warmup_steps=args.lr_warmup_steps,
            decay_steps=args.lr_decay_steps,
            end_value=args.learning_rate * args.lr_end_fraction,
        )
    if args.optimizer == "adamw":
        return optax.adamw(learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "muon":
        from optax.contrib import muon

        return muon(
            learning_rate=learning_rate,
            beta=args.muon_beta,
            weight_decay=args.weight_decay,
            nesterov=True,
            adam_b1=args.muon_adam_b1,
            adam_b2=args.muon_adam_b2,
            adam_weight_decay=args.weight_decay,
            adam_learning_rate=(
                learning_rate if args.muon_adam_lr is None
                else args.muon_adam_lr
            ),
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


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


TRUST_MAP_ELEMENTS = (1, 6, 7, 8, 16, 17)  # H, C, N, O, S, Cl; must match the model


def _interaction_trust_map_loss(
    neural_interaction: jnp.ndarray,   # (B, 1) or (B,) per-structure E_neural(AB)-E_neural(A)-E_neural(B)
    log_lambda: jnp.ndarray,           # (n_el, n_el) learned raw parameter
    *,
    Z: jnp.ndarray,
    R: jnp.ndarray,
    dst_idx: jnp.ndarray,
    src_idx: jnp.ndarray,
    mol_id: jnp.ndarray,
    batch_segments: jnp.ndarray,
    batch_mask: jnp.ndarray,           # (n_edges,) valid-edge mask
    batch_size: int,
    cutoff: float,
    evidence: float,
    hyperprior: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evidence-balanced, per-element-pair shrinkage of the neural interaction energy.

    Each dimer's interaction energy is attributed to element-pair buckets by its
    inter-monomer contacts (weighted by a linear cutoff), giving a per-structure
    effective shrinkage ``Lambda_s = <lambda_{Zi,Zj}>`` over the interface. The loss is
    the negative log-likelihood of a zero-mean Gaussian residual with precision Lambda_s:

        0.5 * Lambda_s * r_s^2  -  0.5 * evidence * log(Lambda_s)

    The data term wants Lambda small (let the residual live); the evidence term wants it
    large (shrink to the prior). The stationary point is Lambda_c ~ evidence / <r^2>_c, so
    the learned lambda is small exactly where the data justifies a large neural
    correction and large where it does not -- that matrix is the trust-map fingerprint. A
    shared hyperprior ties the buckets (var penalty toward their common mean), so
    data-poor buckets borrow strength instead of drifting.

    Returns (loss_term, lambda_matrix) with lambda symmetric and positive.
    """
    elements = jnp.asarray(TRUST_MAP_ELEMENTS)
    lam = jax.nn.softplus(log_lambda)
    lam = 0.5 * (lam + lam.T)  # symmetric

    # Z -> slot index in `elements`, or -1 if not one of the tracked elements. Table size
    # is static (max tracked element + 1) so this traces under jit; any Z beyond it maps
    # to -1 via the clip+compare below.
    table_size = max(TRUST_MAP_ELEMENTS) + 1
    slot_of = -jnp.ones((table_size,), dtype=jnp.int32)
    slot_of = slot_of.at[elements].set(jnp.arange(elements.shape[0], dtype=jnp.int32))
    z_dst = jnp.clip(jnp.take(Z, dst_idx), 0, table_size - 1)
    z_src = jnp.clip(jnp.take(Z, src_idx), 0, table_size - 1)

    r = neural_interaction.reshape(-1)  # (B,)
    si = slot_of[z_dst]
    sj = slot_of[z_src]
    valid = ((si >= 0) & (sj >= 0)).astype(lam.dtype)
    inter = (jnp.take(mol_id, dst_idx) != jnp.take(mol_id, src_idx)).astype(lam.dtype)

    dr = jnp.take(R, dst_idx, axis=0) - jnp.take(R, src_idx, axis=0)
    dist = jnp.linalg.norm(dr, axis=-1)
    contact_w = jnp.clip(1.0 - dist / cutoff, 0.0, 1.0)  # linear cutoff weight

    w = batch_mask * inter * valid * contact_w
    lam_e = lam[jnp.clip(si, 0), jnp.clip(sj, 0)]  # gather; invalid entries zeroed by w

    edge_struct = jnp.take(batch_segments, dst_idx)
    num = jax.ops.segment_sum(lam_e * w, edge_struct, num_segments=batch_size)
    den = jax.ops.segment_sum(w, edge_struct, num_segments=batch_size)
    has_contact = (den > 0).astype(lam.dtype)
    Lambda_s = num / (den + 1e-8)

    nll = 0.5 * Lambda_s * (r ** 2) - 0.5 * evidence * jnp.log(Lambda_s + 1e-8)
    data_term = jnp.sum(nll * has_contact) / (jnp.sum(has_contact) + 1e-8)

    # Shared hyperprior: pull buckets toward their common mean so sparse pairs (which get
    # little gradient from the data term) don't drift.
    hyper = hyperprior * jnp.mean((log_lambda - jnp.mean(log_lambda)) ** 2)
    return data_term + hyper, lam


def _per_structure(value: Any, batch_segments: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Reduce a model output component to a per-structure (batch_size, 1) energy.

    The model surfaces its energy components in mixed layouts: ``energy`` and the
    electrostatics/vdW terms already come back per structure, while the ZBL repulsion is
    per atom. Disabled terms come back as ``None`` or a scalar ``0.0``.
    """
    if value is None:
        return jnp.zeros((batch_size, 1))
    value = jnp.asarray(value)
    if value.ndim == 0:
        return jnp.zeros((batch_size, 1))
    flat = value.reshape(value.shape[0], -1).sum(axis=-1) if value.ndim > 1 else value
    if flat.shape[0] == batch_segments.shape[0]:  # per-atom -> sum onto structures
        return jax.ops.segment_sum(
            flat, segment_ids=batch_segments, num_segments=batch_size
        ).reshape(batch_size, 1)
    return flat.reshape(batch_size, -1).sum(axis=1, keepdims=True)


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
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".json":
        restored = json_to_params(checkpoint_path)
    else:
        restored = ocp.PyTreeCheckpointer().restore(checkpoint_path)
    loaded_params = restored.get("params")
    if loaded_params is None and isinstance(restored.get("model"), Mapping):
        loaded_params = restored["model"].get("params")
    if loaded_params is None:
        raise ValueError(f"Checkpoint {checkpoint_path} has no parameters")

    # Some checkpoints (e.g. orbax_to_json exports) store the flax "params"
    # collection unwrapped, i.e. {"Dense_0": ..., ...} rather than the
    # {"params": {"Dense_0": ...}} shape flax's TrainState.params actually has.
    # Detect and re-wrap so the tree shapes line up before merging, otherwise
    # every leaf silently fails to match at the top level.
    if (
        isinstance(state.params, Mapping)
        and set(state.params.keys()) == {"params"}
        and isinstance(loaded_params, Mapping)
        and "params" not in loaded_params
    ):
        loaded_params = {"params": loaded_params}

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


def save_step_checkpoint(
    workdir: Path,
    epoch: int,
    global_step: int,
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
        "global_step": global_step,
        "metrics": metrics,
    }
    save_training_checkpoint(workdir / f"step-{global_step:08d}", ckpt)


def train(args: argparse.Namespace, cache_path: Path) -> None:
    data = restore_cached_data(cache_path)
    has_mol_id = "mol_id" in data
    has_cgenff_data = (
        "cgenff_master_sigmas" in data
        and "cgenff_master_epsilons" in data
        and "cgenff_type_idx" in data
        and has_mol_id
    )
    has_cgenff_lj = has_cgenff_data and not args.no_cgenff_vdw
    if has_cgenff_data:
        n_types = int(np.asarray(data["cgenff_master_sigmas"]).shape[0])
        if has_cgenff_lj:
            print(
                f"CGenFF LJ data found in cache: {n_types} atom types "
                f"(cgenff_master_sigmas/epsilons, cgenff_type_idx, mol_id) — "
                f"ML/MM residual VdW term is active",
                flush=True,
            )
        else:
            print(
                f"CGenFF LJ data found in cache ({n_types} atom types) but "
                f"--no-cgenff-vdw was passed — empirical LJ term is disabled",
                flush=True,
            )
    else:
        print("No CGenFF LJ data in cache — training a plain ML potential", flush=True)
    n_molecules = int(np.asarray(data["E"]).shape[0])
    max_atoms = int(np.max(np.asarray(data["N"]).reshape(-1)))
    train_idx, valid_idx = split_indices(n_molecules, args.valid_fraction, args.seed)
    train_buckets = bucket_indices_by_natoms(
        data, train_idx, bucket_width=args.atom_bucket_width
    )
    valid_buckets = bucket_indices_by_natoms(
        data, valid_idx, bucket_width=args.atom_bucket_width
    )
    devices = jax.local_devices()[: args.num_devices]
    if len(devices) != args.num_devices:
        raise RuntimeError(
            f"Requested {args.num_devices} devices, but JAX sees {len(jax.local_devices())}: "
            f"{jax.local_devices()}"
        )

    # Resolve restart path and load model architecture config if restarting/resuming
    # or warm-starting from a checkpoint (--init-checkpoint), so architecture-sensitive
    # settings like electrostatics_damping_sigma stay consistent with the source run
    # (and thus with any simulation using the same checkpoint) even when swapping the
    # optimizer for a warm restart.
    restart_path = resolve_restart_path(args)
    source_checkpoint = restart_path
    if source_checkpoint is None and args.init_checkpoint is not None:
        source_checkpoint = Path(args.init_checkpoint).expanduser().resolve()
    if source_checkpoint is not None:
        config_path = source_checkpoint.parent / "run_config.json"
        if config_path.exists():
            print(f"Loading model configuration from {config_path}")
            with config_path.open("r") as fh:
                saved_config = json.load(fh)
            arch_params = [
                "features",
                "max_degree",
                "num_iterations",
                "num_basis_functions",
                "cutoff",
                "max_atomic_number",
                "n_res",
                "predict_charges",
                "no_zbl",
                "trainable_zbl",
                "zbl_cuton",
                "zbl_cutoff",
                "efa",
                "use_energy_bias",
                "electrostatics_damping_sigma",
            ]
            parser = build_parser()
            for param in arch_params:
                if param not in saved_config:
                    continue
                current_val = getattr(args, param)
                if current_val != parser.get_default(param):
                    # User explicitly passed a non-default value on the CLI; an
                    # explicit flag (e.g. turning --predict-charges on to add a
                    # new head) always wins over the checkpoint's saved config.
                    continue
                saved_val = saved_config[param]
                if current_val != saved_val:
                    print(f"  Overriding {param}: {current_val} -> {saved_val} (from checkpoint config)")
                    setattr(args, param, saved_val)
            if (
                "trainable_zbl" not in saved_config
                and not args.no_zbl
                and not args.force_fixed_zbl
            ):
                # Every checkpoint produced before trainable_zbl was recorded
                # used trainable ZBL parameters. Preserve that architecture for
                # legacy restart/warm-starts; new runs remain fixed by default.
                print("  Legacy checkpoint: inferring trainable_zbl=True")
                args.trainable_zbl = True
            # A true restart must restore the same composite definition and
            # optimizer schedule.  Defaults mean "inherit" here; any explicit
            # non-default CLI value remains an intentional override.
            if restart_path is not None:
                resume_params = (
                    "optimizer",
                    "mbd_checkpoint",
                    "mbd_weight",
                    "mbd_ramp_steps",
                    "lr_schedule",
                    "lr_warmup_steps",
                    "lr_decay_steps",
                    "lr_end_fraction",
                    "learning_rate",
                    "muon_adam_lr",
                    "muon_beta",
                    "muon_adam_b1",
                    "muon_adam_b2",
                    "weight_decay",
                    "clip_global_norm",
                )
                for param in resume_params:
                    if param not in saved_config:
                        continue
                    current_val = getattr(args, param)
                    if current_val != parser.get_default(param):
                        continue
                    saved_val = saved_config[param]
                    if current_val != saved_val:
                        print(
                            f"  Restoring {param}: {current_val} -> {saved_val} "
                            "(from restart config)"
                        )
                        setattr(args, param, saved_val)

    if restart_path is not None:
        restart_payload = ocp.PyTreeCheckpointer().restore(restart_path)
        saved_optimizer = restart_payload.get("config", {}).get("optimizer")
        if saved_optimizer is not None and saved_optimizer != args.optimizer:
            raise ValueError(
                f"Checkpoint uses optimizer {saved_optimizer!r}, but --optimizer is "
                f"{args.optimizer!r}. Use the same optimizer with --restart, or use "
                f"--init-checkpoint {restart_path} for a fresh optimizer state."
            )

    # Provisional step count uses the pair heuristic (or the user cap under
    # auto-batch). Recomputed after probing once step fns exist.
    provisional_batch_sizes = resolve_batch_sizes(
        train_buckets,
        per_device_batch_size=args.batch_size_per_device,
        max_pairs_per_device=args.max_pairs_per_device,
        auto_batch=False,
    )
    batches_per_epoch = sum(
        (len(indices) // (provisional_batch_sizes[pad] * args.num_devices))
        for pad, indices in train_buckets.items()
        if provisional_batch_sizes[pad] >= 1
    )
    if args.steps_per_epoch:
        batches_per_epoch = min(batches_per_epoch, args.steps_per_epoch)
    total_planned_steps = args.epochs * max(1, batches_per_epoch)
    if args.lr_schedule == "warmup_cosine" and args.lr_decay_steps == 0:
        args.lr_decay_steps = max(args.lr_warmup_steps + 1, total_planned_steps)
    if args.lr_schedule == "warmup_cosine":
        print(
            f"LR schedule: {args.lr_warmup_steps} warmup steps, cosine decay over "
            f"{args.lr_decay_steps} optimizer steps",
            flush=True,
        )
    print(
        f"Atom buckets: width={args.atom_bucket_width} "
        f"({len(train_buckets)} train pad widths, max pad={max(train_buckets) if train_buckets else 0})",
        flush=True,
    )
    print(
        f"Planned run (provisional): {args.epochs} epochs x {batches_per_epoch} steps/epoch "
        f"= {total_planned_steps} total training steps",
        flush=True,
    )

    max_pad_atoms = pad_atoms_for_count(max_atoms, args.atom_bucket_width)
    model = create_model(args, max_atoms=max_pad_atoms)
    mbd_model = None
    mbd_params = None
    if args.mbd_checkpoint is not None:
        mbd_model, mbd_params = load_mbd_model(Path(args.mbd_checkpoint).expanduser())
        print(
            f"Using frozen MBD correction from {Path(args.mbd_checkpoint).expanduser()} "
            f"with weight {args.mbd_weight:g}: whole-system dispersion added directly to "
            f"the total-energy target (E in this cache scales with atom count, so it's a "
            f"total complex energy, not an interaction energy — no counterpoise decomposition)",
            flush=True,
        )
    multipole_model = None
    multipole_params = None
    if args.multipole_checkpoint is not None:
        from mmml.models.multipoles.electrostatics import load_multipole_model

        multipole_model, multipole_params = load_multipole_model(
            Path(args.multipole_checkpoint).expanduser()
        )
        print(
            f"Using frozen multipole model from {Path(args.multipole_checkpoint).expanduser()} "
            f"with consistency weight {args.multipole_consistency_weight:g}: its predicted "
            f"molecular dipole is added as an auxiliary MSE target alongside (not replacing) "
            f"the DFT-reference dipole loss on the network's own learned-charge dipole",
            flush=True,
        )
    state = init_state(model, data, train_buckets, args)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "run_config.json").open("w") as fh:
        json.dump({**vars(args), "cache_path": str(cache_path)}, fh, indent=2, sort_keys=True)

    print(f"JAX devices: {devices}")
    print(f"Train structures: {len(train_idx):,}; valid structures: {len(valid_idx):,}")
    print(
        f"Max atoms: {max_atoms}; per-device batch cap: {args.batch_size_per_device}; "
        f"atom_bucket_width: {args.atom_bucket_width}"
    )
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
            step_functions[batch_size] = make_steps(
                model,
                step_args,
                devices,
                mbd_model=mbd_model,
                mbd_params=mbd_params,
                multipole_model=multipole_model,
                multipole_params=multipole_params,
            )
        return step_functions[batch_size]

    print(
        f"Resolving per-bucket batch sizes "
        f"(auto_batch={args.auto_batch}, cap={args.batch_size_per_device}, "
        f"prefetch={args.prefetch_batches})...",
        flush=True,
    )
    train_batch_sizes = resolve_batch_sizes(
        train_buckets,
        per_device_batch_size=args.batch_size_per_device,
        max_pairs_per_device=args.max_pairs_per_device,
        auto_batch=args.auto_batch,
        data=data,
        n_devices=args.num_devices,
        steps_for_batch_size=steps_for_batch_size,
        state=state,
        step_functions=step_functions,
    )
    valid_batch_sizes = {
        pad: train_batch_sizes.get(
            pad,
            pairs_budget_batch_size(
                pad,
                per_device_batch_size=args.batch_size_per_device,
                max_pairs_per_device=args.max_pairs_per_device,
            ),
        )
        for pad in valid_buckets
    }
    # Probe may have mutated optimizer state; keep params but that is fine for
    # warm-start. Recompute steps/epoch from the resolved sizes.
    batches_per_epoch = sum(
        (len(indices) // (train_batch_sizes[pad] * args.num_devices))
        for pad, indices in train_buckets.items()
        if train_batch_sizes.get(pad, 0) >= 1
    )
    if args.steps_per_epoch:
        batches_per_epoch = min(batches_per_epoch, args.steps_per_epoch)
    batches_per_epoch = max(1, batches_per_epoch)
    total_planned_steps = args.epochs * batches_per_epoch
    print(
        f"Resolved run: {args.epochs} epochs x {batches_per_epoch} steps/epoch "
        f"= {total_planned_steps} total training steps",
        flush=True,
    )
    print(
        "Batch sizes by pad_atoms: "
        + ", ".join(
            f"{pad}:{train_batch_sizes[pad]}"
            for pad in sorted(train_batch_sizes)
        ),
        flush=True,
    )

    compiled_shapes: set[tuple[int, int]] = set()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_total: dict[str, Any] | None = None
        train_count = 0
        log_total: dict[str, Any] | None = None
        log_count = 0
        train_batches = prefetch_stacked_batches(
            iter_device_batches(
                train_buckets,
                batch_sizes=train_batch_sizes,
                n_devices=args.num_devices,
                rng=rng,
            ),
            data,
            depth=args.prefetch_batches,
        )
        for step, (pad_atoms, batch_size, batch) in enumerate(train_batches, start=1):
            shape = (pad_atoms, batch_size)
            if shape not in compiled_shapes:
                print(
                    f"Compiling steps for pad_atoms={pad_atoms} "
                    f"with per-device batch {batch_size}",
                    flush=True,
                )
                compiled_shapes.add(shape)
            train_step, _ = steps_for_batch_size(batch_size)
            state, metrics = train_step(state, batch)
            train_total = accumulate_metrics(train_total, metrics)
            train_count += 1
            log_total = accumulate_metrics(log_total, metrics)
            log_count += 1
            global_step = (epoch - 1) * batches_per_epoch + step
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                unreplicated = jax_utils.unreplicate(state)
                save_step_checkpoint(
                    workdir,
                    epoch,
                    global_step,
                    unreplicated,
                    model,
                    args,
                    {"train": finalize_metrics(log_total, log_count)},
                )
                print(f"Saved mid-epoch checkpoint at global step {global_step}", flush=True)
            if step % args.log_every_steps == 0:
                m = finalize_metrics(log_total, log_count)
                log_total = None
                log_count = 0
                pct_done = 100.0 * global_step / max(1, total_planned_steps)
                line = (
                    f"epoch {epoch:04d} step {step:06d} "
                    f"[{pct_done:5.1f}% of {total_planned_steps}] "
                    f"loss={m['loss']:.6g} E_MAE={m['energy_mae']:.6g} "
                    f"F_MAE={m['forces_mae']:.6g} "
                    f"D_MAE={m['dipole_mae']:.6g} Q_MAE={m['charge_mae']:.6g}"
                )
                if mbd_model is not None:
                    line += f" MBD_λ={m['mbd_scale']:.3f}"
                if multipole_model is not None:
                    line += f" MultipoleD_MAE={m['multipole_dipole_mae']:.6g}"
                print(line)
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break

        valid_total: dict[str, Any] | None = None
        valid_count = 0
        valid_batches = prefetch_stacked_batches(
            iter_device_batches(
                valid_buckets,
                batch_sizes=valid_batch_sizes,
                n_devices=args.num_devices,
                rng=np.random.default_rng(args.seed + epoch),
            ),
            data,
            depth=args.prefetch_batches,
        )
        for step, (pad_atoms, batch_size, batch) in enumerate(valid_batches, start=1):
            _, eval_step = steps_for_batch_size(batch_size)
            valid_total = accumulate_metrics(valid_total, eval_step(state, batch))
            valid_count += 1
            if args.valid_steps and step >= args.valid_steps:
                break

        train_mean = finalize_metrics(train_total, train_count)
        valid_mean = finalize_metrics(valid_total, valid_count)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:04d} done in {elapsed:.1f}s "
            f"train_loss={train_mean.get('loss', float('nan')):.6g} "
            f"valid_loss={valid_mean.get('loss', float('nan')):.6g} "
            f"valid_E_MAE={valid_mean.get('energy_mae', float('nan')):.6g} "
            f"valid_F_MAE={valid_mean.get('forces_mae', float('nan')):.6g} "
            f"valid_D_MAE={valid_mean.get('dipole_mae', float('nan')):.6g} "
            f"valid_Q_MAE={valid_mean.get('charge_mae', float('nan')):.6g} "
            f"MBD_|E|={valid_mean.get('mbd_energy_abs_mean', 0.0):.6g} "
            f"MBD_|F|={valid_mean.get('mbd_force_abs_mean', 0.0):.6g}"
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
    parser.add_argument(
        "--extxyz",
        default=None,
        help=(
            "Input extxyz file, e.g. so3lr_train.extxyz. Required for --mode cache/"
            "cache-and-train (used to derive the cache path). Not needed for --mode "
            "train if --cache-path is given directly."
        ),
    )
    parser.add_argument("--cache-dir", default=None, help="Directory for Orbax data caches")
    parser.add_argument(
        "--cache-path",
        default=None,
        help=(
            "Exact path to a pre-built Orbax data cache to train from (e.g. one produced "
            "by split_and_inspect_ml_mm_dataset.py's train_cache/). Bypasses the --extxyz/"
            "--cache-dir hash-based lookup. Only valid with --mode train."
        ),
    )
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
    parser.add_argument(
        "--batch-size-per-device",
        type=int,
        default=64,
        help=(
            "Maximum structures per device per step. With --auto-batch (default), "
            "each atom-pad bucket is probed up to this cap."
        ),
    )
    parser.add_argument(
        "--max-pairs-per-device",
        type=int,
        default=18000,
        help=(
            "Fallback per-device batch*n_atoms^2 budget used when --no-auto-batch "
            "is set (also used for provisional LR-step planning)."
        ),
    )
    parser.add_argument(
        "--atom-bucket-width",
        type=int,
        default=4,
        help=(
            "Pad molecules up to multiples of this atom count so nearby sizes "
            "share one compiled shape (1 disables widening)."
        ),
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="Host-side stacked batches to prepare ahead of the training step.",
    )
    parser.add_argument(
        "--auto-batch",
        dest="auto_batch",
        action="store_true",
        default=True,
        help="Probe max per-device batch per pad width (default: on).",
    )
    parser.add_argument(
        "--no-auto-batch",
        dest="auto_batch",
        action="store_false",
        help="Disable memory probing; use --max-pairs-per-device heuristic.",
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
    parser.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs.")
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help=(
            "Additionally save a mid-epoch checkpoint every N optimizer steps "
            "(global step count, across epochs). 0 disables this (default)."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "muon"),
        default="adamw",
        help=(
            "Optimizer for 2D+ weight matrices. 'muon' orthogonalizes matrix "
            "updates via Newton-Schulz and routes non-matrix params (biases, "
            "embeddings) through AdamW internally. Muon's opt_state is not "
            "compatible with an AdamW checkpoint's opt_state; use "
            "--init-checkpoint (not --restart) to warm-start params only when "
            "switching optimizers."
        ),
    )
    parser.add_argument("--muon-beta", type=float, default=0.95, help="Muon momentum decay")
    parser.add_argument(
        "--muon-adam-lr",
        type=float,
        default=None,
        help="Learning rate for Muon's internal AdamW branch (non-matrix params); defaults to --learning-rate",
    )
    parser.add_argument("--muon-adam-b1", type=float, default=0.9)
    parser.add_argument("--muon-adam-b2", type=float, default=0.999)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine"),
        default="warmup_cosine",
        help="Optimizer LR schedule (default: linear warmup followed by cosine decay).",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=1_000,
        help="Linear warmup steps for --lr-schedule warmup_cosine (default: 1000).",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=0,
        help="Cosine-decay horizon; 0 derives it from epochs and train batches.",
    )
    parser.add_argument(
        "--lr-end-fraction",
        type=float,
        default=0.05,
        help="Final LR divided by peak LR for cosine decay (default: 0.05).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-global-norm", type=float, default=10.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=52.91)
    parser.add_argument("--dipole-weight", type=float, default=1.0)
    parser.add_argument("--charges-weight", type=float, default=1.0)
    parser.add_argument(
        "--mbd-checkpoint",
        type=str,
        default=None,
        help=(
            "Frozen QCML MBD checkpoint. Its energy/forces are added to Spooky's "
            "prediction, so Spooky trains the remaining residual."
        ),
    )
    parser.add_argument(
        "--no-cgenff-vdw",
        action="store_true",
        help=(
            "Disable the empirical CGenFF Lennard-Jones term even if cgenff_master_"
            "sigmas/epsilons/cgenff_type_idx are present in the cache. Useful when "
            "training with --mbd-checkpoint instead of empirical LJ dispersion, to "
            "avoid double-counting dispersion physics between the two. mol_id-based "
            "electrostatics masking is unaffected."
        ),
    )
    parser.add_argument(
        "--neural-interaction-l2",
        type=float,
        default=0.0,
        help=(
            "Ridge shrinkage of the neural INTERACTION energy toward zero (lambda). "
            "Penalises lambda * mean[(E_neural(AB) - E_neural(A) - E_neural(B))^2], with "
            "E_neural(A)+E_neural(B) obtained exactly by masking inter-monomer edges out "
            "of the pair list (one extra forward-only pass, ~2x step cost). The target "
            "stays the total energy; this only regularises how loudly the neural term may "
            "speak on top of the MM prior. Without it the neural interaction is 4.1x "
            "LARGER on pairs with <15 training structures than on pairs with >=300, i.e. "
            "loudest where there is no evidence for it. Pairs with --fixed-cgenff-vdw."
        ),
    )
    parser.add_argument(
        "--interaction-trust-map",
        action="store_true",
        help=(
            "Replace the scalar --neural-interaction-l2 with a LEARNED per-element-pair "
            "shrinkage (a 6x6 log-lambda matrix over H,C,N,O,S,Cl), fit by empirical Bayes "
            "so lambda_c ~ evidence/<neural_interaction^2>_c: small where the data justifies "
            "a large neural correction, large where it does not. --neural-interaction-l2 "
            "scales the whole term (use ~1.0). The learned matrix is a per-chemistry TRUST "
            "MAP / data-provenance fingerprint -- dump it with scripts/dump_trust_map.py. "
            "Pairs with --fixed-cgenff-vdw."
        ),
    )
    parser.add_argument(
        "--trust-map-evidence",
        type=float,
        default=1.0,
        help="Evidence weight (gamma) in the trust-map NLL; sets the lambda scale via "
             "lambda ~ gamma/<r^2>. Larger = stronger default shrinkage.",
    )
    parser.add_argument(
        "--trust-map-hyperprior",
        type=float,
        default=0.1,
        help="Shared-hyperprior strength tying the per-element-pair lambdas toward their "
             "common mean, so data-poor buckets borrow strength.",
    )
    parser.add_argument(
        "--fixed-cgenff-vdw",
        action="store_true",
        help=(
            "Pin the CGenFF Lennard-Jones term at its published parameters: disables the "
            "learned global/per-element epsilon scaling AND the network-predicted per-atom "
            "vdW scale. By default the model may rescale the LJ prior freely, and it does — "
            "a trained checkpoint reached global_vdw_scale=0.14 with element scales of 0.10 "
            "(C) and 0.24 (H), i.e. carbon-carbon epsilon at ~1.4% of its physical value, "
            "effectively erasing the force-field prior. With this flag the LJ term becomes "
            "a fixed physical baseline the network can only correct, never scale away."
        ),
    )
    parser.add_argument(
        "--mbd-weight",
        type=float,
        default=1.0,
        help="Scale for the frozen MBD energy and forces (default: 1.0).",
    )
    parser.add_argument(
        "--mbd-ramp-steps",
        type=int,
        default=10_000,
        help=(
            "Linearly introduce the frozen MBD correction over this many optimizer "
            "steps (default: 10000; 0 enables it immediately)."
        ),
    )
    parser.add_argument(
        "--multipole-checkpoint",
        type=str,
        default=None,
        help=(
            "Frozen QCML multipole model checkpoint (Orbax dir or portable JSON). Its "
            "predicted molecular dipole is used as an auxiliary consistency target for "
            "the network's own learned-charge dipole, on top of the DFT-reference dipole "
            "loss (see --multipole-consistency-weight)."
        ),
    )
    parser.add_argument(
        "--multipole-consistency-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the MSE between the network's learned-charge dipole and the "
            "frozen multipole model's predicted dipole (default: 1.0; only used when "
            "--multipole-checkpoint is set)."
        ),
    )
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--num-basis-functions", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-atomic-number", type=int, default=87)
    parser.add_argument("--n-res", type=int, default=2)
    parser.add_argument("--predict-charges", action="store_true", help="Also predict atomic charges/dipoles")
    parser.add_argument("--no-zbl", action="store_true")
    zbl_training = parser.add_mutually_exclusive_group()
    zbl_training.add_argument(
        "--trainable-zbl",
        action="store_true",
        help="Opt in to optimizing ZBL screening parameters (fixed universal ZBL is the default).",
    )
    parser.add_argument(
        "--zbl-cuton",
        type=float,
        default=0.1,
        help="Distance in Å below which fixed ZBL is fully on (default: 0.1).",
    )
    parser.add_argument(
        "--zbl-cutoff",
        type=float,
        default=0.6,
        help="Distance in Å where fixed ZBL reaches exactly zero (default: 0.6).",
    )
    zbl_training.add_argument(
        "--fixed-zbl",
        dest="force_fixed_zbl",
        action="store_true",
        help=(
            "Force universal fixed ZBL when warm-starting a legacy checkpoint; "
            "drops its legacy repulsion parameter leaves during compatible merge."
        ),
    )
    parser.add_argument("--efa", action="store_true")
    parser.add_argument("--use-energy-bias", action="store_true")
    parser.add_argument(
        "--electrostatics-damping-sigma",
        type=float,
        default=4.0,
        help="Apply erf(r/sigma) damping to the learned-charge Coulomb term; set 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10000, help="Structure interval while parsing extxyz")
    parser.add_argument("--log-every-steps", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be non-negative")
    if not 0.0 <= args.lr_end_fraction <= 1.0:
        raise ValueError("--lr-end-fraction must be between 0 and 1")
    if args.mbd_weight < 0.0:
        raise ValueError("--mbd-weight must be non-negative")
    if args.mbd_ramp_steps < 0:
        raise ValueError("--mbd-ramp-steps must be non-negative")
    if args.zbl_cutoff <= 0.0:
        raise ValueError("--zbl-cutoff must be positive")
    if args.zbl_cuton is not None and not 0.0 <= args.zbl_cuton < args.zbl_cutoff:
        raise ValueError("--zbl-cuton must satisfy 0 <= cuton < cutoff")
    if args.cache_path is not None:
        if args.mode != "train":
            raise ValueError("--cache-path is only valid with --mode train")
        cache_path = Path(args.cache_path).expanduser().resolve()
    else:
        if args.extxyz is None:
            raise ValueError("--extxyz is required unless --cache-path is given")
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required unless --cache-path is given")
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
