#!/usr/bin/env python3
"""
Train the spooky PhysNetJAX model on data from:

  (1) An Orbax dataset directory produced by ``sqlite_to_mmml_orbax_cache.py``, or
  (2) A SQLite database directly (will use/create the same orbax cache as that script).

**Default is flat storage** (concatenated ``R``/``Z``/``F`` + ``mol_offsets``), matching
``train_spooky_h5.py`` without ``--legacy-padded`` and ``prepare_h5_datasets_flat``.

**GPU performance (flat):** Random flat batches change atom/pair counts every step →
XLA recompiles. Mitigations: (1) ``--sqlite-layout padded``, or (2) stay flat and use
``--flat-bucketing`` (default): each batch only contains molecules with the **same**
atom count so shapes repeat every step (one compile per distinct atom count). This
script also uses ``jax.device_put`` and optional ``--prefetch``.

Caches without ``mol_offsets`` and with ``R`` of shape ``(n_mol, natoms, 3)`` are
treated as padded (older ``--layout padded`` builds).

Examples
--------
  # From SQLite (default: flat cache under .sqlite_cache)
  python examples/other/train_spooky_from_orbax.py --sqlite /path/to/data.db \\
    --train-size 10000 --valid-size 500

  # From an existing flat orbax dataset directory (*_flat_<hash>)
  python examples/other/train_spooky_from_orbax.py \\
    --orbax-cache /path/to/.sqlite_cache/mydb_flat_abc123def4567890

Prerequisites: jax, flax, optax, e3x, orbax, numpy; and apsw if using --sqlite.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import threading
from pathlib import Path

# Same directory as sqlite_to_mmml_orbax_cache.py
_EX_DIR = Path(__file__).resolve().parent
if str(_EX_DIR) not in sys.path:
    sys.path.insert(0, str(_EX_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils, train_state

from mmml.models.physnetjax.physnetjax.data.data import get_choices, make_dicts
from mmml.models.physnetjax.physnetjax.data.read_h5 import _subset_flat_dataset
from mmml.models.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.models.physnetjax.physnetjax.training.spooky_training import (
    bucket_flat_molecule_indices_by_natoms,
    build_spooky_batch_from_flat_data,
    build_spooky_batch_from_padded_arrays,
    iter_homogeneous_natoms_flat_batches,
    make_spooky_train_step,
    pick_flat_init_indices_homogeneous,
    restart_params_only,
)

from sqlite_to_mmml_orbax_cache import (
    load_or_save_sqlite_orbax_cache,
    max_atoms_in_sqlite,
)


def _device_put_batch(batch: dict) -> dict:
    """Copy array leaves to the default JAX device (async where supported)."""

    def _put(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return jax.device_put(x)
        return x

    return jax.tree.map(_put, batch)


def load_orbax_dataset(cache_dir: Path) -> dict:
    """Restore MMML dataset dict from orbax (flat or padded)."""
    data = ocp.PyTreeCheckpointer().restore(str(cache_dir))
    return {k: np.asarray(v) for k, v in data.items()}


def split_train_valid(
    key: jax.random.PRNGKey,
    data_dict: dict,
    train_size: int,
    valid_size: int,
    *,
    legacy_padded: bool,
) -> tuple[dict, dict, int]:
    n_samples = len(data_dict["E"])
    total = train_size + valid_size
    if total > n_samples:
        raise ValueError(
            f"Requested {train_size} + {valid_size} = {total} samples, "
            f"but dataset has only {n_samples}."
        )

    train_choice, valid_choice = get_choices(key, n_samples, train_size, valid_size)

    if legacy_padded:
        natoms = int(data_dict["R"].shape[1])
        keys = list(data_dict.keys())
        data = [data_dict[k] for k in keys]
        train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)
    else:
        natoms = int(np.max(data_dict["N"]))
        train_data = _subset_flat_dataset(data_dict, train_choice)
        valid_data = _subset_flat_dataset(data_dict, valid_choice)

    return train_data, valid_data, natoms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train spooky PhysNetJAX on orbax/SQLite MMML data (flat by default)."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--orbax-cache",
        type=str,
        help="Directory written by sqlite_to_mmml_orbax_cache (PyTree dataset).",
    )
    src.add_argument("--sqlite", type=str, help="QCML SQLite .db path (uses orbax cache).")

    p.add_argument("--train-size", type=int, default=10_000)
    p.add_argument("--valid-size", type=int, default=500)
    p.add_argument("--natoms", type=int, default=None, help="Only for --sqlite; default auto.")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="SQLite orbax parent dir (same as sqlite_to_mmml_orbax_cache --cache-dir).",
    )
    p.add_argument(
        "--sqlite-layout",
        choices=("flat", "padded"),
        default="flat",
        help="Cache layout when reading/writing SQLite orbax cache (default: flat).",
    )
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument("--charge-filter", type=float, default=None)
    p.add_argument(
        "--spin-mode",
        choices=("unpaired_plus_one", "as_is"),
        default="unpaired_plus_one",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="ckpts_spooky_orbax")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Orbax params dir or "latest" for OUTPUT_DIR/final_params.',
    )
    p.add_argument(
        "--prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefetch the next batch on a background thread while the GPU runs "
            "train_step (default: on). Helps CPU-side batch building; does not fix "
            "XLA recompilation from variable-size flat batches."
        ),
    )
    p.add_argument(
        "--flat-bucketing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flat data only (default: on): group each batch by identical atom count "
            "so JAX sees stable shapes (avoids recompilation per step). Remainders "
            "per bucket are skipped each epoch. Use --no-flat-bucketing for random "
            "mixing (slow JIT)."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    key = jax.random.PRNGKey(42)

    if args.sqlite is not None:
        db = Path(args.sqlite).resolve()
        natoms_cap = args.natoms
        if natoms_cap is None:
            natoms_cap = max_atoms_in_sqlite(db)
            if args.verbose:
                print(f"Auto natoms (max atoms in DB) = {natoms_cap}")
        cache_parent = Path(args.cache_dir).resolve() if args.cache_dir else None
        data_dict, cache_path = load_or_save_sqlite_orbax_cache(
            db,
            natoms=natoms_cap,
            cache_dir=cache_parent,
            max_structures=args.max_structures,
            charge_filter=args.charge_filter,
            spin_mode=args.spin_mode,
            layout=args.sqlite_layout,
            cache=True,
            verbose=args.verbose,
        )
        legacy_padded = args.sqlite_layout == "padded"
        if args.verbose:
            print(f"Data from SQLite / cache: {cache_path}")
    else:
        cache_dir = Path(args.orbax_cache).resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {cache_dir}")
        data_dict = load_orbax_dataset(cache_dir)
        if "mol_offsets" in data_dict:
            legacy_padded = False
        elif data_dict["R"].ndim == 3:
            legacy_padded = True
        else:
            raise ValueError(
                "Unrecognized orbax dataset: expected key 'mol_offsets' (flat) or "
                f"R.ndim==3 (padded); got R.shape={data_dict['R'].shape}."
            )

    train_data, valid_data, natoms = split_train_valid(
        key,
        data_dict,
        args.train_size,
        args.valid_size,
        legacy_padded=legacy_padded,
    )
    key, _ = jax.random.split(key)

    n_train = len(train_data["E"])
    n_valid = len(valid_data["E"])
    layout_name = "padded" if legacy_padded else "flat (mol_offsets)"
    use_flat_bucketing = (not legacy_padded) and args.flat_bucketing
    print(f"Train: {n_train}, Valid: {n_valid}, natoms (max per mol): {natoms}  [{layout_name}]")
    bucket_map: dict[int, np.ndarray] | None = None
    if use_flat_bucketing:
        bucket_map = bucket_flat_molecule_indices_by_natoms(train_data["N"])
        n_full_steps = sum(len(bucket_map[k]) // args.batch_size for k in bucket_map)
        if n_full_steps == 0:
            raise ValueError(
                f"No flat bucket has at least batch_size={args.batch_size} structures. "
                "Lower --batch-size, add data, or use --no-flat-bucketing."
            )
        n_remain = n_train - sum(
            (len(bucket_map[k]) // args.batch_size) * args.batch_size for k in bucket_map
        )
        print(
            f"Flat bucketing: {len(bucket_map)} atom-count buckets, "
            f"~{n_full_steps} full batches/epoch (~{n_remain} samples/epoch not in a full batch).",
            flush=True,
        )
    elif not legacy_padded:
        print(
            "Note: --no-flat-bucketing mixes molecule sizes each step → XLA may recompile "
            "often (low GPU). Prefer default --flat-bucketing or --sqlite-layout padded.",
            flush=True,
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()

    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "latest":
            resume_path = output_dir / "final_params"
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = (Path.cwd() / resume_path).resolve()

    restored_params = None
    restored_config = None
    if resume_path is not None:
        if not resume_path.is_dir():
            raise FileNotFoundError(resume_path)
        print(f"Restoring params from: {resume_path}")
        restored_params, restored_config, _, _, _ = restart_params_only(
            resume_path, checkpointer
        )
        if restored_config is None:
            raise ValueError("Checkpoint has no 'config'.")

    def _to_native(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    if restored_config is not None:
        model_kwargs = {k: _to_native(v) for k, v in restored_config.items()}
        model = SpookyEF(**model_kwargs)
        if int(model.natoms) != int(natoms):
            raise ValueError(
                f"Checkpoint natoms={model.natoms} != dataset natoms={natoms}."
            )
    else:
        model = SpookyEF(
            charges=True,
            natoms=natoms,
            max_atomic_number=87,
            features=64,
            debug=False,
        )

    batch_size = args.batch_size
    init_bs = min(batch_size, n_train)

    if legacy_padded:

        def _make_batch(mol_idx: np.ndarray):
            return build_spooky_batch_from_padded_arrays(
                train_data["Z"][mol_idx],
                train_data["R"][mol_idx],
                train_data["E"][mol_idx],
                train_data["F"][mol_idx],
                train_data["Q"][mol_idx].flatten(),
                train_data["S"][mol_idx].flatten(),
            )

    else:

        def _make_batch(mol_idx: np.ndarray):
            return build_spooky_batch_from_flat_data(train_data, mol_idx)

    if use_flat_bucketing and bucket_map is not None:
        init_batch = _make_batch(pick_flat_init_indices_homogeneous(bucket_map, init_bs))
    else:
        init_batch = _make_batch(np.arange(init_bs, dtype=np.int64))

    if restored_params is not None:
        params = restored_params
    else:
        key, init_key = jax.random.split(key)
        params = model.init(
            init_key,
            atomic_numbers=init_batch["Z"],
            charges=init_batch["Q_atoms"],
            spins=init_batch["S_atoms"],
            positions=init_batch["R"],
            dst_idx=init_batch["dst_idx"],
            src_idx=init_batch["src_idx"],
            batch_segments=init_batch["batch_segments"],
            batch_size=init_batch["batch_size"],
            batch_mask=init_batch["batch_mask"],
            atom_mask=init_batch["atom_mask"],
        )

    tx = optax.adam(args.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    train_step = make_spooky_train_step(
        model,
        forces_weight=52.91,
        energy_weight=1.0,
        batch_size=batch_size,
    )

    rng = np.random.default_rng(0)
    steps_per_epoch = max(n_train // batch_size, 1)

    for epoch in range(args.num_epochs):
        if use_flat_bucketing and bucket_map is not None:
            epoch_idx = list(
                iter_homogeneous_natoms_flat_batches(
                    bucket_map, batch_size, rng, drop_partial=True
                )
            )
            n_ep = len(epoch_idx)
            if n_ep == 0:
                raise RuntimeError("Flat bucketing produced no batches this epoch.")
        else:
            epoch_idx = None  # use perm + steps_per_epoch below

        if use_flat_bucketing and epoch_idx is not None:
            n_steps = n_ep
            if args.prefetch and n_steps > 1:
                q: queue.Queue = queue.Queue(maxsize=2)
                _sentinel = object()

                def _producer_b() -> None:
                    for idx in epoch_idx:
                        q.put(_device_put_batch(_make_batch(idx)))
                    q.put(_sentinel)

                t = threading.Thread(target=_producer_b, daemon=True)
                t.start()
                epoch_loss = 0.0
                n_batches = 0
                while True:
                    batch = q.get()
                    if batch is _sentinel:
                        break
                    state, loss_val, _metrics = train_step(state, batch)
                    epoch_loss += float(loss_val)
                    n_batches += 1
                t.join()
            else:
                epoch_loss = 0.0
                n_batches = 0
                for idx in epoch_idx:
                    batch = _device_put_batch(_make_batch(idx))
                    state, loss_val, _metrics = train_step(state, batch)
                    epoch_loss += float(loss_val)
                    n_batches += 1
        elif args.prefetch and steps_per_epoch > 1:
            perm = rng.permutation(n_train)
            q = queue.Queue(maxsize=2)
            _sentinel = object()

            def _producer() -> None:
                for b in range(steps_per_epoch):
                    idx = perm[b * batch_size : (b + 1) * batch_size]
                    q.put(_device_put_batch(_make_batch(idx)))
                q.put(_sentinel)

            t = threading.Thread(target=_producer, daemon=True)
            t.start()
            epoch_loss = 0.0
            n_batches = 0
            while True:
                batch = q.get()
                if batch is _sentinel:
                    break
                state, loss_val, _metrics = train_step(state, batch)
                epoch_loss += float(loss_val)
                n_batches += 1
            t.join()
        else:
            perm = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            for b in range(steps_per_epoch):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                batch = _device_put_batch(_make_batch(idx))
                state, loss_val, _metrics = train_step(state, batch)
                epoch_loss += float(loss_val)
                n_batches += 1
        print(f"Epoch {epoch + 1}/{args.num_epochs}  loss={epoch_loss / n_batches:.6f}")

    ckpt_dir = output_dir / "final_params"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    payload = {"params": state.params, "config": model.return_attributes()}
    checkpointer.save(
        str(ckpt_dir),
        payload,
        save_args=orbax_utils.save_args_from_target(payload),
    )
    print(f"Saved checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    main()
