#!/usr/bin/env python3
"""
Train the spooky PhysNetJAX model on data from:

  (1) An Orbax dataset directory produced by ``sqlite_to_mmml_orbax_cache.py``, or
  (2) A SQLite database directly (will use/create the same orbax cache as that script).

This uses **padded** arrays ``(n_mol, natoms, …)`` and ``build_spooky_batch_from_padded_arrays``,
equivalent to ``train_spooky_h5.py --legacy-padded``.

The default ``train_spooky_h5.py`` path uses **flat** HDF5 data; do not mix that with this
pipeline without converting formats.

Examples
--------
  # From SQLite (builds/loads cache under .sqlite_cache by default)
  python examples/other/train_spooky_from_orbax.py --sqlite /path/to/data.db \\
    --train-size 10000 --valid-size 500

  # From an existing orbax dataset directory
  python examples/other/train_spooky_from_orbax.py \\
    --orbax-cache /path/to/.sqlite_cache/mydb_abc123def4567890

Prerequisites: jax, flax, optax, e3x, orbax, numpy; and apsw if using --sqlite.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Same directory as sqlite_to_mmml_orbax_cache.py
_EX_DIR = Path(__file__).resolve().parent
if str(_EX_DIR) not in sys.path:
    sys.path.insert(0, str(_EX_DIR))

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils, train_state

from mmml.models.physnetjax.physnetjax.data.data import get_choices, make_dicts
from mmml.models.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.models.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_padded_arrays,
    make_spooky_train_step,
    restart_params_only,
)

from sqlite_to_mmml_orbax_cache import (
    load_or_save_sqlite_orbax_cache,
    max_atoms_in_sqlite,
)


def load_padded_dict_from_orbax(cache_dir: Path) -> dict:
    """Restore MMML padded dataset dict (R, Z, F, E, N, Q, S, [D])."""
    data = ocp.PyTreeCheckpointer().restore(str(cache_dir))
    return {k: np.asarray(v) for k, v in data.items()}


def split_train_valid(
    key: jax.random.PRNGKey,
    data_dict: dict,
    train_size: int,
    valid_size: int,
) -> tuple[dict, dict, int]:
    n_samples = len(data_dict["R"])
    total = train_size + valid_size
    if total > n_samples:
        raise ValueError(
            f"Requested {train_size} + {valid_size} = {total} samples, "
            f"but dataset has only {n_samples}."
        )
    natoms = int(data_dict["R"].shape[1])
    keys = list(data_dict.keys())
    data = [data_dict[k] for k in keys]
    train_choice, valid_choice = get_choices(key, n_samples, train_size, valid_size)
    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)
    return train_data, valid_data, natoms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train spooky PhysNetJAX on orbax/SQLite MMML padded data."
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
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    key = jax.random.PRNGKey(42)

    if args.sqlite is not None:
        db = Path(args.sqlite).resolve()
        natoms = args.natoms
        if natoms is None:
            natoms = max_atoms_in_sqlite(db)
            if args.verbose:
                print(f"Auto natoms = {natoms}")
        cache_parent = Path(args.cache_dir).resolve() if args.cache_dir else None
        data_dict, cache_path = load_or_save_sqlite_orbax_cache(
            db,
            natoms=natoms,
            cache_dir=cache_parent,
            max_structures=args.max_structures,
            charge_filter=args.charge_filter,
            spin_mode=args.spin_mode,
            cache=True,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"Data from SQLite / cache: {cache_path}")
    else:
        cache_dir = Path(args.orbax_cache).resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {cache_dir}")
        data_dict = load_padded_dict_from_orbax(cache_dir)
        if args.verbose:
            print(f"Loaded orbax dataset: {cache_dir}")

    train_data, valid_data, natoms = split_train_valid(
        key, data_dict, args.train_size, args.valid_size
    )
    key, _ = jax.random.split(key)

    n_train = len(train_data["E"])
    n_valid = len(valid_data["E"])
    print(f"Train: {n_train}, Valid: {n_valid}, natoms (padding): {natoms}")

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

    def _make_batch(mol_idx: np.ndarray):
        return build_spooky_batch_from_padded_arrays(
            train_data["Z"][mol_idx],
            train_data["R"][mol_idx],
            train_data["E"][mol_idx],
            train_data["F"][mol_idx],
            train_data["Q"][mol_idx].flatten(),
            train_data["S"][mol_idx].flatten(),
        )

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
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            batch = _make_batch(idx)
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
