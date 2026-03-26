#!/usr/bin/env python3
"""
Example: Train the spooky PhysNetJAX model on qcell HDF5 data.

By default this script uses flat data and batching: concatenated atoms with
``mol_offsets``, and ``build_spooky_batch_from_flat_data`` (sparse pairs per real
molecule size). Pass ``--legacy-padded`` to use zero-padded arrays and
``build_spooky_batch_from_padded_arrays`` instead.

It loads qcell_*.h5 files (single or multiple) and trains the spooky EF model,
which uses system charge and spin multiplicity as inputs.

Prerequisites:
  - jax, flax, optax, e3x, h5py

Usage:
  # Load all qcell_*.h5 from a directory
  python examples/other/train_spooky_h5.py --data-dir /scicore/home/meuwly/boitti0000/qcell

  # Single file
  python examples/other/train_spooky_h5.py --filepath /path/to/qcell_dimers.h5

  # Multiple files explicitly
  python examples/other/train_spooky_h5.py \\
    --filepath /path/to/qcell_sugars.h5 /path/to/qcell_dimers.h5 \\
    --train-size 100000 --valid-size 1000
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
import optax
from flax.training import orbax_utils
import orbax.checkpoint as ocp

from mmml.models.physnetjax.physnetjax.data.read_h5 import (
    prepare_h5_datasets,
    prepare_h5_datasets_flat,
)
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
from flax.training import train_state

OUTPUT_DIR = Path("ckpts_spooky_h5").resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train spooky PhysNetJAX on qcell HDF5 data (single or multi-file)."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--filepath",
        type=str,
        nargs="+",
        help="Path(s) to HDF5 file(s). Pass multiple for multi-file loading.",
    )
    grp.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing qcell_*.h5 files. All matching files are loaded.",
    )
    p.add_argument("--train-size", type=int, default=270_000)
    p.add_argument("--valid-size", type=int, default=1000)
    p.add_argument("--natoms", type=int, default=None, help="Max atoms (auto-detect if omitted)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--charge-filter",
        type=float,
        default=None,
        help="If set, only include structures with this charge (e.g. 0.0 for neutral)",
    )
    p.add_argument("--energy-key", type=str, default="formation_energy")
    p.add_argument("--force-key", type=str, default="total_forces")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument(
        "--resume",
        "--restart",
        dest="resume",
        type=str,
        default=None,
        help=(
            "Orbax checkpoint directory to load params from (params-only; optimizer "
            'starts fresh). Same as --restart. Use "latest" for OUTPUT_DIR/final_params.'
        ),
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--legacy-padded",
        action="store_true",
        help=(
            "Use zero-padded (n_mol, natoms, …) HDF5 arrays and padded batching. "
            "Default is flat concatenated atoms with mol_offsets and per-molecule pair lists."
        ),
    )
    p.add_argument(
        "--flat-bucketing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flat HDF5 only (default: on): each batch contains only molecules with the "
            "same atom count, so JAX sees stable shapes (fewer XLA recompiles). "
            "Use --no-flat-bucketing for fully random mixing."
        ),
    )
    return p.parse_args()


def main(args: argparse.Namespace):
    print("=" * 60)
    print("Spooky PhysNetJAX training on qcell HDF5")
    print("=" * 60)

    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
        filepaths = sorted(data_dir.glob("qcell_*.h5"))
        if not filepaths:
            raise FileNotFoundError(
                f"No qcell_*.h5 files found in {data_dir}"
            )
        filepath_arg = [str(p) for p in filepaths]
        print(f"\nData dir: {data_dir}")
    else:
        filepaths = [Path(p) for p in args.filepath]
        if len(filepaths) == 1:
            filepath_arg = filepaths[0]
        else:
            filepath_arg = [str(p) for p in filepaths]

    print(f"Files: {filepath_arg}")

    key = jax.random.PRNGKey(42)

    # Load data (supports single or multi-file). Default: flat layout + flat batching.
    if args.legacy_padded:
        train_data, valid_data, natoms = prepare_h5_datasets(
            key,
            filepath=filepath_arg,
            train_size=args.train_size,
            valid_size=args.valid_size,
            natoms=args.natoms,
            energy_key=args.energy_key,
            force_key=args.force_key,
            charge_filter=args.charge_filter,
            verbose=args.verbose,
        )
    else:
        train_data, valid_data, natoms = prepare_h5_datasets_flat(
            key,
            filepath=filepath_arg,
            train_size=args.train_size,
            valid_size=args.valid_size,
            natoms=args.natoms,
            energy_key=args.energy_key,
            force_key=args.force_key,
            charge_filter=args.charge_filter,
            verbose=args.verbose,
        )

    n_train = len(train_data["E"])
    n_valid = len(valid_data["E"])
    layout = "legacy padded" if args.legacy_padded else "flat (default)"
    print(f"\nTrain: {n_train}, Valid: {n_valid}, natoms: {natoms}  [{layout}]")
    use_flat_bucketing = (not args.legacy_padded) and args.flat_bucketing
    bucket_map = None
    if use_flat_bucketing:
        bucket_map = bucket_flat_molecule_indices_by_natoms(train_data["N"])
        n_full = sum(len(bucket_map[k]) // args.batch_size for k in bucket_map)
        if n_full == 0:
            raise ValueError(
                f"No atom-count bucket has at least batch_size={args.batch_size} structures. "
                "Lower --batch-size or use --no-flat-bucketing."
            )
        n_remain = n_train - sum(
            (len(bucket_map[k]) // args.batch_size) * args.batch_size for k in bucket_map
        )
        print(
            f"Flat bucketing: {len(bucket_map)} atom-count buckets, "
            f"~{n_full} full batches/epoch (~{n_remain} samples/epoch not in a full batch)."
        )
    elif not args.legacy_padded:
        print(
            "Note: --no-flat-bucketing mixes molecule sizes → frequent XLA recompiles. "
            "Prefer default --flat-bucketing or --legacy-padded."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()

    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "latest":
            resume_path = output_dir / "final_params"
            if not resume_path.is_dir():
                raise ValueError(
                    f'Cannot resume from "latest": {resume_path} does not exist or is not a directory.'
                )
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = (Path.cwd() / resume_path).resolve()

    restored_params = None
    restored_config = None
    if resume_path is not None:
        print(f"\nRestoring checkpoint (params-only, fresh optimizer): {resume_path}")
        restored_params, restored_config, _step, _epoch, _chunk = restart_params_only(
            resume_path, checkpointer
        )
        if restored_config is None:
            raise ValueError(
                f"Checkpoint at {resume_path} has no 'config'; cannot rebuild SpookyEF."
            )

    def _to_native(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    # Build model (from checkpoint when resuming)
    if restored_config is not None:
        model_kwargs = {k: _to_native(v) for k, v in restored_config.items()}
        model = SpookyEF(**model_kwargs)
        if int(model.natoms) != int(natoms):
            raise ValueError(
                f"Checkpoint natoms={model.natoms} does not match dataset natoms={natoms}."
            )
        print(
            f"   Using config from checkpoint: max_atomic_number={model.max_atomic_number}, "
            f"features={model.features}, natoms={model.natoms}"
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
        if args.legacy_padded:
            return build_spooky_batch_from_padded_arrays(
                train_data["Z"][mol_idx],
                train_data["R"][mol_idx],
                train_data["E"][mol_idx],
                train_data["F"][mol_idx],
                train_data["Q"][mol_idx].flatten(),
                train_data["S"][mol_idx].flatten(),
            )
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
        epoch_loss = 0.0
        n_batches = 0
        if use_flat_bucketing and bucket_map is not None:
            for idx in iter_homogeneous_natoms_flat_batches(
                bucket_map, batch_size, rng, drop_partial=True
            ):
                batch = _make_batch(idx)
                state, loss_val, metrics = train_step(state, batch)
                epoch_loss += float(loss_val)
                n_batches += 1
            if n_batches == 0:
                raise RuntimeError("Flat bucketing produced no batches this epoch.")
        else:
            perm = rng.permutation(n_train)
            for b in range(steps_per_epoch):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                batch = _make_batch(idx)
                state, loss_val, metrics = train_step(state, batch)
                epoch_loss += float(loss_val)
                n_batches += 1

        mean_loss = epoch_loss / n_batches
        print(f"Epoch {epoch + 1}/{args.num_epochs}  loss={mean_loss:.6f}")

    # Save final checkpoint
    ckpt_dir = output_dir / "final_params"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    payload = {
        "params": state.params,
        "config": model.return_attributes(),
    }
    checkpointer.save(
        str(ckpt_dir),
        payload,
        save_args=orbax_utils.save_args_from_target(payload),
    )
    print(f"\nSaved checkpoint: {ckpt_dir}")
    print("Done.")


if __name__ == "__main__":
    main(parse_args())
