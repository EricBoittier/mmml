#!/usr/bin/env python3
"""
Example: Train the spooky PhysNetJAX model on QCML TFDS data.

This script demonstrates loading the QCML dft_force_field dataset from
TensorFlow Datasets (TFDS) and training the spooky EF model, which uses
system charge and spin multiplicity as inputs.

Prerequisites:
  - tensorflow, tensorflow_datasets
  - gcloud CLI (for downloading from gs://qcml-datasets)
  - jax, flax, optax, e3x

Usage:
  python examples/other/train_spooky_qcml.py

To use a different energy/forces key (e.g. for other QCML configs):
  build_spooky_batch_from_example(example_np, energy_key="energy", forces_key="forces")
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import jax
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint as ocp

from mmml.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_padded_arrays,
    make_spooky_train_step,
)

# QCML dataset paths
LOCAL_DATA_DIR = "."
QCML_DATA_DIR = "gs://qcml-datasets/tfds"
GCP_PROJECT = "deepmind-opensource"
OUTPUT_DIR = Path("ckpts_spooky_qcml").resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train spooky PhysNetJAX on QCML dft_force_field with online chunked loading."
    )
    p.add_argument("--dataset", type=str, default="qcml/dft_force_field", help="TFDS dataset name.")
    p.add_argument("--split", type=str, default="full", help="TFDS split.")
    p.add_argument("--data-dir", type=str, default=LOCAL_DATA_DIR, help="Local TFDS data dir.")
    p.add_argument("--num-examples", type=int, default=10_000_000, help="Max examples to stream per epoch.")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    p.add_argument("--natoms", type=int, default=30, help="Model natoms and max atom cap.")
    p.add_argument("--num-steps", type=int, default=200000 * 20, help="Total optimization steps.")
    p.add_argument("--log-interval", type=int, default=1000, help="Steps between logs.")
    p.add_argument("--learning-rate", type=float, default=3.1e-4, help="Adam learning rate.")
    p.add_argument("--energy-key", type=str, default="pbe0_formation_energy", help="Energy field key in TFDS examples.")
    p.add_argument("--chunk-examples", type=int, default=200000, help="Examples per streamed chunk.")
    p.add_argument("--chunk-ckpt-interval", type=int, default=10, help="Save chunk checkpoint every N chunks.")
    p.add_argument("--resume", type=str, default=None, help='Checkpoint dir to resume from, or "latest".')
    p.add_argument("--max-atomic-number", type=int, default=87, help="Max atomic number (Z) for embedding. Must match checkpoint when resuming.")
    p.add_argument("--features", type=int, default=64, help="Model feature dimension. Must match checkpoint when resuming.")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Checkpoint output directory.")
    p.add_argument("--qcml-data-dir", type=str, default=QCML_DATA_DIR, help="Remote QCML gs:// root for download.")
    p.add_argument("--gcp-project", type=str, default=GCP_PROJECT, help="GCP project for gcloud storage cp.")
    return p.parse_args()


def download_qcml_dft_force_field(local_data_dir: str, qcml_data_dir: str, gcp_project: str):
    """Download QCML dft_force_field dataset locally."""
    os.system("gcloud config set auth/disable_credentials True")
    os.system(f"mkdir -p {local_data_dir}/qcml/dft_force_field/")
    os.system(
        f"gcloud storage cp -r {qcml_data_dir}/qcml/dft_force_field/1.0.0 "
        f"{local_data_dir}/qcml/dft_force_field/ --project={gcp_project}"
    )


def main(args: argparse.Namespace):
    print("=" * 60)
    print("Spooky PhysNetJAX training on QCML dft_force_field")
    print("=" * 60)
    output_dir = Path(args.output_dir).resolve()

    # 1) Load QCML dataset (download first if needed)
    print("\n1. Loading QCML dft_force_field dataset...")
    try:
        force_field_ds = tfds.load(
            args.dataset, split=args.split, data_dir=args.data_dir
        )
    except Exception as e:
        print(f"   Dataset not found. Run download_qcml_dft_force_field() first: {e}")
        print("   Attempting download...")
        download_qcml_dft_force_field(
            local_data_dir=args.data_dir,
            qcml_data_dir=args.qcml_data_dir,
            gcp_project=args.gcp_project,
        )
        force_field_ds = tfds.load(
            args.dataset, split=args.split, data_dir=args.data_dir
        )

    # Training and loading configuration from CLI.
    num_examples = args.num_examples
    batch_size = args.batch_size
    NATOMS = args.natoms
    num_steps = args.num_steps
    log_interval = args.log_interval
    learning_rate = args.learning_rate
    energy_key = args.energy_key
    chunk_ckpt_interval = args.chunk_ckpt_interval
    resume_checkpoint = args.resume
    # Avoid one huge molecule forcing massive padding.
    max_atoms_cap = NATOMS

    # Stream fixed-size chunks from TFDS to bound host/GPU memory usage.
    chunk_examples = args.chunk_examples
    print(f"\n2. Online chunked loading (chunk_examples={chunk_examples})...")

    def _finalize_chunk(examples_chunk):
        if not examples_chunk:
            return None
        max_atoms = max(int(e["atomic_numbers"].shape[0]) for e in examples_chunk)
        n_samples = len(examples_chunk)
        Z = np.zeros((n_samples, max_atoms), dtype=np.int32)
        R = np.zeros((n_samples, max_atoms, 3), dtype=np.float32)
        F = np.zeros((n_samples, max_atoms, 3), dtype=np.float32)
        E = np.zeros((n_samples,), dtype=np.float32)
        Q = np.zeros((n_samples,), dtype=np.float32)
        S = np.zeros((n_samples,), dtype=np.float32)
        for i, e in enumerate(examples_chunk):
            z = np.asarray(e["atomic_numbers"], dtype=np.int32)
            r = np.asarray(e["positions"], dtype=np.float32)
            f = np.asarray(e["pbe0_forces"], dtype=np.float32)
            n_i = z.shape[0]
            Z[i, :n_i] = z
            R[i, :n_i, :] = r
            F[i, :n_i, :] = f
            E[i] = np.asarray(e[energy_key], dtype=np.float32)
            Q[i] = np.asarray(e["charge"], dtype=np.float32)
            S[i] = np.asarray(e["multiplicity"], dtype=np.float32)
        return {"Z": Z, "R": R, "F": F, "E": E, "Q": Q, "S": S}

    def _stream_epoch_chunks():
        """Yield chunks online for a single pass over up to num_examples."""
        chunk_buf = []
        seen = 0
        kept = 0
        skipped_large = 0
        skipped_outlier = 0
        for example in force_field_ds:
            if seen >= num_examples:
                break
            seen += 1
            e = {k: v.numpy() for k, v in example.items()}
            if bool(e.get("is_outlier", False)):
                skipped_outlier += 1
                continue
            n_atoms = int(e["atomic_numbers"].shape[0])
            if n_atoms > max_atoms_cap:
                skipped_large += 1
                continue
            chunk_buf.append(e)
            kept += 1
            if len(chunk_buf) >= chunk_examples:
                chunk = _finalize_chunk(chunk_buf)
                chunk_buf = []
                if chunk is not None:
                    yield chunk
        if chunk_buf:
            chunk = _finalize_chunk(chunk_buf)
            if chunk is not None:
                yield chunk
        print(
            f"   Epoch stream summary: seen={seen}, kept={kept}, "
            f"outliers={skipped_outlier}, skipped_large={skipped_large}"
        )

    first_epoch_iter = _stream_epoch_chunks()
    first_chunk = next(first_epoch_iter, None)
    if first_chunk is None:
        raise ValueError("No usable examples found after filtering.")
    print(
        f"   First chunk prepared with {first_chunk['Z'].shape[0]} structures, "
        f"max_atoms={first_chunk['Z'].shape[1]}, batch_size={batch_size}"
    )
    checkpointer = ocp.PyTreeCheckpointer()

    # 2.5) If resuming, restore checkpoint early to extract model config
    restored = None
    resume_path = None
    if resume_checkpoint is not None:
        if resume_checkpoint == "latest":
            chunk_root = output_dir / "chunk_checkpoints"
            if not chunk_root.exists():
                raise ValueError(
                    f"Cannot resume from latest: {chunk_root} does not exist."
                )
            ckpt_dirs = sorted([p for p in chunk_root.iterdir() if p.is_dir()])
            if not ckpt_dirs:
                raise ValueError(
                    f"Cannot resume from latest: no checkpoint dirs in {chunk_root}."
                )
            resume_path = ckpt_dirs[-1]
        else:
            resume_path = Path(resume_checkpoint)
            if not resume_path.is_absolute():
                resume_path = (Path.cwd() / resume_path).resolve()

        print(f"\nRestoring checkpoint (for config): {resume_path}")
        restored = checkpointer.restore(str(resume_path))
        if not isinstance(restored, dict):
            restored = {"params": restored}

    # 3) Instantiate spooky model (use config from checkpoint when resuming)
    print("\n3. Instantiating spooky EF model...")
    model_config = restored.get("config") if restored else None
    if model_config is not None:
        # Convert numpy scalars to Python natives for model constructor
        def _to_native(v):
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            return v

        model_kwargs = {k: _to_native(v) for k, v in model_config.items()}
        model = SpookyEF(**model_kwargs)
        print(f"   Using config from checkpoint: max_atomic_number={model.max_atomic_number}, features={model.features}")
    else:
        model = SpookyEF(
            charges=True,
            natoms=NATOMS,
            max_atomic_number=args.max_atomic_number,
            features=args.features,
            debug=False,
        )

    # 4) Build one batched example to initialize params
    print("\n4. Building initialization batch and initializing parameters...")
    n_init = first_chunk["Z"].shape[0]
    init_bs = min(batch_size, n_init)
    init_batch = build_spooky_batch_from_padded_arrays(
        first_chunk["Z"][:init_bs],
        first_chunk["R"][:init_bs],
        first_chunk["E"][:init_bs],
        first_chunk["F"][:init_bs],
        first_chunk["Q"][:init_bs],
        first_chunk["S"][:init_bs],
    )

    key = jax.random.PRNGKey(0)
    params = model.init(
        key,
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

    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    resume_step = 0
    resume_epoch = 1
    resume_chunk = 0

    if restored is not None:
        state = state.replace(params=restored["params"])
        if "opt_state" in restored:
            state = state.replace(opt_state=restored["opt_state"])
        resume_step = int(np.asarray(restored.get("step", 0)))
        resume_epoch = int(np.asarray(restored.get("epoch", 1)))
        resume_chunk = int(np.asarray(restored.get("chunk", 0)))
        print(
            f"   Restored metadata: step={resume_step}, "
            f"epoch={resume_epoch}, chunk={resume_chunk}"
        )

    # 5) Training loop (stream through each chunk before loading next).
    print("\n5. Training...")
    train_step = make_spooky_train_step(
        model, forces_weight=52.91, energy_weight=1.0, batch_size=batch_size
    )
    chunk_ckpt_root = output_dir / "chunk_checkpoints"
    chunk_ckpt_root.mkdir(parents=True, exist_ok=True)

    def _save_chunk_checkpoint(state_to_save, step_to_save, epoch_to_save, chunk_to_save):
        """Save params, metadata, and model config after selected chunks."""
        chunk_dir = chunk_ckpt_root / f"epoch-{epoch_to_save:04d}-chunk-{chunk_to_save:06d}-step-{step_to_save:08d}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        payload = {
            "params": state_to_save.params,
            "opt_state": state_to_save.opt_state,
            "step": np.asarray(step_to_save, dtype=np.int32),
            "epoch": np.asarray(epoch_to_save, dtype=np.int32),
            "chunk": np.asarray(chunk_to_save, dtype=np.int32),
            "config": model.return_attributes(),
        }
        checkpointer.save(
            str(chunk_dir),
            payload,
            save_args=orbax_utils.save_args_from_target(payload),
        )

    rng = np.random.default_rng(0)
    def _train_chunk(chunk, step, chunk_idx, epoch_idx):
        n_chunk = chunk["Z"].shape[0]
        order = rng.permutation(n_chunk)
        full_batches = n_chunk // batch_size
        rem = n_chunk % batch_size

        # Pass through all structures in this chunk once (no replacement).
        for b in range(full_batches):
            idx = order[b * batch_size : (b + 1) * batch_size]
            batch = build_spooky_batch_from_padded_arrays(
                chunk["Z"][idx],
                chunk["R"][idx],
                chunk["E"][idx],
                chunk["F"][idx],
                chunk["Q"][idx],
                chunk["S"][idx],
            )
            state_local, loss_val, metrics = train_step(state_holder["state"], batch)
            state_holder["state"] = state_local
            step += 1
            if step % log_interval == 0 or step == 1 or step == num_steps:
                loss_f = float(loss_val)
                e_loss = float(metrics["e_loss"])
                f_loss = float(metrics["f_loss"])
                print(
                    f"  Step {step:6d}  epoch={epoch_idx:3d}  chunk={chunk_idx:4d}  "
                    f"loss = {loss_f:.6f}  E_loss = {e_loss:.6f}  F_loss = {f_loss:.6f}"
                )
            if step >= num_steps:
                return step

        if rem > 0 and step < num_steps:
            idx = order[full_batches * batch_size :]
            top_up = rng.choice(order, size=(batch_size - rem,), replace=True)
            idx = np.concatenate([idx, top_up], axis=0)
            batch = build_spooky_batch_from_padded_arrays(
                chunk["Z"][idx],
                chunk["R"][idx],
                chunk["E"][idx],
                chunk["F"][idx],
                chunk["Q"][idx],
                chunk["S"][idx],
            )
            state_local, loss_val, metrics = train_step(state_holder["state"], batch)
            state_holder["state"] = state_local
            step += 1
            if step % log_interval == 0 or step == 1 or step == num_steps:
                loss_f = float(loss_val)
                e_loss = float(metrics["e_loss"])
                f_loss = float(metrics["f_loss"])
                print(
                    f"  Step {step:6d}  epoch={epoch_idx:3d}  chunk={chunk_idx:4d}  "
                    f"loss = {loss_f:.6f}  E_loss = {e_loss:.6f}  F_loss = {f_loss:.6f}"
                )
        if (chunk_idx % chunk_ckpt_interval == 0) or (step >= num_steps):
            _save_chunk_checkpoint(state_holder["state"], step, epoch_idx, chunk_idx)
        return step

    step = resume_step
    epoch_idx = resume_epoch
    state_holder = {"state": state}

    if resume_step == 0:
        # Fresh run: continue first epoch from the first chunk already loaded for init.
        chunk_idx = 1
        step = _train_chunk(first_chunk, step, chunk_idx, epoch_idx)
        for chunk in first_epoch_iter:
            if step >= num_steps:
                break
            chunk_idx += 1
            step = _train_chunk(chunk, step, chunk_idx, epoch_idx)
    else:
        # Resume run: skip already completed chunks in the resumed epoch.
        print(
            f"   Skipping first {resume_chunk} chunk(s) of resumed epoch {epoch_idx}."
        )
        chunk_idx = 0
        for chunk in _stream_epoch_chunks():
            if step >= num_steps:
                break
            chunk_idx += 1
            if chunk_idx <= resume_chunk:
                continue
            step = _train_chunk(chunk, step, chunk_idx, epoch_idx)

    # Additional epochs if needed.
    while step < num_steps:
        epoch_idx += 1
        chunk_idx = 0
        for chunk in _stream_epoch_chunks():
            if step >= num_steps:
                break
            chunk_idx += 1
            step = _train_chunk(chunk, step, chunk_idx, epoch_idx)

    state = state_holder["state"]

    # Orbax checkpoint: final trained parameters
    params_ckpt_dir = output_dir / "final_params"
    if params_ckpt_dir.exists():
        shutil.rmtree(params_ckpt_dir)
    checkpointer.save(
        str(params_ckpt_dir),
        state.params,
        save_args=orbax_utils.save_args_from_target(state.params),
    )
    print(f"\nSaved final params checkpoint: {params_ckpt_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main(parse_args())
