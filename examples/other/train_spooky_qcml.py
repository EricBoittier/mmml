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

import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
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
OUTPUT_DIR = Path("ckpts_spooky_qcml")


def download_qcml_dft_force_field():
    """Download QCML dft_force_field dataset locally."""
    os.system("gcloud config set auth/disable_credentials True")
    os.system(f"mkdir -p {LOCAL_DATA_DIR}/qcml/dft_force_field/")
    os.system(
        f"gcloud storage cp -r {QCML_DATA_DIR}/qcml/dft_force_field/1.0.0 "
        f"{LOCAL_DATA_DIR}/qcml/dft_force_field/ --project={GCP_PROJECT}"
    )


def main():
    print("=" * 60)
    print("Spooky PhysNetJAX training on QCML dft_force_field")
    print("=" * 60)

    # 1) Load QCML dataset (download first if needed)
    print("\n1. Loading QCML dft_force_field dataset...")
    try:
        force_field_ds = tfds.load(
            "qcml/dft_force_field", split="full", data_dir=LOCAL_DATA_DIR
        )
    except Exception as e:
        print(f"   Dataset not found. Run download_qcml_dft_force_field() first: {e}")
        print("   Attempting download...")
        download_qcml_dft_force_field()
        force_field_ds = tfds.load(
            "qcml/dft_force_field", split="full", data_dir=LOCAL_DATA_DIR
        )

    # Optional: take a subset for a quick run.
    num_examples = 4096
    batch_size = 32
    num_steps = 1000
    log_interval = 20
    learning_rate = 1e-3

    force_field_ds = force_field_ds.take(num_examples)

    # 2) Materialize a fixed-shape cached dataset for faster training.
    # This removes per-step TF->NumPy conversion overhead and enables larger batches.
    print("\n2. Caching dataset to dense padded arrays...")
    examples = []
    max_atoms = 0
    for example in force_field_ds:
        e = {k: v.numpy() for k, v in example.items()}
        if bool(e.get("is_outlier", False)):
            continue
        examples.append(e)
        max_atoms = max(max_atoms, int(e["atomic_numbers"].shape[0]))

    if not examples:
        raise ValueError("No usable examples found (all filtered as outliers?).")

    n_samples = len(examples)
    Z_all = jnp.zeros((n_samples, max_atoms), dtype=jnp.int32)
    R_all = jnp.zeros((n_samples, max_atoms, 3), dtype=jnp.float32)
    F_all = jnp.zeros((n_samples, max_atoms, 3), dtype=jnp.float32)
    E_all = jnp.zeros((n_samples,), dtype=jnp.float32)
    Q_all = jnp.zeros((n_samples,), dtype=jnp.float32)
    S_all = jnp.zeros((n_samples,), dtype=jnp.float32)

    for i, e in enumerate(examples):
        z = jnp.asarray(e["atomic_numbers"], dtype=jnp.int32)
        r = jnp.asarray(e["positions"], dtype=jnp.float32)
        f = jnp.asarray(e["pbe0_forces"], dtype=jnp.float32)
        n_i = z.shape[0]
        Z_all = Z_all.at[i, :n_i].set(z)
        R_all = R_all.at[i, :n_i, :].set(r)
        F_all = F_all.at[i, :n_i, :].set(f)
        E_all = E_all.at[i].set(jnp.asarray(e["pbe0_energy"], dtype=jnp.float32))
        Q_all = Q_all.at[i].set(jnp.asarray(e["charge"], dtype=jnp.float32))
        S_all = S_all.at[i].set(jnp.asarray(e["multiplicity"], dtype=jnp.float32))

    print(f"   Cached {n_samples} examples, max_atoms={max_atoms}, batch_size={batch_size}")

    # Orbax checkpoint: cached dataset arrays
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_ckpt_dir = OUTPUT_DIR / "dataset_cache"
    if dataset_ckpt_dir.exists():
        shutil.rmtree(dataset_ckpt_dir)
    dataset_ckpt = {
        "Z": Z_all,
        "R": R_all,
        "F": F_all,
        "E": E_all,
        "Q": Q_all,
        "S": S_all,
        "n_samples": jnp.asarray(n_samples),
        "max_atoms": jnp.asarray(max_atoms),
    }
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        str(dataset_ckpt_dir),
        dataset_ckpt,
        save_args=orbax_utils.save_args_from_target(dataset_ckpt),
    )
    print(f"   Saved dataset cache checkpoint: {dataset_ckpt_dir}")

    # 3) Instantiate spooky model
    print("\n3. Instantiating spooky EF model...")
    model = SpookyEF(charges=True, natoms=60, debug=False)

    # 4) Build one batched example to initialize params
    print("\n4. Building initialization batch and initializing parameters...")
    init_bs = min(batch_size, n_samples)
    init_batch = build_spooky_batch_from_padded_arrays(
        Z_all[:init_bs],
        R_all[:init_bs],
        E_all[:init_bs],
        F_all[:init_bs],
        Q_all[:init_bs],
        S_all[:init_bs],
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

    # 5) Training loop (random minibatches from cached dense arrays)
    print("\n5. Training...")
    train_step = make_spooky_train_step(
        model, forces_weight=52.91, energy_weight=1.0
    )

    for step in range(1, num_steps + 1):
        key, subkey = jax.random.split(key)
        replace = batch_size > n_samples
        idx = jax.random.choice(
            subkey, n_samples, shape=(batch_size,), replace=replace
        )
        batch = build_spooky_batch_from_padded_arrays(
            Z_all[idx],
            R_all[idx],
            E_all[idx],
            F_all[idx],
            Q_all[idx],
            S_all[idx],
        )
        state, loss_val, metrics = train_step(state, batch)

        if step % log_interval == 0 or step == 1 or step == num_steps:
            loss_f = float(loss_val)
            e_loss = float(metrics["e_loss"])
            f_loss = float(metrics["f_loss"])
            print(
                f"  Step {step:4d}  loss = {loss_f:.6f}  "
                f"E_loss = {e_loss:.6f}  F_loss = {f_loss:.6f}"
            )

    # Orbax checkpoint: final trained parameters
    params_ckpt_dir = OUTPUT_DIR / "final_params"
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
    main()
