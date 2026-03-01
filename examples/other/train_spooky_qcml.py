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

import jax
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state

from mmml.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_example,
    make_spooky_train_step,
)

# QCML dataset paths
LOCAL_DATA_DIR = "."
QCML_DATA_DIR = "gs://qcml-datasets/tfds"
GCP_PROJECT = "deepmind-opensource"


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

    # Optional: take a small subset for a quick example
    force_field_ds = force_field_ds.take(100)

    # 2) Instantiate spooky model
    print("\n2. Instantiating spooky EF model...")
    model = SpookyEF(charges=True, natoms=60, debug=False)

    # 3) Build one example batch to initialize params
    print("\n3. Building example batch and initializing parameters...")
    example = next(iter(force_field_ds))
    example_np = {k: v.numpy() for k, v in example.items()}
    batch = build_spooky_batch_from_example(
        example_np,
        energy_key="pbe0_energy",
        forces_key="pbe0_forces",
    )

    key = jax.random.PRNGKey(0)
    params = model.init(
        key,
        atomic_numbers=batch["Z"],
        charges=batch["Q_atoms"],
        spins=batch["S_atoms"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch["batch_size"],
        batch_mask=batch["batch_mask"],
        atom_mask=batch["atom_mask"],
    )

    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    # 4) Training loop
    print("\n4. Training...")
    train_step = make_spooky_train_step(
        model, forces_weight=52.91, energy_weight=1.0
    )

    log_interval = 10
    for step, example in enumerate(force_field_ds):
        example_np = {k: v.numpy() for k, v in example.items()}
        batch = build_spooky_batch_from_example(
            example_np,
            energy_key="pbe0_energy",
            forces_key="pbe0_forces",
        )
        state, loss_val, metrics = train_step(state, batch)

        if (step + 1) % log_interval == 0 or step == 0:
            loss_f = float(loss_val)
            e_loss = float(metrics["e_loss"])
            f_loss = float(metrics["f_loss"])
            print(
                f"  Step {step + 1:4d}  loss = {loss_f:.6f}  "
                f"E_loss = {e_loss:.6f}  F_loss = {f_loss:.6f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
