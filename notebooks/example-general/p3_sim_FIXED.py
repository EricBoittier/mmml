#!/usr/bin/env python
"""
Production-ready MM/ML hybrid simulation runner.

Minimal workflow:
  1) Load dataset (NPZ) and batches
  2) Load PhysNet checkpoint (JSON/pickle or orbax fallback)
  3) Build hybrid calculator (ML/MM) with requested cutoffs
  4) Run one quick validation batch (energy/forces)

This script is intentionally compact, logs clearly, and avoids the ad‑hoc
mock arguments and exploratory code that were present before.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mmml.cli.base import (
    load_model_parameters,
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)
from mmml.physnetjax.physnetjax.data.batches import prepare_batches_jit
from mmml.physnetjax.physnetjax.data.data import prepare_datasets


# --------------------------------------------------------------------------- #
# Logging and environment
# --------------------------------------------------------------------------- #

def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_env() -> None:
    # Keep memory usage modest by default
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model_parameters_json(epoch_dir: Path, natoms: int) -> Tuple[dict, object]:
    """
    Load model parameters from JSON first, then pickle. If neither works,
    callers should fall back to orbax via load_model_parameters().
    """
    from mmml.physnetjax.physnetjax.models.model import EF
    import pickle

    def json_to_jax(obj):
        if isinstance(obj, dict):
            return {k: json_to_jax(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) > 0 and isinstance(obj[0], (list, int, float)):
                return jnp.array(obj)
            return [json_to_jax(x) for x in obj]
        return obj

    epoch_dir = Path(epoch_dir)
    json_candidates = [
        epoch_dir / "params.json",
        epoch_dir / "best_params.json",
        epoch_dir / "checkpoint.json",
        epoch_dir / "final_params.json",
        epoch_dir / "json_checkpoint" / "params.json",
    ]

    params = None
    for cand in json_candidates:
        if cand.exists():
            logging.info("Loading parameters from JSON: %s", cand)
            with open(cand, "r") as f:
                data = json.load(f)
            params_data = data.get("params") if isinstance(data, dict) else data
            params = json_to_jax(params_data)
            break

    if params is None:
        pickle_candidates = [
            epoch_dir / "params.pkl",
            epoch_dir / "best_params.pkl",
            epoch_dir / "checkpoint.pkl",
            epoch_dir / "final_params.pkl",
        ]
        for cand in pickle_candidates:
            if cand.exists():
                logging.info("Loading parameters from pickle: %s", cand)
                with open(cand, "rb") as f:
                    data = pickle.load(f)
                params = data.get("params") if isinstance(data, dict) else data
                break

    if params is None:
        raise FileNotFoundError(
            f"Could not find params in {epoch_dir}; looked for JSON/pickle candidates."
        )

    config_candidates = [
        epoch_dir / "model_config.json",
        epoch_dir / "json_checkpoint" / "model_config.json",
    ]
    model_kwargs = {}
    for cand in config_candidates:
        if cand.exists():
            logging.info("Loading model config from: %s", cand)
            with open(cand, "r") as f:
                model_kwargs = json.load(f)
            break

    if not model_kwargs:
        logging.warning("No model_config.json found; using minimal defaults.")
        model_kwargs = {"features": 64, "cutoff": 8.0, "max_degree": 2, "num_iterations": 3}

    model_kwargs["natoms"] = natoms
    model = EF(**model_kwargs)
    model.natoms = natoms
    logging.info("Model loaded (JSON/pickle path).")
    return params, model


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

def load_data(data_path: Path, natoms: int, num_train: int, num_valid: int, seed: int):
    key = jax.random.PRNGKey(seed)
    train_data, valid_data = prepare_datasets(
        key, num_train, num_valid, [str(data_path)], natoms=natoms
    )
    train_batches = prepare_batches_jit(key, train_data, 1, num_atoms=natoms)
    valid_batches = prepare_batches_jit(key, valid_data, 1, num_atoms=natoms)
    logging.info(
        "Loaded data: %s (train=%d, valid=%d)",
        data_path,
        len(train_data["R"]),
        len(valid_data["R"]),
    )
    return train_data, valid_data, train_batches, valid_batches


# --------------------------------------------------------------------------- #
# Calculator setup
# --------------------------------------------------------------------------- #

def build_calculator_factory(
    ATOMS_PER_MONOMER: int,
    N_MONOMERS: int,
    ml_cutoff: float,
    mm_switch_on: float,
    mm_cutoff: float,
    include_mm: bool,
    skip_ml_dimers: bool,
    debug: bool,
    checkpoint_base: Path,
    natoms: int,
    cell: float | None,
):
    CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        N_MONOMERS=N_MONOMERS,
        ml_cutoff_distance=ml_cutoff,
        mm_switch_on=mm_switch_on,
        mm_cutoff=mm_cutoff,
        doML=True,
        doMM=include_mm,
        doML_dimer=not skip_ml_dimers,
        debug=debug,
        model_restart_path=checkpoint_base,
        MAX_ATOMS_PER_SYSTEM=natoms,
        ml_energy_conversion_factor=1.0,
        ml_force_conversion_factor=1.0,
        cell=cell,
    )
    cutoff_params = CutoffParameters(
        ml_cutoff=ml_cutoff, mm_switch_on=mm_switch_on, mm_cutoff=mm_cutoff
    )
    return calculator_factory, cutoff_params


# --------------------------------------------------------------------------- #
# Quick validation
# --------------------------------------------------------------------------- #

def run_quick_validation(calculator_factory, cutoff_params, batch, args) -> None:
    Atoms = setup_ase_imports()
    from mmml.utils.simulation_utils import initialize_simulation_from_batch

    atoms, hybrid_calc = initialize_simulation_from_batch(batch, calculator_factory, cutoff_params, args)
    atoms.calc = hybrid_calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    logging.info("Quick validation: energy=%s eV | forces shape=%s", energy, forces.shape)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MM/ML hybrid simulation runner")
    p.add_argument("--data", type=Path, required=True, help="Path to NPZ dataset")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (epoch dir or file)")
    p.add_argument("--n-monomers", type=int, default=2)
    p.add_argument("--atoms-per-monomer", type=int, default=10)
    p.add_argument("--ml-cutoff", type=float, default=2.0)
    p.add_argument("--mm-switch-on", type=float, default=5.0)
    p.add_argument("--mm-cutoff", type=float, default=1.0)
    p.add_argument("--include-mm", action="store_true", default=True)
    p.add_argument("--skip-ml-dimers", action="store_true", default=False)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--cell", type=float, default=None, help="Cubic cell length (Å); if omitted, no PBC")
    p.add_argument("--num-train", type=int, default=1000)
    p.add_argument("--num-valid", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)
    configure_env()

    logging.info("JAX backend: %s | devices: %s", jax.default_backend(), jax.devices())

    ATOMS_PER_MONOMER = args.atoms_per_monomer
    N_MONOMERS = args.n_monomers
    natoms = ATOMS_PER_MONOMER * N_MONOMERS

    # Resolve checkpoint paths
    checkpoint_base, epoch_dir = resolve_checkpoint_paths(args.checkpoint)
    logging.info("Checkpoint base: %s | epoch: %s", checkpoint_base, epoch_dir)

    # Load model parameters
    try:
        params, model = load_model_parameters_json(epoch_dir, natoms)
    except Exception as json_err:
        logging.warning("JSON/pickle load failed (%s); trying orbax", json_err)
        params, model = load_model_parameters(epoch_dir, natoms)
        model.natoms = natoms
    logging.info("Model ready: %s", model)

    # Load data
    train_data, valid_data, train_batches, valid_batches = load_data(
        args.data, natoms, args.num_train, args.num_valid, args.seed
    )

    # Calculator
    calculator_factory, cutoff_params = build_calculator_factory(
        ATOMS_PER_MONOMER,
        N_MONOMERS,
        args.ml_cutoff,
        args.mm_switch_on,
        args.mm_cutoff,
        args.include_mm,
        args.skip_ml_dimers,
        args.debug,
        checkpoint_base,
        natoms,
        args.cell,
    )
    logging.info("Cutoff parameters: %s", cutoff_params)

    # Quick validation on first training batch (if available)
    if train_batches:
        logging.info("Running quick validation on first training batch.")
        batch0 = train_batches[0]
        run_quick_validation(calculator_factory, cutoff_params, batch0, args)
    else:
        logging.warning("No training batches available for validation.")

    logging.info("Workflow finished successfully.")


if __name__ == "__main__":
    sys.exit(main())

