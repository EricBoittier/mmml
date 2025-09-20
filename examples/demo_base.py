#!/usr/bin/env python3
"""Base functionality for MMML demo scripts.

This module contains common utilities and functions used across different
demo scripts for the MMML package.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def parse_base_args() -> argparse.Namespace:
    """Parse common command line arguments used across demo scripts."""
    parser = argparse.ArgumentParser(
        description="Base arguments for MMML demo scripts"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to the acetone dataset (.npz). Defaults to $MMML_DATA or "
            "mmml/data/fixed-acetone-only_MP2_21000.npz."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory used for the ML model. Defaults to $MMML_CKPT "
            "or mmml/physnetjax/ckpts."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the configuration to evaluate (default: 0).",
    )
    parser.add_argument(
        "--n-monomers",
        type=int,
        default=2,
        help="Number of monomers in the system (default: 2).",
    )
    parser.add_argument(
        "--atoms-per-monomer",
        type=int,
        default=None,
        help=(
            "Number of atoms per monomer. Defaults to total_atoms/n_monomers "
            "derived from the dataset."
        ),
    )
    parser.add_argument(
        "--ml-cutoff",
        type=float,
        default=2.0,
        help="ML cutoff distance passed to the calculator factory (default: 2.0 Å).",
    )
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        default=5.0,
        help="MM switch-on distance for the hybrid calculator (default: 5.0 Å).",
    )
    parser.add_argument(
        "--mm-cutoff",
        type=float,
        default=1.0,
        help="MM cutoff width for the hybrid calculator (default: 1.0 Å).",
    )
    parser.add_argument(
        "--include-mm",
        action="store_true",
        help="Keep MM contributions enabled when evaluating the hybrid calculator.",
    )
    parser.add_argument(
        "--skip-ml-dimers",
        action="store_true",
        help="If set, skip the ML dimer correction in the hybrid calculator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output inside the calculator factory.",
    )
    parser.add_argument(
        "--units",
        choices=("eV", "kcal/mol"),
        default="eV",
        help=(
            "Output units for energies/forces. Use 'kcal/mol' to apply the "
            "ASE conversion factor."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a JSON report containing the comparison results.",
    )
    return parser.parse_args()


def resolve_dataset_path(arg: Path | None) -> Path:
    """Resolve the dataset path from argument or environment variable."""
    candidate = arg or Path(os.environ.get("MMML_DATA", "mmml/data/fixed-acetone-only_MP2_21000.npz"))
    if not candidate.exists():
        sys.exit(f"Dataset not found: {candidate}")
    return candidate


def resolve_checkpoint_paths(arg: Path | None) -> Tuple[Path, Path]:
    """Return (factory_base_dir, epoch_dir) for the supplied checkpoint."""
    from mmml.physnetjax.physnetjax.restart.restart import get_last

    candidate = arg or Path(os.environ.get("MMML_CKPT", "mmml/physnetjax/ckpts"))
    if not candidate.exists():
        sys.exit(f"Checkpoint directory not found: {candidate}")

    candidate = candidate.resolve()
    if not candidate.is_dir():
        sys.exit(f"Checkpoint path is not a directory: {candidate}")

    def last_dir(path: Path) -> Path:
        return Path(get_last(str(path)))

    # Allow pointing directly at an epoch directory containing manifest files
    if (candidate / "manifest.ocdbt").exists():
        return candidate.parent, candidate

    children = [child for child in candidate.iterdir() if child.is_dir()]
    if not children:
        sys.exit("Checkpoint path must contain epoch-* subdirectories or point to an epoch directory.")

    subdir = last_dir(candidate)
    if (subdir / "manifest.ocdbt").exists():
        # The user provided an experiment directory with epoch-* children
        return candidate, subdir

    epoch_dir = last_dir(subdir)
    if (epoch_dir / "manifest.ocdbt").exists():
        # Two-level hierarchy: ckpts/<experiment>/epoch-*
        return subdir, epoch_dir

    sys.exit("Could not locate an epoch directory under the supplied checkpoint path.")


def load_configuration(npz_path: Path, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load a configuration from the dataset."""
    data = np.load(npz_path)
    n_samples = data["R"].shape[0]
    if index < 0 or index >= n_samples:
        sys.exit(f"Sample index {index} out of range (0..{n_samples - 1}).")

    Z = np.asarray(data["Z"][index], dtype=np.int32)
    R = np.asarray(data["R"][index], dtype=np.float64)

    references: Dict[str, np.ndarray] = {}
    for key in ("E", "F"):
        if key in data:
            references[key] = np.asarray(data[key][index])
    return Z, R, references


def load_model_parameters(epoch_dir: Path, natoms: int):
    """Load model parameters from checkpoint."""
    from mmml.physnetjax.physnetjax.restart.restart import get_params_model

    params, model = get_params_model(str(epoch_dir), natoms=natoms)
    if model is None:
        sys.exit(
            "Checkpoint does not contain model attributes; cannot construct PhysNetJax model."
        )
    model.natoms = natoms
    return params, model


def compute_force_metrics(delta_forces: np.ndarray) -> Tuple[float, float]:
    """Compute RMS and maximum absolute force metrics."""
    rms = float(np.sqrt(np.mean(delta_forces**2)))
    max_abs = float(np.abs(delta_forces).max())
    return rms, max_abs


def flatten_array(value: np.ndarray) -> np.ndarray:
    """Flatten an array to 1D."""
    arr = np.asarray(value)
    return arr.reshape(-1)


def setup_ase_imports():
    """Setup ASE imports with error handling."""
    try:
        from ase import Atoms
        return Atoms
    except ModuleNotFoundError as exc:
        sys.exit(f"ASE is required for this example: {exc}")


def setup_mmml_imports():
    """Setup MMML imports with error handling."""
    try:
        from mmml.pycharmmInterface.mmml_calculator import (
            CutoffParameters,
            ev2kcalmol,
            setup_calculator,
        )
        from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
        return CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc
    except ModuleNotFoundError as exc:
        sys.exit(f"Required MMML modules not available: {exc}")


def get_conversion_factors(units: str):
    """Get energy and force conversion factors based on units."""
    if units == "eV":
        energy_factor = 1.0
        force_factor = 1.0
    else:
        _, ev2kcalmol, _, _ = setup_mmml_imports()
        energy_factor = ev2kcalmol
        force_factor = ev2kcalmol
    return energy_factor, force_factor


def get_unit_labels(units: str):
    """Get unit labels for energy and forces."""
    energy_unit_label = units
    force_unit_label = "eV/Å" if units == "eV" else "kcal/mol/Å"
    return energy_unit_label, force_unit_label
