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


def resolve_checkpoint_paths(arg: Path | str | None) -> Tuple[Path, Path]:
    """Return (factory_base_dir, epoch_dir) for the supplied checkpoint.

    Supports both orbax checkpoints (manifest.ocdbt) and JSON checkpoints
    (params.json or a .json file from orbax_to_json).
    """
    try:
        from mmml.models.physnetjax.physnetjax.restart.restart import get_last
    except ModuleNotFoundError:
        from mmml.physnetjax.physnetjax.restart.restart import get_last

    # Convert string to Path if needed
    if arg is None:
        ckpt_env = os.environ.get("MMML_CKPT")
        if ckpt_env:
            candidate = Path(ckpt_env)
        else:
            candidate_models = Path("mmml/models/physnetjax/ckpts")
            candidate_legacy = Path("mmml/physnetjax/ckpts")
            candidate = candidate_models if candidate_models.exists() else candidate_legacy
    elif isinstance(arg, str):
        candidate = Path(arg)
    else:
        candidate = arg

    if not candidate.exists():
        sys.exit(f"Checkpoint directory not found: {candidate}")

    candidate = candidate.resolve()

    # JSON checkpoint: directory with params.json, or direct path to .json file
    if candidate.is_file() and candidate.suffix == ".json":
        # Path to a JSON file (e.g. ckpts_json/DESdimers_params.json)
        # Both base and epoch point to the file so calculator and load_model_parameters can use it
        return candidate, candidate
    if candidate.is_dir():
        params_json = candidate / "params.json"
        if params_json.exists():
            return candidate, candidate

    if not candidate.is_dir():
        sys.exit(f"Checkpoint path is not a directory or JSON file: {candidate}")

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


def resolve_desdimers_checkpoint(script_file: str | Path | None = None) -> Path:
    """Resolve a default DES-family checkpoint path without hardcoding."""
    ckpt_env = os.environ.get("MMML_CKPT")
    if ckpt_env:
        return Path(ckpt_env).expanduser().resolve()

    # Prefer installed package location when available.
    try:
        import mmml as mmml_pkg

        package_root = Path(mmml_pkg.__file__).resolve().parent
        for rel_root in (
            ("models", "physnetjax", "ckpts"),
            ("physnetjax", "ckpts"),
        ):
            for ckpt_name in ("DES", "DESdimers"):
                package_ckpt = package_root.joinpath(*rel_root, ckpt_name)
                if package_ckpt.exists():
                    return package_ckpt.resolve()
    except Exception:
        pass

    # Fallback for local repo execution.
    search_roots: list[Path] = []
    if script_file is not None:
        script_dir = Path(script_file).resolve().parent
        search_roots.extend([script_dir, *script_dir.parents])
    cwd = Path.cwd().resolve()
    search_roots.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        for rel_root in (
            ("mmml", "models", "physnetjax", "ckpts"),
            ("mmml", "physnetjax", "ckpts"),
        ):
            for ckpt_name in ("DES", "DESdimers"):
                candidate = root.joinpath(*rel_root, ckpt_name)
                if candidate.exists():
                    return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate checkpoint. Set MMML_CKPT to a valid checkpoint path."
    )


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
    """Load model parameters from checkpoint (orbax or JSON format)."""
    import jax.numpy as jnp

    epoch_path = Path(epoch_dir)
    is_json = (
        (epoch_path.is_file() and epoch_path.suffix == ".json")
        or (epoch_path.is_dir() and (epoch_path / "params.json").exists())
    )

    if is_json:
        try:
            from mmml.models.physnetjax.physnetjax.models.model import EF as StandardEF
            from mmml.models.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
        except ModuleNotFoundError:
            from mmml.physnetjax.physnetjax.models.model import EF as StandardEF
            from mmml.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
        from mmml.utils.model_checkpoint import load_model_checkpoint

        checkpoint = load_model_checkpoint(
            epoch_path, use_orbax=False, load_params=True, load_config=True
        )
        params = checkpoint.get("params")
        config = checkpoint.get("config", {})

        if params is None:
            sys.exit("Checkpoint does not contain params; cannot load model.")
        if not config:
            sys.exit(
                "JSON checkpoint does not contain model config; cannot construct model. "
                "Use orbax_to_json with a checkpoint that has model_attributes, or "
                "ensure model_config.json exists in the checkpoint directory."
            )

        def _to_jax(obj):
            if isinstance(obj, dict):
                return {k: _to_jax(v) for k, v in obj.items()}
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (list, int, float)):
                return jnp.array(obj)
            if isinstance(obj, list):
                return [_to_jax(x) for x in obj]
            return obj

        params = _to_jax(params)
        if isinstance(params, dict) and "params" not in params:
            params = {"params": params}

        model_attrs = [
            "features", "max_degree", "num_iterations", "num_basis_functions",
            "cutoff", "max_atomic_number", "n_res", "zbl", "efa", "charges",
            "natoms", "total_charge", "n_dcm", "include_pseudotensors",
            "use_energy_bias", "use_pbc", "debug",
        ]
        model_config = {k: v for k, v in config.items() if k in model_attrs}
        model_config["natoms"] = natoms
        epoch_path_str = str(epoch_path).lower()
        is_spooky = (
            str(config.get("model_type", "")).lower() == "spooky"
            or "spooky" in epoch_path_str
        )
        model_cls = SpookyEF if is_spooky else StandardEF
        model = model_cls(**model_config)
        model.natoms = natoms
        return params, model

    try:
        from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model
    except ModuleNotFoundError:
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
        try:
            from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
        except ModuleNotFoundError:
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
