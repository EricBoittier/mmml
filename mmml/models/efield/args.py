"""Argparse for efield-train / efield-evaluate (no JAX).

Help, tab-completion, and ``scripts/generate_cli_docs.py`` import this module.
Keep it stdlib-only so those paths do not initialize GPU runtimes or print
device banners.
"""

from __future__ import annotations

import argparse

EVALUATE_DEFAULTS = {
    "params": "params.json",
    "config": None,
    "data": "data-full.npz",
    "output_dir": "evaluation_results",
    "batch_size": 64,
    "num_test": None,
    "model_config": None,
    "features": None,
    "max_degree": None,
    "num_iterations": None,
    "num_basis_functions": None,
    "cutoff": None,
    "max_atomic_number": None,
    "electrostatics_damping_sigma": None,
    "save_output_npz": False,
    "output_h5": None,
    "test_npz": None,
    "rot_augment": False,
    "rot_perturbation": 1.0,
}


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Single merged NPZ; random train/valid split via --num-train / --num-valid",
    )
    parser.add_argument(
        "--train-npz",
        type=str,
        default=None,
        help="Training split NPZ (R,Z,N,E,F,Ef[,Dxyz|D]) — use with --valid-npz instead of --data",
    )
    parser.add_argument(
        "--valid-npz",
        type=str,
        default=None,
        help="Validation split NPZ (same keys as train)",
    )
    parser.add_argument(
        "--test-npz",
        type=str,
        default=None,
        help="Optional test NPZ: only print shapes (not used for training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for params-*.json, config-*.json, and symlinks",
    )
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=10.0)

    parser.add_argument("--num_train", type=int, default=8000)
    parser.add_argument("--num_valid", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default 256; use 128 or 64 if OOM)",
    )

    parser.add_argument("--clip_norm", type=float, default=10000.0)
    parser.add_argument("--ema_decay", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--reduce_on_plateau_patience", type=int, default=15)
    parser.add_argument("--reduce_on_plateau_cooldown", type=int, default=15)
    parser.add_argument("--reduce_on_plateau_factor", type=float, default=0.9)
    parser.add_argument("--reduce_on_plateau_rtol", type=float, default=1e-4)
    parser.add_argument("--reduce_on_plateau_accumulation_size", type=int, default=5)
    parser.add_argument("--reduce_on_plateau_min_scale", type=float, default=0.01)

    parser.add_argument("--restart", type=str, default=None)

    parser.add_argument(
        "--energy_weight",
        type=float,
        default=1.0,
        help="Weight for energy loss in total loss",
    )
    parser.add_argument(
        "--forces_weight",
        type=float,
        default=100.0,
        help="Weight for forces loss in total loss",
    )
    parser.add_argument(
        "--dipole_weight",
        type=float,
        default=0.1,
        help="Weight for dipole loss in total loss",
    )
    parser.add_argument(
        "--charge_weight",
        type=float,
        default=1000.0,
        help="Weight for charge neutrality loss (sum of charges per molecule squared)",
    )
    parser.add_argument(
        "--polar_weight",
        "--polar-weight",
        type=float,
        default=0.0,
        dest="polar_weight",
        help="Weight for polarizability loss (Bohr³ vs dμ/dEf at Ef=0). 0 disables.",
    )
    parser.add_argument(
        "--polar-at-zero-field",
        dest="polar_at_zero_field",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate dμ/dEf at Ef=0 (default; SPICE-α / isolated DFT)",
    )
    parser.add_argument(
        "--dipole_field_coupling",
        action="store_true",
        help="Add explicit E_total = E_nn + mu·Ef coupling",
    )
    parser.add_argument(
        "--field_scale",
        type=float,
        default=0.001,
        help="Ef_phys = Ef_input * field_scale (au)",
    )
    parser.add_argument(
        "--electrostatics_damping_sigma",
        type=float,
        default=4.0,
        help="Apply erf(r/sigma) damping to learned-charge Coulomb; set 0 to disable",
    )
    parser.add_argument(
        "--zbl",
        action="store_true",
        help="Add ZBL nuclear repulsion for short-range stability",
    )
    parser.add_argument(
        "--include-pseudotensors",
        dest="include_pseudotensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Equivariant parity dimension in e3x MessagePass / tensors (default: on)",
    )
    parser.add_argument(
        "--gradient-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce GPU memory (slower training)",
    )
    parser.add_argument(
        "--rot-augment",
        action="store_true",
        help="Apply random SO(3) rotation augmentation to batches (all splits)",
    )
    parser.add_argument(
        "--rot-perturbation",
        type=float,
        default=1.0,
        help="Rotation perturbation strength in [0, 1] (used with --rot-augment)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug output (e.g. [STRUCT] parameter tree dumps)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        metavar="N",
        help="Save EMA checkpoint every N epochs to params-epoch-NNNN-<uuid>.json (0 = no periodic saves)",
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    defaults = EVALUATE_DEFAULTS
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--params",
        type=str,
        default=defaults["params"],
        help="Path to parameters JSON file (can be params-UUID.json or params.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=defaults["config"],
        help="Path to config JSON file (will be auto-detected from params UUID if not provided)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=defaults["data"],
        help="Path to dataset NPZ file",
    )
    parser.add_argument(
        "--test-npz",
        type=str,
        default=defaults["test_npz"],
        help="Test split NPZ (alias for --data; same as ef-train --test-npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults["output_dir"],
        help="Output directory for plots and metrics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=defaults["num_test"],
        help="Number of test samples to use (None = use all)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=defaults["model_config"],
        help="Path to model config JSON (deprecated: use --config)",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=defaults["features"],
        help="Model features (will be inferred from params/config if not provided)",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=defaults["max_degree"],
        help="Max degree (default: 2)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=defaults["num_iterations"],
        help="Number of iterations (default: 2)",
    )
    parser.add_argument(
        "--num-basis-functions",
        type=int,
        default=defaults["num_basis_functions"],
        help="Number of basis functions (default: 64)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=defaults["cutoff"],
        help="Cutoff radius (default: 10.0)",
    )
    parser.add_argument(
        "--max-atomic-number",
        type=int,
        default=defaults["max_atomic_number"],
        help="Max atomic number (default: 55)",
    )
    parser.add_argument(
        "--electrostatics-damping-sigma",
        type=float,
        default=defaults["electrostatics_damping_sigma"],
        help="Override learned-charge Coulomb erf damping sigma; set 0 to disable",
    )
    parser.add_argument(
        "--save-output-npz",
        action="store_true",
        help="Save evaluation outputs (predictions, targets) to NPZ file",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default=None,
        metavar="PATH",
        help="Write HDF5 trajectory for mmml gui (R,Z,N,E,E_pred,F,F_pred,Dxyz,Dxyz_pred,Ef). Requires h5py.",
    )
    parser.add_argument(
        "--rot-augment",
        action="store_true",
        help="Apply random SO(3) rotation augmentation when building batches (via prepare_batches)",
    )
    parser.add_argument(
        "--rot-perturbation",
        type=float,
        default=defaults["rot_perturbation"],
        help="Rotation perturbation strength in [0, 1] (used with --rot-augment)",
    )
    return parser
