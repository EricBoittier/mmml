"""Argparse for kernnn-train / kernnn-evaluate (no JAX).

Help, tab-completion, and ``scripts/generate_cli_docs.py`` import this module.
Choice names must stay in sync with ``KERNEL_FNS`` / ``DISTANCE_FNS``.
"""

from __future__ import annotations

import argparse

KERNEL_CHOICES = (
    "k20",
    "k21",
    "k22",
    "k23",
    "k24",
    "k25",
    "k26",
    "k30",
    "k31",
    "k32",
    "k33",
    "k34",
    "k35",
    "k36",
)
DISTANCE_SCHEME_CHOICES = ("abcc", "abcc_sym", "acem", "form")

TRAIN_DEFAULTS = {
    "data": None,
    "train_npz": None,
    "valid_npz": None,
    "test_npz": None,
    "workdir": "artifacts/kernnn",
    "ntrain": 3200,
    "nvalid": 400,
    "seed": 42,
    "n_hidden": 64,
    "batch_size": 64,
    "learning_rate": 0.005,
    "f_weight": 10.0,
    "epochs": 1000,
    "patience": 200,
    "ema_decay": 0.999,
    "kernel": "k33",
    "distance_scheme": "abcc",
    "architecture": "ffnet",
    "teacher_checkpoint": None,
    "distill_alpha": 1.0,
    "teacher_energy_offset": None,
    "no_align_teacher_energy": False,
    "teacher_align_n": 256,
}

EVAL_DEFAULTS = {
    "checkpoint": "artifacts/kernnn/best.json",
    "data": "data.npz",
    "output_dir": "artifacts/kernnn/eval",
    "split": "test",
    "seed": 42,
    "ntrain": 3200,
    "nvalid": 400,
    "batch_size": 64,
}


def build_train_parser() -> argparse.ArgumentParser:
    d = TRAIN_DEFAULTS
    p = argparse.ArgumentParser(
        description="Train KerNN (kernel Softplus MLP) on NPZ (R, E, F)"
    )
    p.add_argument(
        "--data",
        type=str,
        default=d["data"],
        help="Single NPZ with R,E,F (random train/valid/test split)",
    )
    p.add_argument("--train-npz", type=str, default=d["train_npz"], help="Train split NPZ")
    p.add_argument("--valid-npz", type=str, default=d["valid_npz"], help="Valid split NPZ")
    p.add_argument("--test-npz", type=str, default=d["test_npz"], help="Optional test split NPZ")
    p.add_argument("--workdir", type=str, default=d["workdir"], help="Output directory")
    p.add_argument("--ntrain", type=int, default=d["ntrain"], help="Training size when using --data")
    p.add_argument("--nvalid", type=int, default=d["nvalid"], help="Validation size when using --data")
    p.add_argument("--seed", type=int, default=d["seed"], help="RNG seed for split/init")
    p.add_argument("--n-hidden", type=int, default=d["n_hidden"], help="Hidden layer width")
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    p.add_argument("--f-weight", type=float, default=d["f_weight"], help="Force loss weight")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument(
        "--patience",
        type=int,
        default=d["patience"],
        help="Early-stop after this many non-improving validation epochs",
    )
    p.add_argument("--ema-decay", type=float, default=d["ema_decay"])
    p.add_argument(
        "--kernel",
        type=str,
        default=d["kernel"],
        choices=KERNEL_CHOICES,
        help="1D kernel name (default k33)",
    )
    p.add_argument(
        "--list-kernels",
        action="store_true",
        help="Print the table of available 1D kernel functions and exit",
    )
    p.add_argument(
        "--distance-scheme",
        type=str,
        default=d["distance_scheme"],
        choices=DISTANCE_SCHEME_CHOICES,
        help="Distance descriptor: abcc, abcc_sym, form (6 atoms), acem (9 atoms)",
    )
    p.add_argument(
        "--architecture",
        type=str,
        default=d["architecture"],
        choices=("ffnet", "dual"),
        help="ffnet (default) or dual (ABCC + dihedral only)",
    )
    p.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=d["teacher_checkpoint"],
        help="PhysNet checkpoint (JSON/Orbax) used as distillation teacher",
    )
    p.add_argument(
        "--distill-alpha",
        type=float,
        default=d["distill_alpha"],
        help="Blend GT vs teacher: loss = alpha*GT + (1-alpha)*teacher (1=pure GT)",
    )
    p.add_argument(
        "--teacher-energy-offset",
        type=float,
        default=d["teacher_energy_offset"],
        help="Add this constant (eV) to teacher energies before distill loss "
        "(overrides auto-align). Use when PhysNet atom refs shift the zero.",
    )
    p.add_argument(
        "--no-align-teacher-energy",
        action="store_true",
        default=d["no_align_teacher_energy"],
        help="Do not auto-fit an additive teacher energy offset vs GT",
    )
    p.add_argument(
        "--teacher-align-n",
        type=int,
        default=d["teacher_align_n"],
        help="Number of train structures used to estimate teacher energy offset",
    )
    return p


def build_evaluate_parser() -> argparse.ArgumentParser:
    d = EVAL_DEFAULTS
    p = argparse.ArgumentParser(description="Evaluate KerNN checkpoint (E/F metrics)")
    p.add_argument("--checkpoint", type=str, default=d["checkpoint"])
    p.add_argument(
        "--data",
        type=str,
        default=d["data"],
        help="NPZ with R, E, F (use --split all for a dedicated test NPZ)",
    )
    p.add_argument("--output-dir", type=str, default=d["output_dir"])
    p.add_argument(
        "--split",
        type=str,
        default=d["split"],
        choices=("train", "valid", "test", "all"),
        help="Which split to evaluate (seed/ntrain/nvalid define the split; "
        "use 'all' for a dedicated test NPZ)",
    )
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--ntrain", type=int, default=d["ntrain"])
    p.add_argument("--nvalid", type=int, default=d["nvalid"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument(
        "--split-json",
        type=str,
        default=None,
        help="Optional data_split.json from training (overrides seed/ntrain/nvalid)",
    )
    p.add_argument(
        "--list-kernels",
        action="store_true",
        help="Print the table of available 1D kernel functions and exit",
    )
    return p
