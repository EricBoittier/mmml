#!/usr/bin/env python3
"""CLI for batched umbrella NVT sampling with PhysNet / SpookyNet.

Usage:
    # Fix C (2), move NH3 rigidly along N–C:
    mmml umbrella-sample \\
      --checkpoint examples/m/kl.json \\
      --structure examples/m/neb/reag_0_opt.xyz \\
      --atoms 2,1 --move-with 1,3,4,5 \\
      --xi-min 1.5 --xi-max 3.5 --n-windows 11 \\
      --k 20 --timestep 0.1 --temperature 300 --nsteps 5000 \\
      -o out/umbrella --overwrite

    # 2D (Cl–C × N–C); invert CH3, avoid 1.5/1.5 corner
    mmml umbrella-sample --checkpoint examples/m/kl.json \\
      --structure examples/m/neb/reag_0_opt.xyz \\
      --atoms 0,2 --atoms2 1,2 \\
      --move-with2 1,3,4,5 --invert-with 6,7,8 \\
      --xi-min 1.8 --xi-max 3.0 --n-windows 4 \\
      --yi-min 1.8 --yi-max 3.0 --n-windows-y 4 \\
      --k 10 --ky 10 -o out/umbrella2d --overwrite

    # NPZ (R, Z) or PDB also work; --seed-mode frames uses consecutive frames as windows
    mmml umbrella-sample --checkpoint ckpt.json --structure data.npz \\
      --atoms 0,1 --targets 1.8,2.0,2.2 --seed-mode frames -o out/umb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.sample import run_umbrella_nvt


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(",")
        return int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected I,J atom indices (got {value!r})"
        ) from exc


def _parse_float_list(value: str) -> tuple[float, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated floats (got {value!r})"
        ) from exc


def _parse_int_list(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers (got {value!r})"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml umbrella-sample",
        description=(
            "Batched distance umbrella sampling with a PhysNet / SpookyNet "
            "checkpoint via JAX-MD NVT Nose-Hoover."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML/JSON UmbrellaConfig; CLI flags override file values when set",
    )
    parser.add_argument("--checkpoint", type=Path, help="PhysNet / SpookyNet / KerNN checkpoint")
    parser.add_argument(
        "--model",
        choices=("physnet", "kernnn"),
        default=None,
        help="ML backend (default: auto-detect KerNN JSON)",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        help="Starting geometry: XYZ, PDB, or NPZ with R/Z arrays",
    )
    parser.add_argument(
        "--structure-index",
        type=int,
        default=None,
        help="Frame index for multi-frame XYZ/PDB/NPZ (default: 0)",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("stretch", "tile", "frames"),
        default=None,
        help=(
            "Window seeding: stretch CV to each ξ₀ (default), tile reference, "
            "or use consecutive frames from --structure"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory for snapshots, trajectories, and summary",
    )
    parser.add_argument(
        "--atoms",
        type=_parse_pair,
        help="0-based atom indices for CV1 distance (I,J)",
    )
    parser.add_argument(
        "--atoms2",
        type=_parse_pair,
        default=None,
        help="0-based atom indices for CV2 distance (K,L); enables 2D umbrella",
    )
    parser.add_argument(
        "--targets",
        type=_parse_float_list,
        help="Comma-separated CV1 centers ξ₀ (Å)",
    )
    parser.add_argument(
        "--targets-y",
        type=_parse_float_list,
        default=None,
        help="Comma-separated CV2 centers η₀ (Å); product grid with --targets",
    )
    parser.add_argument("--xi-min", type=float, help="CV1 grid start (Å) if --targets omitted")
    parser.add_argument("--xi-max", type=float, help="CV1 grid end (Å) if --targets omitted")
    parser.add_argument(
        "--n-windows",
        type=int,
        help="Number of CV1 windows on [xi-min, xi-max]",
    )
    parser.add_argument("--yi-min", type=float, help="CV2 grid start (Å)")
    parser.add_argument("--yi-max", type=float, help="CV2 grid end (Å)")
    parser.add_argument(
        "--n-windows-y",
        type=int,
        help="Number of CV2 windows on [yi-min, yi-max]",
    )
    parser.add_argument(
        "--k",
        dest="k_ev_A2",
        type=float,
        default=None,
        help="CV1 harmonic force constant (eV/Å²); shared across windows (default: 10)",
    )
    parser.add_argument(
        "--ky",
        dest="k_y_ev_A2",
        type=float,
        default=None,
        help="CV2 force constant (eV/Å²); default same as --k",
    )
    parser.add_argument(
        "--move-with",
        type=_parse_int_list,
        default=None,
        help=(
            "Atoms translated rigidly with CV1 atom_j when seeding "
            "(e.g. NH3: --atoms 2,1 --move-with 1,3,4,5 fixes C, moves N+H)"
        ),
    )
    parser.add_argument(
        "--move-with2",
        type=_parse_int_list,
        default=None,
        help="Atoms translated rigidly with CV2 mobile end when seeding",
    )
    parser.add_argument(
        "--invert-with",
        type=_parse_int_list,
        default=None,
        help=(
            "Atoms Walden-blended when seeding a shared-hub 2D stretch "
            "(e.g. CH3 hydrogens: --invert-with 6,7,8)"
        ),
    )
    parser.add_argument(
        "--max-seed-force",
        type=float,
        default=None,
        help="Abort if any window seed max|F| exceeds this (eV/Å; default: 15)",
    )
    parser.add_argument(
        "--thermostat",
        choices=("langevin", "nose-hoover"),
        default=None,
        help=(
            "Packed-batch thermostat (default: langevin). Nose-Hoover shares one "
            "chain across windows and can cascade failures when one replica heats."
        ),
    )
    parser.add_argument(
        "--langevin-gamma",
        type=float,
        default=None,
        help="Langevin friction γ (1/fs in jax-md units; default: 0.1)",
    )
    parser.add_argument(
        "--max-window-temp",
        dest="max_window_temp_K",
        type=float,
        default=None,
        help="Abort if any window kinetic T exceeds this (K; default: 5× --temperature)",
    )
    parser.add_argument(
        "--replica-exchange",
        action="store_true",
        help=(
            "Enable Hamiltonian replica exchange between neighbor umbrella windows "
            "(bias-only Metropolis; even/odd pairs on the 1D/2D grid)"
        ),
    )
    parser.add_argument(
        "--rex-freq",
        type=int,
        default=None,
        help="Attempt RE swaps every this many steps (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature_K",
        type=float,
        default=None,
        help="NVT temperature in K (default: 300)",
    )
    parser.add_argument(
        "--timestep",
        dest="timestep_fs",
        type=float,
        default=None,
        help="Timestep in fs (default: 0.1)",
    )
    parser.add_argument("--nsteps", type=int, default=None, help="NVT steps (default: 1000)")
    parser.add_argument(
        "--printfreq",
        type=int,
        default=None,
        help="Print interval in steps (default: 100)",
    )
    parser.add_argument(
        "--savefreq",
        type=int,
        default=None,
        help="Snapshot save interval (default: same as printfreq)",
    )
    parser.add_argument("--seed", type=int, default=None, help="PRNG seed (default: 42)")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Prefer non-EMA checkpoint params",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    parser.add_argument(
        "--write-window-xyz",
        action="store_true",
        help=(
            "Write per-window XYZ trajectories with mass-weighted CoM at the "
            "origin (slow for large K×N_frames); default off — "
            "umbrella_snapshots.npz is enough for MBAR. "
            "umbrella_bin_minima.traj (lowest E_ML+W per window) is always written"
        ),
    )
    return parser


def _load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    path_keys = ("checkpoint", "structure", "output_dir")
    for key in path_keys:
        value = data.get(key)
        if value is None:
            continue
        expanded = os.path.expandvars(str(value))
        if "$" in expanded:
            raise ValueError(f"unresolved environment variable in config field {key}")
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        data[key] = str(candidate)
    return data


def _config_from_args(args: argparse.Namespace) -> UmbrellaConfig:
    data: dict[str, Any] = {}
    if args.config is not None:
        cfg_path = args.config.expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"--config not found: {cfg_path}")
        data.update(_load_config_file(cfg_path))

    cli_map = {
        "checkpoint": args.checkpoint,
        "structure": args.structure,
        "output_dir": args.output_dir,
        "temperature_K": args.temperature_K,
        "timestep_fs": args.timestep_fs,
        "nsteps": args.nsteps,
        "printfreq": args.printfreq,
        "savefreq": args.savefreq,
        "seed": args.seed,
        "k_ev_A2": args.k_ev_A2,
        "k_y_ev_A2": args.k_y_ev_A2,
        "xi_min": args.xi_min,
        "xi_max": args.xi_max,
        "n_windows": args.n_windows,
        "yi_min": args.yi_min,
        "yi_max": args.yi_max,
        "n_windows_y": args.n_windows_y,
        "structure_index": args.structure_index,
        "seed_mode": args.seed_mode,
        "move_with": args.move_with,
        "move_with2": args.move_with2,
        "invert_with": args.invert_with,
        "max_seed_force": args.max_seed_force,
        "thermostat": args.thermostat,
        "langevin_gamma": args.langevin_gamma,
        "max_window_temp_K": args.max_window_temp_K,
        "rex_freq": args.rex_freq,
    }
    for key, value in cli_map.items():
        if value is not None:
            data[key] = value
    if args.atoms is not None:
        data["atom_i"], data["atom_j"] = args.atoms
    if args.atoms2 is not None:
        data["atom_k"], data["atom_l"] = args.atoms2
    if args.targets is not None:
        data["targets_A"] = args.targets
    if args.targets_y is not None:
        data["targets_y_A"] = args.targets_y
    if args.no_ema:
        data["use_ema"] = False
    if args.overwrite:
        data["overwrite"] = True
    if args.replica_exchange:
        data["replica_exchange"] = True
    if args.write_window_xyz:
        data["write_window_xyz"] = True

    required = ("checkpoint", "structure", "output_dir")
    missing = [name for name in required if not data.get(name)]
    if data.get("atom_i") is None or data.get("atom_j") is None:
        missing.append("atoms")
    if missing:
        raise SystemExit(
            "missing required options: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            + " (or provide them in --config)"
        )

    data.setdefault("targets_A", ())
    data.setdefault("targets_y_A", ())
    data.setdefault("k_ev_A2", 10.0)
    data.setdefault("temperature_K", 300.0)
    data.setdefault("timestep_fs", 0.1)
    data.setdefault("nsteps", 1000)
    data.setdefault("printfreq", 100)
    data.setdefault("seed", 42)
    data.setdefault("use_ema", True)
    data.setdefault("overwrite", False)
    data.setdefault("write_window_xyz", False)
    data.setdefault("structure_index", 0)
    data.setdefault("seed_mode", "stretch")
    data.setdefault("move_with", ())
    data.setdefault("move_with2", ())
    data.setdefault("invert_with", ())
    data.setdefault("max_seed_force", 15.0)
    data.setdefault("thermostat", "langevin")
    data.setdefault("langevin_gamma", 0.1)
    data.setdefault("replica_exchange", False)
    data.setdefault("rex_freq", 100)

    return UmbrellaConfig.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    result = run_umbrella_nvt(config)
    print(
        f"Umbrella sampling done: {result.n_windows} windows, "
        f"{result.n_frames} frames → {result.summary_path}"
    )
    print(f"  snapshots: {result.snapshots_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
