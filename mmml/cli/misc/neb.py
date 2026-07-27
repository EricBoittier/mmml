"""CLI adapter for ASE NEB sampling with an MMML PhysNet checkpoint.

Usage:
    mmml neb \\
      --checkpoint examples/m/kl.json \\
      --initial examples/m/neb/reag_0_opt.xyz \\
      --final examples/m/neb/prod_0_opt.xyz \\
      --output-dir artifacts/nh3_ch3cl/neb \\
      --n-images 11 --fmax 0.05
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mmml.neb import NebConfig, run_neb


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(",")
        return int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected I,J atom indices (got {value!r})"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml neb",
        description=(
            "Nudged elastic band (NEB) path sampling with a PhysNet / MMML "
            "checkpoint as the ASE calculator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML/JSON NebConfig; CLI flags override file values when set",
    )
    parser.add_argument("--checkpoint", type=Path, help="PhysNet / MMML checkpoint")
    parser.add_argument("--initial", type=Path, help="Reactant / initial XYZ")
    parser.add_argument("--final", type=Path, help="Product / final XYZ")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory for neb.traj / neb.xyz / profile / plot / summary",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Total band images including endpoints (default: 11)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Force convergence threshold in eV/Å (default: 0.05)",
    )
    parser.add_argument(
        "--climb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable climbing-image NEB (default: off)",
    )
    parser.add_argument(
        "--interpolate",
        choices=("idpp", "linear"),
        default=None,
        help="Band interpolation method (default: idpp)",
    )
    parser.add_argument(
        "--optimizer",
        choices=("BFGS", "FIRE", "MDMin"),
        default=None,
        help="Band optimizer (default: BFGS)",
    )
    parser.add_argument(
        "--neb-method",
        choices=("improvedtangent", "aseneb", "eb", "spline", "string"),
        default=None,
        help="ASE NEB force method (default: improvedtangent)",
    )
    parser.add_argument(
        "--spring-k",
        type=float,
        default=None,
        help="NEB spring constant (default: 0.1)",
    )
    parser.add_argument(
        "--shared-calculator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Share one ASE calculator across images (default: on)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional optimizer step cap (default: unlimited until fmax)",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write neb_plot.png (default: on)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing artifacts in --output-dir",
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        dest="pairs",
        metavar="I,J",
        help=(
            "Atom-index pair to log as a distance column (repeatable). "
            "Default for 9-atom NH3–CH3Cl: 1,2 (N–C) and 0,2 (Cl–C)."
        ),
    )
    return parser


def _load_config_file(path: Path) -> dict[str, Any]:
    import json

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("--config document must contain a mapping")
    for key in ("initial", "final", "checkpoint", "output_dir"):
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


def _config_from_args(args: argparse.Namespace) -> NebConfig:
    data: dict[str, Any] = {}
    if args.config is not None:
        cfg_path = args.config.expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"--config not found: {cfg_path}")
        data.update(_load_config_file(cfg_path))

    cli_map = {
        "checkpoint": args.checkpoint,
        "initial": args.initial,
        "final": args.final,
        "output_dir": args.output_dir,
        "n_images": args.n_images,
        "fmax": args.fmax,
        "climb": args.climb,
        "interpolate": args.interpolate,
        "optimizer": args.optimizer,
        "neb_method": args.neb_method,
        "spring_k": args.spring_k,
        "shared_calculator": args.shared_calculator,
        "max_steps": args.max_steps,
        "plot": args.plot,
    }
    for key, value in cli_map.items():
        if value is not None:
            data[key] = value
    if args.overwrite:
        data["overwrite"] = True
    if args.pairs:
        data["pair_indices"] = args.pairs

    required = ("checkpoint", "initial", "final", "output_dir")
    missing = [name for name in required if not data.get(name)]
    if missing:
        raise SystemExit(
            "missing required options: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            + " (or provide them in --config)"
        )

    # Apply defaults for optional keys not present in file/CLI.
    data.setdefault("n_images", 11)
    data.setdefault("fmax", 0.05)
    data.setdefault("climb", False)
    data.setdefault("interpolate", "idpp")
    data.setdefault("optimizer", "BFGS")
    data.setdefault("neb_method", "improvedtangent")
    data.setdefault("spring_k", 0.1)
    data.setdefault("shared_calculator", True)
    data.setdefault("plot", True)
    data.setdefault("overwrite", False)

    return NebConfig.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    result = run_neb(config)
    print(
        f"NEB done: barrier={result.barrier_kcal_mol:.4f} kcal/mol "
        f"→ {result.paths['summary']}"
    )
    for key, path in sorted(result.paths.items()):
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
