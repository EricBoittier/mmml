"""CLI adapter for internal-coordinate bond/angle/dihedral scans."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mmml.ic_scan import IcScanConfig, run_ic_scan


SUPPORTED_CALCULATORS = (
    "physnet",
    "spookynet",
    "mbd",
    "multipoles",
    "efield",
    "kernnn",
    "xtb",
    "dftb3-d4",
    "pyscf",
)


def _load_config_document(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("config document must contain a mapping")
    return data


def _resolve_path_fields(data: dict, config_path: Path) -> None:
    for key in ("structure", "checkpoint", "calculator_config", "slako_dir", "workdir"):
        value = data.get(key)
        if not value:
            continue
        expanded = os.path.expandvars(str(value))
        if "$" in expanded:
            raise ValueError(f"unresolved environment variable in config field {key}")
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        data[key] = str(candidate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml ic-scan",
        description=(
            "Prepare and optionally evaluate bond/angle/dihedral scans from a "
            "config that defines DoFs, grids, and 1D or N-D scan combinations."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML/JSON IcScanConfig (structure, dofs, scan_mode/scans, calculator)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write geometries without energy evaluation (overrides evaluate)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even if some energy evaluations fail",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.config.expanduser().resolve()
    try:
        data = _load_config_document(path)
        _resolve_path_fields(data, path)
    except ValueError as exc:
        build_parser().error(str(exc))
    config = IcScanConfig.from_dict(data)
    if args.prepare_only and config.evaluate != "none":
        from dataclasses import replace

        config = replace(config, evaluate="none")
    if args.allow_partial and config.failure_policy != "allow_partial":
        from dataclasses import replace

        config = replace(config, failure_policy="allow_partial")
    result = run_ic_scan(config)
    paths = result.write(args.output, overwrite=args.overwrite)
    print(f"Wrote {len(result.records)} scan points to {paths['manifest']}")
    if result.has_failures and config.failure_policy != "allow_partial":
        print("One or more requested scan points failed; see data.csv for diagnostics.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
