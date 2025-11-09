#!/usr/bin/env python3
"""
Joint training orchestrator for DCMNet CO2 experiments.

Allows launching multiple training jobs concurrently, sharing the same
configuration schema as `trainer.py`.
"""

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_YAML = False

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - pre-3.11 fallback
    tomllib = None  # type: ignore

from trainer import build_argument_parser, run_single_training  # noqa: E402


PATH_FIELDS = {
    "train_efd",
    "train_grid",
    "valid_efd",
    "valid_grid",
    "output_dir",
    "restart",
}


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load a configuration file describing multiple experiments."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text()

    if suffix in {".json"}:
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        if not _HAVE_YAML:
            raise RuntimeError(
                "PyYAML is required to parse YAML configs. Install via `pip install pyyaml` "
                "or use JSON/TOML instead."
            )
        return yaml.safe_load(text)

    if suffix in {".toml"}:
        if tomllib is None:
            raise RuntimeError("TOML parsing requires Python 3.11+ or the `tomli` package.")
        return tomllib.loads(text)  # type: ignore[arg-type]

    raise ValueError(
        f"Unsupported config format '{suffix}'. Use .json, .yaml/.yml, or .toml."
    )


def _merge_with_defaults(
    defaults: Dict[str, Any], experiment: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge experiment settings over global defaults, validating keys."""
    merged = dict(defaults)
    for key, value in experiment.items():
        if key not in merged:
            raise ValueError(f"Unknown configuration key '{key}' in experiment '{experiment}'.")
        merged[key] = value
    return merged


def _convert_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize any path-like entries into Path objects."""
    for key in PATH_FIELDS:
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key]).expanduser()
    return cfg


def _namespace_from_dict(cfg: Dict[str, Any]) -> argparse.Namespace:
    """Construct an argparse.Namespace compatible with trainer.py."""
    return argparse.Namespace(**cfg)


def prepare_experiments(config: Dict[str, Any]) -> List[argparse.Namespace]:
    """Create namespaces for each experiment, including defaults."""
    parser = build_argument_parser()
    defaults_namespace = parser.parse_args([])
    defaults = vars(defaults_namespace)

    global_defaults = config.get("defaults", {})
    experiments_cfg = config.get("experiments")

    if not experiments_cfg:
        raise ValueError("Configuration must contain a non-empty 'experiments' list.")

    # Validate user-provided defaults
    for key in global_defaults:
        if key not in defaults:
            raise ValueError(f"Unknown default key '{key}' in configuration.")

    # Merge defaults
    defaults.update(global_defaults)

    namespaces: List[argparse.Namespace] = []
    for idx, experiment in enumerate(experiments_cfg):
        if not isinstance(experiment, dict):
            raise TypeError(f"Experiment entry at index {idx} must be a mapping/dict.")
        merged = _merge_with_defaults(defaults, experiment)
        merged = _convert_paths(merged)
        namespace = _namespace_from_dict(merged)
        namespaces.append(namespace)
    return namespaces


def _run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry point executed in a child process."""
    namespace = _namespace_from_dict(_convert_paths(cfg))
    try:
        final_path = run_single_training(namespace)
        return {
            "name": namespace.name,
            "status": "completed",
            "final_path": str(final_path),
        }
    except KeyboardInterrupt:  # pragma: no cover - propagated to main
        return {
            "name": namespace.name,
            "status": "cancelled",
            "final_path": None,
        }
    except Exception as exc:  # pragma: no cover - surfaced in main
        return {
            "name": namespace.name,
            "status": "failed",
            "error": repr(exc),
            "final_path": None,
        }


def _as_serializable(namespace: argparse.Namespace) -> Dict[str, Any]:
    """Convert Namespace into a picklable/JSON-friendly dict."""
    payload = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def run_joint_training(
    experiments: Iterable[argparse.Namespace],
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Execute multiple experiments sequentially or in parallel."""
    experiments = list(experiments)
    if not experiments:
        return []

    max_workers = max(1, max_workers)

    # Sequential fallback
    if max_workers == 1 or len(experiments) == 1:
        results = []
        for namespace in experiments:
            cfg_dict = _convert_paths(_as_serializable(namespace))
            results.append(_run_experiment(cfg_dict))
        return results

    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_map = {}
        for namespace in experiments:
            cfg_dict = _as_serializable(namespace)
            future = executor.submit(_run_experiment, cfg_dict)
            future_map[future] = namespace.name

        try:
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - catastrophic failure
                    result = {
                        "name": name,
                        "status": "failed",
                        "error": repr(exc),
                        "final_path": None,
                    }
                results.append(result)
        except KeyboardInterrupt:
            print("\n⚠️  Joint training interrupted by user. Cancelling workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    return results


def build_joint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch multiple DCMNet CO2 training runs concurrently.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON/YAML/TOML configuration describing experiments.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent training processes.",
    )
    return parser


def main() -> None:
    parser = build_joint_parser()
    args = parser.parse_args()

    config = load_config_file(args.config.expanduser())
    experiments = prepare_experiments(config)

    print("=" * 70)
    print(f"Starting joint training for {len(experiments)} experiment(s)")
    print(f"Max concurrent workers: {max(1, args.max_workers)}")
    print("=" * 70)

    try:
        results = run_joint_training(experiments, args.max_workers)
    except KeyboardInterrupt:
        sys.exit(130)

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    cancelled = sum(1 for r in results if r["status"] == "cancelled")

    print("\nSummary:")
    print(f"  ✅ Completed: {completed}")
    if failed:
        print(f"  ❌ Failed:    {failed}")
    if cancelled:
        print(f"  ⚠️  Cancelled:  {cancelled}")

    for res in results:
        name = res["name"]
        status = res["status"]
        if status == "completed":
            print(f"    - {name}: {status} (final checkpoint -> {res['final_path']})")
        elif status == "failed":
            print(f"    - {name}: {status} ({res.get('error')})")
        else:
            print(f"    - {name}: {status}")


if __name__ == "__main__":
    main()

