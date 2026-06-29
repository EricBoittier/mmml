#!/usr/bin/env python3
"""Diagnose learning-curve sweep outliers: seeds, training spikes, and NPZ samples.

``mmml diagnose-lc-outliers`` combines:
  1. Run-level test metrics vs sibling repeats at the same n_train.
  2. Training-curve spike detection on validation metrics.
  3. NPZ sample enrichment: structures exclusive to outlier train splits.

Example:
  mmml diagnose-lc-outliers \\
    --eval-root out/eval/learning_curve/e1000 \\
    --dataset aco \\
    --train-npz out/splits/aco/energies_forces_dipoles_train.npz \\
    --json-out out/eval/learning_curve/e1000/aco_outlier_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402


@dataclass
class Spike:
    metric: str
    epoch: int
    value: float
    reasons: list[str]


@dataclass
class RunRecord:
    dataset: str
    n_train: int
    repeat: int
    seed: int
    run_dir: Path
    test_energy_mae: float | None
    test_forces_mae: float | None
    valid_energy_mae: float | None
    valid_loss: float | None
    metrics_path: Path | None
    spikes: list[Spike] = field(default_factory=list)
    run_outlier: bool = False
    run_outlier_reason: str = ""


def _finite(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def detect_spikes(
    epochs: list[int] | np.ndarray,
    values: list[float] | np.ndarray,
    *,
    metric: str,
    relative_factor: float = 5.0,
    jump_factor: float = 3.0,
) -> list[Spike]:
    """Flag epochs with unusually large validation metric values."""
    y = _finite(np.asarray(values, dtype=float))
    finite = y[np.isfinite(y)]
    if finite.size < 4:
        return []

    tail = finite[finite.size // 2 :]
    ref = float(np.nanmedian(tail))
    if ref <= 0:
        ref = float(np.nanmedian(finite))
    if ref <= 0:
        return []

    spikes: list[Spike] = []
    for i, (epoch, value) in enumerate(zip(epochs, y)):
        if not np.isfinite(value):
            continue
        reasons: list[str] = []
        if value > relative_factor * ref:
            reasons.append(f">{relative_factor:g}x_tail_median")
        if i > 0:
            prev = y[i - 1]
            if np.isfinite(prev) and prev > 0 and value > jump_factor * prev:
                reasons.append(f">{jump_factor:g}x_prev")
        if reasons:
            spikes.append(Spike(metric=metric, epoch=int(epoch), value=float(value), reasons=reasons))
    return spikes


def load_runs(eval_root: Path, dataset: str) -> list[RunRecord]:
    runs: list[RunRecord] = []
    ds_root = eval_root / dataset
    if not ds_root.is_dir():
        raise FileNotFoundError(f"Dataset eval dir not found: {ds_root}")

    for n_dir in sorted(ds_root.glob("n*")):
        if not n_dir.is_dir():
            continue
        n_train = int(n_dir.name[1:])
        for r_dir in sorted(n_dir.glob("r*")):
            if not r_dir.is_dir():
                continue
            repeat = int(r_dir.name[1:])
            summary_path = r_dir / "run_summary.json"
            metrics_path = r_dir / "training_metrics.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text())
            test_eval = summary.get("test_eval", {})
            train_final = summary.get("training_final", {})
            runs.append(
                RunRecord(
                    dataset=dataset,
                    n_train=n_train,
                    repeat=repeat,
                    seed=int(summary.get("seed", 42 + repeat * 1000)),
                    run_dir=r_dir,
                    test_energy_mae=test_eval.get("energy_mae_kcal_mol"),
                    test_forces_mae=test_eval.get("forces_mae_kcal_mol"),
                    valid_energy_mae=train_final.get("valid_energy_mae"),
                    valid_loss=train_final.get("valid_loss"),
                    metrics_path=metrics_path if metrics_path.is_file() else None,
                )
            )
    return runs


def flag_run_outliers(
    runs: list[RunRecord],
    *,
    z_threshold: float = 1.5,
    ratio_threshold: float = 2.0,
) -> None:
    """Mark runs whose test energy MAE is extreme vs sibling repeats."""
    by_size: dict[int, list[RunRecord]] = defaultdict(list)
    for run in runs:
        by_size[run.n_train].append(run)

    for group in by_size.values():
        energies = [r.test_energy_mae for r in group if r.test_energy_mae is not None]
        if len(energies) < 2:
            continue
        median = float(np.median(energies))
        std = float(np.std(energies))
        for run in group:
            e = run.test_energy_mae
            if e is None:
                continue
            reasons: list[str] = []
            if median > 0 and e >= ratio_threshold * median:
                reasons.append(f"test_E>{ratio_threshold:g}x_median({median:.3f})")
            if std > 1e-9 and (e - median) / std >= z_threshold:
                reasons.append(f"test_E_z>={z_threshold:g}")
            if reasons:
                run.run_outlier = True
                run.run_outlier_reason = ", ".join(reasons)


def attach_training_spikes(
    runs: list[RunRecord],
    *,
    relative_factor: float,
    jump_factor: float,
) -> None:
    metric_keys = [
        ("valid_loss", "valid_loss", 1.0),
        ("valid_forces_mae", "valid_forces_mae", 1.0),
        ("valid_energy_mae", "valid_energy_mae", 10.0),
    ]
    for run in runs:
        if run.metrics_path is None:
            continue
        metrics = json.loads(run.metrics_path.read_text())
        epochs = metrics.get("epochs", [])
        for json_key, label, rel_factor in metric_keys:
            if json_key not in metrics:
                continue
            run.spikes.extend(
                detect_spikes(
                    epochs,
                    metrics[json_key],
                    metric=label,
                    relative_factor=relative_factor * rel_factor,
                    jump_factor=jump_factor,
                )
            )


@lru_cache(maxsize=None)
def _full_permutation(seed: int, n_total: int) -> tuple[int, ...]:
    key = jax.random.PRNGKey(seed)
    return tuple(np.asarray(jax.random.choice(key, n_total, shape=(n_total,), replace=False)).tolist())


def warmup_seed_permutations(seeds: set[int], n_total: int, *, verbose: bool = True) -> None:
    for i, seed in enumerate(sorted(seeds), start=1):
        if verbose:
            print(f"  seed {seed} ({i}/{len(seeds)})...", flush=True)
        _full_permutation(seed, n_total)


def reproduce_train_indices(seed: int, n_train: int, n_valid: int, n_total: int) -> np.ndarray:
    del n_valid  # same permutation prefix as physnet-train
    perm = np.asarray(_full_permutation(seed, n_total), dtype=np.int64)
    return perm[:n_train]


def score_npz_samples(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    energy = np.asarray(data["E"], dtype=float)
    forces = np.asarray(data["F"], dtype=float)
    natoms = np.asarray(data["N"], dtype=float)

    f_norm = np.linalg.norm(forces, axis=2)
    fmax = np.max(f_norm, axis=1)
    fmean = np.mean(f_norm, axis=1)
    e_per_atom = energy / np.maximum(natoms, 1.0)

    def zscore(x: np.ndarray) -> np.ndarray:
        med = float(np.median(x))
        std = float(np.std(x))
        if std <= 1e-12:
            return np.zeros_like(x)
        return (x - med) / std

    scores = {
        "energy_z": zscore(energy),
        "energy_per_atom_z": zscore(e_per_atom),
        "fmax_z": zscore(fmax),
        "fmean_z": zscore(fmean),
        "fmax": fmax,
        "abs_energy_z": np.abs(zscore(energy)),
        "composite": np.abs(zscore(energy)) + zscore(fmax) + 0.5 * np.abs(zscore(e_per_atom)),
    }
    if "Q" in data:
        scores["charge_abs_z"] = np.abs(zscore(np.asarray(data["Q"], dtype=float)))
        scores["composite"] = scores["composite"] + 0.25 * scores["charge_abs_z"]
    return scores


def analyze_seed_groups(runs: list[RunRecord]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, list[RunRecord]] = defaultdict(list)
    for run in runs:
        by_seed[run.seed].append(run)

    out: dict[int, dict[str, Any]] = {}
    for seed, seed_runs in sorted(by_seed.items()):
        energies = [r.test_energy_mae for r in seed_runs if r.test_energy_mae is not None]
        out[seed] = {
            "n_runs": len(seed_runs),
            "outlier_runs": sum(1 for r in seed_runs if r.run_outlier),
            "spike_epochs": sum(len(r.spikes) for r in seed_runs),
            "test_energy_mae_mean": float(np.mean(energies)) if energies else None,
            "test_energy_mae_max": float(np.max(energies)) if energies else None,
            "sizes": sorted({r.n_train for r in seed_runs}),
            "runs": [f"n{r.n_train}/r{r.repeat}" for r in sorted(seed_runs, key=lambda x: (x.n_train, x.repeat))],
        }
    return out


def top_global_npz_outliers(train_npz: Path, top_k: int = 10) -> list[dict[str, Any]]:
    data = np.load(train_npz, allow_pickle=True)
    scores = score_npz_samples(train_npz)
    order = np.argsort(scores["composite"])[::-1][:top_k]
    rows = []
    for idx in order:
        rows.append(
            {
                "index": int(idx),
                "composite_score": float(scores["composite"][idx]),
                "energy": float(data["E"][idx]),
                "fmax": float(scores["fmax"][idx]),
                "natoms": int(data["N"][idx]),
            }
        )
    return rows


def enrich_bad_samples(
    runs: list[RunRecord],
    *,
    train_npz: Path,
    n_valid_ratio_numerator: int = 300,
    n_valid_ratio_denominator: int = 8000,
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """Find NPZ indices enriched in outlier runs vs sibling repeats at the same n_train."""
    data = np.load(train_npz, allow_pickle=True)
    n_total = len(data["E"])
    scores = score_npz_samples(train_npz)

    by_size: dict[int, list[RunRecord]] = defaultdict(list)
    for run in runs:
        by_size[run.n_train].append(run)

    suspect_counts: Counter[int] = Counter()
    suspect_seeds: dict[int, set[int]] = defaultdict(set)

    for n_train, group in by_size.items():
        outliers = [r for r in group if r.run_outlier]
        normals = [r for r in group if not r.run_outlier]
        if not outliers or not normals:
            continue
        n_valid = n_train * n_valid_ratio_numerator // n_valid_ratio_denominator
        normal_union: set[int] = set()
        for run in normals:
            normal_union.update(
                reproduce_train_indices(run.seed, n_train, n_valid, n_total).tolist()
            )
        for run in outliers:
            outlier_idx = set(
                reproduce_train_indices(run.seed, n_train, n_valid, n_total).tolist()
            )
            exclusive = outlier_idx - normal_union
            suspect_counts.update(exclusive)
            for idx in exclusive:
                suspect_seeds[idx].add(run.seed)

    if not suspect_counts:
        return []

    candidates: list[dict[str, Any]] = []
    for idx, count in suspect_counts.items():
        candidates.append(
            {
                "index": int(idx),
                "exclusive_to_outlier_runs": int(count),
                "composite_score": float(scores["composite"][idx]),
                "energy": float(data["E"][idx]),
                "fmax": float(scores["fmax"][idx]),
                "natoms": int(data["N"][idx]),
                "outlier_seeds": sorted(suspect_seeds[idx]),
            }
        )

    candidates.sort(
        key=lambda row: (row["exclusive_to_outlier_runs"], row["composite_score"]),
        reverse=True,
    )
    return candidates[:top_k]


def print_report(
    runs: list[RunRecord],
    seed_summary: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
    global_outliers: list[dict[str, Any]],
    *,
    top_spikes: int,
) -> None:
    print(f"\n=== Run outliers ({sum(r.run_outlier for r in runs)}/{len(runs)}) ===")
    print(f"{'run':<16} {'seed':>6} {'test_E':>8} {'valid_E':>8} {'spikes':>6}  reason")
    print("-" * 72)
    for run in sorted(runs, key=lambda r: (r.run_outlier, r.test_energy_mae or -1), reverse=True):
        if not run.run_outlier:
            continue
        test_e = f"{run.test_energy_mae:.3f}" if run.test_energy_mae is not None else "-"
        valid_e = f"{run.valid_energy_mae:.3f}" if run.valid_energy_mae is not None else "-"
        print(
            f"n{run.n_train}/r{run.repeat:<8} {run.seed:6d} {test_e:>8} {valid_e:>8} "
            f"{len(run.spikes):6d}  {run.run_outlier_reason}"
        )

    print("\n=== Seed summary ===")
    print(f"{'seed':>6} {'runs':>5} {'bad':>4} {'spikes':>6} {'mean_E':>8} {'max_E':>8}  sizes")
    print("-" * 72)
    for seed, info in seed_summary.items():
        mean_e = info["test_energy_mae_mean"]
        max_e = info["test_energy_mae_max"]
        print(
            f"{seed:6d} {info['n_runs']:5d} {info['outlier_runs']:4d} "
            f"{info['spike_epochs']:6d} "
            f"{mean_e:8.3f} {max_e:8.3f}  {info['sizes']}"
        )

    if candidates:
        print("\n=== Suspect NPZ samples (exclusive to outlier train splits) ===")
        print(
            f"{'idx':>6} {'excl':>5} {'composite':>9} "
            f"{'E':>10} {'fmax':>8}  outlier_seeds"
        )
        print("-" * 72)
        for row in candidates:
            print(
                f"{row['index']:6d} "
                f"{row['exclusive_to_outlier_runs']:5d} "
                f"{row['composite_score']:9.2f} "
                f"{row['energy']:10.4f} {row['fmax']:8.3f}  {row['outlier_seeds']}"
            )

    if global_outliers:
        print("\n=== Global NPZ outliers (by energy/force score) ===")
        print(f"{'idx':>6} {'composite':>9} {'E':>10} {'fmax':>8}")
        print("-" * 40)
        for row in global_outliers:
            print(
                f"{row['index']:6d} {row['composite_score']:9.2f} "
                f"{row['energy']:10.4f} {row['fmax']:8.3f}"
            )

    print("\n=== Largest training spikes ===")
    all_spikes: list[tuple[RunRecord, Spike]] = []
    for run in runs:
        for spike in run.spikes:
            all_spikes.append((run, spike))
    all_spikes.sort(key=lambda item: item[1].value, reverse=True)
    print(f"{'run':<16} {'seed':>6} {'metric':<18} {'epoch':>6} {'value':>12}  reasons")
    print("-" * 88)
    for run, spike in all_spikes[:top_spikes]:
        print(
            f"n{run.n_train}/r{run.repeat:<8} {run.seed:6d} {spike.metric:<18} "
            f"{spike.epoch:6d} {spike.value:12.4g}  {','.join(spike.reasons)}"
        )


def write_json_report(
    path: Path,
    runs: list[RunRecord],
    seed_summary: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
    global_outliers: list[dict[str, Any]],
) -> None:
    payload = {
        "runs": [
            {
                **asdict(run),
                "run_dir": str(run.run_dir),
                "metrics_path": str(run.metrics_path) if run.metrics_path else None,
                "spikes": [asdict(s) for s in run.spikes],
            }
            for run in runs
        ],
        "seed_summary": {str(seed): info for seed, info in seed_summary.items()},
        "suspect_samples": candidates,
        "global_npz_outliers": global_outliers,
    }
    path.write_text(json.dumps(payload, indent=2))


def run_diagnosis(
    *,
    eval_root: Path,
    dataset: str,
    train_npz: Path | None = None,
    json_out: Path | None = None,
    spike_relative_factor: float = 5.0,
    spike_jump_factor: float = 3.0,
    test_z_threshold: float = 1.5,
    test_ratio_threshold: float = 2.0,
    top_samples: int = 25,
    top_spikes: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run full diagnosis and return structured results."""
    runs = load_runs(eval_root.resolve(), dataset)
    if not runs:
        raise SystemExit(f"No runs found under {eval_root}/{dataset}")
    if verbose:
        print(f"Loaded {len(runs)} runs", flush=True)

    flag_run_outliers(
        runs,
        z_threshold=test_z_threshold,
        ratio_threshold=test_ratio_threshold,
    )
    attach_training_spikes(
        runs,
        relative_factor=spike_relative_factor,
        jump_factor=spike_jump_factor,
    )
    seed_summary = analyze_seed_groups(runs)

    candidates: list[dict[str, Any]] = []
    global_outliers: list[dict[str, Any]] = []
    if train_npz is not None:
        train_npz = train_npz.resolve()
        n_total = len(np.load(train_npz, allow_pickle=True)["E"])
        seeds = {r.seed for r in runs}
        if verbose:
            print(f"Reproducing train splits for {len(seeds)} seeds (n={n_total})...", flush=True)
        warmup_seed_permutations(seeds, n_total, verbose=verbose)
        candidates = enrich_bad_samples(runs, train_npz=train_npz, top_k=top_samples)
        global_outliers = top_global_npz_outliers(train_npz, top_k=min(10, top_samples))

    if verbose:
        print_report(runs, seed_summary, candidates, global_outliers, top_spikes=top_spikes)
    if json_out is not None:
        write_json_report(json_out.resolve(), runs, seed_summary, candidates, global_outliers)
        if verbose:
            print(f"\nWrote {json_out}")

    return {
        "runs": runs,
        "seed_summary": seed_summary,
        "suspect_samples": candidates,
        "global_npz_outliers": global_outliers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose learning-curve sweep outliers (seeds, spikes, NPZ samples).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  mmml diagnose-lc-outliers \\\n"
            "    --eval-root out/eval/learning_curve/e1000 \\\n"
            "    --dataset aco \\\n"
            "    --train-npz out/splits/aco/energies_forces_dipoles_train.npz\n"
        ),
    )
    parser.add_argument("--eval-root", type=Path, required=True, help="Sweep eval root (…/learning_curve/e1000)")
    parser.add_argument("--dataset", default="aco", help="Dataset subdir under eval-root (default: aco)")
    parser.add_argument("--train-npz", type=Path, default=None, help="Train NPZ for split reproduction + scoring")
    parser.add_argument("--json-out", type=Path, default=None, help="Write full JSON report")
    parser.add_argument("--spike-relative-factor", type=float, default=5.0)
    parser.add_argument("--spike-jump-factor", type=float, default=3.0)
    parser.add_argument("--test-z-threshold", type=float, default=1.5)
    parser.add_argument("--test-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--top-samples", type=int, default=25)
    parser.add_argument("--top-spikes", type=int, default=20)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress tables; still writes --json-out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_diagnosis(
        eval_root=args.eval_root,
        dataset=args.dataset,
        train_npz=args.train_npz,
        json_out=args.json_out,
        spike_relative_factor=args.spike_relative_factor,
        spike_jump_factor=args.spike_jump_factor,
        test_z_threshold=args.test_z_threshold,
        test_ratio_threshold=args.test_ratio_threshold,
        top_samples=args.top_samples,
        top_spikes=args.top_spikes,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
