#!/usr/bin/env python3
"""Summarize QCML training run directories from saved epoch metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mmml-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"^epoch-(\d+)$")


def epoch_number(path: Path) -> int | None:
    match = EPOCH_RE.match(path.name)
    return int(match.group(1)) if match else None


def load_epoch_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.iterdir()):
        epoch = epoch_number(path)
        metrics_path = path / "metrics.json"
        if epoch is None or not metrics_path.exists():
            continue
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row.setdefault("epoch", epoch)
        row["checkpoint"] = str(path)
        rows.append(row)
    rows.sort(key=lambda item: int(item["epoch"]))
    return rows


def numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if key in {"epoch", "checkpoint"}:
                continue
            if isinstance(value, int | float) and np.isfinite(value):
                keys.add(key)
    return sorted(keys)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "checkpoint", *numeric_keys(rows)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        key
        for key in numeric_keys(rows)
        if key not in {"learning_rate", "gradient_clip_norm", "charge_weight", "huber_delta"}
        and not key.startswith("target_")
        and not key.startswith("outlier_threshold_")
    ]
    if not keys:
        return
    epochs = np.asarray([row["epoch"] for row in rows], dtype=np.int32)
    cols = 2
    rows_count = int(np.ceil(len(keys) / cols))
    figure, axes = plt.subplots(
        rows_count,
        cols,
        figsize=(10, max(3.0, 2.7 * rows_count)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, key in zip(axes.flat, keys, strict=False):
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        axis.plot(epochs, values, marker="o", linewidth=1.5, markersize=3)
        axis.set_title(key)
        axis.set_xlabel("epoch")
        axis.set_ylabel(key)
        if np.all(values > 0):
            axis.set_yscale("log")
    for axis in axes.flat[len(keys):]:
        axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def summarize_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    metrics = load_epoch_metrics(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics:
        write_csv(output_dir / "epoch_metrics.csv", metrics)
        plot_learning_curves(output_dir / "learning_curves.png", metrics)
    latest = metrics[-1] if metrics else None
    data_split = load_optional_json(run_dir / "data_split.json")
    split_counts = {
        key: len(value)
        for key, value in (data_split or {}).items()
        if isinstance(value, list)
    }
    summary = {
        "run_dir": str(run_dir),
        "num_checkpoints": len(metrics),
        "latest_checkpoint": latest["checkpoint"] if latest else None,
        "latest_epoch": int(latest["epoch"]) if latest else None,
        "latest_metrics": latest,
        "split_counts": split_counts,
        "has_shard_audit": (run_dir / "shard_audit.json").exists(),
        "has_target_scale": (run_dir / "target_scale.json").exists(),
        "has_target_rms": (run_dir / "target_rms.json").exists(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    summaries = []
    for run_dir in args.run_dirs:
        output_dir = args.output_dir or run_dir / "training_summary"
        if len(args.run_dirs) > 1 and args.output_dir is not None:
            output_dir = args.output_dir / run_dir.name
        summary = summarize_run(run_dir, output_dir)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote training summary to {output_dir}")

    if args.output_dir is not None and len(args.run_dirs) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
