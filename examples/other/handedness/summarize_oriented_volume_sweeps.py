"""Aggregate oriented-volume sweep CSVs: mean and std per (max_degree, pseudotensors).

Typical usage from this directory::

  python summarize_oriented_volume_sweeps.py oriented_volume_sweep_seed*.csv
  python summarize_oriented_volume_sweeps.py --glob 'oriented_volume_sweep_seed*.csv' --out summary.csv

If no paths are given, all ``oriented_volume_sweep_seed*.csv`` files next to this script are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _collect_paths(csv_files: list[str], glob_pattern: str | None) -> list[Path]:
  paths: list[Path] = []
  for p in csv_files:
    paths.append(Path(p).resolve())
  if glob_pattern:
    for p in Path().glob(glob_pattern):
      if p.is_file():
        paths.append(p.resolve())
  if not paths:
    default_dir = Path(__file__).resolve().parent
    paths = sorted(default_dir.glob("oriented_volume_sweep_seed*.csv"))
  seen: set[Path] = set()
  unique: list[Path] = []
  for p in paths:
    if p not in seen and p.is_file():
      seen.add(p)
      unique.append(p)
  return unique


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "csv_files",
      nargs="*",
      type=str,
      help="Sweep CSV paths (e.g. oriented_volume_sweep_seed42.csv)",
  )
  parser.add_argument(
      "--glob",
      dest="glob_pattern",
      default=None,
      help="Additional glob relative to the current working directory.",
  )
  parser.add_argument(
      "--out",
      type=str,
      default=None,
      help="Write summary table to this CSV path.",
  )
  args = parser.parse_args()
  paths = _collect_paths(args.csv_files, args.glob_pattern)
  if not paths:
    raise SystemExit("No CSV files found. Pass paths, use --glob, or run from the handedness examples directory.")

  frames = []
  for p in paths:
    df = pd.read_csv(p)
    df["_source_file"] = p.name
    frames.append(df)
  all_df = pd.concat(frames, ignore_index=True)

  required = {"max_degree", "include_pseudotensors", "valid_mae_last10_mean", "valid_mae_final", "seed"}
  missing = required - set(all_df.columns)
  if missing:
    raise SystemExit(f"CSV missing columns {sorted(missing)}; found {list(all_df.columns)}")

  all_df["include_pseudotensors"] = all_df["include_pseudotensors"].map(
      lambda x: x if isinstance(x, bool) else str(x).strip().lower() in ("1", "true", "yes")
  )

  g = all_df.groupby(["max_degree", "include_pseudotensors"], sort=True)
  summary = g.agg(
      n_runs=("seed", "count"),
      last10_mean=("valid_mae_last10_mean", "mean"),
      last10_std=("valid_mae_last10_mean", "std"),
      final_mean=("valid_mae_final", "mean"),
      final_std=("valid_mae_final", "std"),
  ).reset_index()

  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 120)
  print(f"Loaded {len(all_df)} rows from {len(paths)} file(s).\n")
  print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

  if args.out:
    out_path = Path(args.out).resolve()
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
  main()
