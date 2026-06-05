#!/usr/bin/env python3
"""Parse PyCHARMM DYNA> lines and summarize NVE energy conservation / smoothness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DYNA_LINE = re.compile(
    r"^DYNA>\s+(\d+)\s+([\d.]+)\s+"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)


def parse_dyna_lines(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        m = _DYNA_LINE.match(line.strip())
        if not m:
            continue
        rows.append(
            {
                "step": float(m.group(1)),
                "time_ps": float(m.group(2)),
                "total_energy_kcal": float(m.group(3)),
                "kinetic_energy_kcal": float(m.group(4)),
                "potential_energy_kcal": float(m.group(5)),
                "temperature_K": float(m.group(6)),
            }
        )
    return rows


def summarize_nve_energy(rows: list[dict[str, float]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {
            "n_frames": len(rows),
            "status": "fail",
            "notes": "fewer than 2 DYNA> frames",
        }

    etot = np.array([r["total_energy_kcal"] for r in rows], dtype=np.float64)
    epot = np.array([r["potential_energy_kcal"] for r in rows], dtype=np.float64)
    ekin = np.array([r["kinetic_energy_kcal"] for r in rows], dtype=np.float64)
    temp = np.array([r["temperature_K"] for r in rows], dtype=np.float64)
    time_ps = np.array([r["time_ps"] for r in rows], dtype=np.float64)

    step_delta = np.abs(np.diff(etot))
    pot_delta = np.abs(np.diff(epot))

    duration_ps = float(time_ps[-1] - time_ps[0]) if time_ps.size >= 2 else 0.0
    etot_drift = float(etot[-1] - etot[0])
    epot_drift = float(epot[-1] - epot[0])

    out: dict[str, Any] = {
        "n_frames": int(len(rows)),
        "duration_ps": duration_ps,
        "etot_mean_kcal": float(np.mean(etot)),
        "etot_std_kcal": float(np.std(etot)),
        "etot_min_kcal": float(np.min(etot)),
        "etot_max_kcal": float(np.max(etot)),
        "etot_drift_kcal": etot_drift,
        "etot_drift_per_ps_kcal": (
            float(etot_drift / duration_ps) if duration_ps > 0 else float("nan")
        ),
        "max_abs_etot_step_delta_kcal": float(np.max(step_delta)),
        "rms_etot_step_delta_kcal": float(np.sqrt(np.mean(step_delta**2))),
        "epot_std_kcal": float(np.std(epot)),
        "epot_drift_kcal": epot_drift,
        "max_abs_epot_step_delta_kcal": float(np.max(pot_delta)),
        "rms_epot_step_delta_kcal": float(np.sqrt(np.mean(pot_delta**2))),
        "ekin_mean_kcal": float(np.mean(ekin)),
        "temp_mean_K": float(np.mean(temp)),
        "temp_std_K": float(np.std(temp)),
        "smoothness_score": float(
            np.std(etot) + np.max(step_delta) + 0.1 * abs(etot_drift)
        ),
        "status": "pass",
        "notes": "",
    }
    return out


def analyze_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_dyna_lines(text)
    summary = summarize_nve_energy(rows)
    summary["log_path"] = str(log_path)
    if summary.get("n_frames", 0) >= 2:
        summary["status"] = "pass"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="stdout.log from NVE run")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="JSON metrics output path",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Optional NPZ with time series arrays",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"log not found: {args.log}")

    text = args.log.read_text(encoding="utf-8", errors="replace")
    rows = parse_dyna_lines(text)
    summary = summarize_nve_energy(rows)
    summary["log_path"] = str(args.log.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.npz is not None:
        if rows:
            series = {
                k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]
            }
            np.savez_compressed(args.npz, **series)
        else:
            np.savez_compressed(args.npz, n_frames=np.array(0))

    print(json.dumps({k: v for k, v in summary.items() if k != "notes"}, indent=2))
    print(f"Wrote {args.output}")
    return 0 if summary.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
