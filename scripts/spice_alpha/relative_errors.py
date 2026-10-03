#!/usr/bin/env python
"""Validation errors relative to the spread of the validation labels.

Reads efield-train ``history.jsonl`` files and reports, per epoch,
MAE / MAD(labels) for forces, dipoles and polarizability, plus the polar
RMSE / sigma. 1.0 means no better than predicting the dataset mean.
Forces are taken per component over real atoms (Z > 0), in kcal/mol/Å.

Usage:
  python scripts/spice_alpha/relative_errors.py VALID_NPZ CKPT_DIR [CKPT_DIR ...] [--all]
"""

import argparse
import json
from pathlib import Path

import numpy as np

EV2KCAL = 23.060548


def mad(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a - a.mean(axis=0))))


def label_spread(valid_npz: Path) -> dict:
    v = np.load(valid_npz)
    real = v["Z"] > 0
    polar = v["polar"].reshape(len(v["polar"]), -1)
    return {
        "F": mad(v["F"][real] * EV2KCAL),
        "D": mad(v["D"]),
        "P": mad(polar),
        "P_sigma": float(np.std(polar - polar.mean(axis=0))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("valid_npz", type=Path)
    ap.add_argument("ckpt_dirs", type=Path, nargs="+", help="dirs containing history.jsonl")
    ap.add_argument("--all", action="store_true", help="print every epoch (default: first 2 + last 3)")
    args = ap.parse_args()

    s = label_spread(args.valid_npz)
    print(f"valid label spread: MAD_F={s['F']:.2f} kcal/mol/Å  MAD_D={s['D']:.3f}  "
          f"MAD_P={s['P']:.2f} Bohr³ (σ_P={s['P_sigma']:.2f})\n")
    print(f"{'run':32s}{'ep':>4s}{'best':>5s}  {'F/MAD':>6s} {'D/MAD':>6s} {'P/MAD':>6s} {'P rmse/σ':>8s}")
    for d in args.ckpt_dirs:
        hist = d / "history.jsonl"
        if not hist.is_file():
            print(f"{d.name:32s}  (no history.jsonl yet)")
            continue
        rows = [json.loads(line) for line in hist.read_text().splitlines() if line.strip()]
        if not args.all and len(rows) > 5:
            rows = rows[:2] + rows[-3:]
        for h in rows:
            p = h["valid_polar_mae_bohr3"]
            # valid_polar_mse is logged as mean(0.5 * diff**2), hence the factor 2 below.
            # Corrupt polar labels (|alpha| ~ 1e11) make the polar metrics meaningless.
            ok = p < 1e4
            prel = f"{p / s['P']:6.3f}" if ok else "   bad"
            prms = f"{np.sqrt(2 * h['valid_polar_mse_bohr6']) / s['P_sigma']:8.3f}" if ok else "     bad"
            print(f"{d.name:32s}{h['epoch']:4d}{h['best_epoch']:5d}  "
                  f"{h['valid_forces_mae_kcal'] / s['F']:6.3f} {h['valid_dipole_mae'] / s['D']:6.3f} {prel} {prms}")


if __name__ == "__main__":
    main()
