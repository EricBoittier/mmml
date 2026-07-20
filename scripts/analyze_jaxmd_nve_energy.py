#!/usr/bin/env python3
"""Summarize JAX-MD NVE total-energy conservation from an mmml HDF5 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

EV_TO_KCAL_MOL = 23.06054783061903


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as handle:
        time_ps = np.asarray(handle["time_ps"], dtype=np.float64)
        total_ev = np.asarray(handle["total_energy"], dtype=np.float64)
        potential_ev = np.asarray(handle["potential_energy"], dtype=np.float64)
        kinetic_ev = np.asarray(handle["kinetic_energy"], dtype=np.float64)
        temperature = np.asarray(handle["temperature"], dtype=np.float64)
        n_atoms = int(handle.attrs["n_atoms"])

    if total_ev.size < 2 or not all(
        np.all(np.isfinite(x))
        for x in (time_ps, total_ev, potential_ev, kinetic_ev, temperature)
    ):
        raise RuntimeError("NVE trace needs at least two finite frames")

    total_kcal = total_ev * EV_TO_KCAL_MOL
    duration_ps = float(time_ps[-1] - time_ps[0])
    drift_kcal = float(total_kcal[-1] - total_kcal[0])
    slope_kcal_ps = float(np.polyfit(time_ps, total_kcal, 1)[0])
    step_delta = np.diff(total_kcal)
    report = {
        "backend": "jaxmd",
        "n_atoms": n_atoms,
        "n_frames": int(total_ev.size),
        "duration_ps": duration_ps,
        "total_energy_initial_kcal_mol": float(total_kcal[0]),
        "total_energy_final_kcal_mol": float(total_kcal[-1]),
        "endpoint_drift_kcal_mol": drift_kcal,
        "linear_drift_kcal_mol_per_ps": slope_kcal_ps,
        "endpoint_drift_kcal_mol_per_atom_ps": (
            drift_kcal / (n_atoms * duration_ps) if duration_ps > 0.0 else None
        ),
        "std_total_energy_kcal_mol": float(np.std(total_kcal)),
        "max_abs_deviation_from_initial_kcal_mol": float(
            np.max(np.abs(total_kcal - total_kcal[0]))
        ),
        "max_abs_step_delta_kcal_mol": float(np.max(np.abs(step_delta))),
        "rms_step_delta_kcal_mol": float(np.sqrt(np.mean(step_delta**2))),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_min_K": float(np.min(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "finite": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    from mmml.utils.rich_report import print_colored_json

    print_colored_json(report)


if __name__ == "__main__":
    main()
