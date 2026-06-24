#!/usr/bin/env python3
"""Compare MMML calculators against DCM MP2 monomer/dimer reference NPZ files.

The input NPZ must contain padded ``N, Z, R, E`` arrays. The split files written
under ``artifacts/dcm_mp2_round2`` already use the CHARMM-like atom order
``C,Cl,Cl,H,H`` per monomer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms


HARTREE_TO_EV = 27.211386245988
KCAL_MOL_TO_EV = 1.0 / 23.0605


def _energy_to_ev(values: np.ndarray, unit: str) -> np.ndarray:
    unit_l = unit.lower()
    if unit_l in {"ev", "eV".lower()}:
        return values.astype(np.float64)
    if unit_l in {"hartree", "ha"}:
        return values.astype(np.float64) * HARTREE_TO_EV
    if unit_l in {"kcal", "kcal/mol", "kcal_mol"}:
        return values.astype(np.float64) * KCAL_MOL_TO_EV
    raise ValueError(f"Unsupported energy unit: {unit}")


def _select_indices(n_frames: int, max_frames: int | None, stride: int, seed: int) -> np.ndarray:
    indices = np.arange(0, n_frames, int(stride), dtype=int)
    if max_frames is not None and len(indices) > int(max_frames):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=int(max_frames), replace=False))
    return indices


def _load_npz(path: Path, reference_energy_unit: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"N", "Z", "R", "E"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    return {
        "N": np.asarray(data["N"], dtype=int),
        "Z": np.asarray(data["Z"], dtype=np.int32),
        "R": np.asarray(data["R"], dtype=np.float64),
        "E_ref_raw": np.asarray(data["E"], dtype=np.float64),
        "E_ref_eV": _energy_to_ev(np.asarray(data["E"], dtype=np.float64), reference_energy_unit),
        "source_indices": np.asarray(data["source_indices"], dtype=int)
        if "source_indices" in data.files
        else np.arange(len(data["N"]), dtype=int),
    }


def _metrics(errors: np.ndarray) -> dict[str, float]:
    finite = np.asarray(errors[np.isfinite(errors)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "mae_eV": float("nan"), "rmse_eV": float("nan"), "bias_eV": float("nan")}
    return {
        "n": int(finite.size),
        "mae_eV": float(np.mean(np.abs(finite))),
        "rmse_eV": float(np.sqrt(np.mean(finite**2))),
        "bias_eV": float(np.mean(finite)),
    }


class CalculatorRunner:
    def __init__(self, name: str, checkpoint: Path | None, cutoff: float) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.cutoff = float(cutoff)
        self._calculator: Any | None = None
        self._hybrid_cache: dict[tuple[int, bool], Any] = {}

    def _checkpoint_calculator(self):
        if self.checkpoint is None:
            raise ValueError(f"calculator {self.name!r} requires --checkpoint")
        if self._calculator is None:
            from mmml.interfaces.calculators.checkpoint_loading import (
                create_calculator_from_checkpoint,
            )

            self._calculator = create_calculator_from_checkpoint(
                self.checkpoint,
                cutoff=self.cutoff,
            )
        return self._calculator

    def _hybrid_calculator(self, n_atoms: int, *, do_ml_dimer: bool):
        if self.checkpoint is None:
            raise ValueError(f"calculator {self.name!r} requires --checkpoint")
        n_monomers = int(n_atoms) // 5
        if n_monomers * 5 != int(n_atoms):
            raise ValueError(f"hybrid calculators expect DCM 5-atom monomers, got N={n_atoms}")
        key = (n_monomers, bool(do_ml_dimer))
        if key not in self._hybrid_cache:
            from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
            from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

            factory = setup_calculator(
                ATOMS_PER_MONOMER=[5] * n_monomers,
                N_MONOMERS=n_monomers,
                doML=True,
                doMM=False,
                doML_dimer=do_ml_dimer,
                model_restart_path=str(self.checkpoint),
                MAX_ATOMS_PER_SYSTEM=int(n_atoms),
                ml_sparse_dimers=False,
                verbose=False,
            )
            cutoff_params = CutoffParameters(
                ml_switch_width=0.01,
                mm_switch_on=self.cutoff,
                mm_switch_width=0.0,
            )
            calc, _spherical_fn, _get_update_fn = factory(
                atomic_numbers=np.ones(int(n_atoms), dtype=np.int32),
                atomic_positions=np.zeros((int(n_atoms), 3), dtype=np.float64),
                n_monomers=n_monomers,
                cutoff_params=cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=do_ml_dimer,
                backprop=False,
                debug=False,
                verbose=False,
            )
            self._hybrid_cache[key] = calc
        return self._hybrid_cache[key]

    def energy(self, numbers: np.ndarray, positions: np.ndarray) -> float:
        atoms = Atoms(numbers=numbers, positions=positions)
        if self.name == "checkpoint":
            atoms.calc = self._checkpoint_calculator()
        elif self.name == "hybrid-ml":
            atoms.calc = self._hybrid_calculator(len(numbers), do_ml_dimer=True)
        elif self.name == "hybrid-monomer":
            atoms.calc = self._hybrid_calculator(len(numbers), do_ml_dimer=False)
        else:
            raise ValueError(f"Unknown calculator: {self.name}")
        return float(atoms.get_potential_energy())


def _evaluate_file(
    path: Path,
    *,
    calculators: list[CalculatorRunner],
    reference_energy_unit: str,
    max_frames: int | None,
    stride: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _load_npz(path, reference_energy_unit)
    indices = _select_indices(len(data["N"]), max_frames, stride, seed)
    rows: list[dict[str, Any]] = []
    errors_by_calc = {calc.name: [] for calc in calculators}

    for local_idx in indices:
        n_atoms = int(data["N"][local_idx])
        numbers = data["Z"][local_idx, :n_atoms]
        positions = data["R"][local_idx, :n_atoms]
        ref_e = float(data["E_ref_eV"][local_idx])
        base = {
            "dataset": str(path),
            "local_index": int(local_idx),
            "source_index": int(data["source_indices"][local_idx]),
            "n_atoms": n_atoms,
            "reference_energy_raw": float(data["E_ref_raw"][local_idx]),
            "reference_energy_eV": ref_e,
        }
        for calc in calculators:
            row = dict(base)
            row["calculator"] = calc.name
            try:
                pred_e = calc.energy(numbers, positions)
                row["predicted_energy_eV"] = pred_e
                row["error_eV"] = pred_e - ref_e
                row["status"] = "ok"
                errors_by_calc[calc.name].append(row["error_eV"])
            except Exception as exc:
                row["predicted_energy_eV"] = float("nan")
                row["error_eV"] = float("nan")
                row["status"] = f"error: {type(exc).__name__}: {exc}"
            rows.append(row)

    summary = {
        "dataset": str(path),
        "n_available": int(len(data["N"])),
        "n_evaluated": int(len(indices)),
        "reference_energy_unit": reference_energy_unit,
        "metrics": {
            name: _metrics(np.asarray(values, dtype=np.float64))
            for name, values in errors_by_calc.items()
        },
    }
    return rows, summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        required=True,
        help="One or more split NPZ files to evaluate.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint path")
    parser.add_argument(
        "--calculators",
        nargs="+",
        default=["checkpoint"],
        choices=["checkpoint", "hybrid-ml", "hybrid-monomer"],
        help="Calculator modes to evaluate.",
    )
    parser.add_argument(
        "--reference-energy-unit",
        default="hartree",
        choices=["hartree", "ha", "ev", "kcal", "kcal/mol", "kcal_mol"],
    )
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument(
        "--jax-platform",
        choices=["cpu", "gpu", "tpu"],
        default=None,
        help="Set JAX_PLATFORM_NAME before loading calculators.",
    )
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dcm_mp2_round2/calculator_compare"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.jax_platform:
        os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calculators = [
        CalculatorRunner(name=name, checkpoint=args.checkpoint, cutoff=args.cutoff)
        for name in args.calculators
    ]

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in args.data:
        rows, summary = _evaluate_file(
            path,
            calculators=calculators,
            reference_energy_unit=args.reference_energy_unit,
            max_frames=args.max_frames,
            stride=args.stride,
            seed=args.seed,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    csv_path = args.output_dir / "energy_comparison.csv"
    fieldnames = [
        "dataset",
        "local_index",
        "source_index",
        "n_atoms",
        "calculator",
        "reference_energy_raw",
        "reference_energy_eV",
        "predicted_energy_eV",
        "error_eV",
        "status",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": summaries}, indent=2) + "\n")
    print(f"Wrote rows:    {csv_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    for summary in summaries:
        print(f"\n{summary['dataset']}  n={summary['n_evaluated']}", flush=True)
        for name, metrics in summary["metrics"].items():
            print(
                f"  {name:15s} n_ok={metrics['n']:5d} "
                f"MAE={metrics['mae_eV']:.6g} eV "
                f"RMSE={metrics['rmse_eV']:.6g} eV "
                f"bias={metrics['bias_eV']:.6g} eV",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
