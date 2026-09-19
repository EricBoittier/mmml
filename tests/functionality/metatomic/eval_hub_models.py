#!/usr/bin/env python3
"""Evaluate exported PET-MAD / UPET metatomic models through MMML (no CHARMM/MD)."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.interfaces.calculators.ase_fragment_hybrid import (
    evaluate_fragment_hybrid,
    evaluate_whole_system,
)
from mmml.interfaces.calculators.metatomic import (
    have_metatomic,
    is_metatomic_checkpoint,
    load_metatomic_calculator,
)
from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import (
    MetatomicMlpotCalculator,
    build_metatomic_mlpot_model,
)

DEFAULT_MODELS = (
    "pet-mad-s-v1.0.2.pt",
    "pet-mad-xs-v1.5.0.pt",
    "pet-mols-s-v1.0.0.pt",
)


def water_dimer(*, separation_A: float = 2.8) -> Atoms:
    a = molecule("H2O")
    b = molecule("H2O")
    b.positions += np.array([separation_A, 0.0, 0.0], dtype=np.float64)
    return a + b


def summarize_forces(forces: np.ndarray) -> dict[str, float]:
    arr = np.asarray(forces, dtype=np.float64)
    return {
        "max_abs_eV_A": float(np.max(np.abs(arr))),
        "rms_eV_A": float(np.sqrt(np.mean(arr * arr))),
    }


def eval_one(path: Path) -> dict:
    rec: dict = {"model": path.name, "path": str(path), "ok": False}
    rec["is_metatomic_checkpoint"] = bool(is_metatomic_checkpoint(path))
    rec["size_bytes"] = int(path.stat().st_size)
    try:
        calc = load_metatomic_calculator(path, device="cpu")
        rec["calculator_type"] = type(calc).__name__
        rec["calculator_module"] = type(calc).__module__

        mono = molecule("H2O")
        whole_m = evaluate_whole_system(
            calc, mono.get_atomic_numbers(), mono.get_positions()
        )
        rec["monomer_energy_eV"] = float(whole_m.energy_ev)
        rec["monomer_forces"] = summarize_forces(whole_m.forces_ev_per_angstrom)

        dimer = water_dimer()
        z = dimer.get_atomic_numbers()
        r = dimer.get_positions()
        whole_d = evaluate_whole_system(calc, z, r)
        frag = evaluate_fragment_hybrid(
            calc, z, r, [3, 3], do_ml=True, do_ml_dimer=True
        )
        rec["dimer_whole_energy_eV"] = float(whole_d.energy_ev)
        rec["dimer_fragment_energy_eV"] = float(frag.energy_ev)
        rec["dimer_interaction_eV"] = float(frag.energy_ev - 2.0 * whole_m.energy_ev)
        rec["dimer_fragment_n_dimers"] = int(frag.n_dimers_evaluated)
        rec["charmm_energy_kcal_expected"] = float(frag.energy_ev * EV_TO_KCAL_MOL)

        model = build_metatomic_mlpot_model(
            path, z, [3, 3], 2, calculator=calc, do_mm=False, eval_mode="fragments"
        )
        charmm = model.get_pycharmm_calculator(ml_atom_indices=list(range(6)))
        assert isinstance(charmm, MetatomicMlpotCalculator)
        dx = [0.0] * 6
        dy = [0.0] * 6
        dz = [0.0] * 6
        e_kcal = charmm.calculate_charmm(
            6,
            0,
            0,
            None,
            r[:, 0].tolist(),
            r[:, 1].tolist(),
            r[:, 2].tolist(),
            dx,
            dy,
            dz,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        rec["charmm_energy_kcal"] = float(e_kcal)
        rec["charmm_energy_match"] = bool(
            np.isclose(e_kcal, rec["charmm_energy_kcal_expected"], rtol=1e-6, atol=1e-6)
        )
        rec["ok"] = True
        rec["finite"] = bool(np.isfinite(whole_m.energy_ev) and np.isfinite(e_kcal))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/tmp/mmml-metatomic-models"),
        help="Directory of exported .pt files",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Filenames under --model-dir",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = {"have_metatomic": have_metatomic(), "results": []}
    if not have_metatomic():
        print("metatomic extra is not installed (uv sync --extra metatomic)", flush=True)
        return 2
    for name in args.models:
        path = (args.model_dir / name).expanduser()
        if not path.is_file():
            rec = {"model": name, "path": str(path), "ok": False, "error": "missing file"}
        else:
            rec = eval_one(path)
        out["results"].append(rec)
        print(f"[{'OK' if rec.get('ok') else 'FAIL'}] {name}", flush=True)
        if rec.get("error"):
            print(rec["error"], flush=True)
    text = json.dumps(out, indent=2, default=str)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text)
    n_ok = sum(1 for r in out["results"] if r.get("ok") and r.get("finite"))
    return 0 if n_ok == len(args.models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
