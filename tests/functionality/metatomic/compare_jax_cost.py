#!/usr/bin/env python3
"""CHARMM-free CPU cost comparison: metatomic PET vs bundled JAX PhysNet.

Times energy+forces after a warmup on the same geometries. Does not run MD.

Example::

    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \\
      uv run python tests/functionality/metatomic/compare_jax_cost.py \\
      --out /opt/cursor/artifacts/metatomic_vs_physnet_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

from mmml.interfaces.calculators.ase_fragment_hybrid import (
    evaluate_fragment_hybrid,
    evaluate_whole_system,
)

REPO = Path(__file__).resolve().parents[3]
PHYSNET_CKPT = REPO / "examples/ckpts_json/DESdimers_params.json"
SPOOKY_CKPT = REPO / "examples/ckpts_json/spooky_epoch-0004-chunk-000040-step-00584747.json"
ACO_PDB = REPO / "mmml/generate/sample/pdb/aco_monomer.pdb"
DEFAULT_METATOMIC = Path("/tmp/mmml-metatomic-models")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def water_dimer(*, separation_A: float = 2.8) -> Atoms:
    a = molecule("H2O")
    b = molecule("H2O")
    b.positions += np.array([separation_A, 0.0, 0.0], dtype=np.float64)
    return a + b


def acetone_monomer() -> Atoms:
    zs: list[int] = []
    pos: list[list[float]] = []
    for line in ACO_PDB.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        zs.append(8 if name.startswith("O") else 6 if name.startswith("C") else 1)
        pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return Atoms(numbers=np.asarray(zs, dtype=np.int32), positions=np.asarray(pos, dtype=np.float64))


def acetone_dimer(*, spacing_A: float = 5.0) -> Atoms:
    a = acetone_monomer()
    b = acetone_monomer()
    b.positions += np.array([spacing_A, 0.0, 0.0], dtype=np.float64)
    return a + b


def count_leaves(obj) -> int:
    if isinstance(obj, dict):
        return sum(count_leaves(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            arr = np.asarray(obj)
            return int(arr.size) if arr.dtype != object else sum(count_leaves(v) for v in obj)
        return sum(count_leaves(v) for v in obj)
    try:
        return int(np.asarray(obj).size)
    except Exception:
        return 0


def time_calls(fn, *, warmup: int, repeats: int) -> dict:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    last = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "samples_s": [float(x) for x in arr],
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "std_s": float(np.std(arr, ddof=0)),
        "last": last,
    }


def summarize_eval(energy_ev: float, forces: np.ndarray) -> dict:
    f = np.asarray(forces, dtype=np.float64)
    return {
        "energy_eV": float(energy_ev),
        "force_max_abs_eV_A": float(np.max(np.abs(f))),
        "force_rms_eV_A": float(np.sqrt(np.mean(f * f))),
        "finite": bool(np.isfinite(energy_ev) and np.all(np.isfinite(f))),
    }


def load_physnet_ase(n_atoms: int, ckpt: Path):
    from mmml.cli.base import load_physnet_params_and_ef_model
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc

    params, model = load_physnet_params_and_ef_model(ckpt, natoms=int(n_atoms))
    template = Atoms(numbers=[1] * int(n_atoms), positions=np.zeros((int(n_atoms), 3)))
    calc = get_ase_calc(params, model, template)
    n_params = 0
    try:
        import jax

        n_params = int(sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params)))
    except Exception:
        n_params = count_leaves(params)
    return calc, {
        "checkpoint": str(ckpt),
        "size_bytes": int(ckpt.stat().st_size),
        "n_params": n_params,
        "natoms_pad": int(n_atoms),
        "arch": {
            "features": int(getattr(model, "features", -1)),
            "max_degree": int(getattr(model, "max_degree", -1)),
            "num_iterations": int(getattr(model, "num_iterations", -1)),
            "num_basis_functions": int(getattr(model, "num_basis_functions", -1)),
            "cutoff": float(getattr(model, "cutoff", float("nan"))),
            "zbl": bool(getattr(model, "zbl", False)),
            "model_type": type(model).__name__,
        },
    }


def load_metatomic(path: Path):
    from mmml.interfaces.calculators.metatomic import load_metatomic_calculator

    calc = load_metatomic_calculator(path, device=os.environ.get("MMML_METATOMIC_DEVICE", "cpu"))
    n_params = None
    try:
        import torch

        scripted = torch.jit.load(str(path), map_location="cpu")
        n_params = int(sum(p.numel() for p in scripted.parameters()))
    except Exception:
        n_params = None
    return calc, {
        "checkpoint": str(path),
        "size_bytes": int(path.stat().st_size),
        "n_params": n_params,
        "calculator_type": type(calc).__name__,
        "calculator_module": type(calc).__module__,
    }


def _nudge(positions: np.ndarray, seed: int, *, sigma: float = 1.0e-4) -> np.ndarray:
    """Break ASE / model caches without leaving the original geometry."""
    rng = np.random.default_rng(int(seed))
    return np.asarray(positions, dtype=np.float64) + rng.normal(0.0, sigma, size=positions.shape)


def _reset_calculator(calc) -> None:
    reset = getattr(calc, "reset", None)
    if callable(reset):
        reset()
    else:
        calc.results = {}
        calc.atoms = None


def bench_calculator(label: str, calc, atoms: Atoms, *, atoms_per_monomer: list[int], warmup: int, repeats: int) -> dict:
    z = atoms.get_atomic_numbers()
    r0 = np.asarray(atoms.get_positions(), dtype=np.float64)
    rec: dict = {"label": label, "n_atoms": int(len(atoms)), "ok": False}
    call_i = {"n": 0}

    def _next_pos() -> np.ndarray:
        call_i["n"] += 1
        return _nudge(r0, call_i["n"])

    def _whole():
        _reset_calculator(calc)
        out = evaluate_whole_system(calc, z, _next_pos())
        return summarize_eval(out.energy_ev, out.forces_ev_per_angstrom)

    def _frag():
        _reset_calculator(calc)
        out = evaluate_fragment_hybrid(
            calc, z, _next_pos(), atoms_per_monomer, do_ml=True, do_ml_dimer=True
        )
        return summarize_eval(out.energy_ev, out.forces_ev_per_angstrom)

    t_load0 = time.perf_counter()
    t_cold = time.perf_counter()
    cold = _whole()
    rec["cold_whole_s"] = float(time.perf_counter() - t_cold)
    rec["cold_whole_summary"] = cold
    rec["whole"] = time_calls(_whole, warmup=warmup, repeats=repeats)
    rec["fragments"] = time_calls(_frag, warmup=max(1, warmup // 2), repeats=max(3, repeats // 2))
    rec["bench_wall_s"] = float(time.perf_counter() - t_load0)
    rec["whole_summary"] = rec["whole"].pop("last")
    rec["fragments_summary"] = rec["fragments"].pop("last")
    rec["fragment_over_whole"] = float(rec["fragments"]["median_s"] / rec["whole"]["median_s"])
    rec["ok"] = bool(rec["whole_summary"]["finite"] and rec["fragments_summary"]["finite"])
    rec["cache_bust"] = "reset+1e-4A_jitter"
    return rec


def host_info() -> dict:
    return {
        "cpu_model": _cpu_model(),
        "nproc": int(os.cpu_count() or 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
        "MMML_METATOMIC_DEVICE": os.environ.get("MMML_METATOMIC_DEVICE", "cpu"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("metatomic_vs_physnet_cost.json"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_METATOMIC)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["pet-mad-xs-v1.5.0.pt", "pet-mad-s-v1.0.2.pt", "pet-mols-s-v1.0.0.pt"],
    )
    parser.add_argument(
        "--include-spooky",
        action="store_true",
        help="Also try the bundled SpookyNet JSON (expected to fail as a PhysNet load).",
    )
    args = parser.parse_args()

    systems = {
        "water_dimer": (water_dimer(), [3, 3]),
        "acetone_dimer": (acetone_dimer(spacing_A=5.0), [10, 10]),
    }

    report: dict = {
        "host": host_info(),
        "physnet_ckpt": str(PHYSNET_CKPT),
        "spooky_ckpt": str(SPOOKY_CKPT) if SPOOKY_CKPT.is_file() else None,
        "systems": {
            name: {
                "n_atoms": int(len(atoms)),
                "formula": atoms.get_chemical_formula(),
                "atoms_per_monomer": per,
            }
            for name, (atoms, per) in systems.items()
        },
        "results": [],
        "notes": [
            "CPU-only; no CHARMM / no MD.",
            "whole = one energy+forces eval on the dimer (ASE cache reset + 1e-4 A jitter each sample).",
            "fragments = MMML ML/MM USER: E(A)+E(B)+s*(E(AB)-E(A)-E(B)) = 3 sequential model evals.",
            "Production PhysNet MLpot batches those fragments in one jitted apply; this script uses sequential ASE evals for both backends so the model-forward cost is comparable.",
            "PET-MAD/PET-MOLS are universal PET TorchScript models; DESdimers PhysNet is a tiny MPNN (features=32, L=1, 2 iterations, 16 RBF, 6 A cutoff, ZBL).",
            "SpookyNet JSON is bundled but is not a drop-in PhysNet load (embedding 88x64 vs PhysNet 119x32); skipped unless --include-spooky.",
        ],
    }

    jax_entries = [("physnet_desdimers", PHYSNET_CKPT)]
    if args.include_spooky and SPOOKY_CKPT.is_file():
        jax_entries.append(("spookynet", SPOOKY_CKPT))

    for sys_name, (atoms, per) in systems.items():
        n = len(atoms)
        for label, ckpt in jax_entries:
            rec: dict = {"backend": "jax", "model": label, "system": sys_name, "ok": False}
            t0 = time.perf_counter()
            try:
                calc, meta = load_physnet_ase(n, ckpt)
                rec["load_s"] = float(time.perf_counter() - t0)
                rec.update(meta)
                rec.update(bench_calculator(label, calc, atoms, atoms_per_monomer=per, warmup=args.warmup, repeats=args.repeats))
            except Exception as exc:
                rec["load_s"] = float(time.perf_counter() - t0)
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()
            report["results"].append(rec)
            print(json.dumps({k: rec.get(k) for k in ("model", "system", "ok", "error", "whole", "fragments")}, default=str), flush=True)

        for name in args.models:
            path = args.model_dir / name
            rec = {"backend": "metatomic", "model": name, "system": sys_name, "ok": False}
            if not path.is_file():
                rec["error"] = f"missing {path}"
                report["results"].append(rec)
                continue
            t0 = time.perf_counter()
            try:
                calc, meta = load_metatomic(path)
                rec["load_s"] = float(time.perf_counter() - t0)
                rec.update(meta)
                rec.update(bench_calculator(name, calc, atoms, atoms_per_monomer=per, warmup=args.warmup, repeats=args.repeats))
            except Exception as exc:
                rec["load_s"] = float(time.perf_counter() - t0)
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()
            report["results"].append(rec)
            print(json.dumps({k: rec.get(k) for k in ("model", "system", "ok", "error", "whole", "fragments")}, default=str), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}", flush=True)
    n_ok = sum(1 for r in report["results"] if r.get("ok"))
    n_fail = sum(1 for r in report["results"] if not r.get("ok"))
    print(f"ok={n_ok} fail={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
