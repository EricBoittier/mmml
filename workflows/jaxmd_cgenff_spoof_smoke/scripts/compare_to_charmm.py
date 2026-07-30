#!/usr/bin/env python3
"""Compare jax-mm-spoof CGenFF bonded (+ vacuum MM) to native PyCHARMM ENER FORCE.

Runs on GPU nodes (OpenCL). Writes JSON under
``artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# CHARMM uses OpenCL; monomer JAX parity is fine on CPU. Force CPU before any
# jax / jax_md import so empty CUDA nodes do not abort import.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS_COMPARE", "cpu")
os.environ.setdefault("MMML_ALLOW_SELECTIVE_BONDED_BLOCK", "1")

import jax.numpy as jnp
import numpy as np
from jax_md.mm_forcefields.io.charmm import parse_pdb_simple

_SCRIPTS = Path(__file__).resolve().parent
_WORKFLOW = _SCRIPTS.parent
_REPO = _WORKFLOW.parents[1]

# Residue → (psf, pdb, atoms_per_monomer)
CASES: dict[str, dict[str, Any]] = {
    "DCM": {
        "psf": _REPO / "examples/psf/dcm-1.psf",
        "pdb": _REPO / "mmml/data/molecules/dcm_monomer.pdb",
        "n_atoms": 5,
        "smoke_min_pdb": (
            _REPO
            / "artifacts/jaxmd_cgenff_spoof_smoke/dcm_vac_nve/vac_nve_jaxmd_minimized.pdb"
        ),
    },
    "ACO": {
        "psf": _REPO / "examples/psf/aco-1.psf",
        "pdb": _REPO / "tests/functionality/pycharmmETC/pdb/aco.pdb",
        "n_atoms": 10,
        "smoke_min_pdb": (
            _REPO
            / "artifacts/jaxmd_cgenff_spoof_smoke/aco_vac_nve/vac_nve_jaxmd_minimized.pdb"
        ),
    },
}

ENERGY_RTOL = 5e-3
ENERGY_ATOL = 5e-3
FORCE_RTOL = 5e-3
FORCE_ATOL = 5e-3
MM_ENERGY_ATOL = 2e-2
MM_FORCE_ATOL = 5e-2


def _to_float_dict(d: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in d.items()}


def _force_stats(jax_f: np.ndarray, charmm_f: np.ndarray) -> dict[str, float]:
    diff = np.asarray(jax_f, dtype=np.float64) - np.asarray(charmm_f, dtype=np.float64)
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_jax": float(np.max(np.abs(jax_f))),
        "max_abs_charmm": float(np.max(np.abs(charmm_f))),
    }


def _allclose(
    a: float | np.ndarray,
    b: float | np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> bool:
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def _perturb(positions: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(positions, dtype=np.float64) + rng.normal(
        scale=0.03, size=positions.shape
    )


def _load_positions(pdb: Path, n_atoms: int | None = None) -> np.ndarray:
    _, pos = parse_pdb_simple(str(pdb))
    arr = np.asarray(pos, dtype=np.float64)
    if n_atoms is not None:
        arr = arr[: int(n_atoms)]
    return arr


def _prepare_workdir(workdir: Path, psf: Path, pdb: Path) -> tuple[Path, Path]:
    (workdir / "psf").mkdir(parents=True, exist_ok=True)
    (workdir / "pdb").mkdir(parents=True, exist_ok=True)
    psf_dst = workdir / "psf" / psf.name
    pdb_dst = workdir / "pdb" / pdb.name
    shutil.copy2(psf, psf_dst)
    shutil.copy2(pdb, pdb_dst)
    return psf_dst, pdb_dst


def _init_charmm(psf_path: Path) -> None:
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import read_psf_card_file
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF

    with charmm_relaxed_bomlev():
        read.rtf(CGENFF_RTF)
        read.prm(CGENFF_PRM)
        read_psf_card_file(psf_path)


def compare_bonded(
    *,
    label: str,
    psf: Path,
    positions: np.ndarray,
) -> dict[str, Any]:
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_bonded_forces_kcalmol_A,
        run_charmm_bonded_ener_force,
        set_charmm_positions,
        setup_bonded_only_charmm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
        load_monomer_bonded_components_from_psf,
    )

    n_atoms = int(positions.shape[0])
    set_charmm_positions(positions)
    setup_bonded_only_charmm()
    run_charmm_bonded_ener_force(silent=True)
    charmm_e = charmm_bonded_energy_components_kcalmol()
    charmm_f = charmm_bonded_forces_kcalmol_A()

    jax_comp, jax_f = load_monomer_bonded_components_from_psf(
        psf,
        jnp.asarray(positions),
        atoms_per_monomer=n_atoms,
        energy_unit="kcal/mol",
    )
    jax_e = _to_float_dict(jax_comp)
    jax_f_np = np.asarray(jax_f, dtype=np.float64)

    energy_keys = sorted(set(jax_e) | set(charmm_e))
    energy_deltas = {
        k: float(jax_e.get(k, 0.0) - charmm_e.get(k, 0.0)) for k in energy_keys
    }
    ok_e = _allclose(
        jax_e.get("total", 0.0),
        charmm_e.get("total", 0.0),
        rtol=ENERGY_RTOL,
        atol=ENERGY_ATOL,
    )
    ok_f = _allclose(jax_f_np, charmm_f, rtol=FORCE_RTOL, atol=FORCE_ATOL)
    return {
        "label": label,
        "kind": "bonded",
        "n_atoms": n_atoms,
        "pass": bool(ok_e and ok_f),
        "energy_pass": bool(ok_e),
        "force_pass": bool(ok_f),
        "jax_kcalmol": jax_e,
        "charmm_kcalmol": {k: float(v) for k, v in charmm_e.items()},
        "energy_delta_kcalmol": energy_deltas,
        "force_stats": _force_stats(jax_f_np, charmm_f),
        "tolerances": {
            "energy_rtol": ENERGY_RTOL,
            "energy_atol": ENERGY_ATOL,
            "force_rtol": FORCE_RTOL,
            "force_atol": FORCE_ATOL,
        },
    }


def compare_full_mm_vacuum(
    *,
    label: str,
    psf: Path,
    positions: np.ndarray,
) -> dict[str, Any]:
    """Full bonded+nonbonded MM vs CHARMM for a vacuum monomer (large MIC box)."""
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        compare_mm_system_to_charmm,
        run_charmm_bonded_ener_force,
        set_charmm_positions,
        summarize_mm_system_charmm_delta,
    )
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
        mm_system_energy_and_forces,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        VACUUM_CTOFNB,
        VACUUM_CTONNB,
        VACUUM_CUTNB,
        apply_nbonds_kwargs,
        vacuum_nbond_kwargs,
    )

    apply_nbonds_kwargs(vacuum_nbond_kwargs(nbxmod=5))
    apply_charmm_mm_block()
    set_charmm_positions(positions)
    run_charmm_bonded_ener_force(silent=True)

    bonded = load_bonded_system_from_psf(psf, positions, prm_file=CGENFF_PRM)
    nbond_data = load_nonbonded_system_from_charmm(psf, CGENFF_PRM)
    # Large cubic cell ≈ free space for MIC pair loops.
    cell = np.eye(3, dtype=np.float64) * 200.0
    settings = CharmmNbondSettings(
        cutnb=float(VACUUM_CUTNB),
        ctonnb=float(VACUUM_CTONNB),
        ctofnb=float(VACUUM_CTOFNB),
    )
    result = mm_system_energy_and_forces(
        positions,
        bonded,
        nbond_data,
        cell,
        settings,
        include_cmap=False,
    )
    summary = summarize_mm_system_charmm_delta(result)
    try:
        compare_mm_system_to_charmm(
            result,
            energy_rtol=ENERGY_RTOL,
            energy_atol=MM_ENERGY_ATOL,
            force_rtol=FORCE_RTOL,
            force_atol=MM_FORCE_ATOL,
        )
        passed = True
        err = None
    except AssertionError as exc:
        passed = False
        err = str(exc)

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_mm_component_totals_kcalmol,
        charmm_nonbonded_energy_components_kcalmol,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_total_forces_kcalmol_A,
    )

    b_tot, nb_tot, mm_tot = charmm_mm_component_totals_kcalmol()
    charmm_nb = charmm_nonbonded_energy_components_kcalmol()
    charmm_f = np.asarray(charmm_total_forces_kcalmol_A(), dtype=np.float64)
    return {
        "label": label,
        "kind": "full_mm_vacuum",
        "n_atoms": int(positions.shape[0]),
        "pass": passed,
        "summary": summary,
        "error": err,
        "jax_kcalmol": {
            "bonded": float(result.bonded.get("total", 0.0)),
            "vdw": float(result.nonbonded.get("vdw", 0.0)),
            "elec": float(result.nonbonded.get("elec", 0.0)),
            "total": float(result.total_energy),
        },
        "charmm_kcalmol": {
            "bonded": float(b_tot),
            "vdw": float(charmm_nb["vdw"]),
            "elec": float(charmm_nb["elec"]),
            "nb_total": float(nb_tot),
            "total": float(mm_tot),
        },
        "force_stats": _force_stats(np.asarray(result.forces), charmm_f),
        "tolerances": {
            "energy_rtol": ENERGY_RTOL,
            "energy_atol": MM_ENERGY_ATOL,
            "force_rtol": FORCE_RTOL,
            "force_atol": MM_FORCE_ATOL,
        },
    }


def run_case(residue: str, workdir: Path, *, include_mm: bool) -> list[dict[str, Any]]:
    meta = CASES[residue]
    psf_src = Path(meta["psf"])
    pdb_src = Path(meta["pdb"])
    n_atoms = int(meta["n_atoms"])
    if not psf_src.is_file():
        raise FileNotFoundError(psf_src)
    if not pdb_src.is_file():
        raise FileNotFoundError(pdb_src)

    case_dir = workdir / residue.lower()
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    psf_dst, _pdb_dst = _prepare_workdir(case_dir, psf_src, pdb_src)
    os.chdir(case_dir)

    _init_charmm(psf_dst)
    results: list[dict[str, Any]] = []

    # 1) Fixture geometry (slightly perturbed for non-trivial forces).
    pos0 = _perturb(_load_positions(pdb_src, n_atoms), seed=29 if residue == "ACO" else 7)
    results.append(compare_bonded(label=f"{residue}_fixture_bonded", psf=psf_dst, positions=pos0))

    # 2) First monomer from jaxmd spoof minimized smoke geometry (if present).
    smoke_pdb = Path(meta["smoke_min_pdb"])
    if smoke_pdb.is_file():
        pos_smoke = _load_positions(smoke_pdb, n_atoms)
        results.append(
            compare_bonded(
                label=f"{residue}_smoke_min_monomer_bonded",
                psf=psf_dst,
                positions=pos_smoke,
            )
        )

    if include_mm:
        results.append(
            compare_full_mm_vacuum(
                label=f"{residue}_fixture_full_mm",
                psf=psf_dst,
                positions=pos0,
            )
        )
    return results


def _run_one_residue(residue: str, *, include_mm: bool, partial_path: Path) -> int:
    """One CHARMM session per process (PSF cannot be cleanly swapped)."""
    print(f"=== compare {residue} vs native CHARMM ===", flush=True)
    partial: dict[str, Any] = {
        "residue": residue,
        "comparisons": [],
        "error": None,
        "traceback": None,
    }
    try:
        print("  workdir setup…", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"charmm_cmp_{residue}_") as tmp:
            results = run_case(residue, Path(tmp), include_mm=include_mm)
        partial["comparisons"] = results
        for r in results:
            status = "PASS" if r.get("pass") else "FAIL"
            print(
                f"  {r['label']}: {status}  "
                f"ΔE_total={r.get('energy_delta_kcalmol', {}).get('total', r.get('summary'))}  "
                f"max|ΔF|={r.get('force_stats', {}).get('max_abs_diff')}",
                flush=True,
            )
            if r.get("error"):
                print(f"    {r['error']}", flush=True)
    except Exception as exc:
        partial["error"] = str(exc)
        partial["traceback"] = traceback.format_exc()
        print(f"  ERROR {residue}: {exc}", flush=True)
        print(partial["traceback"], flush=True)
        partial_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {partial_path}", flush=True)
        return 1
    partial_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {partial_path}", flush=True)
    return 0 if all(c.get("pass") for c in partial["comparisons"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residues",
        nargs="+",
        default=["DCM", "ACO"],
        choices=sorted(CASES),
    )
    parser.add_argument(
        "--residue",
        choices=sorted(CASES),
        help="Internal: run a single residue in this process.",
    )
    parser.add_argument(
        "--partial-out",
        type=Path,
        help="Internal: write single-residue JSON here.",
    )
    parser.add_argument(
        "--no-mm",
        action="store_true",
        help="Skip full vacuum MM compare (bonded only).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare",
    )
    args = parser.parse_args()

    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.residue:
        if args.partial_out is None:
            parser.error("--residue requires --partial-out")
        return _run_one_residue(
            args.residue,
            include_mm=not args.no_mm,
            partial_path=args.partial_out.resolve(),
        )

    import subprocess

    report: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": str(_REPO),
        "residues": list(args.residues),
        "include_mm": not args.no_mm,
        "comparisons": [],
        "errors": [],
    }

    py = os.environ.get("MMML_PYTHON", sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    fail = 0
    for residue in args.residues:
        partial_path = out_dir / f"partial_{residue.lower()}.json"
        cmd = [
            py,
            "-u",
            str(Path(__file__).resolve()),
            "--residue",
            residue,
            "--partial-out",
            str(partial_path),
            "--out-dir",
            str(out_dir),
        ]
        if args.no_mm:
            cmd.append("--no-mm")
        print(f"\n## spawning {residue}", flush=True)
        rc = subprocess.call(cmd, env=env, cwd=str(_REPO))
        if rc != 0:
            fail = 1
        if partial_path.is_file():
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            report["comparisons"].extend(partial.get("comparisons") or [])
            if partial.get("error"):
                report["errors"].append(
                    {
                        "residue": residue,
                        "error": partial["error"],
                        "traceback": partial.get("traceback"),
                    }
                )
        else:
            fail = 1
            report["errors"].append(
                {"residue": residue, "error": f"missing partial report (rc={rc})"}
            )

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["pass"] = fail == 0 and not report["errors"]
    n_ok = sum(1 for c in report["comparisons"] if c.get("pass"))
    n_tot = len(report["comparisons"])
    report["summary"] = {"ok": n_ok, "total": n_tot, "errors": len(report["errors"])}

    out_path = out_dir / "compare_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    print(f"summary: ok={n_ok}/{n_tot} errors={len(report['errors'])}", flush=True)
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
