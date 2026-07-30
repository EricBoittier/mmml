#!/usr/bin/env python3
"""Print jax-mm-spoof vs native CHARMM compare (+ optional MD side-by-side)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).resolve().parents[1]
_REPO = _WORKFLOW.parents[1]


def _load_suite_energy(job_dir: Path) -> dict | None:
    for name in (
        "suite_summary_jaxmd.json",
        "suite_summary_pycharmm.json",
        "suite_summary.json",
        "md_summary.json",
        "calculator_summary.json",
    ):
        p = job_dir / name
        if not p.is_file():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _fmt_energy(payload: dict | None) -> str:
    if not payload:
        return "n/a"
    pre = payload.get("pre_md_minimization") or {}
    if "best_energy_eV" in pre:
        return f"E_min={pre['best_energy_eV']:.6f} eV  fmax={pre.get('best_force_fmax_eVA')}"
    if "energy_eV" in payload:
        return f"E={payload['energy_eV']}"
    return "present"


def main() -> int:
    compare_path = (
        _REPO / "artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare/compare_report.json"
    )
    print("=== jax-mm-spoof CGenFF vs native CHARMM (ENER FORCE) ===")
    if not compare_path.is_file():
        print(f"  missing {compare_path}")
        print("  run: python workflows/jaxmd_cgenff_spoof_smoke/scripts/compare_to_charmm.py")
        ef_ok = False
    else:
        report = json.loads(compare_path.read_text(encoding="utf-8"))
        ef_ok = bool(report.get("pass"))
        for c in report.get("comparisons") or []:
            status = "PASS" if c.get("pass") else "FAIL"
            kind = c.get("kind")
            label = c.get("label")
            if kind == "bonded":
                dE = (c.get("energy_delta_kcalmol") or {}).get("total")
                dF = (c.get("force_stats") or {}).get("max_abs_diff")
                print(
                    f"  {status}  {label}: ΔE_total={dE:+.6e} kcal/mol  "
                    f"max|ΔF|={('n/a' if dF is None else f'{dF:.6e}')} kcal/mol/Å"
                )
                je = c.get("jax_kcalmol") or {}
                ce = c.get("charmm_kcalmol") or {}
                print(
                    f"         jax_total={je.get('total')}  "
                    f"charmm_total={ce.get('total')}"
                )
            else:
                print(f"  {status}  {label}: {c.get('summary')}")
                if c.get("error"):
                    print(f"         {c['error']}")
        for err in report.get("errors") or []:
            print(f"  ERROR {err.get('residue')}: {err.get('error')}")
        s = report.get("summary") or {}
        print(f"  E/F summary: ok={s.get('ok')}/{s.get('total')} errors={s.get('errors')}")

    print()
    print("=== MD smoke side-by-side (jaxmd spoof vs native pycharmm) ===")
    spoof_cfg = yaml.safe_load((_WORKFLOW / "config.yaml").read_text(encoding="utf-8")) or {}
    native_cfg = (
        yaml.safe_load((_WORKFLOW / "config.native_charmm.yaml").read_text(encoding="utf-8"))
        or {}
    )
    spoof_root = _REPO / str(
        spoof_cfg.get("output_root", "artifacts/jaxmd_cgenff_spoof_smoke")
    )
    native_root = _REPO / str(
        native_cfg.get("output_root", "artifacts/jaxmd_cgenff_spoof_smoke_native_charmm")
    )
    jobs = list((spoof_cfg.get("jobs") or {}).keys())
    md_missing = 0
    for job_id in jobs:
        spoof_dir = spoof_root / job_id
        native_dir = native_root / job_id
        spoof_rep = spoof_dir / "smoke_report.json"
        native_rep = native_dir / "smoke_report.json"
        spoof_rc = (
            json.loads(spoof_rep.read_text(encoding="utf-8")).get("returncode")
            if spoof_rep.is_file()
            else None
        )
        native_rc = (
            json.loads(native_rep.read_text(encoding="utf-8")).get("returncode")
            if native_rep.is_file()
            else None
        )
        if spoof_rc is None or native_rc is None:
            md_missing += 1
        print(
            f"  {job_id}: spoof_rc={spoof_rc} native_rc={native_rc}\n"
            f"      spoof  {_fmt_energy(_load_suite_energy(spoof_dir))}\n"
            f"      native {_fmt_energy(_load_suite_energy(native_dir))}"
        )

    return 0 if ef_ok and md_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
