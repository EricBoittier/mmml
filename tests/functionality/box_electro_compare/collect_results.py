#!/usr/bin/env python3
"""Collect energies and NVE metrics from box_electro_compare runs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "workflows" / "dcm5_md_benchmark" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_lib import load_config  # noqa: E402
from collect_benchmark import _parse_ase, _parse_jaxmd, job_metadata  # noqa: E402

from mmml.interfaces.pycharmmInterface.lr_solver_grms_compare import (  # noqa: E402
    read_hybrid_grms_from_output_dir,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _job_include_mm(job: dict[str, Any], manifest: dict[str, Any] | None) -> bool:
    extra = job.get("extra_args") or []
    if "--no-include-mm" in extra:
        return False
    if "--include-mm" in extra:
        return True
    if manifest:
        args = manifest.get("args") or {}
        if args.get("include_mm") is False:
            return False
        if args.get("include_mm") is True:
            return True
    return True


def _manifest_exit_code(out_dir: Path) -> int | None:
    manifest = _read_json(out_dir / "run_manifest.json")
    if not manifest:
        return None
    code = manifest.get("exit_code")
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _mini_energy_eV(out_dir: Path) -> float | None:
    candidates = [
        out_dir / name
        for name in (
            "mini_energy.json",
            "mini_00_energy.json",
            "02_mini_energy.json",
            "01_mini_energy.json",
        )
    ]
    candidates.extend(sorted(out_dir.glob("*mini_energy.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        payload = _read_json(path)
        if payload is None:
            continue
        for key in ("hybrid_energy_eV", "energy_eV", "total_energy_eV"):
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    pass
    snapshots = _read_json(out_dir / "snapshots.json")
    if snapshots:
        for snap in snapshots.get("snapshots") or []:
            meta = snap.get("meta") or {}
            for key in ("hybrid_energy_eV", "energy_eV"):
                if key in meta:
                    try:
                        return float(meta[key])
                    except (TypeError, ValueError):
                        pass
    return None


def _best_pre_md_energy_eV(out_dir: Path, backend: str) -> float | None:
    if backend == "ase":
        summary = _read_json(out_dir / "suite_summary.json")
        if summary:
            pre = summary.get("pre_md") or {}
            if "best_energy_eV" in pre:
                return float(pre["best_energy_eV"])
    if backend == "jaxmd":
        summary = _read_json(out_dir / "suite_summary_jaxmd.json")
        if summary:
            pre = summary.get("pre_md") or {}
            if "best_energy_eV" in pre:
                return float(pre["best_energy_eV"])
    return None


def _parse_nve_row(out_dir: Path, cfg: dict[str, Any], job_id: str) -> dict[str, str]:
    job = cfg["jobs"][job_id]
    backend = str(job["backend"])
    meta = job_metadata(cfg, job_id)
    exit_code = _manifest_exit_code(out_dir)

    if backend == "ase":
        parsed = _parse_ase(out_dir, job, meta)
    elif backend == "jaxmd":
        parsed = _parse_jaxmd(out_dir, meta)
    else:
        parsed = dict(meta)
        parsed["status"] = "unknown"
        parsed["notes"] = f"unexpected NVE backend {backend!r}"

    status = str(parsed.get("status", "unknown"))
    notes = str(parsed.get("notes", "") or "")

    if exit_code is not None and exit_code != 0:
        if status in {"missing", "skipped", "unknown"}:
            status = "failed"
        if not notes:
            notes = f"exit_code={exit_code}"
        elif "exit_code=" not in notes:
            notes = f"{notes}; exit_code={exit_code}"

    return {
        "status": status,
        "etot_drift_eV": str(parsed.get("energy_drift", "") or ""),
        "notes": notes,
    }


def _parse_row(cfg: dict, job_id: str, out_dir: Path) -> dict[str, Any]:
    job = cfg["jobs"][job_id]
    backend = str(job["backend"])
    manifest = _read_json(out_dir / "run_manifest.json") if out_dir.is_dir() else None
    exit_code = _manifest_exit_code(out_dir) if out_dir.is_dir() else None
    row: dict[str, Any] = {
        "job_id": job_id,
        "backend": backend,
        "setup": job.get("setup", ""),
        "include_mm": _job_include_mm(job, manifest),
        "lr_solver": "",
        "jax_pme_method": "",
        "status": "missing",
        "hybrid_grms_kcalmol_A": "",
        "energy_eV": "",
        "etot_drift_eV": "",
        "notes": "",
    }
    extra = job.get("extra_args") or []
    for i, tok in enumerate(extra):
        if tok == "--lr-solver" and i + 1 < len(extra):
            row["lr_solver"] = extra[i + 1]
        if tok == "--jax-pme-method" and i + 1 < len(extra):
            row["jax_pme_method"] = extra[i + 1]
    if not row["include_mm"]:
        row["lr_solver"] = "n/a"
    elif backend in {"ase", "jaxmd"} and not row["lr_solver"]:
        row["lr_solver"] = "mic (default)"
        if "jax_pme" in job_id:
            row["notes"] = "ASE/JAX-MD: jax_pme not available; runs MIC"

    if not out_dir.is_dir():
        row["notes"] = "output dir missing"
        return row

    grms = read_hybrid_grms_from_output_dir(out_dir)
    if grms is not None:
        row["hybrid_grms_kcalmol_A"] = f"{grms:.6f}"

    e_mini = _mini_energy_eV(out_dir)
    e_pre = _best_pre_md_energy_eV(out_dir, backend)
    energy = e_mini if e_mini is not None else e_pre
    if energy is not None:
        row["energy_eV"] = f"{energy:.8f}"

    if job_id.startswith("energy_"):
        has_snapshots = out_dir.joinpath("snapshots.json").is_file()
        if has_snapshots and (exit_code in (None, 0)):
            row["status"] = "ok"
        elif exit_code == 0 and energy is not None:
            row["status"] = "ok"
        elif exit_code is not None and exit_code != 0:
            row["status"] = "failed"
            row["notes"] = row["notes"] or f"exit_code={exit_code}"
        elif has_snapshots or energy or grms is not None:
            row["status"] = "incomplete"
            if exit_code is not None:
                row["notes"] = row["notes"] or f"exit_code={exit_code}"
        else:
            row["status"] = "incomplete"
        return row

    nve = _parse_nve_row(out_dir, cfg, job_id)
    row["status"] = nve["status"]
    row["etot_drift_eV"] = nve["etot_drift_eV"]
    if nve["notes"]:
        row["notes"] = nve["notes"]
    if energy is None and nve["notes"]:
        m = re.search(r"etot_drift_eV=([0-9.+-eE]+)", nve["notes"])
        if m:
            row["etot_drift_eV"] = m.group(1)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_root = args.run_root.expanduser().resolve()
    rows = []
    for job_id in cfg["jobs"]:
        out_dir = run_root / job_id
        rows.append(_parse_row(cfg, job_id, out_dir))

    fields = [
        "job_id",
        "backend",
        "setup",
        "include_mm",
        "lr_solver",
        "jax_pme_method",
        "status",
        "energy_eV",
        "hybrid_grms_kcalmol_A",
        "etot_drift_eV",
        "notes",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(f, "")) for f in fields))
    args.out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# DCM:5 L=25 Å electrostatics / backend comparison",
        "",
        f"Run root: `{run_root}`",
        "",
        "| job | backend | MM | lr_solver | E (eV) | GRMS | NVE drift | status |",
        "|-----|---------|----|-----------|--------|------|-----------|--------|",
    ]
    for row in rows:
        md.append(
            f"| {row['job_id']} | {row['backend']} | {row['include_mm']} | "
            f"{row['lr_solver'] or '—'} | {row['energy_eV'] or '—'} | "
            f"{row['hybrid_grms_kcalmol_A'] or '—'} | {row['etot_drift_eV'] or '—'} | "
            f"{row['status']} |"
        )
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_tsv} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
