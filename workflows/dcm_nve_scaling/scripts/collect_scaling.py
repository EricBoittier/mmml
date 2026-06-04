#!/usr/bin/env python3
"""Aggregate DCM NVE scaling outputs into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scaling_lib import (  # noqa: E402
    composition_string,
    load_config,
    nve_inbfrq_values,
    paths_for_size,
    workflow_root,
)

_CSV_FIELDS = [
    "n_monomers",
    "inbfrq",
    "composition",
    "status",
    "n_frames",
    "max_com_disp_A",
    "cluster_com_disp_A",
    "cluster_msd_A2",
    "max_internal_rmsd_A",
    "worst_monomer_1based",
    "outlier_ratio",
    "sparse_dimer_ok",
    "forces_npz",
    "notes",
]


def _load_com_npz(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    except Exception:
        return None


def _load_audit(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _row_for_size(cfg: dict[str, Any], n: int, *, inbfrq: int) -> dict[str, Any]:
    paths = paths_for_size(cfg, n, inbfrq=inbfrq)
    row: dict[str, Any] = {
        "n_monomers": n,
        "inbfrq": inbfrq,
        "composition": composition_string(n, prefix=str(cfg.get("composition_prefix", "DCM"))),
        "status": "missing",
        "n_frames": "",
        "max_com_disp_A": "",
        "cluster_com_disp_A": "",
        "cluster_msd_A2": "",
        "max_internal_rmsd_A": "",
        "worst_monomer_1based": "",
        "outlier_ratio": "",
        "sparse_dimer_ok": "",
        "forces_npz": paths["forces_npz"].is_file(),
        "notes": "",
    }

    if not (paths["out_dir"] / "done.txt").is_file():
        row["notes"] = "done.txt missing"
        return row

    com = _load_com_npz(paths["com_npz"])
    audit = _load_audit(paths["audit_json"])
    if com is None:
        row["notes"] = "com_analysis.npz missing"
        return row

    max_disp = np.asarray(com.get("max_disp_per_monomer", []), dtype=float)
    if max_disp.size == 0:
        row["notes"] = "empty com analysis"
        return row

    worst = int(np.argmax(max_disp)) + 1
    median = float(np.median(max_disp))
    outlier = float(max_disp[worst - 1] / median) if median > 1e-8 else float("inf")

    row["status"] = "pass" if bool(com.get("ok", True)) else "fail"
    row["n_frames"] = int(com.get("n_frames", max_disp.shape[0] if max_disp.ndim == 2 else 0))
    row["max_com_disp_A"] = f"{float(np.max(max_disp)):.4f}"
    row["cluster_com_disp_A"] = f"{float(com.get('max_cluster_com_disp_A', np.nan)):.4f}"
    row["cluster_msd_A2"] = f"{float(com.get('mean_msd_cluster_A2', np.nan)):.4f}"
    row["max_internal_rmsd_A"] = f"{float(com.get('max_internal_rmsd_A', np.nan)):.4f}"
    row["worst_monomer_1based"] = worst
    row["outlier_ratio"] = f"{outlier:.2f}"
    fail_reasons = com.get("fail_reasons")
    if fail_reasons is not None and len(fail_reasons) and row["status"] == "fail":
        row["notes"] = ",".join(str(x) for x in np.atleast_1d(fail_reasons))[:120]
    if audit is not None:
        row["sparse_dimer_ok"] = audit.get("sparse_dimer", {}).get("ok", "")
        if audit.get("verdict"):
            row["notes"] = str(audit["verdict"])[:120]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=workflow_root() / "config.yaml")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sizes = [int(x) for x in cfg["cluster_sizes"]]
    inbfrqs = nve_inbfrq_values(cfg)
    rows = [
        _row_for_size(cfg, n, inbfrq=ib)
        for n in sizes
        for ib in inbfrqs
    ]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# DCM NVE scaling summary",
        "",
        f"Config: `{args.config}`",
        "",
        "| N | inbfrq | status | cluster disp (Å) | outlier ratio | max internal RMSD |",
        "|---|--------|--------|------------------|---------------|-------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_monomers']} | {r['inbfrq']} | {r['status']} | "
            f"{r['cluster_com_disp_A']} | {r['outlier_ratio']} | {r['max_internal_rmsd_A']} |"
        )
    lines.append("")
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv} and {args.md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
