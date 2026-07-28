#!/usr/bin/env python3
"""Aggregate campaign status.json files into results/summary.{csv,md}."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "path",
    "job",
    "completed",
    "solvent",
    "variant",
    "basin",
    "elapsed_seconds",
    "barrier_kcal_mol",
    "delta_e_product_kcal_mol",
    "error",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    return p.parse_args()


def _iter_statuses(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("status.json") if p.is_file())


def _row(path: Path, root: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path.relative_to(root)) if root in path.parents else str(path),
            "job": "",
            "completed": False,
            "solvent": "",
            "variant": "",
            "basin": "",
            "elapsed_seconds": "",
            "barrier_kcal_mol": "",
            "delta_e_product_kcal_mol": "",
            "error": f"read_error: {exc}",
        }
    rel = str(path.relative_to(root)) if root in path.parents or path.parent == root else str(path)
    return {
        "path": rel,
        "job": data.get("job", ""),
        "completed": bool(data.get("completed")),
        "solvent": data.get("solvent") or "",
        "variant": data.get("variant") or "",
        "basin": data.get("basin") or "",
        "elapsed_seconds": data.get("elapsed_seconds", ""),
        "barrier_kcal_mol": data.get("barrier_kcal_mol", ""),
        "delta_e_product_kcal_mol": data.get("delta_e_product_kcal_mol", ""),
        "error": data.get("error") or "",
    }


def main() -> int:
    args = _parse_args()
    root = args.artifact_root.resolve()
    rows = [_row(p, root) for p in _iter_statuses(root)]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    done = sum(1 for r in rows if r["completed"])
    lines = [
        "# NH₃–CH₃Cl reaction-path campaign",
        "",
        f"Completed: **{done}/{len(rows)}** status files under `{root}`",
        "",
        "| job | solvent | variant | basin | status | elapsed (s) | notes |",
        "|---|---|---|---|:---:|---:|---|",
    ]
    for r in rows:
        icon = "✅" if r["completed"] else f"❌ {r['error']}"
        notes = ""
        if r.get("barrier_kcal_mol") not in ("", None):
            notes = f"barrier={r['barrier_kcal_mol']}"
        lines.append(
            f"| {r['job']} | {r['solvent']} | {r['variant']} | {r['basin']} | "
            f"{icon} | {r['elapsed_seconds']} | {notes} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv} and {args.markdown} ({done}/{len(rows)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
