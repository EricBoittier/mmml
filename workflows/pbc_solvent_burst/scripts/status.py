#!/usr/bin/env python3
"""Summarize pbc_solvent_burst job completion from done.txt markers."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    cell_run_tag,
    iter_matrix_cells,
    load_config,
    paths_for_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "status.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    rows: list[dict[str, str]] = []
    for cell in iter_matrix_cells(cfg):
        paths = paths_for_run(cfg, cell)
        done = paths["done"].is_file()
        summary = paths["campaign_summary"].is_file()
        handoff = paths["final_handoff"].is_file()
        status = "done" if done and handoff else ("partial" if summary else "pending")
        rows.append(
            {
                "run_tag": cell_run_tag(cell),
                "solvent": cell.solvent,
                "n_monomers": str(cell.n_monomers),
                "temperature": str(cell.temperature),
                "box_size": str(cell.box_size),
                "status": status,
                "done": str(done),
                "summary": str(summary),
                "final_handoff": str(handoff),
                "out_dir": str(paths["out_dir"]),
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "solvent",
        "n_monomers",
        "temperature",
        "box_size",
        "status",
        "done",
        "summary",
        "final_handoff",
        "out_dir",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['run_tag']:22}  {row['status']:7}  "
            f"T={row['temperature']} L={row['box_size']}Å  handoff={row['final_handoff']}"
        )
    n_done = sum(1 for r in rows if r["status"] == "done")
    print(f"\n{n_done}/{len(rows)} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
