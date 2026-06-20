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

from campaign_lib import load_config, paths_for_run, repo_root  # noqa: E402


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
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", [])]
    sizes = [int(x) for x in cfg.get("cluster_sizes", [])]

    rows: list[dict[str, str]] = []
    for sol in solvents:
        for n in sizes:
            paths = paths_for_run(cfg, sol, n)
            done = paths["done"].is_file()
            summary = paths["campaign_summary"].is_file()
            handoff = paths["final_handoff"].is_file()
            status = "done" if done and handoff else ("partial" if summary else "pending")
            rows.append(
                {
                    "solvent": sol,
                    "n_monomers": str(n),
                    "status": status,
                    "done": str(done),
                    "summary": str(summary),
                    "final_handoff": str(handoff),
                    "out_dir": str(paths["out_dir"]),
                }
            )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solvent",
                "n_monomers",
                "status",
                "done",
                "summary",
                "final_handoff",
                "out_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['solvent']}:{row['n_monomers']:>3}  {row['status']:7}  "
            f"handoff={row['final_handoff']}  {row['out_dir']}"
        )
    n_done = sum(1 for r in rows if r["status"] == "done")
    print(f"\n{ n_done}/{len(rows)} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
