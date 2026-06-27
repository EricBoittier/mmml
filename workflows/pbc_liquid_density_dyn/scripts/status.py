#!/usr/bin/env python3
"""Summarize liquid-density dynamics campaign progress."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    campaign_job_order,
    cell_run_tag,
    iter_matrix_cells,
    load_config,
    paths_for_run,
    repo_root,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--failed", action="store_true")
    p.add_argument("--csv", type=Path, default=None)
    return p.parse_args()


def _leg_status(out_dir: Path, job_id: str) -> str:
    leg = out_dir / job_id
    handoff = leg / "handoff" / "state.npz"
    if handoff.is_file():
        return "done"
    if leg.is_dir() and any(leg.iterdir()):
        return "partial"
    return "pending"


def _row(cfg: dict, cell) -> dict:
    paths = paths_for_run(cfg, cell)
    tag = cell_run_tag(cell, cfg)
    order = campaign_job_order(cfg)
    legs = {jid: _leg_status(paths["out_dir"], jid) for jid in order}
    done_n = sum(1 for s in legs.values() if s == "done")
    summary_ok = False
    if paths["campaign_summary"].is_file():
        try:
            payload = json.loads(paths["campaign_summary"].read_text(encoding="utf-8"))
            jobs = payload.get("jobs", [])
            summary_ok = all(int(j.get("exit_code", 1)) == 0 for j in jobs)
        except (json.JSONDecodeError, TypeError):
            summary_ok = False
    health = "OK" if paths["done"].is_file() else ("FAIL" if done_n and not summary_ok else "RUN")
    stdout = paths["out_dir"] / "stdout.log"
    has_log = stdout.is_file()
    return {
        "tag": tag,
        "solvent": cell.solvent,
        "n": cell.n_monomers,
        "T_K": int(round(cell.temperature)),
        "L_A": int(round(cell.box_size)),
        "legs_done": f"{done_n}/{len(order)}",
        "health": health,
        "done_txt": paths["done"].is_file(),
        "prep_ladder": paths["prep_ladder"].is_dir(),
        "has_stdout": has_log,
        "final_handoff": paths["final_handoff"].is_file(),
    }


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    rows = []
    for cell in iter_matrix_cells(cfg):
        tag = cell_run_tag(cell, cfg)
        if args.tag and tag != args.tag:
            continue
        row = _row(cfg, cell)
        if args.failed and row["health"] == "OK":
            continue
        rows.append(row)

    if not rows:
        print("No matching runs.")
        return 0

    headers = list(rows[0].keys())
    print(f"{'tag':<24} {'n':>4} {'T':>4} {'L':>4} {'legs':>8} {'health':>6} handoff prep")
    for r in rows:
        print(
            f"{r['tag']:<24} {r['n']:4d} {r['T_K']:4d} {r['L_A']:4d} "
            f"{r['legs_done']:>8} {r['health']:>6} "
            f"{'Y' if r['final_handoff'] else 'n'} "
            f"{'Y' if r['prep_ladder'] else 'n'}"
        )

    csv_path = args.csv or (repo_root() / "workflows/pbc_liquid_density_dyn/results/status.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
