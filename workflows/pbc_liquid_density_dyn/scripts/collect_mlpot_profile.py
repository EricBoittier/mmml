#!/usr/bin/env python3
"""Collect MLpot profile lines and JAX compile timers from profile campaign logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    repo_root,
)

_PROFILE_RE = re.compile(
    r"MLpot profile: (?P<calls>\d+) ML callbacks, "
    r"ML=(?P<ml>[0-9.]+)s \((?P<ml_pct>[0-9.]+)%\), "
    r"CHARMM\+overhead=(?P<charmm>[0-9.]+)s"
)
_JAX_TIMER_RE = re.compile(
    r"mmml: JAX compile timers — estimated compile=(?P<compile>[0-9.]+)s, run=(?P<run>[0-9.]+)s"
)
_WARMUP_RE = re.compile(r"warmup-mlpot-jax: done in (?P<sec>[0-9.]+)s")


def _resolve_config(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    workflow = _SCRIPTS.parent
    candidate = workflow / raw
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"config not found: {raw}")


def _tail_matches(path: Path, pattern: re.Pattern[str]) -> re.Match[str] | None:
    if not path.is_file():
        return None
    hits: list[re.Match[str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            hits.append(m)
    return hits[-1] if hits else None


def _scan_cell(cfg: dict, cell) -> dict:
    paths = paths_for_run(cfg, cell)
    tag = cell_run_tag(cell, cfg)
    stdout = paths["out_dir"] / "stdout.log"
    n_ml = int(cell.n_monomers) * 5
    row: dict = {
        "tag": tag,
        "solvent": cell.solvent,
        "n_monomers": int(cell.n_monomers),
        "n_ml_atoms": n_ml,
        "box_A": float(cell.box_size),
        "stdout": str(stdout),
        "has_stdout": stdout.is_file(),
        "done": paths["done"].is_file(),
    }
    prof = _tail_matches(stdout, _PROFILE_RE)
    if prof:
        row.update(
            {
                "ml_callbacks": int(prof.group("calls")),
                "ml_seconds": float(prof.group("ml")),
                "ml_pct": float(prof.group("ml_pct")),
                "charmm_seconds": float(prof.group("charmm")),
            }
        )
    warmup = _tail_matches(stdout, _WARMUP_RE)
    if warmup:
        row["warmup_seconds"] = float(warmup.group("sec"))
    jax = _tail_matches(stdout, _JAX_TIMER_RE)
    if jax:
        row["jax_compile_seconds"] = float(jax.group("compile"))
        row["jax_run_seconds"] = float(jax.group("run"))
    meta = paths["out_dir"] / "profile_git_metadata.json"
    if meta.is_file():
        try:
            row["profile_metadata"] = str(meta)
        except OSError:
            pass
    for leg in paths["out_dir"].glob("pycharmm_*/profile_git_metadata.json"):
        row.setdefault("profile_metadata_legs", []).append(str(leg))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.profile.gpu.yaml",
        help="Workflow config (default: config.profile.gpu.yaml)",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)
    rows = [_scan_cell(cfg, cell) for cell in iter_matrix_cells(cfg)]

    print(f"{'tag':<22} {'N':>4} {'ML_atoms':>8} {'L':>4} {'ML%':>6} {'ML_s':>8} {'warmup_s':>9} done")
    for r in rows:
        ml_pct = f"{r.get('ml_pct', 0):.1f}" if "ml_pct" in r else "-"
        ml_s = f"{r.get('ml_seconds', 0):.3f}" if "ml_seconds" in r else "-"
        warm = f"{r.get('warmup_seconds', 0):.1f}" if "warmup_seconds" in r else "-"
        print(
            f"{r['tag']:<22} {r['n_monomers']:4d} {r['n_ml_atoms']:8d} "
            f"{int(r['box_A']):4d} {ml_pct:>6} {ml_s:>8} {warm:>9} "
            f"{'Y' if r['done'] else 'n'}"
        )

    out_root = repo_root() / "workflows/pbc_liquid_density_dyn/results"
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (out_root / "mlpot_profile_scaling.csv")
    json_path = args.json or (out_root / "mlpot_profile_scaling.json")

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys() if k != "profile_metadata_legs"})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {csv_path}")
        print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
