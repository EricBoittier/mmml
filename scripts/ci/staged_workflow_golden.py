#!/usr/bin/env python3
"""Characterization harness for ``run_staged_workflow``.

``run_staged_workflow`` is a 2469-line function holding ~700 uncovered
statements. Decomposing it is only safe with a *behavioural* baseline, because
its failure mode is not a crash: a stage that resumes from the wrong restart, or
a segment counter that stops advancing, produces a complete run with a quietly
different trajectory.

This captures what a staged run *did* -- stage order, step counts, restart
lineage, thermodynamics, and which artifacts were written -- into a golden
record, and compares a later run against it.

    # before touching the function
    python scripts/ci/staged_workflow_golden.py capture RUN_DIR -o golden.json
    # after each extraction step
    python scripts/ci/staged_workflow_golden.py compare RUN_DIR golden.json

Fields that legitimately vary between identical runs (wall time, absolute paths,
host) are normalised away. Everything that encodes *what the workflow decided*
is compared exactly; float observables are compared with an explicit tolerance,
because MD on different hardware is not bitwise reproducible but a correct
refactor must not move a mean temperature by more than noise.

Exit status is 0 when the runs agree and 1 when they do not, so it drops
straight into a refactor loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Recorded per stage. These encode decisions (what ran, in what order, resuming
# from what) rather than timings, so they must match exactly.
_EXACT_STAGE_FIELDS = (
    "stage",
    "backend",
    "setup",
    "status",
    "integrator",
    "nsteps_requested",
    "nsteps_completed",
    "frames_written",
    "record_every_steps",
)

# Physical observables: compared with a relative tolerance.
_NUMERIC_STAGE_FIELDS = (
    "dt_fs",
    "ps_requested",
    "ps_completed",
    "temperature_K",
    "temperature_mean_K",
    "temperature_final_K",
    "temperature_first_K",
    "pressure_atm",
    "pressure_mean_atm",
    "box_A_initial",
    "box_A_final",
    "volume_A3_final",
    "density_g_cm3_final",
    "density_g_cm3_mean",
)

# Deliberately excluded: wall_time_s, job_id, description (free text), and any
# absolute path. See _artifact_names for how artifacts are normalised.
_ARTIFACT_SUFFIXES = (".res", ".dcd", ".crd", ".pdb", ".psf")


def _artifact_names(stage: dict[str, Any]) -> list[str]:
    """Artifact *basenames* for one stage, sorted.

    Only the names matter: the directory is run-specific, but which restart and
    trajectory files a stage produced is exactly the lineage a refactor must
    preserve.
    """
    raw = stage.get("artifacts") or []
    names: list[str] = []
    if isinstance(raw, dict):
        raw = list(raw.values())
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            names.append(Path(item).name)
        elif isinstance(item, dict):
            for key in ("path", "file", "name"):
                if isinstance(item.get(key), str):
                    names.append(Path(item[key]).name)
                    break
    return sorted(names)


def _on_disk_manifest(run_dir: Path) -> list[str]:
    """Non-empty restart/trajectory files actually present, as ``name:sizeclass``.

    Exact byte sizes differ run to run (timestamps inside DCD headers), so size
    is bucketed by order of magnitude: enough to catch "the file is now empty"
    or "we wrote 10x fewer frames" without failing on noise.
    """
    out: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _ARTIFACT_SUFFIXES:
            continue
        size = path.stat().st_size
        if size <= 0:
            out.append(f"{path.name}:empty")
            continue
        magnitude = len(str(size))
        out.append(f"{path.name}:1e{magnitude - 1}")
    return out


def capture(run_dir: Path) -> dict[str, Any]:
    """Build a golden record from a completed staged-workflow output directory."""
    summary_path = run_dir / "stage_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"no stage_summary.json under {run_dir}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    stages = []
    for stage in payload.get("stages") or []:
        record: dict[str, Any] = {f: stage.get(f) for f in _EXACT_STAGE_FIELDS}
        record["_numeric"] = {f: stage.get(f) for f in _NUMERIC_STAGE_FIELDS}
        record["_artifacts"] = _artifact_names(stage)
        stages.append(record)

    return {
        "schema": 1,
        "backend": payload.get("backend"),
        "setup": payload.get("setup"),
        "exit_code": payload.get("exit_code"),
        "stage_order": [s.get("stage") for s in payload.get("stages") or []],
        "stages": stages,
        "manifest": _on_disk_manifest(run_dir),
    }


def _cmp_numbers(name: str, want: Any, got: Any, rtol: float) -> str | None:
    if want is None and got is None:
        return None
    if want is None or got is None:
        return f"{name}: {want!r} -> {got!r} (one side missing)"
    try:
        w, g = float(want), float(got)
    except (TypeError, ValueError):
        return None if want == got else f"{name}: {want!r} -> {got!r}"
    if w == 0.0:
        return None if abs(g) <= rtol else f"{name}: {w} -> {g}"
    if abs(g - w) / abs(w) > rtol:
        return f"{name}: {w:.6g} -> {g:.6g} (rel {abs(g - w) / abs(w):.2e} > {rtol:g})"
    return None


def compare(golden: dict[str, Any], current: dict[str, Any], *, rtol: float) -> list[str]:
    """Return one message per behavioural difference (empty list == unchanged)."""
    diffs: list[str] = []

    for key in ("backend", "setup", "exit_code"):
        if golden.get(key) != current.get(key):
            diffs.append(f"{key}: {golden.get(key)!r} -> {current.get(key)!r}")

    if golden.get("stage_order") != current.get("stage_order"):
        diffs.append(
            f"stage order: {golden.get('stage_order')} -> {current.get('stage_order')}"
        )
        return diffs  # per-stage comparison is meaningless once the order moved

    for g_stage, c_stage in zip(golden.get("stages", []), current.get("stages", [])):
        label = g_stage.get("stage", "?")
        for field in _EXACT_STAGE_FIELDS:
            if g_stage.get(field) != c_stage.get(field):
                diffs.append(
                    f"[{label}] {field}: {g_stage.get(field)!r} -> {c_stage.get(field)!r}"
                )
        for field, want in (g_stage.get("_numeric") or {}).items():
            msg = _cmp_numbers(f"[{label}] {field}", want, (c_stage.get("_numeric") or {}).get(field), rtol)
            if msg:
                diffs.append(msg)
        if g_stage.get("_artifacts") != c_stage.get("_artifacts"):
            diffs.append(
                f"[{label}] artifacts: {g_stage.get('_artifacts')} -> {c_stage.get('_artifacts')}"
            )

    if golden.get("manifest") != current.get("manifest"):
        missing = sorted(set(golden.get("manifest", [])) - set(current.get("manifest", [])))
        added = sorted(set(current.get("manifest", [])) - set(golden.get("manifest", [])))
        if missing:
            diffs.append(f"artifacts no longer written: {missing}")
        if added:
            diffs.append(f"unexpected new artifacts: {added}")

    return diffs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="write a golden record for a run directory")
    cap.add_argument("run_dir", type=Path)
    cap.add_argument("-o", "--out", type=Path, required=True)

    cmp_ = sub.add_parser("compare", help="compare a run directory against a golden record")
    cmp_.add_argument("run_dir", type=Path)
    cmp_.add_argument("golden", type=Path)
    cmp_.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="relative tolerance for physical observables (default: 1e-6, i.e. "
        "a pure refactor; raise to ~1e-3 when comparing across hardware)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "capture":
        record = capture(args.run_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"golden record written to {args.out} "
            f"({len(record['stages'])} stages, {len(record['manifest'])} artifacts)"
        )
        return 0

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    current = capture(args.run_dir)
    diffs = compare(golden, current, rtol=args.rtol)
    if not diffs:
        print(f"staged workflow unchanged vs {args.golden} ({len(current['stages'])} stages)")
        return 0
    print(f"::error::staged workflow diverged from {args.golden}:", file=sys.stderr)
    for line in diffs:
        print(f"  - {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
