#!/usr/bin/env python3
"""Judge a training run against explicit postconditions and exit accordingly.

Use this instead of reading a SLURM job state. A job can report COMPLETED while
training a partly-random model, and TIMEOUT after the thing under test already
succeeded — both were observed in the Q0 campaign on 2026-08-02.

    python scripts/check_training_run.py \
        --workdir artifacts/spooky_q0_distill_smoke \
        --log artifacts/spooky_q0_distill_smoke/slurm-206104.out \
        --require-steps 40 --require-distillation

Writes ``<workdir>/run_verdict.json`` and exits non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmml.utils.training_run_check import check_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        default=None,
        help="Log file to parse; repeatable. Defaults to <workdir>/slurm-*.out.",
    )
    parser.add_argument(
        "--require-steps",
        type=int,
        default=1,
        help="Minimum training step the run must have reached.",
    )
    parser.add_argument("--no-require-checkpoint", dest="require_checkpoint", action="store_false")
    parser.add_argument(
        "--allow-partial-warm-start",
        dest="require_full_warm_start",
        action="store_false",
        help="Accept a warm-start that left parameters randomly initialized.",
    )
    parser.add_argument("--require-distillation", action="store_true")
    parser.add_argument(
        "--max-force-mae",
        type=float,
        default=None,
        help="Fail if the final force MAE (eV/A) exceeds this.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir: Path = args.workdir
    logs = args.log or sorted(workdir.glob("slurm-*.out"))
    lines: list[str] = []
    for path in logs:
        try:
            lines.extend(Path(path).read_text(errors="replace").splitlines())
        except OSError as exc:
            print(f"could not read {path}: {exc}", file=sys.stderr)

    verdict = check_run(
        workdir,
        lines,
        require_steps=args.require_steps,
        require_checkpoint=args.require_checkpoint,
        require_full_warm_start=args.require_full_warm_start,
        require_distillation=args.require_distillation,
        max_force_mae=args.max_force_mae,
    )

    out = args.output or (workdir / "run_verdict.json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)

    print(verdict.render())
    if verdict.failures:
        print(
            "\nThe SLURM job state is not the answer here — these postconditions are.",
            file=sys.stderr,
        )
    return 0 if verdict.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
