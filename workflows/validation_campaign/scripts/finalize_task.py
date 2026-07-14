#!/usr/bin/env python3
"""Write provenance and terminal status from inside a running job.

Separate from the driver so that a driver which crashes hard still leaves an
honest terminal receipt behind.

This deliberately does NOT invent proof. The only check it may certify is
``exit_zero``, which it observes directly. Every other acceptance check must be
asserted by the scientific driver in ``proof.json``; if the driver does not
write one, the task stays INCOMPLETE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_lib as lib  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["running", "finished"], required=True)
    parser.add_argument("--exit-code", type=int, default=None)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "running":
        lib.write_json(out / "provenance.json", lib.runtime_provenance())
        lib.write_json(
            out / "status.json", {"state": "RUNNING", "started_utc": lib.utcnow()}
        )
        return 0

    exit_code = args.exit_code
    lib.write_json(
        out / "status.json",
        {
            "state": "COMPLETED" if exit_code == 0 else "FAILED",
            "exit_code": exit_code,
            "finished_utc": lib.utcnow(),
        },
    )

    # exit_zero is the one check the harness observes for itself.
    request = lib.read_json(out / "request.json") or {}
    acceptance = request.get("acceptance") or []
    proof_path = out / "proof.json"
    if acceptance == ["exit_zero"] and not proof_path.is_file():
        lib.write_json(
            proof_path,
            {
                "passed": exit_code == 0,
                "checks": {"exit_zero": exit_code == 0},
                "asserted_by": "harness (observed process exit status)",
                "sources": ["status.json", "stdout.log", "stderr.log"],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
