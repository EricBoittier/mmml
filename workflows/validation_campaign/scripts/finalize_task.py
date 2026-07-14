#!/usr/bin/env python3
"""Create terminal status and minimal proof for a dispatched task."""

import argparse
import datetime as dt
import json
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--exit-code", type=int, required=True)
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
request = json.loads((args.output_dir / "request.json").read_text())
status = {
    "state": "COMPLETED" if args.exit_code == 0 else "FAILED",
    "exit_code": args.exit_code,
    "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
}
(args.output_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
proof_path = args.output_dir / "proof.json"
if not proof_path.exists() and request.get("acceptance") == ["exit_zero"]:
    proof_path.write_text(json.dumps({
        "passed": args.exit_code == 0,
        "checks": {"exit_zero": args.exit_code == 0},
        "sources": ["stdout.log", "stderr.log", "status.json"],
    }, indent=2, sort_keys=True) + "\n")

