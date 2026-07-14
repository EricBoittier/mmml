#!/usr/bin/env python3
"""Fail clearly for catalogued tasks whose scientific driver is not built yet."""

import argparse
import json
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--output-dir", type=Path, required=True)
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "status.json").write_text(json.dumps({
    "state": "NEEDS_DRIVER",
    "task_id": args.task,
    "message": "Scientific driver and proof checks must be implemented before submission.",
}, indent=2) + "\n")
raise SystemExit(2)

