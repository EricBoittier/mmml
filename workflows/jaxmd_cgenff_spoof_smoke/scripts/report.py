#!/usr/bin/env python3
"""Summarize jaxmd CGenFF spoof smoke job outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[1]
_REPO = _WORKFLOW.parents[1]


def main() -> int:
    import yaml

    cfg = yaml.safe_load((_WORKFLOW / "config.yaml").read_text(encoding="utf-8")) or {}
    root = _REPO / str(cfg.get("output_root", "artifacts/jaxmd_cgenff_spoof_smoke"))
    jobs = list((cfg.get("jobs") or {}).keys())
    print(f"=== jaxmd CGenFF spoof smoke report ({root}) ===")
    ok = 0
    fail = 0
    missing = 0
    for job_id in jobs:
        report_path = root / job_id / "smoke_report.json"
        if not report_path.is_file():
            print(f"  {job_id}: MISSING (not run)")
            missing += 1
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rc = int(payload.get("returncode", 1))
        elapsed = payload.get("elapsed_s")
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        if rc == 0:
            ok += 1
        else:
            fail += 1
        print(f"  {job_id}: {status}  elapsed={elapsed}s  out={payload.get('output_dir')}")
        job_yaml = Path(payload.get("job_yaml", ""))
        if job_yaml.is_file():
            y = yaml.safe_load(job_yaml.read_text(encoding="utf-8")) or {}
            print(
                f"      composition={y.get('composition')} setup={y.get('setup')} "
                f"jax_mm_spoof={y.get('jax_mm_spoof')} backend={y.get('backend')}"
            )
    print(f"summary: ok={ok} fail={fail} missing={missing} total={len(jobs)}")
    return 0 if fail == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
