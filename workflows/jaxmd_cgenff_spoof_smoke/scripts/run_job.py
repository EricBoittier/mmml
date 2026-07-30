#!/usr/bin/env python3
"""Build and run one jaxmd CGenFF spoof smoke job via mmml md-system."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_SCRIPTS = Path(__file__).resolve().parent
_WORKFLOW = _SCRIPTS.parent
_REPO = _WORKFLOW.parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (_WORKFLOW / "config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_checkpoint(cfg: dict[str, Any]) -> Path:
    env = os.environ.get("MMML_CKPT", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if path.exists():
            return path
    raw = str(cfg.get("checkpoint", "examples/ckpts_json/DESdimers_params.json"))
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = (_REPO / path).resolve()
    if not path.exists():
        # Spoof does not load PhysNet; fall back to any existing portable JSON.
        for cand in (
            _REPO / "examples/ckpts_json/DESdimers_params.json",
            Path("/mmhome/boittier/home/mmml/examples/ckpts_json/DESdimers_params.json"),
        ):
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError(
            f"Placeholder checkpoint not found: {path}. Set MMML_CKPT or copy "
            "examples/ckpts_json/DESdimers_params.json into this clone."
        )
    return path


def job_output_dir(cfg: dict[str, Any], job_id: str) -> Path:
    root = _REPO / str(cfg.get("output_root", "artifacts/jaxmd_cgenff_spoof_smoke"))
    return (root / job_id).resolve()


def write_job_config(cfg: dict[str, Any], job_id: str, out_dir: Path) -> Path:
    jobs = cfg.get("jobs") or {}
    if job_id not in jobs:
        raise KeyError(f"Unknown job_id {job_id!r}; known: {sorted(jobs)}")
    job = dict(jobs[job_id])
    defaults = dict(cfg.get("defaults") or {})
    merged = {**defaults, **{k: v for k, v in job.items() if k != "description"}}
    merged["checkpoint"] = str(resolve_checkpoint(cfg))
    merged["output_dir"] = str(out_dir)
    merged["packmol_cache_dir"] = str(out_dir / ".packmol_cache")
    backend = str(merged.get("backend", "jaxmd")).strip().lower()
    if backend in {"pycharmm", "charmm", "native"}:
        merged["backend"] = "pycharmm"
        merged["jax_mm_spoof"] = False
    else:
        # Default smoke path: jaxmd + CGenFF bonded spoof.
        merged["backend"] = "jaxmd"
        merged["jax_mm_spoof"] = bool(merged.get("jax_mm_spoof", True))
    # Drop workflow-only keys that md-system rejects.
    for drop in ("output_root", "description", "jobs", "defaults"):
        merged.pop(drop, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "job.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)
    return path


def mmml_cmd(md_argv: list[str]) -> list[str]:
    py = os.environ.get("MMML_PYTHON", sys.executable)
    return [py, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Job key from config.yaml jobs:")
    parser.add_argument(
        "--config",
        type=Path,
        default=_WORKFLOW / "config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = job_output_dir(cfg, args.job_id)
    job_yaml = write_job_config(cfg, args.job_id, out_dir)
    report = {
        "job_id": args.job_id,
        "job_yaml": str(job_yaml),
        "output_dir": str(out_dir),
        "repo": str(_REPO),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.chdir(_REPO)
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("JAX_PLATFORMS", env.get("JAX_PLATFORMS", "cpu"))
    # Prefer this clone's package tree over any other install.
    env["PYTHONPATH"] = str(_REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    cmd = mmml_cmd(["--config", str(job_yaml)])
    report["command"] = cmd
    print(f"=== {args.job_id} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    report["elapsed_s"] = round(time.time() - t0, 3)
    report["returncode"] = rc
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Capture a few useful artifacts if present.
    for name in ("run_manifest.json", "md_summary.json", "campaign_summary.json"):
        p = out_dir / name
        if p.is_file():
            report[name] = str(p)
    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if rc != 0:
        print(f"FAIL {args.job_id} rc={rc}", file=sys.stderr)
        return rc
    print(f"OK {args.job_id} ({report['elapsed_s']} s) → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
