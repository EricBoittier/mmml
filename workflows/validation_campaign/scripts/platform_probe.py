#!/usr/bin/env python3
"""Write a compact, machine-readable platform proof artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(*args: str) -> str | None:
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imports = {}
    jax_data = {}
    for module in ("numpy", "ase", "jax", "mmml"):
        try:
            mod = __import__(module)
            imports[module] = {"ok": True, "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            imports[module] = {"ok": False, "error": repr(exc)}
    try:
        import jax
        jax_data = {
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "default_backend": jax.default_backend(),
            "devices": [str(x) for x in jax.devices()],
        }
    except Exception as exc:
        jax_data = {"error": repr(exc), "x64_enabled": False}
    provenance = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "environment": {key: os.environ.get(key) for key in ("JAX_ENABLE_X64", "MMML_ML_DTYPE", "CUDA_VISIBLE_DEVICES")},
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "imports": imports,
        "jax": jax_data,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checks = {
        "provenance_complete": bool(provenance["hostname"] and provenance["git_commit"]),
        "jax_x64_enabled": bool(jax_data.get("x64_enabled")),
        "imports_pass": all(row["ok"] for row in imports.values()),
    }
    proof = {"passed": all(checks.values()), "checks": checks, "sources": ["provenance.json"]}
    (args.output_dir / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps({"jax_device_count": len(jax_data.get("devices", []))}, indent=2) + "\n")
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

