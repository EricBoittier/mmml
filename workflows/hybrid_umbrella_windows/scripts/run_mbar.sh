#!/usr/bin/env bash
# MBAR on a finished hybrid umbrella run_dir; write mbar/status.json marker.
set -euo pipefail

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir|--run-dir) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${OUT}" ]]; then
  echo "FAIL: --run-dir required" >&2
  exit 2
fi

mkdir -p "${OUT}/mbar"
uv run mmml umbrella-mbar --run-dir "${OUT}"

uv run python - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
summary = json.loads((out / "umbrella_summary.json").read_text())
mbar = summary.get("mbar") or {}
if "error" in mbar:
    raise SystemExit(f"MBAR failed: {mbar['error']}")
if "pmf_rel_kcal_mol" not in mbar:
    raise SystemExit("MBAR block missing pmf_rel_kcal_mol in umbrella_summary.json")
status = {
    "ok": True,
    "run_dir": str(out),
    "n_windows_used": mbar.get("n_windows_used"),
    "failed_windows": mbar.get("failed_windows"),
}
(out / "mbar" / "status.json").write_text(json.dumps(status, indent=2) + "\n")
print("PASS: mbar ->", out / "mbar" / "status.json")
xi0 = mbar.get("xi0") or []
pmf = mbar.get("pmf_rel_kcal_mol") or []
for x, f in zip(xi0, pmf):
    if f is None:
        continue
    print(f"  xi0={float(x):7.3f}  PMF={float(f):8.3f} kcal/mol")
PY
