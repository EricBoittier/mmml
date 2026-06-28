#!/usr/bin/env bash
# Preflight for des_dimer_pair_scans workflow.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

"${MMML_PYTHON}" - <<'PY'
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path("workflows/des_dimer_pair_scans/scripts")))
from scan_lib import iter_pairs, load_config, validate_compositions, validate_species_residues, workflow_root

cfg = load_config(workflow_root() / "config.yaml")
missing = validate_species_residues(cfg)
if missing:
    raise SystemExit(f"Unknown CGENFF residues: {missing}")
validate_compositions(cfg)
pairs = list(iter_pairs(cfg))
print(f"OK: {len(pairs)} dimer pairs, {len(cfg['species'])} species")
for backend, pkg in [("xtb", "tblite"), ("charmm", "pycharmm")]:
    try:
        __import__(pkg)
        print(f"  {backend}: available")
    except ImportError:
        print(f"  {backend}: MISSING ({pkg})")
orca = shutil.which("orca") or shutil.which("ORCA")
print(f"  orca_mp2: {'available' if orca else 'MISSING (set ORCA=path/to/orca)'}")
ckpt = cfg.get("reference_checkpoint")
if ckpt:
    p = Path(ckpt)
    if not p.is_file() and not (Path.cwd() / ckpt).is_file():
        print(f"  warning: reference checkpoint not found: {ckpt}")
PY
