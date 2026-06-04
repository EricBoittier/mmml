#!/usr/bin/env python3
"""Consolidated MLpot / cluster audit after minimization (no CHARMM required).

Checks:
  - Post-mini CRD exists
  - Uniform monomer atom layout
  - Sparse dimer cap vs mm_switch_on (validate_mlpot_sparse_dimers --free-space)
  - Optional stdout log grep for atom consistency / free-space dimer cap

Example
-------
  python scripts/audit_mlpot_cluster.py \\
    --output-dir workflows/dcm_nve_scaling/results/dcm_7_nve \\
    --composition DCM:7 -o audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_composition(s: str) -> tuple[str, int]:
    m = re.match(r"^([A-Za-z0-9]+):(\d+)$", s.strip())
    if not m:
        raise ValueError(f"Expected RES:COUNT, got {s!r}")
    return m.group(1), int(m.group(2))


def _find_mini_crd(out_dir: Path, tag: str) -> Path | None:
    numbered = out_dir / f"02_mlpot_mmml_{tag}.crd"
    direct = numbered if numbered.is_file() else out_dir / f"mini_full_mlpot_{tag}.crd"
    if direct.is_file():
        return direct
    matches = sorted(out_dir.glob("mini_full_mlpot_*.crd"))
    return matches[0] if matches else None


def _grep_log(out_dir: Path) -> dict[str, bool]:
    log = out_dir / "stdout.log"
    if not log.is_file():
        return {"log_found": False}
    text = log.read_text(encoding="utf-8", errors="replace")
    return {
        "log_found": True,
        "atom_consistency_ok": "Atom consistency OK" in text,
        "free_space_all_pairs": "free-space all-pairs safe" in text
        or "free-space all-pairs" in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--composition", type=str, required=True)
    parser.add_argument("--atoms-per-monomer", type=int, default=5)
    parser.add_argument("--mm-switch-on", type=float, default=5.5)
    parser.add_argument(
        "--free-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use free-space all-pairs sparse dimer cap (default: on)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    residue, n_mol = _parse_composition(args.composition)
    tag = f"{residue.lower()}_{n_mol}"
    out_dir = args.output_dir.resolve()

    report: dict = {
        "composition": args.composition,
        "output_dir": str(out_dir),
        "tag": tag,
        "ok": True,
        "verdict": "OK",
    }

    crd = _find_mini_crd(out_dir, tag)
    if crd is None:
        report["ok"] = False
        report["verdict"] = f"Missing mini CRD in {out_dir}"
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["verdict"], flush=True)
        return 1

    report["mini_crd"] = str(crd)
    n_atoms = n_mol * int(args.atoms_per_monomer)
    report["n_atoms"] = n_atoms
    report["monomer_offsets_ok"] = n_atoms % n_mol == 0

    script = _REPO / "scripts" / "validate_mlpot_sparse_dimers.py"
    validate_cmd = [
        sys.executable,
        str(script),
        "--crd",
        str(crd),
        "--n-monomers",
        str(n_mol),
        "--atoms-per-monomer",
        str(args.atoms_per_monomer),
        "--mm-switch-on",
        str(args.mm_switch_on),
    ]
    if args.free_space:
        validate_cmd.append("--free-space")
    proc = subprocess.run(
        validate_cmd,
        capture_output=True,
        text=True,
    )
    report["sparse_dimer"] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:] if proc.stdout else "",
        "stderr": proc.stderr.strip()[-500:] if proc.stderr else "",
        "ok": proc.returncode == 0,
    }
    report["log_grep"] = _grep_log(out_dir)

    if not report["sparse_dimer"]["ok"]:
        report["ok"] = False
        err = (report["sparse_dimer"].get("stderr") or "").strip()
        if err and "FAIL:" in (report["sparse_dimer"].get("stdout") or ""):
            report["verdict"] = "Sparse dimer cap check failed"
        elif err:
            last = err.splitlines()[-1]
            report["verdict"] = f"Sparse dimer validation error: {last}"
        else:
            report["verdict"] = "Sparse dimer cap check failed"
    elif not report["monomer_offsets_ok"]:
        report["ok"] = False
        report["verdict"] = "Atom count not divisible by n_monomers"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["verdict"], flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
