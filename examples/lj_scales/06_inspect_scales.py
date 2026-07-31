#!/usr/bin/env python
"""Step 06 — read what training actually learned.

`hybrid_mm.json` (not the checkpoint) is what MD consumes. This step reports the
learned scales and, crucially, maps them the way MD will: onto CHARMM's atom-type
ordering, filling 1.0 for types that were never trained.

Sanity guidance:

* A scale still at exactly 1.0 means that type got no gradient — it was absent
  from the training data. Expected for solvent types; not a bug.
* A scale sitting exactly on a bound means the fit wanted to go further and was
  stopped. Read that as a warning, not a result: the LJ is standing in for
  something else (missing long-range electrostatics, a bad handoff cutoff, or an
  ML term that has not converged).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_BOUNDS,
    MM_LJ_SIGMA_SCALE_BOUNDS,
    find_learnable_lj_scales_sidecar,
    scales_to_atc,
)

print("=== 06: learned LJ scales ===")


def resolve_ckpt_dir() -> Path:
    """Mirror the ``LJ_CKPT_DIR`` default in ``_env.sh``.

    ``05_train.sh`` runs in a subshell, so its exports are gone by the time this
    step runs unless ``_env.sh`` was sourced into the interactive shell. Without
    a fallback the empty string becomes ``Path('.')`` and this step quietly
    searches the current directory — which finds nothing and blames training.

    A bare ``ARTIFACTS_DIR`` is deliberately not consulted: other examples export
    it for their own studies, and following it here would look for this ladder's
    checkpoints in theirs.
    """
    ckpt_dir = os.environ.get("LJ_CKPT_DIR", "").strip()
    if ckpt_dir:
        return Path(ckpt_dir)
    artifacts = os.environ.get("LJ_ARTIFACTS_DIR", "").strip()
    if artifacts:
        return Path(artifacts) / "ckpts"
    return Path(__file__).resolve().parents[2] / "artifacts" / "lj_scales" / "ckpts"


ckpt_dir = resolve_ckpt_dir()
explicit = os.environ.get("LJ_SIDECAR", "")

sidecar = None
if explicit:
    sidecar = find_learnable_lj_scales_sidecar(scales_file=explicit)
elif ckpt_dir.is_dir():
    # Newest first: several runs share the checkpoint dir, and an older one is
    # not what you just trained.
    candidates = sorted(
        ckpt_dir.rglob("hybrid_mm.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        sidecar = find_learnable_lj_scales_sidecar(scales_file=candidate)
        if sidecar is not None:
            break

if sidecar is None:
    where = explicit if explicit else f"{ckpt_dir}{'' if ckpt_dir.is_dir() else ' (does not exist)'}"
    print(
        f"ERROR: no hybrid_mm.json with learnable scales under {where}\n"
        "       Run 05_train.sh first, or point this step at the run you trained:\n"
        "         source examples/lj_scales/_env.sh   # same shell as step 05\n"
        "         LJ_CKPT_DIR=/path/to/ckpts uv run python examples/lj_scales/06_inspect_scales.py\n"
        "         LJ_SIDECAR=/path/to/hybrid_mm.json uv run python examples/lj_scales/06_inspect_scales.py",
        file=sys.stderr,
    )
    sys.exit(2)

payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
names = payload["cgenff_type_names"]
sig = np.asarray(payload["mm_lj_sigma_scale"], dtype=float)
eps = np.asarray(payload["mm_lj_epsilon_scale"], dtype=float)

print(f"\nsidecar: {sidecar}")
print(f"types  : {len(names)}")

moved = [i for i in range(len(names))
         if abs(sig[i] - 1.0) > 1e-3 or abs(eps[i] - 1.0) > 1e-3]
print(f"moved  : {len(moved)} of {len(names)} types received a gradient\n")

sig_lo, sig_hi = MM_LJ_SIGMA_SCALE_BOUNDS
eps_lo, eps_hi = MM_LJ_EPSILON_SCALE_BOUNDS
print(f"  bounds : sigma [{sig_lo:g}, {sig_hi:g}]   epsilon [{eps_lo:g}, {eps_hi:g}]\n")

print(f"  {'type':10s} {'s_sigma':>9s} {'s_epsilon':>10s}")
saturated = 0
for i in moved:
    at_bound = [
        label
        for label, value, lo, hi in (
            ("sigma", sig[i], sig_lo, sig_hi),
            ("epsilon", eps[i], eps_lo, eps_hi),
        )
        if abs(value - lo) < 1e-6 or abs(value - hi) < 1e-6
    ]
    flag = f"   <- {'/'.join(at_bound)} pinned at bound" if at_bound else ""
    saturated += bool(at_bound)
    print(f"  {names[i]:10s} {sig[i]:9.4f} {eps[i]:10.4f}{flag}")
if not moved:
    print("  (none — every scale is still 1.0, i.e. stock CGenFF)")
if saturated:
    print(
        f"\n  {saturated} type(s) hit a bound. The fit wanted LJ the bounds do not\n"
        "  allow — treat those numbers as a symptom, not a parameter."
    )

# --- how MD will see them ---------------------------------------------------
# MD does not use master-table ordering; it remaps onto param.get_atc(). Types
# absent from training fall back to 1.0, which is what makes it safe to deploy a
# dimer-trained model into a solvated box.
print("\n-- as deployed (example CHARMM ATC ordering) --")
atc = list(dict.fromkeys([*[names[i] for i in moved][:3], "OT", "HT"]))
ep_scale, sig_scale = scales_to_atc(names, sig, eps, atc)
print(f"  {'ATC type':10s} {'sig_scale':>10s} {'ep_scale':>10s}")
for name, s, e in zip(atc, sig_scale, ep_scale):
    tag = "" if abs(s - 1) < 1e-9 and abs(e - 1) < 1e-9 else "  <- trained"
    print(f"  {name:10s} {s:10.4f} {e:10.4f}{tag}")

print("\n06: OK")
