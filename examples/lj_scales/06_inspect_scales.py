#!/usr/bin/env python
"""Step 06 — read what training actually learned.

`hybrid_mm.json` (not the checkpoint) is what MD consumes. This step reports the
learned scales and, crucially, maps them the way MD will: onto CHARMM's atom-type
ordering, filling 1.0 for types that were never trained.

Sanity guidance:

* A scale still at exactly 1.0 means that type got no gradient — it was absent
  from the training data. Expected for solvent types; not a bug.
* Scales far from 1.0 (outside roughly 0.5-2.0) deserve suspicion. That is
  usually compensation for something else being wrong rather than a genuine
  parameter correction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from mmml.models.mm_lj_scales import find_learnable_lj_scales_sidecar, scales_to_atc

print("=== 06: learned LJ scales ===")

ckpt_dir = Path(os.environ.get("LJ_CKPT_DIR", ""))
explicit = os.environ.get("LJ_SIDECAR", "")

sidecar = None
if explicit:
    sidecar = find_learnable_lj_scales_sidecar(scales_file=explicit)
elif ckpt_dir.is_dir():
    for candidate in sorted(ckpt_dir.rglob("hybrid_mm.json")):
        sidecar = find_learnable_lj_scales_sidecar(scales_file=candidate)
        if sidecar is not None:
            break

if sidecar is None:
    print(
        f"ERROR: no hybrid_mm.json with learnable scales under {ckpt_dir or '(unset)'}\n"
        "       Run 05_train.sh first, or set LJ_SIDECAR to an explicit path.",
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

print(f"  {'type':10s} {'s_sigma':>9s} {'s_epsilon':>10s}")
for i in moved:
    flag = ""
    if not (0.5 <= sig[i] <= 2.0) or not (0.5 <= eps[i] <= 2.0):
        flag = "   <- outside 0.5-2.0, check for compensation"
    print(f"  {names[i]:10s} {sig[i]:9.4f} {eps[i]:10.4f}{flag}")
if not moved:
    print("  (none — every scale is still 1.0, i.e. stock CGenFF)")

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
