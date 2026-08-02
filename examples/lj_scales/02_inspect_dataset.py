#!/usr/bin/env python
"""Step 02 — look at the data before spending GPU hours on it.

Two things matter here:

1. **PSF ordering.** CGenFF typing walks the CHARMM topology, so atom order must
   match the PSF. Two files can hold identical data in different orders and only
   one is usable — and the wrong one does not crash, it mis-assigns types.
2. **Which types you are actually fitting.** Only types present in the data get a
   gradient; everything else stays at scale 1.0 forever. That is correct
   behaviour, but it means the type histogram tells you what training can and
   cannot learn.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def atom_order(path: Path) -> str:
    d = np.load(path, allow_pickle=True)
    z = np.asarray(d["Z"][0])
    z = z[z > 0]
    return " ".join(SYMBOL.get(int(v), str(v)) for v in z)


print("=== 02: dataset inspection ===")

raw = Path(os.environ.get("LJ_DATASET", ""))
enriched = Path(os.environ.get("LJ_ENRICHED", ""))

if not raw.is_file():
    print(f"ERROR: LJ_DATASET not found: {raw}", file=sys.stderr)
    sys.exit(2)

print(f"\n-- atom ordering --\n  {raw.name}\n    frame 0: {atom_order(raw)}")

# Any sibling NPZ of the same size is very likely the same data in another
# order; surfacing it here is what stops someone grabbing the wrong file.
siblings = [
    p for p in raw.parent.glob("*.npz")
    if p != raw and abs(p.stat().st_size - raw.stat().st_size) < raw.stat().st_size * 0.5
]
for sib in siblings[:4]:
    try:
        print(f"  {sib.name}\n    frame 0: {atom_order(sib)}")
    except Exception:
        continue
if siblings:
    print(
        "\n  Heavy atoms first (e.g. 'C Cl Cl H H') is PSF order.\n"
        "  Hydrogens interleaved (e.g. 'C H H Cl Cl') is NOT — it will mis-assign types."
    )

if not enriched.is_file():
    print(f"\n(no enriched NPZ yet at {enriched} — run 01_prepare_dataset.sh)")
    print("02: OK (raw inspection only)")
    sys.exit(0)

# --- what the assignment produced ------------------------------------------
d = np.load(enriched, allow_pickle=True)
idx = np.asarray(d["cgenff_type_idx"])
chg = np.asarray(d["cgenff_charge"])
mol = np.asarray(d["mol_id"])

real = idx >= 0
print(f"\n-- assignment ({enriched.name}) --")
print(f"  frames          : {len(d['E'])}")
print(f"  atoms/frame     : {idx.shape[1]}  ({real[0].sum()} real, rest padding)")
print(f"  monomers        : {len(np.unique(mol[mol >= 0]))}")
print(f"  net charge/frame: {chg[0][real[0]].sum():+.4f} e")

try:
    from mmml.models.mm_lj_scales import cgenff_type_names_from_prm

    names = cgenff_type_names_from_prm()
    counts = Counter(int(v) for v in idx[real])
    print(f"\n-- types present ({len(counts)} of {len(names)} in the master table) --")
    print("  These are the only types that can receive a gradient:")
    for t, n in counts.most_common():
        frac = n / real.sum()
        print(f"    {names[t]:10s} {n:9d} atoms  ({frac:5.1%})")
except Exception as exc:  # pragma: no cover
    print(f"  (could not resolve type names: {exc})")

print("\n02: OK")
