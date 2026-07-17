#!/usr/bin/env python3
"""Detect spurious minima in hybrid ML/MM dimer scans (1D cuts over a 2D grid).

A rigid dimer scan at fixed orientation should have ONE minimum: attraction in,
repulsion out. Extra minima, or a non-monotonic repulsive wall, are artifacts --
and MD will find them, because a gradient descent needs no thermal barrier.

The point of this tool is to say WHICH term is responsible, because the fix
differs completely:

* **handoff-induced** -- the taper blends ML out and MM in badly, leaving a dip
  or bump in the switching window ``[mm_switch_on - ml_switch_width, mm_switch_on]``.
  Mitigation: move ``mm_switch_on`` outward (and ``--cutoff`` with it) so the
  handoff happens where the interaction is already negligible; or widen
  ``ml_switch_width`` so the blend is gentler.
* **model-intrinsic** -- the feature lives where the taper is ~1 (pure ML) and
  survives with the handoff disabled. Moving the handoff CANNOT fix it. The
  model itself is wrong there; mitigations are retraining (more/denser data in
  that region, a repulsive prior such as ZBL, or a smoothness penalty), not
  cutoff tuning.
* **wall-induced** -- the short-range wall is doing it (``wall_E`` non-zero).
  Mitigation: soften ``k`` or lower ``r_on``.

Classification is by evidence, not assertion: each detected extremum is tagged
with the region it sits in and the value of every term there, so the term
carrying the feature is visible rather than inferred.

Noise: ``prominence`` filters float32 wiggle at the asymptote (~1e-4 eV) which
would otherwise show up as dozens of "minima".

    python scripts/check_spurious_minima.py --scan-dir dimer_scans
    python scripts/check_spurious_minima.py --scan-dir a --compare-dir b
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.0605


def load_scan(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    out = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    out["E_int"] = out["E_total"] - out["E_total"][-1]
    return out


def find_extrema(r: np.ndarray, e: np.ndarray, prominence: float):
    """Interior local minima/maxima whose depth exceeds ``prominence``.

    Prominence is measured against the neighbouring turning points, so a
    shoulder on a monotonic curve is not reported and float32 ripple at the
    asymptote is filtered out.
    """
    mins, maxs = [], []
    for i in range(1, len(e) - 1):
        if e[i] < e[i - 1] and e[i] < e[i + 1]:
            left = np.max(e[:i]) if i else e[i]
            right = np.max(e[i + 1:])
            if min(left, right) - e[i] >= prominence:
                mins.append(i)
        elif e[i] > e[i - 1] and e[i] > e[i + 1]:
            left = np.min(e[:i]) if i else e[i]
            right = np.min(e[i + 1:])
            if e[i] - max(left, right) >= prominence:
                maxs.append(i)
    return mins, maxs


def region_of(r: float, mm_switch_on: float, ml_switch_width: float, mm_switch_width: float) -> str:
    if r < mm_switch_on - ml_switch_width:
        return "ML-only"
    if r <= mm_switch_on:
        return "HANDOFF"
    if r <= mm_switch_on + mm_switch_width:
        return "MM-tail"
    return "beyond"


def attribute(d: dict, i: int) -> str:
    """Which term carries the feature at index i?"""
    terms = {
        "ml_2b": abs(d["ml_2b_E"][i]),
        "mm": abs(d["mm_E"][i]),
        "wall": abs(d.get("wall_E", np.zeros_like(d["r_com"]))[i]),
    }
    top = max(terms, key=terms.get)
    if terms[top] < 1e-6:
        return "none (flat)"
    return top


def analyse(name: str, d: dict, args) -> dict:
    r, e = d["r_com"], d["E_int"]
    mins, maxs = find_extrema(r, e, args.prominence)
    print(f"\n=== {name} ===  n={len(r)}  spacing={r[1]-r[0]:.3f} A  "
          f"prominence>{args.prominence:g} eV")
    print(f"  handoff window: [{args.mm_switch_on - args.ml_switch_width:g}, "
          f"{args.mm_switch_on:g}] A")

    verdicts = []
    for i in mins:
        reg = region_of(float(r[i]), args.mm_switch_on, args.ml_switch_width, args.mm_switch_width)
        car = attribute(d, i)
        print(f"  MIN  r={r[i]:6.2f}  E_int={e[i]:9.4f} eV ({e[i]*EV_TO_KCAL:7.2f} kcal/mol)"
              f"  [{reg:8s}] carried by {car:6s}"
              f"  ml_2b={d['ml_2b_E'][i]:8.4f} mm={d['mm_E'][i]:8.4f}")
        verdicts.append((float(r[i]), reg, car))
    for i in maxs:
        reg = region_of(float(r[i]), args.mm_switch_on, args.ml_switch_width, args.mm_switch_width)
        # A barrier at positive E_int inside the repulsive wall means the wall is
        # non-monotonic -- an atom can be pulled INWARD. Call that out.
        flag = "  <- non-monotonic repulsion" if e[i] > 0 else ""
        print(f"  MAX  r={r[i]:6.2f}  E_int={e[i]:9.4f} eV ({e[i]*EV_TO_KCAL:7.2f} kcal/mol)"
              f"  [{reg:8s}]{flag}")

    n_min = len(mins)
    if n_min <= 1:
        print(f"  VERDICT: OK ({n_min} minimum)")
        return {"name": name, "ok": True, "n_min": n_min, "verdicts": verdicts}

    print(f"  VERDICT: SPURIOUS -- {n_min} minima where a rigid scan admits one")
<<<<<<< HEAD
    # Report a cause PER feature: a scan can have several, with different fixes.
    # Collapsing them into one verdict hides that.
    for r_i, reg, car in verdicts:
        if car == "mm" or reg == "HANDOFF" and car != "ml_2b":
            print(f"    r={r_i:.2f} HANDOFF-INDUCED (carried by {car}): move --mm-switch-on "
                  f"outward with --cutoff, or widen --ml-switch-width.")
        elif car == "wall":
            print(f"    r={r_i:.2f} WALL-INDUCED: soften k / lower r_on in short_range_wall.py.")
        elif car == "ml_2b":
            print(f"    r={r_i:.2f} MODEL-INTRINSIC (carried by ml_2b, taper ~1 here).")
            print("           Moving --mm-switch-on on THIS checkpoint cannot fix it. But the")
            print("           model is a function of the cutoff it was TRAINED at: rerun this")
            print("           check on a checkpoint trained at a longer cutoff before")
            print("           concluding the artifact is inherent (--compare-dir).")
=======
    regions = {v[1] for v in verdicts}
    carriers = {v[2] for v in verdicts}
    if regions <= {"HANDOFF"} or carriers == {"mm"}:
        print("    cause: HANDOFF. Move --mm-switch-on outward (and --cutoff with it),")
        print("           or widen --ml-switch-width for a gentler blend.")
    elif "ML-only" in regions and "ml_2b" in carriers:
        print("    cause: MODEL-INTRINSIC. The feature sits where the taper is ~1 and is")
        print("           carried by ml_2b, so moving --mm-switch-on CANNOT fix it.")
        print("           Mitigate by retraining: denser/short-range data, a repulsive")
        print("           prior (ZBL), or a smoothness penalty. Cutoff tuning is a no-op.")
    elif "wall" in carriers:
        print("    cause: WALL. Soften k or lower r_on in mmml/models/short_range_wall.py.")
>>>>>>> 8cd5d69b4 (spur)
    return {"name": name, "ok": False, "n_min": n_min, "verdicts": verdicts}


def main() -> int:
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-dir", required=True, help="dir of scan_*.csv from scan_hybrid_dimer.py")
    p.add_argument("--compare-dir", default=None,
                   help="second scan dir (e.g. a different cutoff/model) to compare against")
    p.add_argument("--prominence", type=float, default=1e-3,
                   help="minimum depth (eV) for a feature to count; filters float32 ripple")
    p.add_argument("--ml-switch-width", type=float, default=DEFAULT_ML_SWITCH_WIDTH)
    p.add_argument("--mm-switch-on", type=float, default=DEFAULT_MM_SWITCH_ON)
    p.add_argument("--mm-switch-width", type=float, default=DEFAULT_MM_SWITCH_WIDTH)
    args = p.parse_args()

    scans = sorted(Path(args.scan_dir).glob("scan_*.csv"))
    if not scans:
        print(f"no scan_*.csv in {args.scan_dir}", file=sys.stderr)
        return 1

    bad = 0
    for s in scans:
        res = analyse(f"{s.stem} [{args.scan_dir}]", load_scan(s), args)
        bad += not res["ok"]
        if args.compare_dir:
            other = Path(args.compare_dir) / s.name
            if other.exists():
                res2 = analyse(f"{s.stem} [{args.compare_dir}]", load_scan(other), args)
                if res["n_min"] != res2["n_min"]:
                    print(f"  >>> DIFFERS: {res['n_min']} minima vs {res2['n_min']} -- the "
                          f"artifact depends on the setting being compared")
                else:
                    print(f"  >>> SAME minima count ({res['n_min']}) in both -- the artifact is "
                          f"INDEPENDENT of the compared setting")

    print(f"\n{'FAIL' if bad else 'OK'}: {bad}/{len(scans)} scans show spurious minima")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
