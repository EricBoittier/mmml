#!/usr/bin/env python3
"""Survey the chemical space of an SO3LR-format dimer extxyz, and how much of it
the hybrid ML/MM CGenFF assignment can actually type.

Two questions in one streaming pass over the file:

1. **What is in there?**  Split each frame into covalent components, render each
   as a Hill formula, and tally monomers, unordered monomer pairs, elements and
   frame sizes.
2. **What can hybrid ML/MM run on?**  Feed every ``--cgenff-stride``-th frame
   through :func:`mmml.data.cgenff_dataset.assign_frame_cgenff` — the same call
   ``mmml prepare-mm-dataset`` makes — and record the matched CGenFF ``RESI``
   pair, or the reason the frame would be dropped.

The JSON this writes is the input to
``scripts/gen_docs_des_chemspace_figures.py``.

Usage (on the cluster that holds the data)::

    python scripts/scan_des_chemical_space.py ~/qcell/qcell_xyz/qcell_dimers.xyz \\
        --out artifacts/des_chemspace/qcell_dimers_scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers

from mmml.data.cgenff_dataset import (
    assign_frame_cgenff,
    find_covalent_components,
    format_composition,
    load_reference,
)


def iter_frames(path: Path, stride: int, limit: int | None):
    """Yield ``(Z, R)`` for every ``stride``-th frame of an extxyz file.

    Streamed rather than read via ASE: these files are ~750 MB and only the
    species and positions are needed here.
    """
    with open(path) as fh:
        seen = kept = 0
        while True:
            head = fh.readline()
            if not head:
                return
            head = head.strip()
            if not head:
                continue
            n_at = int(head)
            fh.readline()  # comment / property line
            body = [fh.readline() for _ in range(n_at)]
            take = (seen % stride) == 0
            seen += 1
            if not take:
                continue
            z = np.empty(n_at, dtype=np.int32)
            r = np.empty((n_at, 3), dtype=np.float64)
            for i, row in enumerate(body):
                f = row.split()
                z[i] = atomic_numbers[f[0]]
                r[i] = (float(f[1]), float(f[2]), float(f[3]))
            yield z, r
            kept += 1
            if limit and kept >= limit:
                return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz", type=Path, help="SO3LR-format dimer extxyz")
    ap.add_argument("--out", type=Path, required=True, help="JSON summary to write")
    ap.add_argument("--stride", type=int, default=1,
                    help="composition survey: keep every Nth frame (default 1)")
    ap.add_argument("--cgenff-stride", type=int, default=20,
                    help="run the CGenFF assignment on every Nth *kept* frame "
                         "(default 20; the assignment is ~100x the cost of a "
                         "composition tally)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N kept frames")
    a = ap.parse_args(argv)

    ref = load_reference()
    print(f"CGenFF reference: {len(ref.residues)} RESI templates, "
          f"{len(ref.sigmas)} nonbonded types", file=sys.stderr)

    # Composition survey
    pairs, monomers, elements = Counter(), Counter(), Counter()
    natoms_hist, ncomp_hist = Counter(), Counter()
    # CGenFF assignment
    pairs_ok, pairs_fail = Counter(), Counter()
    reasons, resi_pairs = Counter(), Counter()
    monomer_ok, monomer_fail = Counter(), Counter()
    resi_of_formula = defaultdict(Counter)

    n_frames = n_typed_attempts = 0
    for z, r in iter_frames(a.xyz, a.stride, a.limit):
        comps = find_covalent_components(z, r)
        # Component order matters below: assign_frame_cgenff re-derives the same
        # components and returns res_names in that order, so the formula -> RESI
        # map must not be built from the sorted view.
        forms_by_comp = [format_composition(z[c]) for c in comps]
        forms = sorted(forms_by_comp)
        key = " + ".join(forms)

        ncomp_hist[len(comps)] += 1
        natoms_hist[len(z)] += 1
        elements.update({int(v) for v in np.unique(z)})
        for f in forms:
            monomers[f] += 1
        pairs[key] += 1

        if n_frames % a.cgenff_stride == 0:
            n_typed_attempts += 1
            assignment, reason = assign_frame_cgenff(z, r, ref, compute_mm=False)
            if assignment is None:
                pairs_fail[key] += 1
                # Bucket by reason class, dropping the frame-specific tail.
                reasons[reason.split("(")[0].strip()[:90]] += 1
                for f in forms:
                    monomer_fail[f] += 1
            else:
                pairs_ok[key] += 1
                resi_pairs[" + ".join(sorted(assignment.res_names))] += 1
                for f, res in zip(forms_by_comp, assignment.res_names):
                    monomer_ok[f] += 1
                    resi_of_formula[f][res] += 1

        n_frames += 1
        if n_frames % 20000 == 0:
            ok = sum(pairs_ok.values())
            print(f"  {n_frames} frames; CGenFF {ok}/{n_typed_attempts} typed",
                  file=sys.stderr, flush=True)

    from ase.data import chemical_symbols
    summary = {
        "source": str(a.xyz),
        "stride": a.stride,
        "cgenff_stride": a.cgenff_stride,
        "frames_scanned": n_frames,
        "elements": sorted(((chemical_symbols[z], c) for z, c in elements.items()),
                           key=lambda kv: -kv[1]),
        "natoms_hist": sorted(natoms_hist.items()),
        "ncomponents_hist": sorted(ncomp_hist.items()),
        "monomers": monomers.most_common(),
        "pairs": pairs.most_common(),
        "cgenff": {
            "frames_attempted": n_typed_attempts,
            "n_typed": sum(pairs_ok.values()),
            "n_failed": sum(pairs_fail.values()),
            "pairs_ok": pairs_ok.most_common(),
            "pairs_fail": pairs_fail.most_common(),
            "reasons": reasons.most_common(),
            "resi_pairs": resi_pairs.most_common(),
            "monomer_ok": monomer_ok.most_common(),
            "monomer_fail": monomer_fail.most_common(),
            "resi_of_formula": {k: v.most_common() for k, v in resi_of_formula.items()},
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1))

    ok = summary["cgenff"]["n_typed"]
    print(f"{n_frames} frames: {len(monomers)} monomers, {len(pairs)} pairs")
    print(f"CGenFF: {ok}/{n_typed_attempts} frames typeable "
          f"({100 * ok / max(n_typed_attempts, 1):.1f}%), "
          f"{len(resi_pairs)} distinct RESI pairs")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
