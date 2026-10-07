"""Drop dimer frames whose monomers are unphysically close, into a NEW npz.

Why
---
On the DES dimer set, the top 0.1% of force components owned **95%** of the
validation force MSE, and every one of the worst frames was a 0.9-1.3 A
intermolecular contact. The labels there are fine (|F|max over the whole set is
248 kcal/mol/A); it is the *model* that blows up, predicting up to 1880. Those
geometries are steeply repulsive, are not sampled in 300 K MD, and sit exactly
where the ML/MM handoff is least constrained.

Measured effect of the cut on validation metrics (job 19360535 checkpoint):

    min-contact cut   frames kept   E MAE   E RMSE   F MAE   F RMSE
                0.0          8480   3.754    9.245   1.885   10.995
                1.5          8059   3.762    9.211   1.651    2.359
                2.0          7062   3.918    9.684   1.503    2.120

So 1.5 A removes the force tail for ~5% of frames. Past ~2.0 A the energy
metrics get *worse* -- you start deleting real interaction physics, not
pathologies. Hence the 1.5 A default.

The criterion is the minimum distance between atoms of *different* monomers
(``mol_id``), over real atoms only (``Z > 0``). At 1.5 A this still keeps
hydrogen bonds, whose H...O contacts are ~1.6-1.8 A.

Never overwrites the input: the filtered set is a new file, so the unfiltered
dataset remains available for comparison.

Usage
-----
    uv run python -m mmml.cli.misc.filter_close_contacts \\
        --in  artifacts/lj_scales_des/des_dimers_cgenff_top50.npz \\
        --out artifacts/lj_scales_des/des_dimers_cgenff_top50_min15.npz \\
        --min-contact 1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EV_PER_ANGSTROM_TO_KCAL = 23.060548012069496


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", required=True, type=Path)
    p.add_argument("--min-contact", type=float, default=1.5,
                   help="drop frames whose closest intermolecular contact is "
                        "below this (Angstrom); default 1.5")
    p.add_argument("--max-force", type=float, default=None,
                   help="also drop frames whose max |F| label exceeds this "
                        "(eV/Angstrom); off by default -- the labels are clean, "
                        "the close contacts are the problem")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be dropped, write nothing")
    return p.parse_args(argv)


def min_intermolecular_distance(R, Z, mol_id):
    """Per-frame closest contact between different monomers; inf if undefined."""
    n = R.shape[0]
    out = np.full(n, np.inf)
    for i in range(n):
        real = Z[i] > 0
        r, m = R[i][real], mol_id[i][real]
        a = np.where(m == 0)[0]
        b = np.where(m == 1)[0]
        if len(a) == 0 or len(b) == 0:
            continue
        d = np.linalg.norm(r[a][:, None, :] - r[b][None, :, :], axis=-1)
        out[i] = d.min()
    return out


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.out.resolve() == args.inp.resolve():
        raise SystemExit("--out must differ from --in (refusing to overwrite)")
    if args.out.exists() and not args.dry_run:
        raise SystemExit(f"--out already exists: {args.out} (move it first)")

    d = np.load(args.inp, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    R, Z, mol_id = data["R"], data["Z"], data["mol_id"]
    n = R.shape[0]
    print(f"input : {args.inp}  ({n} frames)")

    dmin = min_intermolecular_distance(R, Z, mol_id)
    keep = dmin >= float(args.min_contact)
    dropped_contact = int((~keep).sum())

    dropped_force = 0
    if args.max_force is not None:
        F = data["F"]
        fmax = np.array([
            np.linalg.norm(F[i][Z[i] > 0], axis=1).max() if (Z[i] > 0).any() else 0.0
            for i in range(n)
        ])
        force_ok = fmax <= float(args.max_force)
        dropped_force = int((keep & ~force_ok).sum())
        keep &= force_ok

    n_keep = int(keep.sum())
    finite = dmin[np.isfinite(dmin)]
    print(f"\nclosest-contact distribution (A):")
    for q, v in zip((0, 0.1, 1, 5, 50), np.percentile(finite, [0, 0.1, 1, 5, 50])):
        print(f"  p{q:<5} {v:.3f}")
    print(f"\ncut at {args.min_contact} A -> dropped {dropped_contact} frames "
          f"({100*dropped_contact/n:.2f}%)")
    if args.max_force is not None:
        print(f"max|F| > {args.max_force} eV/A -> dropped {dropped_force} more")
    print(f"keeping {n_keep} / {n} frames ({100*n_keep/n:.2f}%)")

    if n_keep < 0.5 * n:
        raise SystemExit(
            f"Refusing to write: the filter would remove {100*(1-n_keep/n):.1f}% "
            "of the dataset. That is a different dataset, not a cleanup -- "
            "re-check --min-contact."
        )

    report = {
        "input": str(args.inp), "output": str(args.out),
        "min_contact_A": float(args.min_contact),
        "max_force_eV_per_A": args.max_force,
        "n_in": n, "n_kept": n_keep,
        "dropped_close_contact": dropped_contact,
        "dropped_high_force": dropped_force,
        "min_contact_kept_A": float(dmin[keep].min()) if n_keep else None,
    }

    if args.dry_run:
        print("\n(dry run: nothing written)")
        print(json.dumps(report, indent=2))
        return 0

    # Per-frame arrays are filtered; per-type tables (cgenff_master_*) are not
    # indexed by frame and must be copied through unchanged.
    filtered = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        filtered[key] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == n else arr

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **filtered)
    print(f"\nwrote {args.out}")
    print(f"closest surviving contact: {report['min_contact_kept_A']:.3f} A")

    report_path = args.report or args.out.with_suffix(".filter.json")
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
