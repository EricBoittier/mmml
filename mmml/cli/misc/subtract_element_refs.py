"""Remove per-element reference energies (fitted on THIS dataset) into a new npz.

Why
---
Training the DES dimer set from scratch, energy MAE sat at ~1200 kcal/mol for
six epochs and never moved, while forces converged normally (579 -> 16). The
cause is scale, not optimisation: raw E has std 805.8 kcal/mol, and a plain
per-element least-squares fit on element counts explains **99.97%** of its
variance. The network was being asked to rediscover ~800 kcal/mol of pure
composition offset through `energy_bias`, which starts at zero -- Adam at
lr=1e-3 does not travel that far in 40 epochs. A warm start hides the problem
by carrying those offsets in its weights, which is exactly why the warm-started
baseline reached 3.75 kcal/mol and the from-scratch run did not.

    raw      : std = 805.8 kcal/mol
    residual : std =  12.8 kcal/mol   MAE = 9.3   max = 79.2

After subtraction the model only fits real chemistry (~13 kcal/mol), so a
< 1 kcal/mol target is a 13:1 reduction rather than 800:1.

Not the same as ``--subtract-atom-energies``: that flag uses *free-atom*
reference energies from ``mmml.data.units``, which are in a different
convention than these labels -- applying it moves the mean from -57 eV to
+8591 eV, i.e. makes the problem far worse. The references here are fitted on
the dataset itself and are therefore self-consistent by construction.

Effect on downstream use: the shift is a constant per composition, so forces
are untouched and energy *differences* within a fixed system (all that MD and
the dimer interaction energy depend on) are unchanged. The fitted coefficients
are stored in the output npz as ``element_ref_Z`` / ``element_ref_E_eV`` so the
absolute scale can always be restored.

Usage
-----
    uv run python -m mmml.cli.misc.subtract_element_refs \\
        --in  artifacts/.../des_dimers_cgenff_top50_min15.npz \\
        --out artifacts/.../des_dimers_cgenff_top50_min15_eref.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EV_TO_KCAL_MOL = 23.060548012069496


def fit_element_refs(E, Z):
    """Least-squares per-element energy from element counts. Returns (zs, coef)."""
    E = np.asarray(E, dtype=np.float64).ravel()
    Z = np.asarray(Z)
    zs = np.unique(Z[Z > 0])
    counts = np.stack([(Z == z).sum(axis=1) for z in zs], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(counts, E, rcond=None)
    return zs, coef, counts


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", required=True, type=Path)
    p.add_argument("--min-variance-explained", type=float, default=0.99,
                   help="abort if the per-element fit explains less than this "
                        "fraction of the energy variance (default 0.99); below "
                        "it, composition is not the dominant term and this "
                        "transform is not the right tool")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.out.resolve() == args.inp.resolve():
        raise SystemExit("--out must differ from --in (refusing to overwrite)")
    if args.out.exists() and not args.dry_run:
        raise SystemExit(f"--out already exists: {args.out} (move it first)")

    d = np.load(args.inp, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    E = np.asarray(data["E"], dtype=np.float64).ravel()
    Z = np.asarray(data["Z"])

    zs, coef, counts = fit_element_refs(E, Z)
    residual = E - counts @ coef
    var_explained = 1.0 - residual.var() / E.var()

    print(f"input : {args.inp}  ({len(E)} frames)")
    print(f"elements fitted: {zs.tolist()}")
    print(f"  {'Z':>4}  {'E_ref (eV)':>14}")
    for z, c in zip(zs, coef):
        print(f"  {int(z):>4}  {c:>14.6f}")
    print(f"\nvariance explained by composition: {100*var_explained:.4f}%")
    print(f"raw      : std={E.std()*EV_TO_KCAL_MOL:10.1f} kcal/mol")
    print(f"residual : std={residual.std()*EV_TO_KCAL_MOL:10.1f} kcal/mol  "
          f"max|dev|={np.abs(residual - residual.mean()).max()*EV_TO_KCAL_MOL:.1f}")

    if var_explained < args.min_variance_explained:
        raise SystemExit(
            f"Composition explains only {100*var_explained:.2f}% of the energy "
            f"variance (< {100*args.min_variance_explained:.2f}%). Subtracting "
            "per-element refs would remove little and could mask a real signal."
        )

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    data["E"] = residual.reshape(-1, 1)
    # Keep the fit so the absolute scale is always recoverable.
    data["element_ref_Z"] = zs.astype(np.int32)
    data["element_ref_E_eV"] = coef.astype(np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"\nwrote {args.out}")

    report = {
        "input": str(args.inp), "output": str(args.out),
        "n_frames": int(len(E)),
        "variance_explained": float(var_explained),
        "raw_std_kcal": float(E.std() * EV_TO_KCAL_MOL),
        "residual_std_kcal": float(residual.std() * EV_TO_KCAL_MOL),
        "element_refs_eV": {int(z): float(c) for z, c in zip(zs, coef)},
    }
    rp = args.out.with_suffix(".eref.json")
    rp.write_text(json.dumps(report, indent=2))
    print(f"wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
