#!/usr/bin/env python
"""Copy an efield splits directory with per-element reference energies removed from E.

SPICE-alpha E labels are absolute total energies (~1e3-1e5 in label units).
The model's element_bias starts at zero, so it cannot reach those offsets by
gradient descent. Fit E ~ sum_i e_{Z_i} by least squares on the TRAIN split
and subtract it from every split; forces are unchanged. The fitted e_Z go to
atomic_energy_refs.json so predictions can be shifted back.

Usage:
  python scripts/spice_alpha/subtract_atomic_energies.py SRC_SPLITS DST_SPLITS
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def composition(Z: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """(n_frames, n_elements) element counts, ignoring Z=0 padding."""
    return (Z[:, :, None] == elements[None, None, :]).sum(axis=1).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()

    train = np.load(args.src / "energies_forces_dipoles_train.npz")
    elements = np.unique(train["Z"][train["Z"] > 0])
    A = composition(train["Z"], elements)
    e_ref, *_ = np.linalg.lstsq(A, train["E"].astype(np.float64), rcond=None)

    args.dst.mkdir(parents=True, exist_ok=True)
    report = {"src": str(args.src), "fit_on": "train",
              "e_ref": {int(z): float(e) for z, e in zip(elements, e_ref)}, "splits": {}}
    for src in sorted(args.src.glob("*.npz")):
        data = np.load(src, allow_pickle=False)
        out = {k: data[k] for k in data.files}
        unknown = set(np.unique(data["Z"][data["Z"] > 0])) - set(elements.tolist())
        if unknown:
            raise SystemExit(f"{src.name}: elements {sorted(unknown)} absent from train fit")
        resid = data["E"] - composition(data["Z"], elements) @ e_ref
        out["E"] = resid.astype(data["E"].dtype)
        np.savez(args.dst / src.name, **out)
        report["splits"][src.name] = {"E_std_before": float(data["E"].std()),
                                      "E_std_after": float(resid.std()),
                                      "E_mad_after": float(np.mean(np.abs(resid - resid.mean())))}
        print(f"{src.name}: E std {data['E'].std():.4g} -> {resid.std():.4g}")
    for extra in args.src.glob("*.json"):
        shutil.copy2(extra, args.dst / extra.name)
    (args.dst / "atomic_energy_refs.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
