"""Score checkpoints against the PBE0-D3BJ reference PES.

Two complementary metrics, both restricted to pairs that exist in the training data:
  - RMSE over the binding window (-10 < E_ref < +5 kcal/mol)
  - well-depth error at the reference minimum geometry (physically interpretable)

Usage:  python local_validation/rescore.py LABEL=surfaces/foo.csv [LABEL=... ...]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

KEY = ["molecule_a", "molecule_b", "distance_angstrom", "offset_angstrom"]

# training coverage in the v1 cache; DCM-MEOH (0 structures) is excluded entirely
N_TRAIN = {
    "ACE-TIP3": 1632, "MEOH-MEOH": 1407, "TIP3-TIP3": 885, "BENZ-TIP3": 809,
    "TIP3-MEOH": 317, "ACE-ACE": 125, "ACE-MEOH": 29, "BENZ-BENZ": 13,
    "DCM-TIP3": 10, "DCM-BENZ": 4, "BENZ-MEOH": 3, "DCM-DCM": 3,
    "ACE-BENZ": 2, "DCM-ACE": 2,
}


def load_reference() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob("local_validation/reference_pes*.csv"))]
    ref = pd.concat(frames, ignore_index=True).rename(columns={"energy_kcal_mol": "E_ref"})
    return ref.drop_duplicates(subset=KEY)


def main() -> None:
    specs = [s.split("=", 1) for s in sys.argv[1:]]
    if not specs:
        sys.exit(__doc__)

    ref = load_reference()
    R = ref.set_index(KEY).E_ref

    models = {}
    for label, path in specs:
        d = pd.read_csv(path if path.startswith("local_validation") else f"local_validation/{path}")
        d = d[d.backend.str.startswith("spookynet_hybrid")]
        models[label] = d.set_index(KEY).comp_Eint_kcal_mol

    def window_rmse(label: str, pair: tuple[str, str]) -> float:
        m = models[label]
        j = m.index.intersection(R.index)
        mm, rr = m.loc[j], R.loc[j]
        sel = ((mm.index.get_level_values(0) == pair[0])
               & (mm.index.get_level_values(1) == pair[1])
               & (rr > -10) & (rr < 5))
        err = (mm[sel] - rr[sel]).dropna()
        return float(np.sqrt((err ** 2).mean())) if len(err) else np.nan

    def well_error(label: str, pair: tuple[str, str]) -> float:
        """Model minus reference at the reference's own minimum geometry."""
        g = ref[(ref.molecule_a == pair[0]) & (ref.molecule_b == pair[1])]
        if g.empty:
            return np.nan
        row = g.loc[g.E_ref.idxmin()]
        key = (pair[0], pair[1], row.distance_angstrom, row.offset_angstrom)
        val = models[label].get(key, np.nan)
        return float(val - row.E_ref)

    labels = [lab for lab, _ in specs]
    pairs = [tuple(p.split("-")) for p in N_TRAIN]

    print("=== RMSE in binding window (kcal/mol), trained pairs only ===")
    print(f"{'pair':<11}{'n_train':>8}" + "".join(f"{l:>16}" for l in labels))
    rmse_all: dict[str, list[float]] = {l: [] for l in labels}
    rmse_rich: dict[str, list[float]] = {l: [] for l in labels}
    for pair in sorted(pairs, key=lambda p: -N_TRAIN[f"{p[0]}-{p[1]}"]):
        name = f"{pair[0]}-{pair[1]}"
        line = f"{name:<11}{N_TRAIN[name]:>8}"
        for lab in labels:
            v = window_rmse(lab, pair)
            line += f"{v:16.2f}"
            rmse_all[lab].append(v)
            if N_TRAIN[name] >= 300:
                rmse_rich[lab].append(v)
        print(line)

    print(f"\n{'MEAN (all trained)':<19}" + "".join(f"{np.nanmean(rmse_all[l]):16.2f}" for l in labels))
    print(f"{'MEAN (n>=300)':<19}" + "".join(f"{np.nanmean(rmse_rich[l]):16.2f}" for l in labels))

    print("\n=== Well-depth error at the reference minimum (model - ref, kcal/mol) ===")
    print(f"{'pair':<11}{'n_train':>8}" + "".join(f"{l:>16}" for l in labels))
    for pair in sorted(pairs, key=lambda p: -N_TRAIN[f"{p[0]}-{p[1]}"]):
        name = f"{pair[0]}-{pair[1]}"
        line = f"{name:<11}{N_TRAIN[name]:>8}"
        for lab in labels:
            line += f"{well_error(lab, pair):+16.2f}"
        print(line)
    print(f"\n{'MEAN |well err|':<19}"
          + "".join(f"{np.nanmean([abs(well_error(l, p)) for p in pairs]):16.2f}" for l in labels))

    print("\n=== RANKING (mean binding-window RMSE over trained pairs) ===")
    for i, (lab, v) in enumerate(
        sorted(((l, np.nanmean(rmse_all[l])) for l in labels), key=lambda x: x[1]), 1
    ):
        print(f"  {i}. {lab:<22} {v:.3f} kcal/mol")


if __name__ == "__main__":
    main()
