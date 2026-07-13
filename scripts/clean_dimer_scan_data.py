#!/usr/bin/env python3
"""Remove geometrically invalid and clearly failed dimer-scan evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["molecule_a", "molecule_b", "backend"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-contact", type=float, default=1.5)
    parser.add_argument("--min-interaction-kcal", type=float, default=-100.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    energy = pd.to_numeric(df["energy_kcal_mol"], errors="coerce")
    contact = pd.to_numeric(df["min_contact_angstrom"], errors="coerce")
    df = df.copy()
    df["_energy_numeric"] = energy

    # A common far-separation reference per pair/backend avoids treating
    # monomer-energy constants as interaction energy.  Median over all offsets
    # at the largest sampled separation is robust to one failed endpoint.
    far = df.groupby(GROUPS, dropna=False)["distance_angstrom"].transform("max")
    far_rows = df[np.isclose(df["distance_angstrom"], far)]
    refs = far_rows.groupby(GROUPS, dropna=False)["_energy_numeric"].median()
    ref_index = pd.MultiIndex.from_frame(df[GROUPS])
    df["interaction_kcal_mol_for_cleaning"] = energy - refs.reindex(ref_index).to_numpy()

    reason = pd.Series("", index=df.index, dtype=object)
    reason[~np.isfinite(energy)] = "nonfinite_energy"
    reason[(reason == "") & (~np.isfinite(contact))] = "nonfinite_contact"
    reason[(reason == "") & (contact < args.min_contact)] = "close_contact"
    extreme = df["interaction_kcal_mol_for_cleaning"] < args.min_interaction_kcal
    failed_groups = set(
        map(
            tuple,
            df.loc[extreme, GROUPS].drop_duplicates().itertuples(index=False, name=None),
        )
    )
    group_keys = list(df[GROUPS].itertuples(index=False, name=None))
    failed_pair = pd.Series(
        [tuple(key) in failed_groups for key in group_keys], index=df.index
    )
    reason[(reason == "") & failed_pair] = "failed_backend_pair_extreme_attraction"

    rejected = df[reason != ""].copy()
    rejected["rejection_reason"] = reason[reason != ""]
    cleaned = df[reason == ""].drop(columns=["_energy_numeric"])
    rejected = rejected.drop(columns=["_energy_numeric"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    rejected.to_csv(args.audit, index=False)
    print(f"Kept {len(cleaned)}/{len(df)} rows -> {args.output}")
    print(rejected["rejection_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
