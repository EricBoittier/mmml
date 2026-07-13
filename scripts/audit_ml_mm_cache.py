#!/usr/bin/env python3
"""Audit an ML/MM Orbax cache for composition and target pathologies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp


SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def stats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    good = x[finite]
    out = {"count": int(x.size), "finite": int(finite.sum()), "nonfinite": int((~finite).sum())}
    if good.size:
        qs = np.quantile(good, [0, .001, .01, .5, .99, .999, 1])
        out.update({k: float(v) for k, v in zip(("min", "q001", "q01", "median", "q99", "q999", "max"), qs)})
        out["mean"] = float(good.mean())
        out["std"] = float(good.std())
    return out


def formula(z: np.ndarray) -> str:
    counts = Counter(map(int, z))
    order = [6, 1] + sorted(k for k in counts if k not in (6, 1))
    return "".join(f"{SYMBOLS.get(k, f'Z{k}')}{counts[k] if counts[k] != 1 else ''}" for k in order if k in counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = ocp.PyTreeCheckpointer().restore(str(Path(args.cache).expanduser().resolve()))
    arrays = {k: np.asarray(v) for k, v in data.items()}
    z = arrays["Z"].reshape(-1)
    offsets = arrays["mol_offsets"].reshape(-1).astype(np.int64)
    n_frames = offsets.size - 1
    formulas = []
    water_monomer, water_dimer, ho_only = [], [], []
    for i in range(n_frames):
        zi = z[offsets[i]:offsets[i + 1]]
        f = formula(zi)
        formulas.append(f)
        if f == "H2O": water_monomer.append(i)
        if f == "H4O2": water_dimer.append(i)
        if set(map(int, zi)) <= {1, 8}: ho_only.append(i)

    report = {
        "cache": str(Path(args.cache).expanduser().resolve()),
        "frames": n_frames,
        "atoms": int(z.size),
        "keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "top_formulas": dict(Counter(formulas).most_common(100)),
        "water": {"H2O_frames": len(water_monomer), "H4O2_frames": len(water_dimer), "HO_only_frames": len(ho_only)},
        "targets": {},
    }
    for key in ("E", "E_cgenff_mm", "F", "F_cgenff_mm", "Q", "D", "cgenff_charge", "cgenff_master_sigmas", "cgenff_master_epsilons"):
        if key in arrays:
            report["targets"][key] = stats(arrays[key])
    if "E" in arrays and "E_cgenff_mm" in arrays:
        report["targets"]["E_minus_E_cgenff_mm"] = stats(arrays["E"] - arrays["E_cgenff_mm"])
    if "F" in arrays and "F_cgenff_mm" in arrays:
        report["targets"]["F_minus_F_cgenff_mm"] = stats(arrays["F"] - arrays["F_cgenff_mm"])

    for label, indices in (("water_dimer", water_dimer), ("HO_only", ho_only)):
        subset = {}
        idx = np.asarray(indices, dtype=np.int64)
        for key in ("E", "E_cgenff_mm", "Q", "D"):
            if key in arrays and idx.size:
                subset[key] = stats(arrays[key][idx])
        if idx.size and "E" in arrays and "E_cgenff_mm" in arrays:
            subset["E_minus_E_cgenff_mm"] = stats(arrays["E"][idx] - arrays["E_cgenff_mm"][idx])
        atom_idx = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices]) if indices else np.array([], dtype=int)
        if atom_idx.size and "cgenff_charge" in arrays:
            for atomic_number in (1, 8):
                chosen = atom_idx[z[atom_idx] == atomic_number]
                subset[f"charge_{SYMBOLS[atomic_number]}"] = stats(arrays["cgenff_charge"].reshape(-1)[chosen])
        report["water"][label] = subset

    e = arrays.get("E", np.array([])).reshape(-1)
    emm = arrays.get("E_cgenff_mm", np.array([])).reshape(-1)
    for key, values in (("E", e), ("E_cgenff_mm", emm), ("E_minus_E_cgenff_mm", e - emm if e.size and emm.size else np.array([]))):
        if values.size:
            order = np.argsort(np.abs(values))[-20:][::-1]
            report.setdefault("largest_absolute", {})[key] = [
                {"frame": int(i), "formula": formulas[int(i)], "value": float(values[int(i)])} for i in order
            ]

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"water": report["water"], "targets": report["targets"], "largest_absolute": report.get("largest_absolute", {})}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
