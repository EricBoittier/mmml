#!/usr/bin/env python3
"""Quick overnight summary for dense_dt_campaign H5s (manuscript §§7–8)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/lj_scales/dense_dt_campaign"


def _series(h5: h5py.File, *names: str):
    for n in names:
        if n in h5:
            return np.asarray(h5[n][...], dtype=float)
        if "thermo" in h5 and n in h5["thermo"]:
            return np.asarray(h5["thermo"][n][...], dtype=float)
    return None


def summarize_h5(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        keys = sorted(list(h.keys()) + ([f"thermo/{k}" for k in h["thermo"].keys()] if "thermo" in h else []))
        etot = _series(h, "E_tot", "e_tot", "energy")
        hnhc = _series(h, "invariant", "H_NHC", "h_nhc")
        dens = _series(h, "density_g_cm3", "density")
        out = {"path": str(path), "n_frames": int(etot.shape[0]) if etot is not None else 0, "keys_sample": keys[:40]}
        if etot is not None and etot.size:
            out["E_tot_start"] = float(etot[0])
            out["E_tot_end"] = float(etot[-1])
            out["dE_tot"] = float(etot[-1] - etot[0])
        if hnhc is not None and hnhc.size:
            out["H_start"] = float(hnhc[0])
            out["H_end"] = float(hnhc[-1])
            out["dH"] = float(hnhc[-1] - hnhc[0])
        if dens is not None and dens.size:
            out["rho_mean"] = float(np.nanmean(dens))
            out["rho_end"] = float(dens[-1])
        return out


def main() -> int:
    rows = []
    for tag_dir in sorted(OUT.glob("*/")):
        if tag_dir.name == "logs":
            continue
        h5s = list(tag_dir.glob("*.h5"))
        if not h5s:
            continue
        for h5 in h5s:
            try:
                rows.append({"tag": tag_dir.name, **summarize_h5(h5)})
            except Exception as exc:  # noqa: BLE001
                rows.append({"tag": tag_dir.name, "path": str(h5), "error": str(exc)})
    out_path = OUT / "summary.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
