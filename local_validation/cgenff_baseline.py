"""Compute the pure CGenFF interaction energy (LJ + Coulomb) on the dimer scan grid.

Bypasses the CHARMM backend in run_dimer_scan_campaign, whose cluster build leaves a
persistent steric clash for several pairs (a constant ~4e5 kcal/mol offset in the VDW
term). Here the CGenFF energy is evaluated directly from the validated per-atom types
and charges, on exactly the geometries the model is scanned on.

Writes a CSV with the columns the rescore script expects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

K_COULOMB = 332.06371  # e^2/A -> kcal/mol


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scan = _load("scan", ROOT / "scripts" / "run_dimer_scan_campaign.py")
prep = _load("prep", ROOT / "scripts" / "prepare_ml_mm_dataset.py")

SIG, EPS = prep._CGENFF_SIGMAS, prep._CGENFF_EPSILONS


def cgenff_interaction(z_a, r_a, z_b, r_b) -> float:
    """Inter-monomer CGenFF LJ + Coulomb, in kcal/mol."""
    _, t_a, q_a = prep.match_cgenff_template_fast(z_a, r_a, target_charge=0.0)
    _, t_b, q_b = prep.match_cgenff_template_fast(z_b, r_b, target_charge=0.0)

    d = np.linalg.norm(r_a[:, None, :] - r_b[None, :, :], axis=-1)
    sig_ij = 0.5 * (SIG[t_a][:, None] + SIG[t_b][None, :])
    eps_ij = np.sqrt(np.abs(EPS[t_a][:, None] * EPS[t_b][None, :]))
    q_ij = q_a[:, None] * q_b[None, :]

    sr6 = (sig_ij / d) ** 6
    e_lj = float(np.sum(4.0 * eps_ij * (sr6**2 - sr6)))
    e_el = float(np.sum(K_COULOMB * q_ij / d))
    return e_lj + e_el


def main() -> None:
    rows = []
    for (label_a, label_b), cfg in sorted(scan.PAIR_SCAN_CONFIG.items()):
        distances, _ = scan.build_pair_distance_grid(label_a, label_b)
        offsets = cfg["offsets_angstrom"]
        for g in scan.make_oriented_scan_geometries(label_a, label_b, distances, offsets):
            z = g.atoms.get_atomic_numbers()
            pos = g.atoms.get_positions()
            idx_a, idx_b = g.fragments
            e = cgenff_interaction(
                np.asarray(z)[list(idx_a)], np.asarray(pos)[list(idx_a)],
                np.asarray(z)[list(idx_b)], np.asarray(pos)[list(idx_b)],
            )
            rows.append({
                "molecule_a": label_a,
                "molecule_b": label_b,
                "distance_angstrom": g.distance_angstrom,
                "offset_angstrom": g.offset_angstrom,
                "comp_Eint_kcal_mol": e,
                "backend": "spookynet_hybrid_cgenff_direct",
            })
    df = pd.DataFrame(rows)
    out = ROOT / "local_validation" / "surfaces" / "cgenff_direct.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {df.groupby(['molecule_a','molecule_b']).ngroups} pairs)")
    print("\nCGenFF well depth per pair (kcal/mol):")
    print(df.groupby(["molecule_a", "molecule_b"]).comp_Eint_kcal_mol.min().round(2).to_string())


if __name__ == "__main__":
    main()
