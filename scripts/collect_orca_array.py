#!/usr/bin/env python3
"""Collect an ORCA array run into a training npz (hybrid-MM ready).

Reads the compact .dat files written by run_array.sh, converts to the training
convention, and re-attaches the CGenFF fields from the source npz -- the
geometries are the same, so the types/charges/master tables carry over exactly.

Units: ORCA gives Hartree and Hartree/Bohr. This writes eV and eV/Angstrom, with
force = -gradient. This codebase has already eaten one silent 23x unit error;
the conversion is explicit and recorded in _mmml_units.

Per-element references are refitted here rather than reused from the GFN2 set:
they are level-of-theory specific, and the GFN2 refs would leave a residual that
is neither an atomization energy nor small.

    python scripts/collect_orca_array.py --run-dir orca_run \\
        --source gfn2_nms15_train.npz gfn2_nms15_valid.npz gfn2_nms15_test.npz \\
        --out pbe0_nms15.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.make_dimer_scan_dataset import fit_atom_refs  # noqa: E402


def parse_dat(path: Path, n_at: int):
    """(energy Eh, gradient (n,3) Eh/bohr, dipole (3,) au) or None."""
    txt = path.read_text().splitlines()
    try:
        e = float(txt[1])
    except (IndexError, ValueError):
        return None
    g, dip, mode = [], np.zeros(3), None
    for line in txt[2:]:
        s = line.strip()
        if s.startswith("# gradient"):
            mode = "g"; continue
        if s.startswith("# dipole"):
            mode = "d"; continue
        if not s:
            continue
        if mode == "g":
            try:
                g.append(float(s))
            except ValueError:
                pass
        elif mode == "d":
            p = s.split()
            if len(p) >= 3:
                try:
                    dip = np.array([float(x) for x in p[:3]])
                except ValueError:
                    pass
    if len(g) != 3 * n_at:
        return None
    return e, np.asarray(g).reshape(n_at, 3), dip


def main() -> int:
    from mmml.data.units import (
        BOHR_TO_ANGSTROM,
        HARTREE_BOHR_TO_EV_ANGSTROM,
        HARTREE_TO_EV,
    )

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--source", nargs="+", required=True,
                   help="the npz(s) the geometries came from (for CGenFF fields)")
    p.add_argument("--out", default="pbe0_nms15.npz")
    p.add_argument("--method", default="PBE0-D4/def2-TZVP")
    args = p.parse_args()

    run = Path(args.run_dir)
    idx = np.load(run / "index.npz", allow_pickle=True)
    Z_all, N_all = np.asarray(idx["Z"]), np.asarray(idx["N"])
    names_all = np.array([str(x) for x in idx["res_name"]])
    n_tot, n_at = Z_all.shape

    src = [dict(np.load(f, allow_pickle=True)) for f in args.source]
    T_all = np.concatenate([np.asarray(s["cgenff_type_idx"]) for s in src])
    Q_all = np.concatenate([np.asarray(s["cgenff_charge"]) for s in src])
    M_all = np.concatenate([np.asarray(s["mol_id"]) for s in src])
    R_all = np.concatenate([np.asarray(s["R"]) for s in src])
    if len(T_all) != n_tot:
        print(f"source has {len(T_all)} structures, index has {n_tot} -- the "
              f"--source list must match --data given to make_orca_array.py",
              file=sys.stderr)
        return 1

    E = np.full(n_tot, np.nan)
    F = np.zeros((n_tot, n_at, 3))
    D = np.zeros((n_tot, 3))
    n_missing = 0
    for i in range(n_tot):
        f = run / "dat" / f"{i:06d}.dat"
        if not f.is_file():
            n_missing += 1
            continue
        got = parse_dat(f, int(N_all[i]))
        if got is None:
            n_missing += 1
            continue
        e, g, dip = got
        E[i] = e * HARTREE_TO_EV
        F[i, : int(N_all[i])] = -g * HARTREE_BOHR_TO_EV_ANGSTROM  # force = -grad
        D[i] = dip * BOHR_TO_ANGSTROM                             # a.u. -> e*A

    keep = ~np.isnan(E)
    n_fail = len(list((run / "failed").glob("*.txt")))
    print(f"{keep.sum()}/{n_tot} collected  ({n_missing} missing/unparsed, "
          f"{n_fail} marked failed by the runner)")
    if keep.sum() == 0:
        return 1
    if n_missing:
        print("  NOTE: the array is resumable -- rerun sbatch to fill the gaps "
              "before collecting, rather than training on a partial set.")

    E_raw = E[keep]
    refs, fitted, elems = fit_atom_refs(E_raw, Z_all[keep], N_all[keep])
    E_ref = E_raw - fitted
    print(f"per-element refs ({args.method}, fitted): "
          + ", ".join(f"Z={int(z)}:{refs[int(z)]:.3f}" for z in elems))
    print("  (rank-deficient with two species: a minimum-norm split, NOT physical "
          "atom energies; only the residuals matter and they will not extrapolate "
          "to a new composition)")
    print(f"  E before: {E_raw.min():10.2f} .. {E_raw.max():10.2f} eV "
          f"(spread {np.ptp(E_raw):.1f})")
    print(f"  E after : {E_ref.min():10.2f} .. {E_ref.max():10.2f} eV "
          f"(spread {np.ptp(E_ref):.1f})")

    common = dict(
        atom_ref_energies=refs,
        cgenff_master_sigmas=np.asarray(src[0]["cgenff_master_sigmas"]),
        cgenff_master_epsilons=np.asarray(src[0]["cgenff_master_epsilons"]),
        _mmml_units=np.array(["energy=eV", "forces=eV/Angstrom", "coords=Angstrom",
                              "dipole=e*Angstrom", f"method={args.method}",
                              "E=atom-referenced", "E_total=raw"]),
    )
    out = Path(args.out)
    rng = np.random.default_rng(0)
    nk = int(keep.sum())
    perm = rng.permutation(nk)
    n_tr, n_va = int(0.8 * nk), int(0.1 * nk)
    base = str(out.with_suffix(""))
    Rk, Zk, Nk = R_all[keep], Z_all[keep], N_all[keep]
    Tk, Qk, Mk = T_all[keep], Q_all[keep], M_all[keep]
    nk_names = names_all[keep]
    for tag, sel in (("train", perm[:n_tr]), ("valid", perm[n_tr:n_tr + n_va]),
                     ("test", perm[n_tr + n_va:])):
        f = Path(f"{base}_{tag}.npz")
        np.savez(f, R=Rk[sel], Z=Zk[sel], N=Nk[sel],
                 E=E_ref[sel].reshape(-1, 1), E_total=E_raw[sel].reshape(-1, 1),
                 F=F[keep][sel], D=D[keep][sel], mol_id=Mk[sel],
                 cgenff_type_idx=Tk[sel], cgenff_charge=Qk[sel],
                 res_name=nk_names[sel], **common)
        print(f"-> {f}  ({len(sel)} structures)")
    print(f"\n{args.method}; eV, eV/A, dipole e*A. Same geometries as the GFN2 set, "
          f"so the orientation-scan gate transfers directly.")
    print("Ready for: mmml physnet-train --hybrid-mm --data <base>_train.npz "
          "--valid-data <base>_valid.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
