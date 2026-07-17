#!/usr/bin/env python3
"""Build a dimer-scan dataset: GFN2-xTB (default) or HF on GPU (gpu4pyscf).

WHY this exists. The current dimer data are MD snapshots of a liquid, so they
are overwhelmingly *separated* pairs: DCM,DCM has a median r_com of 9.76 A and a
median closest contact of 7.71 A, with only ~1% of dimers below 2.64 A contact.
The binding region -- the part that decides whether MD is stable -- is a thin
tail. Models fit on it score a 0.053 eV validation MAE while being wrong by
1.4 eV on close approaches they never saw (checked against GFN2-xTB), and MD
then falls into those holes. Validation MAE cannot see this: the metric and the
defect live in disjoint regions.

So this samples the region the old data lacks, deliberately and uniformly:
S^2 approach directions x SO(3) monomer orientations x r_com, with r weighted
toward the binding/repulsive range rather than the thermal one.

Geometry sampling is imported from scan_dimer_orientations, so a dataset built
here is directly comparable to the diagnostics that motivated it.

Units: pyscf returns Hartree and Hartree/Bohr; this writes **eV and eV/Angstrom**
to match the training pipeline (mmml/data/units.py declares eV canonical).
Getting that wrong is a silent 27x error -- it has already happened once in this
codebase.

Output is a raw npz (R/Z/N/E/F/D/mol_id/res_name). Run
scripts/prepare_ml_mm_dataset.py on it afterwards to attach the CGenFF fields
and make the splits -- that path is already proven; do not reimplement it here.

    # size the job first -- it prints a real estimate from a timed sample
    python scripts/make_dimer_scan_dataset.py --resids DCM,ACO --dry-run

    python scripts/make_dimer_scan_dataset.py --resids DCM,ACO \\
        --n-directions 6 --n-orientations 12 --out scan_hf.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.scan_dimer_orientations import (  # noqa: E402
    fibonacci_sphere,
    quat_to_matrix,
    super_fibonacci,
)


def r_grid(r_min: float, r_max: float, n: int, r_dense_to: float) -> np.ndarray:
    """Separations weighted toward the binding/repulsive region.

    A uniform grid spends most of its points where the interaction is already
    zero -- which is exactly the mistake the MD-snapshot data made. Two thirds
    of the points go below ``r_dense_to``.
    """
    n_near = int(round(n * 2 / 3))
    near = np.linspace(r_min, r_dense_to, n_near, endpoint=False)
    far = np.linspace(r_dense_to, r_max, n - n_near)
    return np.concatenate([near, far])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="out_combined_dedup/energies_forces_dipoles_test.npz",
                   help="source of the rigid monomer geometries + res_name")
    p.add_argument("--resids", default="DCM,ACO")
    p.add_argument("--n-directions", type=int, default=6)
    p.add_argument("--n-orientations", type=int, default=12)
    p.add_argument("--n-r", type=int, default=20)
    p.add_argument("--r-min", type=float, default=2.8)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--r-dense-to", type=float, default=6.0,
                   help="two thirds of the r points fall below this (Angstrom)")
    p.add_argument("--min-contact", type=float, default=1.1,
                   help="skip geometries closer than this (SCF stops converging; "
                        "and nothing physical needs them)")
    p.add_argument("--method", default="gfn2", choices=("gfn2", "hf"),
                   help=(
                       "gfn2 (default): GFN2-xTB via tblite. ~0.010 s/geom with OpenMP set, "
                       "and it HAS D4 dispersion. hf: HF via gpu4pyscf, ~1.2 s/geom (120x "
                       "slower) and NO dispersion at all -- for DCM, whose attraction is "
                       "largely dispersion, HF barely binds, so an HF set teaches a "
                       "qualitatively wrong PES. Neither is MP2: see the note at the end."
                   ))
    p.add_argument("--basis", default="def2-SVP", help="HF only")
    p.add_argument("--omp-threads", type=int, default=8,
                   help=(
                       "OpenMP threads for tblite. NOT optional: left unset, tblite "
                       "oversubscribes every core and runs 166x slower (1.666 vs 0.010 "
                       "s/geom measured on 32 cores). Must be set before tblite is imported."
                   ))
    p.add_argument("--include-monomers", action="store_true", default=True,
                   help="monomers carry the intramolecular energy the model must learn")
    p.add_argument("--out", default="dimer_scan_hf.npz")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="write partial results this often; the run is resumable")
    p.add_argument("--dry-run", action="store_true",
                   help="count geometries and time a real sample, then stop")
    args = p.parse_args()

    # Must precede the tblite import: OpenMP reads it at load time.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))

    from mmml.data.units import HARTREE_BOHR_TO_EV_ANGSTROM, HARTREE_TO_EV

    raw = dict(np.load(args.data, allow_pickle=True))
    res = np.array([str(x) for x in raw["res_name"]])
    resids = [r.strip() for r in args.resids.split(",") if r.strip()]

    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = r_grid(args.r_min, args.r_max, args.n_r, args.r_dense_to)

    # --- build every geometry (cheap; the QM is the cost) -------------------
    geoms = []  # (res_name, Z, R, mol_id)
    for resid in resids:
        k = int(np.where(res == resid)[0][0])
        n = int(raw["N"][k])
        Z1 = np.asarray(raw["Z"][k])[:n]
        R1 = np.asarray(raw["R"][k])[:n]
        R1 = R1 - R1.mean(axis=0)
        if args.include_monomers:
            geoms.append((resid, Z1, R1.copy(), np.zeros(n, np.int32)))
        skipped = 0
        for d in dirs:
            for q in quats:
                Rb0 = R1 @ quat_to_matrix(q).T
                for r in rs:
                    a = R1 - 0.5 * r * d
                    b = Rb0 + 0.5 * r * d
                    if np.linalg.norm(a[:, None] - b[None, :], axis=-1).min() < args.min_contact:
                        skipped += 1
                        continue
                    geoms.append((f"{resid},{resid}",
                                  np.concatenate([Z1, Z1]),
                                  np.concatenate([a, b]),
                                  np.concatenate([np.zeros(n, np.int32), np.ones(n, np.int32)])))
        print(f"{resid}: {len(dirs)}x{len(quats)}x{len(rs)} = "
              f"{len(dirs) * len(quats) * len(rs)} dimers, {skipped} skipped "
              f"(contact < {args.min_contact} A)")

    n_tot = len(geoms)
    print(f"\n{n_tot} geometries total; r grid {rs.min():.2f}-{rs.max():.2f} A, "
          f"{(rs < args.r_dense_to).sum()}/{len(rs)} below {args.r_dense_to:g}")

    # --- time a real sample rather than guessing ---------------------------
    if args.method == "hf":
        from gpu4pyscf.scf import RHF
        from pyscf import gto

        def run_one(Z, R):
            mol = gto.M(atom=[(int(z), tuple(float(x) for x in r)) for z, r in zip(Z, R)],
                        basis=args.basis, verbose=0)
            mf = RHF(mol).density_fit()
            e = mf.kernel()
            if not mf.converged:
                return None
            g = mf.nuc_grad_method().kernel()
            d = mf.dip_moment(unit="Debye", verbose=0)
            # Hartree -> eV, Hartree/Bohr -> eV/A, force = -gradient
            return (float(e) * HARTREE_TO_EV,
                    -np.asarray(g) * HARTREE_BOHR_TO_EV_ANGSTROM,
                    np.asarray(d, dtype=float))
    else:
        from tblite.interface import Calculator

        BOHR = 1.8897261254535  # Angstrom -> Bohr; tblite works in Bohr

        def run_one(Z, R):
            try:
                c = Calculator("GFN2-xTB", np.asarray(Z), np.asarray(R) * BOHR)
                c.set("verbosity", 0)
                r = c.singlepoint()
            except Exception:
                return None  # non-convergence surfaces as an exception here
            e = float(r.get("energy"))
            g = np.asarray(r.get("gradient"))  # Hartree/Bohr
            try:
                d = np.asarray(r.get("dipole"), dtype=float) * 2.541746  # a.u. -> Debye
            except Exception:
                d = np.zeros(3)
            return (e * HARTREE_TO_EV,
                    -g * HARTREE_BOHR_TO_EV_ANGSTROM,
                    d)

    sample = geoms[:: max(1, n_tot // 3)][:3]
    t0 = time.time()
    for _, Z, R, _m in sample:
        run_one(Z, R)
    per = (time.time() - t0) / max(len(sample), 1)
    print(f"timed {len(sample)} geometries: {per:.2f}s each (first includes JIT warmup)")
    label = f"HF/{args.basis}" if args.method == "hf" else "GFN2-xTB"
    print(f"ESTIMATE: {n_tot} x {per:.3f}s = {n_tot * per / 60:.1f} min at {label} "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    if args.dry_run:
        print("\n--dry-run: stopping before the QM.")
        return 0

    # --- resume ------------------------------------------------------------
    out = Path(args.out)
    part = out.with_suffix(".partial.npz")
    E = np.full(n_tot, np.nan)
    F = np.zeros((n_tot, max(len(g[1]) for g in geoms), 3))
    D = np.zeros((n_tot, 3))
    done = 0
    if part.exists():
        prev = np.load(part)
        k = int(prev["done"])
        E[:k] = prev["E"][:k]; F[:k] = prev["F"][:k]; D[:k] = prev["D"][:k]
        done = k
        print(f"resuming from {part}: {done}/{n_tot} already done")

    t0 = time.time()
    n_fail = 0
    for i in range(done, n_tot):
        _res, Z, R, _m = geoms[i]
        r = run_one(Z, R)
        if r is None:
            n_fail += 1  # SCF did not converge; left as NaN and dropped at the end
        else:
            e, f, d = r
            E[i] = e; F[i, :len(Z)] = f; D[i] = d
        if (i + 1) % args.checkpoint_every == 0 or i == n_tot - 1:
            np.savez(part, E=E, F=F, D=D, done=i + 1)
            el = time.time() - t0
            rate = (i + 1 - done) / max(el, 1e-9)
            print(f"  {i + 1}/{n_tot}  {rate:.2f} geom/s  "
                  f"ETA {(n_tot - i - 1) / max(rate, 1e-9) / 60:.1f} min  "
                  f"({n_fail} SCF failures)", flush=True)

    # --- assemble ----------------------------------------------------------
    keep = ~np.isnan(E)
    print(f"\n{keep.sum()}/{n_tot} converged ({n_fail} SCF failures dropped)")
    n_at = F.shape[1]
    R_out = np.zeros((int(keep.sum()), n_at, 3))
    Z_out = np.zeros((int(keep.sum()), n_at), np.int32)
    M_out = np.full((int(keep.sum()), n_at), -1, np.int32)
    N_out = np.zeros(int(keep.sum()), np.int32)
    names = []
    j = 0
    for i in range(n_tot):
        if not keep[i]:
            continue
        rn, Z, R, m = geoms[i]
        R_out[j, :len(Z)] = R; Z_out[j, :len(Z)] = Z; M_out[j, :len(Z)] = m
        N_out[j] = len(Z); names.append(rn); j += 1

    np.savez(
        out, R=R_out, Z=Z_out, N=N_out, E=E[keep].reshape(-1, 1), F=F[keep],
        D=D[keep], mol_id=M_out, res_name=np.array(names),
        _mmml_units=np.array(["energy=eV", "forces=eV/Angstrom", "coords=Angstrom",
                              f"method={label}"]),
    )
    print(f"-> {out}  ({keep.sum()} structures, {label}, eV & eV/A)")
    print("\nNext: scripts/prepare_ml_mm_dataset.py to attach CGenFF fields + split.")
    if args.method == "hf":
        print("NOTE: HF has NO dispersion and this applies no counterpoise correction. "
              "For DCM the attraction is largely dispersion, so these wells are "
              "qualitatively wrong. Do not mix with the existing MP2 data.")
    else:
        print("NOTE: GFN2-xTB is semi-empirical. A model trained on this reproduces "
              "GFN2, not MP2, so do NOT mix it with the existing MP2 data -- the two "
              "are different potentials. It is the right tool for proving the pipeline "
              "(dense coverage -> no spurious minima) cheaply; re-run at MP2 for a "
              "production set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
