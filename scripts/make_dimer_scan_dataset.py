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


def nms_conformers(Z, R0, n_samples, temperature_K, rng, eig_min=1e-4):
    """Thermal normal-mode samples around a GFN2-relaxed geometry.

    Why not uniform random noise: isotropic displacement is chemically blind --
    it stretches stiff bonds as readily as soft torsions, so most samples are
    absurd, high in energy, and say nothing about what the molecule does. Normal
    modes displace along the actual vibrations with thermal amplitudes: soft
    modes move far, stiff bonds barely move. (This is what ANI-1 used.)

    Sampling is the classical harmonic ensemble, done in mass-weighted
    coordinates so the units close:

        H_mw = H / sqrt(m_i m_j)                    eV / (A^2 amu)
        lambda_i, v_i = eig(H_mw)                   lambda_i = omega_i^2
        <Q_i^2> = kT / lambda_i                     -> A sqrt(amu)
        dR = sum_i Q_i v_i / sqrt(m)                -> A

    (Doing this from mode *energies* instead -- sqrt(kT)/E -- is dimensionally
    sqrt(eV)/eV, not a length, and produces ~3.5 A displacements that destroy the
    molecule. The rmsd printed by the caller is the check on that.)

    The input is an MD snapshot, NOT a stationary point, and a Hessian there
    gives meaningless modes -- so it is relaxed first. The 6 near-zero
    eigenvalues (translation/rotation) are dropped; their 1/lambda amplitude
    would blow up.
    """
    import tempfile

    import ase.units as u
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations
    from tblite.ase import TBLite

    atoms = Atoms(numbers=Z, positions=R0)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
    BFGS(atoms, logfile=None).run(fmax=0.005, steps=500)
    R_eq = atoms.get_positions()

    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(atoms, name=f"{tmp}/vib")
        vib.run()
        data = vib.get_vibrations()

    H = np.asarray(data.get_hessian_2d())            # eV / A^2
    m = atoms.get_masses()
    w = np.repeat(m, 3) ** -0.5
    H_mw = H * w[:, None] * w[None, :]
    lam, V = np.linalg.eigh(H_mw)

    keep = lam > eig_min                             # drops trans/rot + noise
    lam, V = lam[keep], V[:, keep]
    if lam.size == 0:
        raise RuntimeError("no positive modes -- optimisation failed?")

    kT = u.kB * temperature_K                        # eV
    sigma = np.sqrt(kT / lam)                        # A sqrt(amu)
    freqs = np.sqrt(lam) * 1e10 / (2 * np.pi * u._c) * np.sqrt(u._e / u._amu)  # cm^-1

    out = np.empty((n_samples, len(Z), 3))
    for k in range(n_samples):
        Q = rng.normal(0.0, sigma)
        out[k] = R_eq + (V @ Q * w).reshape(-1, 3)
    return R_eq, out, freqs


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
    p.add_argument("--monomer-conformers", type=int, default=64,
                   help=(
                       "Thermal normal-mode conformers per species. Rigid monomers "
                       "(=1) leave the intramolecular ML term untrained: it sees one "
                       "geometry and learns a constant. Each dimer draws two "
                       "conformers independently, and the conformers also enter the "
                       "set as monomer structures."
                   ))
    p.add_argument("--nms-temperature", type=float, default=300.0,
                   help="temperature (K) setting the normal-mode amplitudes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="dimer_scan_hf.npz")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="write partial results this often; the run is resumable")
    p.add_argument("--dry-run", action="store_true",
                   help="count geometries and time a real sample, then stop")
    args = p.parse_args()

    # Must precede the tblite import: OpenMP reads it at load time.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))

    from mmml.data.units import (
        BOHR_TO_ANGSTROM,
        HARTREE_BOHR_TO_EV_ANGSTROM,
        HARTREE_TO_EV,
    )

    raw = dict(np.load(args.data, allow_pickle=True))
    res = np.array([str(x) for x in raw["res_name"]])
    resids = [r.strip() for r in args.resids.split(",") if r.strip()]

    rng = np.random.default_rng(args.seed)
    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = r_grid(args.r_min, args.r_max, args.n_r, args.r_dense_to)

    # --- build every geometry (cheap; the QM is the cost) -------------------
    geoms = []  # (res_name, Z, R, mol_id, cgenff_type_idx, cgenff_charge)
    for resid in resids:
        k = int(np.where(res == resid)[0][0])
        n = int(raw["N"][k])
        Z1 = np.asarray(raw["Z"][k])[:n]
        R1 = np.asarray(raw["R"][k])[:n]
        R1 = R1 - R1.mean(axis=0)
        # Built from a monomer whose CGenFF assignment the source dataset already
        # carries (graph-isomorphism-validated at prep time), so carry it through
        # rather than re-deriving. Atom order is the monomer's own, which is what
        # hybrid_forward expects (the MD calculator wants PSF order instead).
        t1 = np.asarray(raw["cgenff_type_idx"][k])[:n]
        q1 = np.asarray(raw["cgenff_charge"][k])[:n]
        R_eq, confs, freqs = nms_conformers(
            Z1, R1, args.monomer_conformers, args.nms_temperature, rng)
        # rmsd vs the EQUILIBRIUM: ~0.1 A at 300 K. If this is angstroms, the
        # amplitude formula is wrong and the molecules are being destroyed.
        rmsd = float(np.sqrt(((confs - R_eq) ** 2).sum(-1).mean()))
        confs = confs - confs.mean(axis=1, keepdims=True)
        print(f"{resid}: relaxed, {len(freqs)} modes {freqs.min():.0f}-{freqs.max():.0f} cm^-1, "
              f"{len(confs)} conformers @ {args.nms_temperature:g} K (rmsd {rmsd:.3f} A)")
        if args.include_monomers:
            for c in confs:
                geoms.append((resid, Z1, c.copy(), np.zeros(n, np.int32), t1, q1))
        skipped = 0
        for d in dirs:
            for q in quats:
                for r in rs:
                    # independent conformers per monomer, so the dimer term cannot
                    # memorise one frozen intramolecular geometry
                    a = confs[rng.integers(len(confs))] - 0.5 * r * d
                    b = confs[rng.integers(len(confs))] @ quat_to_matrix(q).T + 0.5 * r * d
                    if np.linalg.norm(a[:, None] - b[None, :], axis=-1).min() < args.min_contact:
                        skipped += 1
                        continue
                    geoms.append((f"{resid},{resid}",
                                  np.concatenate([Z1, Z1]),
                                  np.concatenate([a, b]),
                                  np.concatenate([np.zeros(n, np.int32), np.ones(n, np.int32)]),
                                  np.concatenate([t1, t1]),
                                  np.concatenate([q1, q1])))
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
                # a.u. (e*bohr) -> e*Angstrom, matching how the model forms its
                # own dipole (sum q_i r_i). The existing dataset's D does NOT
                # agree: its ACO monomer |D| is 0.207 where 2.88 D = 0.600 e*A,
                # and DCM is off by a different factor -- so it is not a unit
                # conversion. Do not train dipoles against both sets.
                d = np.asarray(r.get("dipole"), dtype=float) * BOHR_TO_ANGSTROM
            except Exception:
                d = np.zeros(3)
            return (e * HARTREE_TO_EV,
                    -g * HARTREE_BOHR_TO_EV_ANGSTROM,
                    d)

    sample = geoms[:: max(1, n_tot // 3)][:3]
    t0 = time.time()
    for _, Z, R, _m, _t, _q in sample:
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
        _res, Z, R, _m, _t, _q = geoms[i]
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
    nk = int(keep.sum())
    R_out = np.zeros((nk, n_at, 3))
    Z_out = np.zeros((nk, n_at), np.int32)
    M_out = np.full((nk, n_at), -1, np.int32)
    T_out = np.full((nk, n_at), -1, np.int32)
    Q_out = np.zeros((nk, n_at))
    N_out = np.zeros(nk, np.int32)
    names = []
    j = 0
    for i in range(n_tot):
        if not keep[i]:
            continue
        rn, Z, R, m, t, q = geoms[i]
        R_out[j, :len(Z)] = R; Z_out[j, :len(Z)] = Z; M_out[j, :len(Z)] = m
        T_out[j, :len(Z)] = t; Q_out[j, :len(Z)] = q
        N_out[j] = len(Z); names.append(rn); j += 1

    common = dict(
        cgenff_master_sigmas=np.asarray(raw["cgenff_master_sigmas"]),
        cgenff_master_epsilons=np.asarray(raw["cgenff_master_epsilons"]),
        _mmml_units=np.array(["energy=eV", "forces=eV/Angstrom", "coords=Angstrom",
                              "dipole=e*Angstrom", f"method={label}"]),
    )
    rng = np.random.default_rng(0)
    perm = rng.permutation(nk)
    n_tr, n_va = int(0.8 * nk), int(0.1 * nk)
    base = str(out.with_suffix(""))
    for tag, sel in (("train", perm[:n_tr]), ("valid", perm[n_tr:n_tr + n_va]),
                     ("test", perm[n_tr + n_va:])):
        f = Path(f"{base}_{tag}.npz")
        np.savez(f, R=R_out[sel], Z=Z_out[sel], N=N_out[sel],
                 E=E[keep][sel].reshape(-1, 1), F=F[keep][sel], D=D[keep][sel],
                 mol_id=M_out[sel], cgenff_type_idx=T_out[sel], cgenff_charge=Q_out[sel],
                 res_name=np.array(names)[sel], **common)
        print(f"-> {f}  ({len(sel)} structures)")
    print(f"\n{label}; eV, eV/A, dipole e*A. CGenFF types/charges carried from the "
          f"source monomers; master tables ({len(common['cgenff_master_sigmas'])} types) copied.")
    print("Ready for: mmml physnet-train --hybrid-mm --data <base>_train.npz "
          "--valid-data <base>_valid.npz")
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
