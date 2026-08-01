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


def nms_conformers(Z, R0, n_samples, temperature_K, rng, freq_min_cm=200.0):
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
    H_mw = H * w[:, None] * w[None, :]               # eV / (A^2 amu)
    lam, V = np.linalg.eigh(H_mw)                    # lam = omega^2

    # Drop translation+rotation by COUNT, not by threshold: a threshold let a
    # near-zero mode through for DCM (10 modes where 5 atoms have 3N-6=9) and its
    # 1/lambda amplitude produced a 2.1 A rmsd.
    n_tr = 6 if len(Z) > 2 else 5
    order = np.argsort(lam)
    lam, V = lam[order][n_tr:], V[:, order][:, n_tr:]
    if np.any(lam <= 0):
        raise RuntimeError(f"{(lam <= 0).sum()} non-positive modes after dropping "
                           f"{n_tr} -- not a minimum")
    # lambda [eV/(A^2 amu)] -> wavenumber: sqrt(eV/(A^2 amu) -> s^-2) / (2 pi c).
    # Derived once rather than trusted: 9.6485e27 = (eV->J)/((A->m)^2 (amu->kg)).
    CM_PER_SQRT_LAM = np.sqrt(9.6485e27) / (2 * np.pi * 2.99792458e10)   # ~521.5
    freqs = CM_PER_SQRT_LAM * np.sqrt(lam)

    # Near-free rotors (acetone's 112 cm^-1 methyl torsion) are NOT harmonic. The
    # harmonic amplitude sqrt(kT/lambda) diverges as lambda->0 and tears the
    # molecule apart -- SCF then fails to converge. Drop them; MD samples those
    # motions properly if they are ever needed.
    soft = freqs < freq_min_cm
    if soft.any():
        print(f"  (dropping {soft.sum()} mode(s) below {freq_min_cm:g} cm^-1: "
              f"{np.round(freqs[soft], 0)} -- near-free rotors, not harmonic)")
    lam, V, freqs = lam[~soft], V[:, ~soft], freqs[~soft]

    kT = u.kB * temperature_K                        # eV
    sigma = np.sqrt(kT / lam)                        # A sqrt(amu)

    out = np.empty((n_samples, len(Z), 3))
    for k in range(n_samples):
        Q = rng.normal(0.0, sigma)
        out[k] = R_eq + (V @ Q * w).reshape(-1, 3)
    return R_eq, out, freqs


def fit_atom_refs(E, Z_all, N_all):
    """Per-element reference energies fitted from the data (least squares).

    A NN asked to predict a -666 eV total, when the physics of interest is
    ~0.05 eV, spends all its capacity on a composition-dependent constant. The
    standard fix is to subtract per-element references so the target becomes an
    atomization-like energy of order eV.

    Fitted here rather than taken from the repo's ATOM_ENERGIES_HARTREE: those
    are referenced at a different level of theory, and subtracting DFT atom
    energies from GFN2 totals leaves a residual that is neither an atomization
    energy nor small. Fitting keeps the reference self-consistent with the data.

    NOTE the composition space is rank-deficient: dimers are exactly 2x their
    monomers, so with only two species there are 2 independent compositions for
    4 elements. lstsq returns the minimum-norm solution -- non-unique, but the
    RESIDUALS are unique, and those are all the model sees. It is equivalent to
    per-species referencing for this dataset, and will NOT extrapolate to a new
    composition.

    Safe for the hybrid assembly: E_AB - E_A - E_B cancels the references
    exactly, and (1-s)(E_A+E_B) + s*E_AB carries the same A+B reference at
    either end of the taper, so the handoff is unaffected.
    """
    elems = np.unique(Z_all[Z_all > 0])
    C = np.zeros((len(E), len(elems)))
    for i in range(len(E)):
        z = Z_all[i][: N_all[i]]
        for j, e in enumerate(elems):
            C[i, j] = (z == e).sum()
    coef, *_ = np.linalg.lstsq(C, E, rcond=None)
    refs = np.zeros(int(elems.max()) + 1)
    refs[elems] = coef
    return refs, C @ coef, elems


def _mono_energy(Z, R):
    """GFN2 energy (eV) of one geometry -- used only for the equipartition check."""
    from tblite.interface import Calculator
    c = Calculator("GFN2-xTB", np.asarray(Z), np.asarray(R) * 1.8897261254535)
    c.set("verbosity", 0)
    return float(c.singlepoint().get("energy")) * 27.211386


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
    p.add_argument("--no-include-monomers", action="store_false", dest="include_monomers",
                   help="omit monomer NMS frames from the geometry bank")
    p.add_argument("--include-hetero", action="store_true", default=True,
                   help="emit A,B heterodimers for every unordered pair of --resids "
                        "(independent NMS draws per monomer, same grid as homodimers)")
    p.add_argument("--no-include-hetero", action="store_false", dest="include_hetero",
                   help="homodimers + monomers only")
    p.add_argument("--monomer-conformers", type=int, default=64,
                   help=(
                       "Thermal normal-mode conformers per species. Must be >= 2: "
                       "rigid monomers (=1) leave the intramolecular ML term untrained "
                       "and bias every dimer to one frozen pose. Each dimer draws two "
                       "conformers independently, and the conformers also enter the "
                       "set as monomer structures."
                   ))
    p.add_argument("--nms-temperature", type=float, default=300.0,
                   help="temperature (K) setting the normal-mode amplitudes")
    p.add_argument("--nms-freq-min", type=float, default=200.0,
                   help=(
                       "Exclude modes below this (cm^-1). They are near-free rotors "
                       "(acetone's 112 cm^-1 methyl torsion), not harmonic oscillators: "
                       "the harmonic amplitude diverges as omega->0 and breaks the "
                       "molecule (SCF stops converging)."
                   ))
    p.add_argument("--geometry-only", action="store_true",
                   help=(
                       "Write the geometry bank (R/Z/N + CGenFF fields) and stop — no "
                       "GFN2/HF labels. Use when ORCA (or another LoT) will relabel."
                   ))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="dimer_scan_hf.npz")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="write partial results this often; the run is resumable")
    p.add_argument("--dry-run", action="store_true",
                   help="count geometries and time a real sample, then stop")
    args = p.parse_args()

    if args.monomer_conformers < 2:
        print(
            f"ERROR: --monomer-conformers must be >= 2 (got {args.monomer_conformers}). "
            "Thermal NMS is required to reduce intramolecular bias; isotropic noise "
            "is not a substitute.",
            file=sys.stderr,
        )
        return 2

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
    if not resids:
        print("ERROR: --resids is empty", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = r_grid(args.r_min, args.r_max, args.n_r, args.r_dense_to)

    # --- build every geometry (cheap; the QM is the cost) -------------------
    # bank[resid] = dict(Z, confs, t, q)
    bank: dict[str, dict] = {}
    geoms = []  # (res_name, Z, R, mol_id, cgenff_type_idx, cgenff_charge)
    for resid in resids:
        hits = np.where(res == resid)[0]
        if hits.size == 0:
            print(f"ERROR: resid {resid!r} not found in {args.data}", file=sys.stderr)
            return 2
        k = int(hits[0])
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
            Z1, R1, args.monomer_conformers, args.nms_temperature, rng,
            freq_min_cm=args.nms_freq_min)
        # rmsd vs the EQUILIBRIUM: ~0.1 A at 300 K. If this is angstroms, the
        # amplitude formula is wrong and the molecules are being destroyed.
        rmsd = float(np.sqrt(((confs - R_eq) ** 2).sum(-1).mean()))
        # Equipartition is the real check on the amplitudes: a classical harmonic
        # sample at T must sit (3N-6)/2 * kT above the minimum. rmsd alone cannot
        # tell a correct soft-mode excursion from a broken formula.
        e_eq = _mono_energy(Z1, R_eq)
        e_conf = np.array([_mono_energy(Z1, c) for c in confs[: min(16, len(confs))]])
        expect = 0.5 * len(freqs) * 8.617333e-5 * args.nms_temperature  # only sampled modes
        print(f"{resid}: <E-E_min> = {(e_conf - e_eq).mean():.3f} eV  "
              f"(equipartition expects {expect:.3f} eV)")
        confs = confs - confs.mean(axis=1, keepdims=True)
        print(f"{resid}: relaxed, {len(freqs)} modes {freqs.min():.0f}-{freqs.max():.0f} cm^-1, "
              f"{len(confs)} conformers @ {args.nms_temperature:g} K (rmsd {rmsd:.3f} A)")
        bank[resid] = {"Z": Z1, "confs": confs, "t": t1, "q": q1}
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
        print(f"{resid},{resid}: {len(dirs)}x{len(quats)}x{len(rs)} = "
              f"{len(dirs) * len(quats) * len(rs)} dimers, {skipped} skipped "
              f"(contact < {args.min_contact} A)")

    if args.include_hetero and len(resids) >= 2:
        for i, ra in enumerate(resids):
            for rb in resids[i + 1:]:
                Za, confa, ta, qa = (
                    bank[ra]["Z"], bank[ra]["confs"], bank[ra]["t"], bank[ra]["q"])
                Zb, confb, tb, qb = (
                    bank[rb]["Z"], bank[rb]["confs"], bank[rb]["t"], bank[rb]["q"])
                na, nb = len(Za), len(Zb)
                skipped = 0
                for d in dirs:
                    for q in quats:
                        for r in rs:
                            a = confa[rng.integers(len(confa))] - 0.5 * r * d
                            b = (confb[rng.integers(len(confb))] @ quat_to_matrix(q).T
                                 + 0.5 * r * d)
                            if (np.linalg.norm(a[:, None] - b[None, :], axis=-1).min()
                                    < args.min_contact):
                                skipped += 1
                                continue
                            geoms.append((
                                f"{ra},{rb}",
                                np.concatenate([Za, Zb]),
                                np.concatenate([a, b]),
                                np.concatenate([
                                    np.zeros(na, np.int32),
                                    np.ones(nb, np.int32),
                                ]),
                                np.concatenate([ta, tb]),
                                np.concatenate([qa, qb]),
                            ))
                print(f"{ra},{rb}: {len(dirs)}x{len(quats)}x{len(rs)} heterodimers, "
                      f"{skipped} skipped (contact < {args.min_contact} A)")

    n_tot = len(geoms)
    print(f"\n{n_tot} geometries total; r grid {rs.min():.2f}-{rs.max():.2f} A, "
          f"{(rs < args.r_dense_to).sum()}/{len(rs)} below {args.r_dense_to:g}")

    if args.geometry_only:
        out = Path(args.out)
        n_at = max(len(g[1]) for g in geoms)
        R_out = np.zeros((n_tot, n_at, 3))
        Z_out = np.zeros((n_tot, n_at), np.int32)
        M_out = np.full((n_tot, n_at), -1, np.int32)
        T_out = np.full((n_tot, n_at), -1, np.int32)
        Q_out = np.zeros((n_tot, n_at))
        N_out = np.zeros(n_tot, np.int32)
        names = []
        for j, (rn, Z, R, m, t, q) in enumerate(geoms):
            R_out[j, :len(Z)] = R
            Z_out[j, :len(Z)] = Z
            M_out[j, :len(Z)] = m
            T_out[j, :len(Z)] = t
            Q_out[j, :len(Z)] = q
            N_out[j] = len(Z)
            names.append(rn)
        common = {}
        if "cgenff_master_sigmas" in raw:
            common["cgenff_master_sigmas"] = np.asarray(raw["cgenff_master_sigmas"])
            common["cgenff_master_epsilons"] = np.asarray(raw["cgenff_master_epsilons"])
        np.savez(
            out,
            R=R_out, Z=Z_out, N=N_out,
            # Placeholders so collectors / prep see a complete schema; ORCA overwrites.
            E=np.full((n_tot, 1), np.nan),
            F=np.zeros((n_tot, n_at, 3)),
            D=np.zeros((n_tot, 3)),
            mol_id=M_out, cgenff_type_idx=T_out, cgenff_charge=Q_out,
            res_name=np.array(names),
            _mmml_units=np.array([
                "coords=Angstrom", "geometry_only=true",
                "labels=pending (ORCA / other LoT)",
            ]),
            **common,
        )
        print(f"\n--geometry-only: wrote {n_tot} frames -> {out}")
        print("Next: scripts/make_orca_array.py --data", out, "--out orca_run ...")
        return 0

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

    # --- per-element reference subtraction ---------------------------------
    E_raw = E[keep]
    refs, fitted, elems = fit_atom_refs(E_raw, Z_out, N_out)
    E_ref = E_raw - fitted
    print(f"\nper-element refs (eV, fitted): "
          + ", ".join(f"Z={int(z)}:{refs[int(z)]:.3f}" for z in elems))
    print("  (rank-deficient fit: these are a minimum-norm split, NOT physical atom "
          "energies -- only the residuals are meaningful)")
    print(f"  E before: {E_raw.min():10.2f} .. {E_raw.max():10.2f}  "
          f"(spread {np.ptp(E_raw):.1f} eV)")
    print(f"  E after : {E_ref.min():10.2f} .. {E_ref.max():10.2f}  "
          f"(spread {np.ptp(E_ref):.1f} eV)")
    # Per-species residual spread is the number that matters: it is what the model
    # must actually learn, once the composition constant is gone.
    for sp in sorted(set(names)):
        m = np.array([x == sp for x in names])
        if m.sum():
            print(f"    {sp:9s} n={int(m.sum()):5d}  E_ref {E_ref[m].min():8.2f} .. "
                  f"{E_ref[m].max():8.2f} eV")

    common = dict(
        atom_ref_energies=refs,
        cgenff_master_sigmas=np.asarray(raw["cgenff_master_sigmas"]),
        cgenff_master_epsilons=np.asarray(raw["cgenff_master_epsilons"]),
        _mmml_units=np.array(["energy=eV", "forces=eV/Angstrom", "coords=Angstrom",
                              "dipole=e*Angstrom", f"method={label}",
                              "E=atom-referenced (add atom_ref_energies back for totals)",
                              "E_total=raw"]),
    )
    rng = np.random.default_rng(0)
    perm = rng.permutation(nk)
    n_tr, n_va = int(0.8 * nk), int(0.1 * nk)
    base = str(out.with_suffix(""))
    for tag, sel in (("train", perm[:n_tr]), ("valid", perm[n_tr:n_tr + n_va]),
                     ("test", perm[n_tr + n_va:])):
        f = Path(f"{base}_{tag}.npz")
        np.savez(f, R=R_out[sel], Z=Z_out[sel], N=N_out[sel],
                 E=E_ref[sel].reshape(-1, 1), E_total=E_raw[sel].reshape(-1, 1),
                 F=F[keep][sel], D=D[keep][sel],
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
