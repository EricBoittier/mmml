#!/usr/bin/env python3
"""Run counterpoise-corrected HF/DFT dimer scans via GPU4PySCF (in-process, no subprocess).

Same idea as ``run_orca_hf_mp2_scan.py`` — reuses the same adaptive per-pair
distance grid (``build_pair_distance_grid``, optionally extended inward via
``--close-floor``) and writes a CSV that plugs directly into
``plot_2d_pes.py`` / ``plot_1d_slices_by_offset.py`` — but calls GPU4PySCF's
Python API directly instead of shelling out to an external binary, and uses
density fitting (the ``--aux-basis``) for speed the way GPU4PySCF is meant
to be used.

Counterpoise (Boys-Bernardi) correction uses PySCF's ghost-atom convention
(``X-`` prefix on the element symbol — basis functions only, no nuclear
charge/electrons), matching the existing pattern in
``mmml/interfaces/pyscf4gpuInterface/calcs.py:compute_interaction_energy``.

``--methods`` accepts ``HF`` or any PySCF/GPU4PySCF XC functional keyword
(e.g. ``PBE0``, ``B3LYP``, ``wB97M-V``).

Usage
-----
    python scripts/run_gpu4pyscf_scan.py --pairs TIP3:TIP3 BENZ:BENZ \\
        --methods HF PBE0 --basis def2-SVP --aux-basis def2-universal-jkfit \\
        --close-floor 1.2 --output-dir results/dimer_scan_campaign/gpu4pyscf

Falls back to plain (CPU) PySCF with ``--cpu`` if no GPU/gpu4pyscf is
available — useful for a quick correctness smoke test off-cluster.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mmml.analysis.dimer_molecules import PAIR_SCAN_CONFIG, make_oriented_scan_geometries
from mmml.analysis.dimer_scans import min_fragment_contact_distance
from scripts.run_dimer_scan_campaign import build_pair_distance_grid
from scripts.run_orca_hf_mp2_scan import _cache_key, load_cached_results

HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL_MOL = 23.060548867
HARTREE_TO_KCAL_MOL = HARTREE_TO_EV * EV_TO_KCAL_MOL


def _build_atom_list(symbols: list[str], positions: np.ndarray, ghost_mask: np.ndarray) -> list[tuple]:
    """PySCF ``atom=`` spec with ghost atoms as PySCF's ``X-`` prefix convention."""
    return [
        (f"X-{sym}" if is_ghost else sym, tuple(float(v) for v in pos))
        for sym, pos, is_ghost in zip(symbols, positions, ghost_mask)
    ]


def _rhf(mol, use_gpu: bool):
    if use_gpu:
        from gpu4pyscf.scf import RHF
    else:
        from pyscf.scf import RHF
    return RHF(mol)


def _mean_field(mol, method: str, use_gpu: bool):
    """Build the SCF reference for HF or a DFT functional (not MP2 — see _energy_for)."""
    if method.upper() == "HF":
        return _rhf(mol, use_gpu)
    if use_gpu:
        from gpu4pyscf.dft import rks
    else:
        from pyscf.dft import rks
    return rks.RKS(mol, xc=method)


def _energy_for(
    symbols: list[str], positions: np.ndarray, ghost_mask: np.ndarray,
    *, method: str, basis: str, aux_basis: str | None, charge: int, spin: int,
    use_gpu: bool, scf_conv_tol: float, scf_max_cycle: int, dispersion: str | None = None,
) -> float:
    import pyscf

    atom_list = _build_atom_list(symbols, positions, ghost_mask)
    mol = pyscf.M(atom=atom_list, basis=basis, charge=charge, spin=spin, unit="Ang", verbose=0)

    if method.upper() == "MP2":
        # MP2 isn't a single mean-field kernel() call: run the RHF reference
        # first (dispersion, if any, would apply to that reference — DFT-D
        # parameterizations don't cover MP2 itself, so dispersion is a no-op
        # here; ghost atoms work identically since DF-MP2 just correlates
        # whatever occupied/virtual orbitals come out of the RHF reference).
        if use_gpu:
            from gpu4pyscf.mp import dfmp2
        else:
            from pyscf.mp import dfmp2
        mf = _rhf(mol, use_gpu).density_fit(auxbasis=aux_basis)
        mf.conv_tol = scf_conv_tol
        mf.max_cycle = scf_max_cycle
        e_hf = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"SCF (MP2 reference) did not converge ({basis})")
        pt = dfmp2.DFMP2(mf)
        e_corr, _ = pt.kernel()
        return float(e_hf + e_corr)

    mf = _mean_field(mol, method, use_gpu)
    if dispersion and dispersion.lower() != "none":
        mf.disp = dispersion.lower()
    mf = mf.density_fit(auxbasis=aux_basis)
    mf.conv_tol = scf_conv_tol
    mf.max_cycle = scf_max_cycle
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge ({method}/{basis})")
    return float(energy)


def evaluate_pyscf_scan(
    geometries, label_a: str, label_b: str, *,
    method: str, basis: str, aux_basis: str | None, counterpoise: bool,
    charge: int, spin: int, use_gpu: bool, scf_conv_tol: float, scf_max_cycle: int,
    backend_name: str, skip_keys: set[tuple] | None = None, dispersion: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    isolated_cache: dict[str, float] = {}
    skip_keys = skip_keys or set()

    for geom in geometries:
        cache_key = _cache_key(label_a, label_b, backend_name, geom.distance_angstrom, geom.offset_angstrom)
        if cache_key in skip_keys:
            continue

        symbols = geom.atoms.get_chemical_symbols()
        positions = geom.atoms.get_positions()
        idx_a, idx_b = geom.fragments
        n = len(symbols)
        mask_a = np.zeros(n, dtype=bool)
        mask_a[idx_a] = True

        common = dict(
            method=method, basis=basis, aux_basis=aux_basis, charge=charge, spin=spin,
            use_gpu=use_gpu, scf_conv_tol=scf_conv_tol, scf_max_cycle=scf_max_cycle,
            dispersion=dispersion,
        )
        try:
            e_dimer = _energy_for(symbols, positions, np.zeros(n, dtype=bool), **common)

            if counterpoise:
                # Fragment A real, fragment B as ghost basis (and vice versa) —
                # both computed *in the full dimer geometry* so the ghost
                # basis functions sit exactly where the real atoms are.
                e_a = _energy_for(symbols, positions, ~mask_a, **common)
                e_b = _energy_for(symbols, positions, mask_a, **common)
            else:
                # Isolated monomer energies don't depend on separation/offset
                # (rigid monomers) — compute once per pair and reuse.
                if "A" not in isolated_cache:
                    isolated_cache["A"] = _energy_for(
                        [symbols[i] for i in idx_a], positions[idx_a],
                        np.zeros(len(idx_a), dtype=bool), **common,
                    )
                if "B" not in isolated_cache:
                    isolated_cache["B"] = _energy_for(
                        [symbols[i] for i in idx_b], positions[idx_b],
                        np.zeros(len(idx_b), dtype=bool), **common,
                    )
                e_a = isolated_cache["A"]
                e_b = isolated_cache["B"]

            e_int_hartree = e_dimer - e_a - e_b
            rows.append(
                {
                    "molecule_a": label_a,
                    "molecule_b": label_b,
                    "distance_angstrom": geom.distance_angstrom,
                    "offset_angstrom": geom.offset_angstrom,
                    "energy_ev": e_int_hartree * HARTREE_TO_EV,
                    "energy_kcal_mol": e_int_hartree * HARTREE_TO_KCAL_MOL,
                    "comp_Edimer_hartree": e_dimer,
                    "comp_EfragA_hartree": e_a,
                    "comp_EfragB_hartree": e_b,
                    "min_contact_angstrom": min_fragment_contact_distance(geom.atoms, geom.fragments),
                    "backend": backend_name,
                }
            )
            print(
                f"    d={geom.distance_angstrom:.2f} off={geom.offset_angstrom:.2f}  "
                f"E_int={e_int_hartree * HARTREE_TO_KCAL_MOL:+.3f} kcal/mol"
            )
        except Exception as e:
            print(f"    Warning: {method} failed at d={geom.distance_angstrom:.2f} off={geom.offset_angstrom:.2f}: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pairs", nargs="+", default=["TIP3:TIP3", "BENZ:BENZ"],
        help="Pairs to scan as LABEL_A:LABEL_B (default: TIP3:TIP3 BENZ:BENZ)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["HF", "PBE0"],
        help=(
            "'HF', 'MP2' (density-fitted, RHF reference), or any PySCF/GPU4PySCF XC functional "
            "keyword (default: HF PBE0). GPU4PySCF makes larger basis sets (def2-TZVP, "
            "aug-cc-pVDZ, ...) much more tractable than the ORCA/CPU runs — worth trying via --basis."
        ),
    )
    parser.add_argument(
        "--basis", default="def2-SVP",
        help="Orbital basis set (default: def2-SVP). Larger bases (def2-TZVP, aug-cc-pVDZ) are "
        "reasonable here given GPU acceleration — remember to bump --aux-basis to match.",
    )
    parser.add_argument(
        "--aux-basis", default="def2-universal-jkfit",
        help=(
            "Density-fitting auxiliary basis (default: def2-universal-jkfit — a general-purpose "
            "Coulomb/exchange fitting basis that works across all def2 orbital bases). Also used "
            "as the RI basis for MP2 correlation. Pass '' / none to let PySCF auto-select instead."
        ),
    )
    parser.add_argument(
        "--dispersion", default="none", choices=["none", "d3bj", "d3zero", "d4"],
        help=(
            "Empirical dispersion correction via pyscf-dispersion (default: none). No-op for "
            "--methods MP2 (dispersion parameterizations are DFT/HF-specific, not meaningful "
            "for a method that already computes correlation directly). Requires the "
            "pyscf-dispersion package with a working DFT-D3/D4 library for your platform."
        ),
    )
    parser.add_argument(
        "--no-counterpoise", action="store_true",
        help="Skip Boys-Bernardi counterpoise correction (cheaper, but BSSE will distort close-range results)",
    )
    parser.add_argument("--charge", type=int, default=0, help="Total charge per fragment/system (default 0)")
    parser.add_argument("--spin", type=int, default=0, help="PySCF spin = n_alpha - n_beta (default 0, closed shell)")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Use plain CPU PySCF instead of GPU4PySCF (useful for a correctness smoke test without a GPU)",
    )
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10, help="SCF convergence tolerance (default 1e-10)")
    parser.add_argument("--scf-max-cycle", type=int, default=100, help="SCF max iterations (default 100)")
    parser.add_argument("--min-contact", type=float, default=1.5, help="Grid-anchoring contact distance (default 1.5)")
    parser.add_argument(
        "--close-floor", type=float, default=None,
        help="Extend the grid inward to this centre-to-centre distance (Å), same as run_dimer_scan_campaign.py",
    )
    parser.add_argument("--n-close", type=int, default=6, help="Extra points between --close-floor and grid start")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dimer_scan_campaign/gpu4pyscf"))
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing scan_results.csv in --output-dir and recompute everything from scratch",
    )
    args = parser.parse_args()

    aux_basis = args.aux_basis if args.aux_basis and args.aux_basis.lower() != "none" else None

    if args.cpu:
        try:
            import pyscf  # noqa: F401
        except ImportError:
            print("pyscf not importable. Install it (pip install pyscf) or drop --cpu to use GPU4PySCF.")
            sys.exit(1)
        print("Using CPU PySCF (--cpu passed).")
    else:
        try:
            import gpu4pyscf  # noqa: F401
        except ImportError:
            print(
                "gpu4pyscf not importable. Install gpu4pyscf-cudaXXx for your CUDA version, "
                "or pass --cpu to fall back to plain PySCF."
            )
            sys.exit(1)
        print("Using GPU4PySCF.")

    basis_slug = args.basis.lower().replace("-", "").replace("*", "s").replace("(", "").replace(")", "").replace(",", "")
    cp_suffix = "_cp" if not args.no_counterpoise else ""
    engine_tag = "pyscfcpu" if args.cpu else "gpu4pyscf"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scan_results.csv"
    if args.no_resume:
        results, seen_keys = [], set()
    else:
        results, seen_keys = load_cached_results(csv_path)

    for pair_str in args.pairs:
        label_a, label_b = pair_str.split(":")
        if (label_a, label_b) not in PAIR_SCAN_CONFIG:
            print(f"Skipping {pair_str}: not in PAIR_SCAN_CONFIG")
            continue
        cfg = PAIR_SCAN_CONFIG[(label_a, label_b)]
        offsets = cfg["offsets_angstrom"]
        distances, safe_d = build_pair_distance_grid(
            label_a, label_b, min_contact=args.min_contact,
            close_floor=args.close_floor, n_close=args.n_close,
        )
        geometries = list(make_oriented_scan_geometries(label_a, label_b, distances, offsets))
        print(f"{label_a}+{label_b}: {cfg['description']}")
        print(
            f"  safe contact clears at d≈{safe_d:.2f} Å — grid spans "
            f"{distances.min():.2f}–{distances.max():.2f} Å ({len(distances)} distances × {len(offsets)} offsets)"
        )

        for method in args.methods:
            method_slug = method.lower().replace("-", "")
            is_mp2 = method.upper() == "MP2"
            disp_suffix = f"_{args.dispersion}" if (args.dispersion != "none" and not is_mp2) else ""
            backend_name = f"{method_slug}_{basis_slug}_{engine_tag}{disp_suffix}{cp_suffix}"
            disp_note = "" if is_mp2 else f" disp={args.dispersion}"
            print(f"  Evaluating {method}/{args.basis} (aux={aux_basis or 'auto'}){disp_note}...")
            rows = evaluate_pyscf_scan(
                geometries, label_a, label_b,
                method=method, basis=args.basis, aux_basis=aux_basis,
                counterpoise=not args.no_counterpoise,
                charge=args.charge, spin=args.spin, use_gpu=not args.cpu,
                scf_conv_tol=args.scf_conv_tol, scf_max_cycle=args.scf_max_cycle,
                backend_name=backend_name, skip_keys=seen_keys,
                dispersion=None if is_mp2 else args.dispersion,
            )
            results.extend(rows)
            seen_keys.update(
                _cache_key(r["molecule_a"], r["molecule_b"], r["backend"], r["distance_angstrom"], r["offset_angstrom"])
                for r in rows
            )

            # Save incrementally so partial progress survives a crash/timeout
            # and can be resumed with the same command later.
            pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
