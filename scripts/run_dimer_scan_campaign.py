#!/usr/bin/env python3
"""Run the molecular dimer scan campaign with learned multipoles, MBD, xTB,
SpookyNet, and (optionally) CHARMM/CGenFF — all sharing one distance/offset
grid per pair so every backend lands in a single combined CSV.

The distance grid is chosen per pair: a cheap geometry-only sweep
(``find_safe_min_distance``) locates where fragment atoms actually stop
overlapping (on-axis, offset=0) and anchors the grid there, instead of using
one fixed floor that's unsafe for bulky/asymmetric pairs (e.g. ACE+ACE needs
~5 Å before atoms clear) and wasteful for compact ones.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mmml.analysis.dimer_molecules import (
    MOLECULES,
    ORIENTED_MONOMERS,
    PAIR_SCAN_CONFIG,
    make_oriented_scan_geometries,
)
from mmml.analysis.dimer_scans import (
    evaluate_scan,
    evaluate_scan_monomer_decomposed,
    find_safe_min_distance,
    make_xtb_calculator,
    min_fragment_contact_distance,
    molecule_pair_labels,
)
from mmml.models.mbd import QCMLMBDCalculator
from mmml.models.multipoles import LearnedMolecularMultipoleElectrostatics
from mmml.models.spookynet_calc import SpookyNetCalculator

EV_TO_KCAL_MOL = 23.060548867

# Residue name mapping + explicit geometries/atom-order for CHARMM PSF construction
# (bypasses CHARMM's IC-build code, which is unstable for these small monomers).
CHARMM_RESIDUES = {
    "DCM": "DCM",
    "ACE": "ACO",
    "BENZ": "BENZ",
    "TIP3": "TIP3",
    "MEOH": "MEOH",
}


def _charmm_residue_geometries() -> dict:
    return {
        "DCM": (
            MOLECULES["DCM"].positions[[0, 3, 4, 1, 2]],
            ["C", "H1", "H2", "CL1", "CL2"],
            np.array([6, 1, 1, 17, 17]),
        ),
        "ACO": (
            MOLECULES["ACE"].positions[[3, 0, 1, 2, 4, 5, 6, 7, 8, 9]],
            ["O1", "C1", "C2", "C3", "H21", "H22", "H23", "H31", "H32", "H33"],
            np.array([8, 6, 6, 6, 1, 1, 1, 1, 1, 1]),
        ),
        "BENZ": (
            MOLECULES["BENZ"].positions[[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]],
            ["CG", "HG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HZ"],
            np.array([6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1]),
        ),
        "TIP3": (
            MOLECULES["TIP3"].positions[[0, 1, 2]],
            ["OH2", "H1", "H2"],
            np.array([8, 1, 1]),
        ),
        "MEOH": (
            MOLECULES["MEOH"].positions[[0, 1, 2, 3, 4, 5]],
            ["CB", "OG", "HG1", "HB1", "HB2", "HB3"],
            np.array([6, 8, 1, 1, 1, 1]),
        ),
    }


def _init_charmm():
    """Import + quiet-initialize PyCHARMM. Returns the helper callables needed below."""
    import pycharmm

    pycharmm.settings.set_bomb_level(-5)

    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_from_composition
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_energy_row
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        setup_default_nbonds,
        sync_charmm_positions,
    )

    pycharmm_quiet()
    return _build_cluster_from_composition, setup_default_nbonds, sync_charmm_positions, charmm_energy_row


def evaluate_charmm_scan(geometries, label_a, label_b, charmm_fns) -> list[dict]:
    """Evaluate a scan's geometries with CHARMM/CGenFF (PSF built once per pair)."""
    build_cluster, setup_nbonds, sync_positions, energy_row = charmm_fns
    res_a = CHARMM_RESIDUES[label_a]
    res_b = CHARMM_RESIDUES[label_b]
    build_cluster(
        composition=[(res_a, 1), (res_b, 1)],
        spacing=5.0,
        residue_geometries=_charmm_residue_geometries(),
    )
    setup_nbonds()

    import pycharmm

    rows: list[dict] = []
    for geom in geometries:
        try:
            sync_positions(geom.atoms.positions)
            pycharmm.lingo.charmm_script("ENER")
            terms = energy_row()
            elec = float(terms.get("ELEC", np.nan))
            vdw = float(terms.get("VDW", np.nan))
            tot = float(terms.get("ENER", np.nan))
            rows.append(
                {
                    "molecule_a": label_a,
                    "molecule_b": label_b,
                    "distance_angstrom": geom.distance_angstrom,
                    "offset_angstrom": geom.offset_angstrom,
                    "energy_ev": tot / EV_TO_KCAL_MOL,
                    "energy_kcal_mol": tot,
                    "backend": "charmm",
                    "charmm_ELEC_kcal": elec,
                    "charmm_VDW_kcal": vdw,
                    "min_contact_angstrom": min_fragment_contact_distance(geom.atoms, geom.fragments),
                }
            )
        except Exception as e:
            print(f"    Warning: CHARMM failed at d={geom.distance_angstrom} Å offset={geom.offset_angstrom} Å: {e}")
    return rows


def build_pair_distance_grid(
    label_a: str, label_b: str, *, min_contact: float = 1.5,
    n_near: int = 11, n_far: int = 8, near_span: float = 2.5, far_span: float = 9.5,
) -> tuple[np.ndarray, float]:
    """Per-pair distance grid anchored to where fragment atoms actually clear contact."""
    cfg = PAIR_SCAN_CONFIG[(label_a, label_b)]
    monomers = ORIENTED_MONOMERS[(label_a, label_b)]
    safe_d = find_safe_min_distance(
        monomers["a"], monomers["b"],
        transverse_axis=cfg["transverse_axis"], min_contact=min_contact,
    )
    d_start = max(2.0, safe_d - 0.75)  # a bit before the safe point to still show the repulsive wall onset
    distances = np.concatenate([
        np.linspace(d_start, d_start + near_span, n_near),
        np.linspace(d_start + near_span + 0.5, d_start + far_span, n_far),
    ])
    return distances, safe_d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multipole-checkpoint",
        type=Path,
        default=None,
        help="Path to multipoles model checkpoint folder",
    )
    parser.add_argument(
        "--mbd-checkpoint",
        type=Path,
        default=None,
        help="Path to MBD model checkpoint folder",
    )
    parser.add_argument("--max-ell", type=int, default=3, help="Maximum multipole rank (0-3)")
    parser.add_argument(
        "--spookynet-checkpoint",
        type=Path,
        default=None,
        help="Path to a SpookyNet JSON checkpoint (params + config)",
    )
    parser.add_argument(
        "--with-charmm",
        action="store_true",
        help="Also evaluate CHARMM/CGenFF energies (requires pycharmm)",
    )
    parser.add_argument(
        "--min-contact",
        type=float,
        default=1.5,
        help="Contact distance (Å) used to anchor each pair's distance grid (default 1.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign"),
        help="Output directory",
    )
    args = parser.parse_args()

    # Prevent JAX GPU preallocation issue
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize calculators
    print("Initializing calculators...")

    use_multipole = False
    multipole_calc = None
    if args.multipole_checkpoint is not None:
        try:
            multipole_calc = LearnedMolecularMultipoleElectrostatics(
                checkpoint=args.multipole_checkpoint,
                max_ell=args.max_ell,
                origin="nuclear_charge_centroid",
                softening_bohr=0.5,
            )
            use_multipole = True
            print("  Learned multipole calculator initialized successfully.")
        except Exception as e:
            print(f"  Error loading multipole model: {e}")
            sys.exit(1)
    else:
        print("  No multipole checkpoint provided. Skipping multipole backend.")

    use_mbd = False
    mbd_calc = None
    if args.mbd_checkpoint is not None:
        try:
            mbd_calc = QCMLMBDCalculator(checkpoint=args.mbd_checkpoint)
            use_mbd = True
            print("  Learned MBD calculator initialized successfully.")
        except Exception as e:
            print(f"  Error loading MBD model: {e}")
            sys.exit(1)
    else:
        print("  No MBD checkpoint provided. Skipping MBD backend.")

    use_spookynet = False
    spookynet_calc = None
    if args.spookynet_checkpoint is not None:
        try:
            spookynet_calc = SpookyNetCalculator(checkpoint=args.spookynet_checkpoint)
            use_spookynet = True
            print("  SpookyNet calculator initialized successfully.")
        except Exception as e:
            print(f"  Error loading SpookyNet model: {e}")
            sys.exit(1)
    else:
        print("  No SpookyNet checkpoint provided. Skipping spookynet/spookynet_hybrid backends.")

    # Check for xTB
    use_xtb = False
    try:
        xtb_calc = make_xtb_calculator(method="GFN2-xTB")
        use_xtb = True
        print("  xTB calculator initialized successfully.")
    except Exception as e:
        print(f"  xTB calculator not available: {e}. Skipping xTB backend.")

    use_charmm = False
    charmm_fns = None
    if args.with_charmm:
        try:
            charmm_fns = _init_charmm()
            use_charmm = True
            print("  CHARMM/CGenFF initialized successfully.")
        except Exception as e:
            print(f"  Error initializing CHARMM: {e}")
            sys.exit(1)

    if not (use_multipole or use_mbd or use_xtb or use_spookynet or use_charmm):
        print("No backends are available or enabled. Exiting.")
        sys.exit(0)

    labels = list(MOLECULES.keys())
    pairs = molecule_pair_labels(labels, include_homodimers=True)

    print(f"Will scan {len(pairs)} unique pairs (per-pair distance grid, up to 5 offsets, 2D).")

    results = []

    for idx, (label_a, label_b) in enumerate(pairs, 1):
        pair_cfg = PAIR_SCAN_CONFIG[(label_a, label_b)]
        offsets = pair_cfg["offsets_angstrom"]
        distances, safe_d = build_pair_distance_grid(label_a, label_b, min_contact=args.min_contact)
        print(f"[{idx}/{len(pairs)}] {label_a}+{label_b}: {pair_cfg['description']}")
        print(
            f"  safe contact clears at d≈{safe_d:.2f} Å (offset=0) — grid spans "
            f"{distances.min():.2f}–{distances.max():.2f} Å"
        )
        print(f"  {len(distances)} distances × {len(offsets)} offsets = {len(distances) * len(offsets)} geometries")
        geometries = list(make_oriented_scan_geometries(label_a, label_b, distances, offsets))

        # Evaluate Multipoles
        if use_multipole:
            print(f"  Evaluating learned multipole (max_ell={args.max_ell})...")
            try:
                mp_rows = evaluate_scan(geometries, lambda: multipole_calc)
                for r in mp_rows:
                    r["backend"] = "learned_multipole"
                results.extend(mp_rows)
            except Exception as e:
                print(f"    Error: {e}")

        # Evaluate MBD
        if use_mbd:
            print("  Evaluating learned MBD...")
            try:
                mbd_rows = evaluate_scan(geometries, lambda: mbd_calc)
                for r in mbd_rows:
                    r["backend"] = "learned_mbd"
                results.extend(mbd_rows)
            except Exception as e:
                print(f"    Error: {e}")

        # Evaluate SpookyNet (raw dimer energy)
        if use_spookynet:
            print("  Evaluating SpookyNet...")
            try:
                sn_rows = evaluate_scan(geometries, lambda: spookynet_calc)
                for r in sn_rows:
                    r["backend"] = "spookynet"
                results.extend(sn_rows)
            except Exception as e:
                print(f"    Error: {e}")

            print("  Evaluating SpookyNet hybrid (dimer/monomer decomposition)...")
            try:
                sn_hybrid_rows = evaluate_scan_monomer_decomposed(
                    geometries, lambda: spookynet_calc
                )
                for r in sn_hybrid_rows:
                    r["backend"] = "spookynet_hybrid"
                results.extend(sn_hybrid_rows)
            except Exception as e:
                print(f"    Error: {e}")

        # Evaluate xTB
        if use_xtb:
            print("  Evaluating xTB GFN2...")
            try:
                xtb_rows = evaluate_scan(geometries, lambda: xtb_calc)
                for r in xtb_rows:
                    r["backend"] = "xtb_gfn2"
                results.extend(xtb_rows)
            except Exception as e:
                print(f"    Error: {e}")

        # Evaluate CHARMM/CGenFF
        if use_charmm:
            print("  Evaluating CHARMM/CGenFF...")
            if label_a not in CHARMM_RESIDUES or label_b not in CHARMM_RESIDUES:
                print(f"    Skipping: no CHARMM residue mapping for {label_a}/{label_b}")
            else:
                results.extend(evaluate_charmm_scan(geometries, label_a, label_b, charmm_fns))

    df = pd.DataFrame(results)
    csv_path = args.output_dir / "scan_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Generate plots
    print("Generating plots...")
    for (label_a, label_b), group in df.groupby(["molecule_a", "molecule_b"]):
        plt.figure(figsize=(7, 5))
        for backend, sub in group.groupby("backend"):
            sub = sub.sort_values("distance_angstrom")
            plt.plot(
                sub["distance_angstrom"], sub["energy_kcal_mol"], marker="o", label=backend
            )
        plt.title(f"Dimer Scan: {label_a} + {label_b}")
        plt.xlabel("Center distance / Å")
        plt.ylabel("Energy / kcal mol$^{-1}$")
        plt.legend(frameon=False)
        plt.tight_layout()
        plot_path = args.output_dir / f"{label_a}_{label_b}.png"
        plt.savefig(plot_path)
        plt.close()

    print(f"Plots saved in {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
