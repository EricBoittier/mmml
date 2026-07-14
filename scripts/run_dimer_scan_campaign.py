#!/usr/bin/env python3
"""Run the molecular dimer scan campaign with learned multipoles, MBD, xTB,
SpookyNet, optional DFTB3-D4, and (optionally) CHARMM/CGenFF — all sharing one
distance/offset grid per pair so every backend lands in a single combined CSV.

The distance grid is chosen per pair: a cheap geometry-only sweep
(``find_safe_min_distance``) locates where fragment atoms actually stop
overlapping (on-axis, offset=0) and anchors the grid there, instead of using
one fixed floor that's unsafe for bulky/asymmetric pairs (e.g. ACE+ACE needs
~5 Å before atoms clear) and wasteful for compact ones.
"""

import argparse
import json
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
    make_dftb3_d4_calculator,
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
            MOLECULES["ACE"].positions[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
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


def _charmm_component_rows(
    common: dict, *, total_kcal: float, elec_kcal: float, vdw_kcal: float
) -> list[dict]:
    """Materialize CHARMM nonbond components as plot-compatible backends."""
    rows: list[dict] = []
    for backend, component in (
        ("charmm", total_kcal),
        ("charmm_electrostatics", elec_kcal),
        ("charmm_lj", vdw_kcal),
    ):
        rows.append(
            {
                **common,
                "energy_ev": component / EV_TO_KCAL_MOL,
                "energy_kcal_mol": component,
                "backend": backend,
            }
        )
    return rows


def evaluate_charmm_scan(geometries, label_a, label_b, charmm_fns) -> list[dict]:
    """Evaluate CHARMM total, electrostatics-only, and LJ-only scan surfaces."""
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
            common = {
                    "molecule_a": label_a,
                    "molecule_b": label_b,
                    "distance_angstrom": geom.distance_angstrom,
                    "offset_angstrom": geom.offset_angstrom,
                    "charmm_ELEC_kcal": elec,
                    "charmm_VDW_kcal": vdw,
                    "min_contact_angstrom": min_fragment_contact_distance(geom.atoms, geom.fragments),
                }
            # Materialize the components as ordinary backends so the existing
            # reference/interaction-energy and 2D plotting pipeline can render
            # them with exactly the same geometry masks as the total surface.
            rows.extend(
                _charmm_component_rows(
                    common, total_kcal=tot, elec_kcal=elec, vdw_kcal=vdw
                )
            )
        except Exception as e:
            print(f"    Warning: CHARMM failed at d={geom.distance_angstrom} Å offset={geom.offset_angstrom} Å: {e}")
    return rows


def build_pair_distance_grid(
    label_a: str, label_b: str, *, min_contact: float = 1.5,
    n_near: int = 11, n_far: int = 8, near_span: float = 2.5, far_span: float = 9.5,
    close_floor: float | None = None, n_close: int = 6,
) -> tuple[np.ndarray, float]:
    """Per-pair distance grid anchored to where fragment atoms actually clear contact.

    If *close_floor* is given, extra points are prepended from *close_floor*
    up to the grid's normal (safe) start — deliberately probing distances
    where fragment atoms overlap for at least some backends. Useful for
    diagnosing whether a backend without a repulsive-wall term (e.g. a bare
    electrostatic multipole/MBD model) has a real bounded minimum or just
    keeps favouring ever-closer contact; other backends (xTB, CHARMM) will
    likely produce huge/clash-filtered energies there, which is expected.
    """
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
    if close_floor is not None and close_floor < d_start:
        close_pts = np.linspace(close_floor, d_start, n_close, endpoint=False)
        distances = np.concatenate([close_pts, distances])
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
        "--spookynet-tag",
        type=str,
        default=None,
        help=(
            "Suffix appended to the spookynet/spookynet_hybrid backend names "
            "(e.g. 'muon_ep7' -> 'spookynet_muon_ep7'), so multiple checkpoints "
            "can be evaluated into the same CSV without colliding under the "
            "same backend name."
        ),
    )
    parser.add_argument(
        "--spookynet-mbd-checkpoint",
        type=Path,
        default=None,
        help=(
            "Override the MBD checkpoint used alongside --spookynet-checkpoint. "
            "By default, if the SpookyNet checkpoint was trained with a frozen "
            "MBD correction (scripts/train_so3lr_spooky_extxyz.py --mbd-checkpoint), "
            "that same path (as recorded in the checkpoint's own config) is used "
            "automatically — this only needs setting if that recorded path doesn't "
            "exist on this machine, or to force a different MBD checkpoint."
        ),
    )
    parser.add_argument(
        "--spookynet-no-mbd",
        action="store_true",
        help=(
            "Force Spooky-only evaluation even if the checkpoint's config references "
            "an MBD correction (e.g. to isolate/debug the Spooky component alone). "
            "Note this will NOT match how the checkpoint was actually trained."
        ),
    )
    parser.add_argument(
        "--spookynet-mbd-weight",
        type=float,
        default=None,
        help="Override the MBD correction weight (default: use the value recorded in the checkpoint's config)",
    )
    parser.add_argument(
        "--with-charmm",
        action="store_true",
        help="Also evaluate CHARMM/CGenFF energies (requires pycharmm)",
    )
    parser.add_argument(
        "--skip-xtb",
        action="store_true",
        help="Don't evaluate xTB, even if available (e.g. it's being run separately elsewhere)",
    )
    parser.add_argument(
        "--with-dftb3-d4",
        action="store_true",
        help=(
            "Evaluate DFTB3-D4 through the external DFTB+ executable using "
            "the 3ob-3-1 Slater-Koster parameter set."
        ),
    )
    parser.add_argument(
        "--dftb-sk-dir",
        type=Path,
        default=None,
        help="Path to the 3ob-3-1 Slater-Koster directory (required with --with-dftb3-d4)",
    )
    parser.add_argument(
        "--dftb-command",
        type=str,
        default="dftb+",
        help="DFTB+ executable to run (default: dftb+)",
    )
    parser.add_argument(
        "--min-contact",
        type=float,
        default=1.5,
        help="Contact distance (Å) used to anchor each pair's distance grid (default 1.5)",
    )
    parser.add_argument(
        "--close-floor",
        type=float,
        default=None,
        help=(
            "Extend each pair's grid inward to this centre-to-centre distance (Å), "
            "below the normal safe-contact start. Probes whether backends without a "
            "repulsive-wall term (multipoles/MBD) have a real minimum or diverge "
            "unbounded at close range; other backends will likely be clash-filtered there."
        ),
    )
    parser.add_argument(
        "--n-close",
        type=int,
        default=6,
        help="Number of extra points between --close-floor and the normal grid start (default 6)",
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
            spookynet_calc = SpookyNetCalculator(
                checkpoint=args.spookynet_checkpoint,
                mbd_checkpoint=(False if args.spookynet_no_mbd else args.spookynet_mbd_checkpoint),
                mbd_weight=args.spookynet_mbd_weight,
            )
            use_spookynet = True
            print("  SpookyNet calculator initialized successfully.")
        except Exception as e:
            print(f"  Error loading SpookyNet model: {e}")
            sys.exit(1)
    else:
        print("  No SpookyNet checkpoint provided. Skipping spookynet/spookynet_hybrid backends.")

    # Check for xTB
    use_xtb = False
    if args.skip_xtb:
        print("  --skip-xtb passed. Skipping xTB backend.")
    else:
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

    use_dftb3_d4 = False
    dftb3_d4_calc = None
    if args.with_dftb3_d4:
        if args.dftb_sk_dir is None:
            parser.error("--with-dftb3-d4 requires --dftb-sk-dir pointing to 3ob-3-1")
        try:
            dftb3_d4_calc = make_dftb3_d4_calculator(
                slako_dir=args.dftb_sk_dir,
                workdir=args.output_dir / "_dftb3_d4_work",
                command=args.dftb_command,
            )
            use_dftb3_d4 = True
            print("  DFTB3-D4 calculator initialized successfully.")
        except Exception as e:
            print(f"  Error initializing DFTB3-D4: {e}")
            sys.exit(1)

    if not (use_multipole or use_mbd or use_xtb or use_spookynet or use_charmm or use_dftb3_d4):
        print("No backends are available or enabled. Exiting.")
        sys.exit(0)

    labels = list(MOLECULES.keys())
    pairs = molecule_pair_labels(labels, include_homodimers=True)

    print(f"Will scan {len(pairs)} unique pairs (per-pair distance grid, up to 5 offsets, 2D).")

    results = []

    for idx, (label_a, label_b) in enumerate(pairs, 1):
        pair_cfg = PAIR_SCAN_CONFIG[(label_a, label_b)]
        offsets = pair_cfg["offsets_angstrom"]
        distances, safe_d = build_pair_distance_grid(
            label_a, label_b, min_contact=args.min_contact,
            close_floor=args.close_floor, n_close=args.n_close,
        )
        print(f"[{idx}/{len(pairs)}] {label_a}+{label_b}: {pair_cfg['description']}")
        print(
            f"  safe contact clears at d≈{safe_d:.2f} Å (offset=0) — grid spans "
            f"{distances.min():.2f}–{distances.max():.2f} Å"
        )
        print(f"  {len(distances)} distances × {len(offsets)} offsets = {len(distances) * len(offsets)} geometries")
        geometries = list(make_oriented_scan_geometries(label_a, label_b, distances, offsets))

        if use_spookynet:
            from mmml.analysis.dimer_cgenff import attach_cgenff_dimer_metadata
            from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

            for geometry in geometries:
                attach_cgenff_dimer_metadata(
                    geometry.atoms,
                    geometry.pair,
                    geometry.fragments,
                    prm_path=CGENFF_PRM,
                )

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
            spookynet_backend = f"spookynet_{args.spookynet_tag}" if args.spookynet_tag else "spookynet"
            spookynet_hybrid_backend = (
                f"spookynet_hybrid_{args.spookynet_tag}" if args.spookynet_tag else "spookynet_hybrid"
            )
            print(f"  Evaluating SpookyNet (backend={spookynet_backend})...")
            try:
                sn_rows = evaluate_scan(geometries, lambda: spookynet_calc)
                for r in sn_rows:
                    r["backend"] = spookynet_backend
                results.extend(sn_rows)
            except Exception as e:
                print(f"    Error: {e}")

            print(f"  Evaluating SpookyNet hybrid (backend={spookynet_hybrid_backend})...")
            try:
                sn_hybrid_rows = evaluate_scan_monomer_decomposed(
                    geometries, lambda: spookynet_calc
                )
                for r in sn_hybrid_rows:
                    r["backend"] = spookynet_hybrid_backend
                results.extend(sn_hybrid_rows)

                component_backends = {
                    "electrostatics_energy": "electrostatics",
                    "cgenff_vdw_energy": "cgenff_lj",
                    "zbl_repulsion_energy": "zbl",
                    "neural_energy": "neural",
                    "mbd_energy": "mbd",
                }
                for key, suffix in component_backends.items():
                    value_key = f"comp_Eint_{key}_ev"
                    for source in sn_hybrid_rows:
                        value_ev = float(source.get(value_key, 0.0))
                        component_row = {
                            "molecule_a": source["molecule_a"],
                            "molecule_b": source["molecule_b"],
                            "distance_angstrom": source["distance_angstrom"],
                            "offset_angstrom": source["offset_angstrom"],
                            "energy_ev": value_ev,
                            "energy_kcal_mol": value_ev * EV_TO_KCAL_MOL,
                            "min_contact_angstrom": source["min_contact_angstrom"],
                            "backend": (
                                f"spookynet_{suffix}_{args.spookynet_tag}"
                                if args.spookynet_tag
                                else f"spookynet_{suffix}"
                            ),
                        }
                        results.append(component_row)
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

        # Evaluate DFTB3-D4.  As for xTB, the plotting workflow obtains the
        # interaction energy by referencing every offset curve to its largest
        # separation, so retain the raw total energy in the standard columns.
        if use_dftb3_d4:
            print("  Evaluating DFTB3-D4 (3ob-3-1)...")
            try:
                dftb_rows = evaluate_scan(geometries, lambda: dftb3_d4_calc)
                for r in dftb_rows:
                    r["backend"] = "dftb3_d4"
                results.extend(dftb_rows)
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
    if csv_path.exists():
        # Accumulate with whatever's already in this output dir instead of
        # clobbering it — e.g. re-running with a different --spookynet-tag
        # (or any other backend combo) into the same --output-dir would
        # otherwise silently overwrite prior results with none of this run's
        # backends in them. Keep this run's rows on any (pair, backend,
        # distance, offset) collision (an intentional re-run of the same
        # backend refreshes it); everything else from before is preserved.
        df_prior = pd.read_csv(csv_path)
        key_cols = ["molecule_a", "molecule_b", "backend", "distance_angstrom", "offset_angstrom"]
        df = pd.concat([df_prior, df], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path} ({len(df)} total rows, backends: {sorted(df['backend'].unique())})")

    if use_spookynet:
        # Proof-of-work for the standalone adapter: this report is written
        # after evaluation, so it records whether annotated CGenFF inputs were
        # actually consumed rather than merely supported by the class.
        spookynet_calc.write_energy_function_report(
            args.output_dir / "calculator_energy_function.json"
        )
        hybrid = df[df["backend"] == spookynet_hybrid_backend].copy()
        component_columns = [
            "comp_Eint_neural_energy_ev",
            "comp_Eint_electrostatics_energy_ev",
            "comp_Eint_cgenff_vdw_energy_ev",
            "comp_Eint_zbl_repulsion_energy_ev",
            "comp_Eint_mbd_energy_ev",
        ]
        available = [name for name in component_columns if name in hybrid.columns]
        reconstructed = hybrid[available].fillna(0.0).sum(axis=1)
        target = hybrid["comp_Eint_ev"]
        lj_values = hybrid["comp_Eint_cgenff_vdw_energy_ev"].fillna(0.0)
        audit = {
            "checkpoint": str(args.spookynet_checkpoint.resolve()),
            "backend": spookynet_hybrid_backend,
            "n_points": int(len(hybrid)),
            "component_columns": available,
            "max_abs_component_reconstruction_error_ev": float(
                np.max(np.abs(reconstructed - target)) if len(hybrid) else np.nan
            ),
            "cgenff_lj_nonzero_points": int(np.count_nonzero(np.abs(lj_values) > 1e-12)),
            "cgenff_lj_min_ev": float(lj_values.min()) if len(hybrid) else np.nan,
            "cgenff_lj_max_ev": float(lj_values.max()) if len(hybrid) else np.nan,
            "cgenff_inputs_consumed": bool(spookynet_calc.cgenff_lj_inputs_supplied),
            "jax_enable_x64": bool(__import__("jax").config.jax_enable_x64),
            "mmml_ml_dtype": os.environ.get("MMML_ML_DTYPE"),
        }
        (args.output_dir / "component_reconstruction_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "Spooky component audit: "
            f"CGenFF LJ nonzero at {audit['cgenff_lj_nonzero_points']}/{audit['n_points']} points; "
            "max reconstruction error "
            f"{audit['max_abs_component_reconstruction_error_ev']:.3e} eV"
        )

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
