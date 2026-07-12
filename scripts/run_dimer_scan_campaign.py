#!/usr/bin/env python3
"""Run the molecular dimer scan campaign with learned multipoles, MBD, and xTB."""

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
    PAIR_SCAN_CONFIG,
    make_oriented_scan_geometries,
)
from mmml.analysis.dimer_scans import (
    evaluate_scan,
    make_xtb_calculator,
    molecule_pair_labels,
)
from mmml.models.mbd import QCMLMBDCalculator
from mmml.models.multipoles import LearnedMolecularMultipoleElectrostatics


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

    # Check for xTB
    use_xtb = False
    try:
        xtb_calc = make_xtb_calculator(method="GFN2-xTB")
        use_xtb = True
        print("  xTB calculator initialized successfully.")
    except Exception as e:
        print(f"  xTB calculator not available: {e}. Skipping xTB backend.")

    if not (use_multipole or use_mbd or use_xtb):
        print("No backends are available or enabled. Exiting.")
        sys.exit(0)

    # Distance grid: fine spacing near contact (asymptotics), coarser at long range
    distances = np.concatenate([
        np.linspace(2.5, 5.0, 11),   # close-range: 2.5–5.0 Å in 0.25 Å steps
        np.linspace(5.5, 12.0, 8),   # long-range: 5.5–12.0 Å in ~0.93 Å steps
    ])
    labels = list(MOLECULES.keys())
    pairs = molecule_pair_labels(labels, include_homodimers=True)

    print(f"Will scan {len(pairs)} unique pairs × {len(distances)} distances × up to 5 offsets (2D).")

    results = []

    for idx, (label_a, label_b) in enumerate(pairs, 1):
        pair_cfg = PAIR_SCAN_CONFIG[(label_a, label_b)]
        offsets = pair_cfg["offsets_angstrom"]
        print(f"[{idx}/{len(pairs)}] {label_a}+{label_b}: {pair_cfg['description']}")
        print(f"  {len(distances)} distances × {len(offsets)} offsets = {len(distances)*len(offsets)} geometries")
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
