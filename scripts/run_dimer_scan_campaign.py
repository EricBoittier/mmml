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

from ase import Atoms

from mmml.analysis.dimer_scans import (
    distance_scan_geometries,
    evaluate_scan,
    make_xtb_calculator,
    molecule_pair_labels,
)
from mmml.models.mbd import QCMLMBDCalculator
from mmml.models.multipoles import LearnedMolecularMultipoleElectrostatics

# Monomer registry from notebooks/qcml_dimer_scan_prototype.py
MOLECULES = {
    "DCM": {
        "atoms": Atoms(
            "CCl2H2",
            positions=[
                [0.000, 0.000, 0.000],
                [1.760, 0.000, 0.000],
                [-1.760, 0.000, 0.000],
                [0.000, 0.950, 0.720],
                [0.000, -0.950, 0.720],
            ],
        ),
    },
    "ACE": {
        "atoms": Atoms(
            "C3OH6",
            positions=[
                [0.000, 0.000, 0.000],
                [1.520, 0.000, 0.000],
                [-1.520, 0.000, 0.000],
                [0.000, 1.220, 0.000],
                [2.050, 0.900, 0.000],
                [2.050, -0.450, 0.780],
                [2.050, -0.450, -0.780],
                [-2.050, 0.900, 0.000],
                [-2.050, -0.450, 0.780],
                [-2.050, -0.450, -0.780],
            ],
        ),
    },
    "BENZ": {
        "atoms": Atoms(
            "C6H6",
            positions=[
                [1.397, 0.000, 0.000],
                [0.699, 1.210, 0.000],
                [-0.699, 1.210, 0.000],
                [-1.397, 0.000, 0.000],
                [-0.699, -1.210, 0.000],
                [0.699, -1.210, 0.000],
                [2.480, 0.000, 0.000],
                [1.240, 2.148, 0.000],
                [-1.240, 2.148, 0.000],
                [-2.480, 0.000, 0.000],
                [-1.240, -2.148, 0.000],
                [1.240, -2.148, 0.000],
            ],
        ),
    },
    "TIP3": {
        "atoms": Atoms(
            "OH2",
            positions=[
                [0.000000, 0.000000, 0.000000],
                [0.957200, 0.000000, 0.000000],
                [-0.239987, 0.926627, 0.000000],
            ],
        ),
    },
    "MEOH": {
        "atoms": Atoms(
            "COH4",
            positions=[
                [0.000, 0.000, 0.000],
                [1.430, 0.000, 0.000],
                [1.770, 0.910, 0.000],
                [-0.540, 0.900, 0.000],
                [-0.540, -0.450, 0.780],
                [-0.540, -0.450, -0.780],
            ],
        ),
    },
}


def make_pair_scan(label_a: str, label_b: str, distances: np.ndarray) -> list:
    return list(
        distance_scan_geometries(
            MOLECULES[label_a]["atoms"],
            MOLECULES[label_b]["atoms"],
            distances,
            pair=(label_a, label_b),
            axis=(1.0, 0.0, 0.0),
            center="centroid",
            mol_id_array="mol_id",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multipole-checkpoint",
        type=Path,
        required=True,
        help="Path to multipoles model checkpoint folder",
    )
    parser.add_argument(
        "--mbd-checkpoint", type=Path, required=True, help="Path to MBD model checkpoint folder"
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
    try:
        multipole_calc = LearnedMolecularMultipoleElectrostatics(
            checkpoint=args.multipole_checkpoint,
            max_ell=args.max_ell,
            origin="nuclear_charge_centroid",
            softening_bohr=0.5,
        )
        print("  Learned multipole calculator initialized successfully.")
    except Exception as e:
        print(f"  Error loading multipole model: {e}")
        sys.exit(1)

    try:
        mbd_calc = QCMLMBDCalculator(checkpoint=args.mbd_checkpoint)
        print("  Learned MBD calculator initialized successfully.")
    except Exception as e:
        print(f"  Error loading MBD model: {e}")
        sys.exit(1)

    # Check for xTB
    use_xtb = False
    try:
        xtb_calc = make_xtb_calculator(method="GFN2-xTB")
        use_xtb = True
        print("  xTB calculator initialized successfully.")
    except Exception as e:
        print(f"  xTB calculator not available: {e}. Skipping xTB backend.")

    # Define spacing grids
    distances = np.linspace(3.0, 12.0, 19)
    labels = list(MOLECULES.keys())
    pairs = molecule_pair_labels(labels, include_homodimers=True)

    print(f"Will scan {len(pairs)} unique pairs across {len(distances)} distances.")

    results = []

    for idx, (label_a, label_b) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Scanning {label_a} + {label_b}...")
        geometries = make_pair_scan(label_a, label_b, distances)

        # Evaluate Multipoles
        print(f"  Evaluating learned multipole (max_ell={args.max_ell})...")
        try:
            mp_rows = evaluate_scan(geometries, lambda: multipole_calc)
            for r in mp_rows:
                r["backend"] = "learned_multipole"
            results.extend(mp_rows)
        except Exception as e:
            print(f"    Error: {e}")

        # Evaluate MBD
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
