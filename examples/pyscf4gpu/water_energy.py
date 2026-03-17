#!/usr/bin/env python
"""
Example: GPU-accelerated DFT energy and gradient for water using pyscf4gpuInterface.

Requires: gpu4pyscf, pyscf (install via micromamba-create-full or quantum-gpu extra)

Usage:
    uv run python examples/pyscf4gpu/water_energy.py
    make pyscf-example
"""

from pathlib import Path

# Add project root for imports when run as script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmml.interfaces.pyscf4gpuInterface.calcs import (
    setup_mol,
    compute_dft,
    save_output,
    get_dummy_args,
)
from mmml.interfaces.pyscf4gpuInterface.enums import CALCS


def main():
    # Water molecule in Angstrom (PySCF format: "symbol x y z")
    mol_str = """
    O    0.000000    0.000000    0.000000
    H    0.957000    0.000000    0.000000
    H   -0.240000    0.927000    0.000000
    """

    # Build args (same as CLI would parse)
    args = get_dummy_args(mol_str.strip(), [CALCS.ENERGY, CALCS.GRADIENT])
    args.basis = "def2-tzvp"
    args.xc = "PBE0"
    args.output = "water_dft_output.pkl"

    print("Computing DFT energy and gradient for water (PBE0/def2-TZVP)")
    print("-" * 60)

    output = compute_dft(args, [CALCS.ENERGY, CALCS.GRADIENT], extra=None)

    print("-" * 60)
    print(f"Energy: {output['energy']:.10f} Hartree")
    print(f"Gradient shape: {output['gradient'].shape}")

    save_output(args.output, output, "pkl")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
