#!/usr/bin/env python
"""
Example: DFT energy + gradient via programmatic interface.

Run from project root: uv run python examples/pyscf4gpu/01_pyscf_dft_programmatic.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmml.interfaces.pyscf4gpuInterface.calcs import (
    compute_dft,
    get_dummy_args,
    save_pyscf_results,
)
from mmml.interfaces.pyscf4gpuInterface.enums import CALCS


def main():
    mol_str = "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0"
    args = get_dummy_args(mol_str, [CALCS.ENERGY, CALCS.GRADIENT])
    args.basis = "def2-TZVP"  # default is def2-SVP
    args.xc = "PBE0"
    args.output = "examples/pyscf4gpu/out/01_results"

    print("=== 01: DFT programmatic (PBE0/def2-TZVP) ===")
    output = compute_dft(args, [CALCS.ENERGY, CALCS.GRADIENT])
    print(f"Energy: {output['energy']} Hartree")

    save_pyscf_results(args.output, output)
    print(f"Output: {args.output}.npz and .h5")


if __name__ == "__main__":
    main()
