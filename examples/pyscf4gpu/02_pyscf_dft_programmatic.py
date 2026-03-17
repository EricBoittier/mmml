#!/usr/bin/env python
"""
Example: DFT full (energy, gradient, hessian, harmonic, thermo) via programmatic interface.

Run from project root: uv run python examples/pyscf4gpu/02_pyscf_dft_programmatic.py
Note: Hessian/harmonic/thermo are expensive; use small molecules.
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
    calcs = [CALCS.ENERGY, CALCS.GRADIENT, CALCS.HESSIAN, CALCS.HARMONIC, CALCS.THERMO]
    args = get_dummy_args(mol_str, calcs)
    args.basis = "def2-SVP"
    args.xc = "PBE0"
    args.output = "examples/pyscf4gpu/out/02_results"

    print("=== 02: DFT full programmatic (energy, gradient, hessian, harmonic, thermo) ===")
    output = compute_dft(args, calcs)
    print(f"Energy: {output['energy']} Hartree")

    save_pyscf_results(args.output, output)
    print(f"Output: {args.output}.npz and .h5")


if __name__ == "__main__":
    main()
