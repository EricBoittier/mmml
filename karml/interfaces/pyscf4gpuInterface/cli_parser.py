"""Argparse for ``mmml pyscf-dft`` (no PySCF / gpu4pyscf).

Help and docs generation import this module. The GPU DFT implementation stays
in ``calcs.py`` and is loaded only when a calculation is requested.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # molecule
    parser.add_argument("--mol", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--log_file", type=str, default="pyscf.log")
    parser.add_argument("--monomer_a", type=str, default="")
    parser.add_argument("--monomer_b", type=str, default="")
    parser.add_argument("--basis", type=str, default="def2-SVP")
    parser.add_argument("--xc", type=str, default="PBE0")
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)
    # flags to do certain calcs
    parser.add_argument("--energy", default=False, action="store_true")
    parser.add_argument("--optimize", default=False, action="store_true")
    parser.add_argument("--gradient", default=False, action="store_true")
    parser.add_argument("--hessian", default=False, action="store_true")
    parser.add_argument("--harmonic", default=False, action="store_true")
    parser.add_argument("--thermo", default=False, action="store_true")
    parser.add_argument("--interaction", default=False, action="store_true")
    parser.add_argument("--dens_esp", default=False, action="store_true")
    parser.add_argument("--ir", default=False, action="store_true")
    parser.add_argument("--shielding", default=False, action="store_true")
    parser.add_argument("--polarizability", default=False, action="store_true")
    parser.add_argument(
        "--ir-efield",
        default=False,
        action="store_true",
        help="IR + Hessian pipeline in a uniform E-field; scan fields from --efield-points",
    )
    parser.add_argument(
        "--efield-points",
        type=str,
        default="0,0,0",
        help="Semicolon-separated Ex,Ey,Ez in a.u., e.g. '0,0,0;0,0,0.001;0,0,-0.001'",
    )
    parser.add_argument(
        "--efield-fd-axis",
        type=int,
        default=2,
        help="Cartesian axis (0=x,1=y,2=z) for finite-difference dμ/dE from the scan",
    )
    parser.add_argument(
        "--efield-scf",
        default=False,
        action="store_true",
        help="SCF only in uniform E-field: energy, dipole, forces (use --efield-points); no IR/Hessian",
    )
    parser.add_argument(
        "--efield-scf-no-forces",
        default=False,
        action="store_true",
        help="With --efield-scf, skip nuclear gradient (energy + dipole only)",
    )
    parser.add_argument(
        "--efield-dipole-unit",
        type=str,
        default="DEBYE",
        help="Dipole unit for --efield-scf (e.g. DEBYE, AU)",
    )
    parser.add_argument(
        "--no-efield-include-nuclear-energy",
        dest="efield_include_nuclear_energy",
        action="store_false",
        help=(
            "After SCF in a uniform field, omit nuclear-field energy (use mf.kernel energy only)."
        ),
    )
    parser.set_defaults(efield_include_nuclear_energy=True)
    parser.add_argument("--save_option", type=str, default="hdf5")
    return parser
