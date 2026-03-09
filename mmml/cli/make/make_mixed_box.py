"""Create a simulation box containing a mixture of different residue types.

Generates individual residue PDB files via ``make_res``, packs them into a
single box with packmol, then sets up the PyCHARMM PSF / coordinates.

Usage from Python::

    from mmml.cli.make_mixed_box import main_loop
    import argparse

    args = argparse.Namespace(
        residues=["MEOH", "ACET"],
        counts=[10, 10],
        side_length=23.0,
        skip_energy_show=False,
    )
    result = main_loop(args)
    # result["atoms_per_monomer"]  ->  [6, 6, ..., 4, 4, ...]
    # result["pdb_path"]           ->  "pdb/init-packmol.pdb"

CLI::

    python -m mmml.cli.make_mixed_box --residues MEOH ACET --counts 10 10 --side_length 23.0
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mixed-residue simulation box with packmol."
    )
    parser.add_argument(
        "--residues",
        nargs="+",
        required=True,
        help="Residue names (e.g. MEOH ACET).",
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        required=True,
        help="Number of molecules for each residue (same order as --residues).",
    )
    parser.add_argument(
        "--side_length",
        type=float,
        default=30.0,
        help="Cubic box side length in Å (default: 30.0).",
    )
    parser.add_argument(
        "--skip-energy-show",
        dest="skip_energy_show",
        action="store_true",
        help="Skip CHARMM energy.show() calls (avoids segfault on some clusters).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Packmol distance tolerance in Å (default: 2.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _generate_residue_pdbs(
    residues: List[str],
    skip_energy_show: bool = False,
) -> Dict[str, Any]:
    """Generate a PDB file for each unique residue type.

    Returns a dict mapping residue name -> {"pdb": Path, "n_atoms": int, "atoms": Atoms}.
    """
    from mmml.cli.make import make_res
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM

    os.makedirs("pdb", exist_ok=True)
    os.makedirs("xyz", exist_ok=True)
    os.makedirs("psf", exist_ok=True)

    info: Dict[str, Any] = {}
    for i, res in enumerate(dict.fromkeys(residues)):  # unique, preserving order
        if i > 0:
            CLEAR_CHARMM()
        res_upper = res.upper()
        args_res = argparse.Namespace(res=res_upper, skip_energy_show=skip_energy_show)
        atoms = make_res.main_loop(args_res)
        pdb_path = Path("pdb") / f"{res_upper.lower()}.pdb"
        shutil.copy("pdb/initial.pdb", pdb_path)
        info[res_upper] = {
            "pdb": pdb_path,
            "n_atoms": len(atoms),
            "atoms": atoms,
        }
        print(f"[make_mixed_box] Residue {res_upper}: {len(atoms)} atoms -> {pdb_path}")
    return info


def _run_packmol_mixed(
    residue_info: Dict[str, Any],
    residues: List[str],
    counts: List[int],
    side_length: float,
    tolerance: float = 2.0,
    output_pdb: str = "pdb/init-packmol.pdb",
) -> str:
    """Write a packmol input with one ``structure`` block per residue type and run it.

    Returns the path to the output PDB.
    """
    from mmml.interfaces.pycharmmInterface.setupBox import PACKMOL_PATH

    os.makedirs("packmol", exist_ok=True)
    os.makedirs(str(Path(output_pdb).parent), exist_ok=True)

    L = side_length
    blocks = []
    for res, n in zip(residues, counts):
        pdb = residue_info[res.upper()]["pdb"]
        blocks.append(
            f"structure {pdb}\n"
            f"  chain A\n"
            f"  resnumbers 2\n"
            f"  number {n}\n"
            f"  inside box 0.0 0.0 0.0 {L} {L} {L}\n"
            f"end structure"
        )

    packmol_input = (
        f"seed {np.random.randint(1_000_000)}\n"
        f"output {output_pdb}\n"
        f"filetype pdb\n"
        f"tolerance {tolerance}\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )

    inp_path = Path("packmol") / "packmol_mixed.inp"
    inp_path.write_text(packmol_input)

    packmol_bin = os.path.expanduser(PACKMOL_PATH)
    cmd = f"{packmol_bin} < {inp_path}"
    print(f"[make_mixed_box] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"packmol failed with exit code {ret}")
    print(f"[make_mixed_box] packmol done -> {output_pdb}")
    return output_pdb


def _setup_charmm_box(
    pdb_path: str,
    side_length: float,
    tag: str,
) -> None:
    """Read the mixed PDB into PyCHARMM, set up PSF / PBC, and minimise."""
    from mmml.interfaces.pycharmmInterface import setupBox
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        pycharmm,
        reset_block,
        reset_block_no_internal,
        safe_energy_show,
    )
    from mmml.interfaces.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM

    CLEAR_CHARMM()
    setupBox.setup_box_generic(pdb_path, side_length=side_length, tag=tag)

    reset_block()
    reset_block_no_internal()
    reset_block()
    nbonds_script = """
nbonds atom cutnb 14.0 ctofnb 12.0 ctonnb 10.0 -
vswitch NBXMOD 3 -
inbfrq -1 imgfrq -1
"""
    pycharmm.lingo.charmm_script(nbonds_script)
    safe_energy_show()
    setupBox.minimize_box()
    print("[make_mixed_box] Box setup & minimisation done.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main_loop(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a mixed-residue box and return metadata.

    Args:
        args: Namespace with ``residues``, ``counts``, ``side_length``,
              and optionally ``skip_energy_show`` and ``tolerance``.

    Returns:
        Dict with keys:
            ``atoms_per_monomer`` – list of atom counts per monomer (in box order)
            ``n_atoms_per_type``  – dict mapping residue name -> atom count
            ``pdb_path``          – path to the packed PDB
            ``n_monomers``        – total number of monomers
            ``side_length``       – box side length
    """
    residues: List[str] = [r.upper() for r in args.residues]
    counts: List[int] = list(args.counts)
    side_length: float = args.side_length
    skip_energy_show: bool = getattr(args, "skip_energy_show", False)
    tolerance: float = getattr(args, "tolerance", 2.0)

    if len(residues) != len(counts):
        raise ValueError(
            f"--residues and --counts must have the same length; "
            f"got {len(residues)} residues and {len(counts)} counts"
        )

    # 1. Generate one PDB per unique residue type
    residue_info = _generate_residue_pdbs(residues, skip_energy_show=skip_energy_show)

    # 2. Pack mixed box
    output_pdb = "pdb/init-packmol.pdb"
    _run_packmol_mixed(
        residue_info, residues, counts,
        side_length=side_length,
        tolerance=tolerance,
        output_pdb=output_pdb,
    )

    # 3. Setup PyCHARMM box
    tag = "_".join(r.lower() for r in residues)
    _setup_charmm_box(output_pdb, side_length=side_length, tag=tag)

    # 4. Build the atoms_per_monomer list (packmol places types in order)
    atoms_per_monomer: List[int] = []
    n_atoms_per_type: Dict[str, int] = {}
    for res, n in zip(residues, counts):
        na = residue_info[res]["n_atoms"]
        atoms_per_monomer.extend([na] * n)
        n_atoms_per_type[res] = na

    result = {
        "atoms_per_monomer": atoms_per_monomer,
        "n_atoms_per_type": n_atoms_per_type,
        "pdb_path": output_pdb,
        "n_monomers": sum(counts),
        "side_length": side_length,
    }
    print(f"[make_mixed_box] atoms_per_monomer = {atoms_per_monomer}")
    print(f"[make_mixed_box] total_atoms = {sum(atoms_per_monomer)}")
    return result


def main() -> None:
    args = parse_args()
    print(args)
    main_loop(args)


if __name__ == "__main__":
    main()
