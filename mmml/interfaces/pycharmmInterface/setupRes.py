# Standard library imports
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

# Third-party scientific computing
import numpy as np

# ASE imports
import ase
import ase.io
from ase import Atoms
from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block
from mmml.interfaces.pycharmmInterface.import_pycharmm import safe_energy_show
from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF, CHARMM_HOME, CHARMM_LIB_DIR
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf, set_up_directories

os.environ["CHARMM_HOME"] = CHARMM_HOME
os.environ["CHARMM_LIB_DIR"] = CHARMM_LIB_DIR

print("CHARMM_HOME: ", CHARMM_HOME)
print("CHARMM_LIB_DIR: ", CHARMM_LIB_DIR)

sys.path.append(str(Path(CHARMM_HOME) / "tool" / "pycharmm"))


# CHARMM imports
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.minimize as minimize
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.lingo


try:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

    apply_charmm_verbosity(prnlev=5, warnlev=5, bomlev=-2)
except Exception:
    pass


problem_symbols = ["HO", "CA", "CM", ]


def iupac_2_number(iupac):
    from mendeleev import element

    allowed = ["H", "C", "N", "O", "F", "S", "CL"]
    number = []
    for i in iupac:
        if i[0:1] == "CL":
            number.append(element(i[0:1]).atomic_number)
        elif i[0] in allowed:
            number.append(element(i[0]).atomic_number)
        else:
            print("Element not supported by ANI2X models: atom {}".format(i))
            exit()
    return number


def get_residue_atoms() -> dict[str, list[list[float]]]:
    with open(
        CGENFF_RTF
    ) as f:
        lines = f.readlines()

    """Reads the RTF file and returns a dictionary of residue atoms and the atom names of their internal coordinates"""
    resatoms = [
        _.strip("\n").split()
        for _ in lines
        if _.startswith("RESI") or _.startswith("IC")
    ]
    resatoms = [
        _.strip("\n").split()
        for _ in lines
        if _.startswith("RESI") or _.startswith("IC")
    ]
    x = {}
    for _ in resatoms:
        if _[0] == "RESI":
            x[_[1]] = []
            k = _[1]
        elif _[0] == "IC":
            x[k].append(_[1:4])


def generate_residue(resid) -> None:
    """Generates a residue from the RTF file"""
    print("*" * 5, "Generating residue", "*" * 5)
    s = """DELETE ATOM SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        CGENFF_PRM_BOMLEV,
        ic_prm_fill,
        read_cgenff_prm,
    )

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        read.rtf(CGENFF_RTF)
        read_cgenff_prm(bomlev=False)
    read.sequence_string(resid)
    gen.new_segment(seg_name=resid, setup_ic=True)
    ic_prm_fill(replace_all=True)
    reset_block()





def _show_energy(skip_energy_show: bool) -> None:
    if skip_energy_show:
        print("Skipping energy.show() (--skip-energy-show).")
        return
    safe_energy_show()


def _has_resolved_geometry(atoms: Atoms, *, min_axis_span_A: float = 0.2) -> bool:
    """True when coordinates are finite and span at least two dimensions.

    Guards against failed IC builds that leave CHARMM's 9999 placeholder coords,
    NaNs, or collapse the molecule onto a single point or line. Genuinely planar
    molecules (e.g. a single water) are valid geometry, so the extent test is
    orientation-independent: it measures spans along the *principal* axes and
    only requires the second-largest to exceed ``min_axis_span_A``. A collapsed
    point spans zero axes and a line spans one, so both are rejected, while any
    real 2D/3D geometry passes regardless of how it happens to sit relative to
    the Cartesian axes (the previous all-three-Cartesian-axes test spuriously
    failed axis-aligned planar molecules).
    """
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        return False
    if len(positions) <= 2:
        # Fewer than 3 atoms cannot define a 2D extent; a finite point/diatomic
        # is the best geometry available, so accept it.
        return True
    centered = positions - positions.mean(axis=0)
    # Project onto principal axes, then measure the Å-valued span along each so
    # the threshold stays in the same units as before.
    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
    principal_spans = np.sort(np.ptp(centered @ principal_axes.T, axis=0))[::-1]
    if principal_spans[1] < float(min_axis_span_A):
        return False
    # Reject accidental y/z duplication (bad PDB parsing or coor mix-ups).
    if np.allclose(positions[:, 1], positions[:, 2], rtol=0.0, atol=1e-6):
        return False
    return True


def _charmm_xyz_array() -> np.ndarray:
    return coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=np.float64)


def _set_charmm_xyz(positions: np.ndarray) -> None:
    import pandas as pd

    arr = np.asarray(positions, dtype=np.float64)
    coor.set_positions(pd.DataFrame(arr, columns=["x", "y", "z"]))


def _needs_coordinate_seed(positions: np.ndarray) -> bool:
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        return True
    return bool(np.any(np.abs(positions) > 9000.0))


def generate_coordinates(
    skip_energy_show: bool = False,
    validate: bool = True,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Build coordinates for the live CHARMM residue.

    ``rng`` makes the result **reproducible**. Both random draws below feed a
    minimization, so an unseeded generator lands in a different local minimum on
    every call: for trialanine the radius of gyration wandered between 3.29 and
    3.88 Å across runs of identical code. That non-determinism reached CI as a
    flaky pair-count assertion in ``test_cg_jaxmd_unified`` (see PR #180), and it
    silently undermines any baseline built on a structure from here.

    ``None`` keeps the historical unseeded behavior so existing callers are
    unaffected; pass a seeded generator when you need the same structure twice.
    """
    print("*" * 5, "Generating coordinates", "*" * 5)

    set_up_directories()

    if rng is None:
        rng = np.random.default_rng()

    ic.build()
    pycharmm_quiet()

    pos = _charmm_xyz_array()
    if _needs_coordinate_seed(pos):
        # ``ic.build()`` can leave CHARMM's 9999 placeholder coords; seed finite
        # values before minimization (use numpy — not in-place DataFrame ops).
        pos = rng.uniform(0.5, 2.5, size=pos.shape)
        _set_charmm_xyz(pos)

    mini(nbxmod=1, skip_energy_show=skip_energy_show)

    # Light jitter breaks symmetric IC seeds before the production exclusion list.
    pos = _charmm_xyz_array().copy()
    pos *= rng.uniform(0.85, 1.15, size=pos.shape)
    _set_charmm_xyz(pos)
    mini(nbxmod=5, skip_energy_show=skip_energy_show)
    # end_energy = pycharmm.lingo.get_energy_value("ENER")
    # energy_diff = end_energy - start_energy
    # if energy_diff > 0:
        # print("WARNING: Energy difference is positive, something may have gone wrong")

    # save pycharmm coordinates as pdb file
    write.coor_pdb("pdb/initial.pdb")

    # read pdb file
    mol = ase.io.read("pdb/initial.pdb")
    Z = get_Z_from_psf()
    mol.set_atomic_numbers(Z)

    atoms = ase.Atoms(
        symbols=mol.get_chemical_symbols(),
        positions=mol.get_positions(),
        cell=mol.get_cell(),
        pbc=mol.get_pbc(),
    )
    if validate and not _has_resolved_geometry(atoms):
        raise RuntimeError(
            "PyCHARMM residue coordinate generation produced unresolved geometry. "
            "Ensure generate_residue() has loaded topology/parameters before minimization."
        )
    return atoms




def mini(nbxmod=5, skip_energy_show: bool = False):
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        apply_nbonds_kwargs,
        vacuum_nbond_kwargs,
    )

    print("*" * 5, "Minimizing", "*" * 5)
    with charmm_relaxed_bomlev(-2):
        pycharmm_quiet()
        apply_nbonds_kwargs(vacuum_nbond_kwargs(nbxmod=nbxmod))

        # equivalent CHARMM scripting command: minimize abnr nstep 1000 tole 1e-3 tolgr 1e-3
        minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
        # equivalent CHARMM scripting command: energy
        _show_energy(skip_energy_show)


def write_psf(resid: str) -> None:
    print("*" * 5, "Writing PSF", "*" * 5)
    print(f"psf/{resid.lower()}-1.psf")
    write.psf_card("psf/initial.psf")
    write.psf_card(f"psf/{resid.lower()}-1.psf")


def main(resid: str, skip_energy_show: bool = False, max_attempts: int = 2) -> Atoms:
    """Main function"""
    resid = resid.upper()
    print("*" * 5, f"Generating residue from residue name ({resid})", "*" * 5)
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            generate_residue(resid)
            atoms = generate_coordinates(skip_energy_show=skip_energy_show)
            break
        except RuntimeError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            print(
                f"Residue coordinate generation failed on attempt {attempt}/{max_attempts}: {exc}. "
                "Retrying from a fresh PyCHARMM residue setup."
            )
    else:
        raise RuntimeError(f"Failed to generate residue {resid}") from last_error
    write_psf(resid)

    # copy pdb/initial.pdb to pdb/resid.pdb
    shutil.copy("pdb/initial.pdb", f"pdb/{resid.lower()}.pdb")

    # create an xyz file
    ase.io.write("xyz/initial.xyz", atoms)
    print(f"xyz/{resid.lower()}.xyz")
    shutil.copy("xyz/initial.xyz", f"xyz/{resid.lower()}.xyz")

    print("Done")
    return atoms


def cli():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resid", type=str, required=True)
    args = parser.parse_args()
    atoms = main(args.resid)
    print(atoms)


if __name__ == "__main__":
    cli()
