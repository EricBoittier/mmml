# Standard library imports
import os
import sys
import shutil
import pickle
import itertools
from pathlib import Path
from io import BytesIO

# Third-party scientific computing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ASE imports
import ase
from ase import Atoms, io
from ase.data import covalent_radii
from ase.io.pov import get_bondpairs, set_high_bondorder_pairs
from ase.visualize.plot import plot_atoms
from ase.io import read
from ase.visualize import view

from import_pycharmm import *
from import_pycharmm import CGENFF_RTF

# CHARMM imports
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.select as select
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.param as param
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.shake as shake
import pycharmm.scalar as scalar
import pycharmm.lingo


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
    resi2atoms = x


def generate_residue(resid) -> None:
    """Generates a residue from the RTF file"""
    print("*" * 5, "Generating residue", "*" * 5)
    s = """DELETE ATOM SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")
    read.sequence_string(resid)
    gen.new_segment(seg_name=resid, setup_ic=True)
    ic.prm_fill(replace_all=True)


def generate_coordinates() -> Atoms:
    print("*" * 5, "Generating coordinates", "*" * 5)

    # make pdb directory
    os.makedirs("pdb", exist_ok=True)
    os.makedirs("res", exist_ok=True)
    os.makedirs("dcd", exist_ok=True)
    os.makedirs("psf", exist_ok=True)
    os.makedirs("xyz", exist_ok=True)
    ic.build()
    coor.show()

    xyz = coor.get_positions()
    print("positions:")
    print(xyz)

    start_energy = pycharmm.lingo.get_energy_value("ENER")
    print("start_energy:")
    print(start_energy)

    xyz *= 0
    xyz += 3 * np.random.random(xyz.to_numpy().shape)
    coor.set_positions(xyz)
    coor.get_positions()
    s = """
    ENERGY
    mini sd nstep 1000
    ENERGY
    """
    pycharmm.lingo.charmm_script("BOMLEV -1")
    pycharmm.lingo.charmm_script(s)

    print("positions:")
    print(xyz)

    xyz = coor.get_positions()
    xyz *= 3 * np.random.random(xyz.to_numpy().shape)
    coor.set_positions(xyz)
    pycharmm.lingo.charmm_script("BOMLEV -1")
    pycharmm.lingo.charmm_script(s)

    print("positions:")
    print(xyz)

    end_energy = pycharmm.lingo.get_energy_value("ENER")
    print("end_energy:")
    print(end_energy)
    energy_diff = end_energy - start_energy
    print(f"Energy difference: {energy_diff}")
    if energy_diff > 0:
        print("WARNING: Energy difference is positive, something may have gone wrong")

    # save pycharmm coordinates as pdb file
    write.coor_pdb("pdb/initial.pdb")

    # read pdb file
    mol = ase.io.read("pdb/initial.pdb")
    e = mol.get_chemical_symbols()
    e = [_[:1] if _.upper() in problem_symbols else _ for _ in e]
    print(e)
    e = [_ if _[0] != "H" else "H" for _ in e]
    print(e)
    # atomic numbers
    an = [ase.data.chemical_symbols.index(_) for _ in e]
    print(an)
    mol.set_atomic_numbers(an)

    atoms = ase.Atoms(
        symbols=e,
        positions=mol.get_positions(),
        cell=mol.get_cell(),
        pbc=mol.get_pbc(),
    )
    return atoms


def mini():
    print("*" * 5, "Minimizing", "*" * 5)
    # Specify nonbonded python object called my_nbonds - this just sets it up
    # equivalant CHARMM scripting command: nbonds cutnb 18 ctonnb 13 ctofnb 17 cdie eps 1 atom vatom fswitch vfswitch
    my_nbonds = pycharmm.NonBondedScript(
        cutnb=18.0,
        ctonnb=13.0,
        ctofnb=17.0,
        eps=1.0,
        cdie=True,
        atom=True,
        vatom=True,
        fswitch=True,
        vfswitch=True,
    )

    # Implement these non-bonded parameters by "running" them.
    my_nbonds.run()

    # equivalent CHARMM scripting command: minimize abnr nstep 1000 tole 1e-3 tolgr 1e-3
    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
    # equivalent CHARMM scripting command: energy
    energy.show()


def write_psf(resid: str) -> None:
    print("*" * 5, "Writing PSF", "*" * 5)
    print(f"psf/{resid.lower()}-1.psf")
    write.psf_card("psf/initial.psf")
    write.psf_card(f"psf/{resid.lower()}-1.psf")


def main(resid: str) -> None:
    """Main function"""
    resid = resid.upper()
    print("*" * 5, f"Generating residue from residue name ({resid})", "*" * 5)
    generate_residue(resid)
    atoms = generate_coordinates()
    mini()
    write_psf(resid)

    # copy pdb/initial.pdb to pdb/resid.pdb
    shutil.copy("pdb/initial.pdb", f"pdb/{resid.lower()}.pdb")

    # create an xyz file
    ase.io.write("xyz/initial.xyz", atoms)
    print(f"xyz/{resid.lower()}.xyz")
    shutil.copy("xyz/initial.xyz", f"xyz/{resid.lower()}.xyz")

    print("Done")


def cli():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resid", type=str, required=True)
    args = parser.parse_args()
    main(args.resid)


if __name__ == "__main__":
    cli()
