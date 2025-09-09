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

from mmml.pycharmmInterface.import_pycharmm import (
    CGENFF_RTF, CGENFF_PRM, CHARMM_HOME, CHARMM_LIB_DIR
)
os.environ["CHARMM_HOME"] = CHARMM_HOME
os.environ["CHARMM_LIB_DIR"] = CHARMM_LIB_DIR

print(CHARMM_HOME)
print(CHARMM_LIB_DIR)
import sys
sys.path.append(str(Path(CHARMM_HOME) / "tool" / "pycharmm"))

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

# import simple scripts
from mmml.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM


# unit registry
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

PACKMOL_PATH = "packmol"
cwd = Path(__file__).parent
water_pdb_path = cwd / ".." / "data" / "tip3.pdb"
octanol_pdb_path = cwd / ".." / "data" / "ocoh.pdb"
ase_water = ase.io.read(water_pdb_path)
ase_octanol = ase.io.read(octanol_pdb_path)

def correct_names(atoms: Atoms) -> Atoms:
    """
    Corrects the names of the atoms in the atoms object
    """
    problem_symbols = ["CL", "HO"]
    e = atoms.get_chemical_symbols()
    e = [_[:1] if _.upper() in problem_symbols else _ for _ in e]
    print(e)
    e = [_ if _[0] != "H" else "H" for _ in e]
    print(e)
    # atomic numbers
    an = [ase.data.chemical_symbols.index(_) for _ in e]
    print(an)
    atoms.set_atomic_numbers(an)
    return atoms

water = correct_names(ase_water)    
octanol = correct_names(ase_octanol)

solvents_ase = {
    "water": water,
    "octanol": octanol,
}
solvents_density = {
    "water": 1000,
    "octanol": 824,
}

def read_initial_pdb(cwd: Path) -> Atoms:
    """Reads the initial PDB file and returns an ASE Atoms object"""
    mol = ase.io.read(cwd / "pdb" / "initial.pdb")
    e = mol.get_chemical_symbols()
    print(mol)
    print(e)
    mol.set_chemical_symbols(
        [
            (
                _[:1]
                if _.upper()
                not in [
                    "CL",
                ]
                else _
            )
            for _ in e
        ]
    )
    return mol


def determine_box_size_from_mol(mol: Atoms) -> float:
    """Determines the box size based on the maximum distance between any two atoms"""
    dists = np.linalg.norm(
        mol.positions[:, None, :] - mol.positions[None, :, :], axis=-1
    )
    return np.max(dists)


def setup_box(mol: Atoms) -> None:
    """Sets up the box"""
    box_size = determine_box_size_from_mol(mol)
    print(f"Box size: {box_size}")


def determine_n_molecules_from_density(
    density: float, mol: Atoms, side_length: float = 35,
    solvent: str = None
) -> float:
    if solvent is not None:
        atoms = solvents_ase[solvent]
        density = solvents_density[solvent]
    else:
        atoms = mol
    masses = atoms.get_masses()

    molecular_weight = masses.sum()
    molecular_formula = atoms.get_chemical_formula(mode="reduce")

    # note use of two lines to keep length of line reasonable
    s = f"The molecular weight of {molecular_formula} is {molecular_weight:1.2f} gm/mol."
    print(s)

    box_size = side_length * ureg.angstrom
    volume = box_size**3  # Volume of the box in cm^3

    print("Volume of the box: ", volume)

    density = density * (ureg.kilogram / ureg.meter**3)
    molecular_weight = molecular_weight * (ureg.gram / ureg.mole)  # g/mol

    # Calculate mass of the substance in the box
    mass = density * volume  # mass = density * volume
    print(mass.to("g"))
    # Calculate moles in the box
    moles = mass.to("g") / molecular_weight.to("g/mol")
    print(moles)
    # Define Avogadro's number (molecules per mole)
    avogadro_number = 6.022e23 * ureg.molecule / ureg.mole

    # Calculate number of molecules
    num_molecules = moles * avogadro_number
    n_molecules = int(num_molecules.magnitude)
    # Display the result
    print(f"Number of molecules in the box: {n_molecules}")
    return n_molecules


def run_packmol_solvation(n_molecules: int, side_length: float, solvent: str) -> None:
    
    if solvent not in solvents_ase:
        raise ValueError(f"Solvent {solvent} not found in {solvents_ase.keys()}")

    solvent_pdb = solvents_ase[solvent]
    solvent_pdb_path = f"pdb/{solvent}.pdb"
    solvent_pdb.write(solvent_pdb_path)

    packmol_input = f"""

    output pdb/init-{solvent}box.pdb
    filetype pdb
    tolerance 2.0
    structure pdb/initial.pdb 
    number 1
    resnumbers 2
    chain A
    inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}
    end structure
    structure pdb/{solvent}.pdb 
    number {n_molecules}
    resnumbers 2
    chain A
    inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}
    end structure


    """
    import os
    os.makedirs("packmol", exist_ok=True)
    randint = np.random.randint(1000000)
    packmol_script = packmol_input.split("\n")
    packmol_script[1] = f"seed {randint}"
    packmol_script = "\n".join(packmol_script)
    with open(f"packmol/packmol-{solvent}.inp", "w") as f:
        f.writelines(packmol_script)

    import subprocess
    import os

    print(f"{PACKMOL_PATH} < packmol/packmol-{solvent}.inp")
    output = os.system(
        " ".join(
            [PACKMOL_PATH, " < ", f"packmol/packmol-{solvent}.inp"]
        )
    )
    print(output)
    print(f"Generated init-{solvent}box.pdb")


def run_packmol(n_molecules: int, side_length: float) -> None:
    packmol_input = f"""

    output pdb/init-packmol.pdb
    filetype pdb
    tolerance 2.0
    structure pdb/initial.pdb 
    chain A
    number {n_molecules}
    inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}
    end structure
    """
    import os
    os.makedirs("packmol", exist_ok=True)
    randint = np.random.randint(1000000)
    packmol_script = packmol_input.split("\n")
    packmol_script[1] = f"seed {randint}"
    packmol_script = "\n".join(packmol_script)
    with open("packmol/packmol.inp", "w") as f:
        f.writelines(packmol_script)

    import subprocess
    import os

    print(f"{PACKMOL_PATH} < packmol/packmol.inp")
    output = os.system(
        " ".join(
            [PACKMOL_PATH, " < ", "packmol/packmol.inp"]
        )
    )
    print(output)
    print("Generated initial.pdb")


def setup_box_generic(pdb_path, rtf=CGENFF_RTF, prm=CGENFF_PRM, side_length: float = 30, tag=""):
    """
    Sets up the box
    """
    CLEAR_CHARMM()
    read.rtf(rtf)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(prm)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")
    header = f"""bomlev -2
    prnlev 4
    wrnlev 4
    OPEN UNIT 1 READ FORM NAME {pdb_path}
    READ SEQU PDB UNIT 1
    CLOSE UNIT 1
    GENERATE SYS FIRST NONE LAST NONE SETUP 

    OPEN UNIT 1 READ FORM NAME {pdb_path}
    READ COOR PDB UNIT 1
    CLOSE UNIT 1
    
    """
    pycharmm.lingo.charmm_script(header)
    print("read header")
    pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=side_length))
    print("read pbcset")
    pycharmm.lingo.charmm_script(pbcs)
    print("read pbcs")
    energy.show()
    write.psf_card(f"psf/system-{tag}.psf")
    write.coor_pdb(f"pdb/init-{tag}.pdb")
    print(f"wrote pdb/init-{tag}.pdb")



def initialize_psf(resid: str, n_molecules: int, side_length: float, solvent: str):
    """
    Initializes the PSF file
    """
    CLEAR_CHARMM()

    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")

    if solvent is not None:
        resstr = " ".join([solvent.upper()]*(n_molecules-1))    
        resstr = f"{resid.upper()} {solvent.upper()}"
        pdb_path = f"pdb/init-{solvent}box.pdb"
    else:
        resstr = " ".join([resid.upper()]*n_molecules)
        pdb_path = f"pdb/init-packmol.pdb"

    header = f"""bomlev -2
    prnlev 4
    wrnlev 4
    OPEN UNIT 1 READ FORM NAME {pdb_path}
    READ SEQU PDB UNIT 1
    CLOSE UNIT 1
    GENERATE SYS FIRST NONE LAST NONE SETUP 

    OPEN UNIT 1 READ FORM NAME {pdb_path}
    READ COOR PDB UNIT 1
    CLOSE UNIT 1
    
    """
    pycharmm.lingo.charmm_script(header)
    print("read header")
    pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=side_length))
    print("read pbcset")
    pycharmm.lingo.charmm_script(pbcs)
    print("read pbcs")
    energy.show()
    print("read energy")
    # pycharmm.lingo.charmm_script(write_system_psf)
    if solvent is not None:
        write.psf_card(f"psf/{resid}-{solvent}-{n_molecules}.psf")
        write.psf_card(f"psf/system-{solvent}.psf")
        write.coor_pdb(f"pdb/init-{solvent}box.pdb")
        print("wrote pdb/init-{solvent}box.pdb")
    else:
        write.psf_card(f"psf/system-packmol.psf")
        write.psf_card(f"psf/system.psf")
        write.coor_pdb(f"pdb/init-packmol.pdb")
        print("wrote pdb/init-packmol.pdb")

def minimize_box():
    nbonds = """!#########################################
    ! Bonded/Non-bonded Options & Constraints
    !#########################################

    ! Non-bonding parameters
    nbonds atom ewald pmewald kappa 0.43  -
    fftx 32 ffty 32 fftz 32 order 4 -
    cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
    lrc vdw vswitch -
    inbfrq -1 imgfrq -1

    """
    pycharmm.lingo.charmm_script(nbonds)

    # equivalent CHARMM scripting command: minimize abnr nstep 1000 tole 1e-3 tolgr 1e-3
    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
    # equivalent CHARMM scripting command: energy
    energy.show()


def main(density: float, side_length: float, residue: str, solvent: str):
    cwd = Path(os.getcwd())
    mol = read_initial_pdb(cwd)
    print(mol)
    print(mol.get_chemical_symbols())
    print(solvent)
    if solvent is None:
        n_molecules = determine_n_molecules_from_density(density, mol, side_length, solvent=None)
        run_packmol(n_molecules, side_length)
    else:
        n_molecules = determine_n_molecules_from_density(density, mol, side_length, solvent)
        run_packmol_solvation(n_molecules, side_length, solvent)
    initialize_psf(residue, n_molecules, side_length, solvent)
    # minimize_box()


def cli():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--density", type=float, required=True, 
        help="Density of the box in kg/m^3"   )
    parser.add_argument("-l", "--side_length", type=float, required=True, 
        help="Side length of the box in angstrom")
    parser.add_argument("-r", "--residue", type=str, required=True, 
        help="Residue name")
    parser.add_argument("-s", "--solvent", type=str, required=False, default=None,
        help="Solvent name")
    args = parser.parse_args()
    if args.solvent == "None":
        args.solvent = None
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)} {type(getattr(args, arg))}")
    main(args.density, args.side_length, args.residue, args.solvent)


if __name__ == "__main__":
    cli()
