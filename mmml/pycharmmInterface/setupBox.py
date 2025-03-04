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

# Environment setup (before loading pycharmm)
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


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


# Constants
packmol_input = str(Path("packmol.inp").absolute())

def read_initial_pdb(path: Path) -> Atoms:
    """Reads the initial PDB file and returns an ASE Atoms object"""
    write.coor_pdb('pdb/initial.pdb')
    mol = ase.io.read("pdb/initial.pdb")
    e = mol.get_chemical_symbols()
    print(mol)
    print(e)
    mol.set_chemical_symbols([_[:1] if _.upper() not in ["CL",] else _ for _ in e])
    return mol

def determine_box_size_from_mol(mol: Atoms) -> float:
    """Determines the box size based on the maximum distance between any two atoms"""
    dists = np.linalg.norm(mol.positions[:, None, :] - mol.positions[None, :, :], axis=-1)
    return np.max(dists)

def setup_box(mol: Atoms) -> None:
    """Sets up the box"""
    box_size = determine_box_size_from_mol(mol)
    print(f"Box size: {box_size}")

def determine_n_molecules_from_density(density: float, mol: Atoms, side_length: float = 35) -> float:
    atoms = mol
    masses = atoms.get_masses()

    molecular_weight = masses.sum()
    molecular_formula = atoms.get_chemical_formula(mode='reduce')

    # note use of two lines to keep length of line reasonable
    s = f'The molecular weight of {molecular_formula} is {molecular_weight:1.2f} gm/mol.'
    print(s)

    box_size = side_length * ureg.angstrom
    volume = box_size**3  # Volume of the box in cm^3

    print("Volume of the box: ", volume)

    # Assume density and molecular weight (you can replace these with actual values)
    # Example: Water (H2O)
    density = 0.8 * ureg.gram / ureg.centimeter**3  # g/cm^3
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
    num_molecules =  moles * avogadro_number
    n_molecules = int(num_molecules.magnitude)
    # Display the result
    print(f"Number of molecules in the box: {n_molecules}")
    return n_molecules

def run_packmol(n_molecules: int, side_length: float) -> None:
    packmol_input = f"""

    output init.pdb
    filetype pdb
    tolerance 2.0
    structure pdb/initial.pdb 
    number {n_molecules}
    inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}
    end structure
    """
    randint = np.random.randint(1000000)
    packmol_script = packmol_input.split("\n")
    packmol_script[1] = f"seed {randint}"
    packmol_script = "\n".join(packmol_script)
    with open("packmol.inp", "w") as f:
        f.writelines(packmol_script)

    import subprocess
    import os
    output = os.system(" ".join(["/pchem-data/meuwly/boittier/home/packmol/packmol", " < ", "packmol.inp"])) 

def initialize_psf():
    s="""DELETE ATOM SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    s="""DELETE PSF SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    header = """bomlev -2
    prnlev 3
    wrnlev 1

    !#########################################
    ! Tasks
    !#########################################

    ! 0:    Do it, or
    ! Else: Do not, there is no try!
    set mini 0
    set heat 0
    set equi 0
    set ndcd 1
    ! Start Production at dcd number n
    set ndcd 0

    OPEN UNIT 1 READ FORM NAME pdb/initial.pdb
    READ SEQU PDB UNIT 1
    CLOSE UNIT 1
    GENERATE DCM FIRST NONE LAST NONE SETUP 

    OPEN UNIT 1 READ FORM NAME pdb/initial.pdb
    READ COOR PDB UNIT 1
    CLOSE UNIT 1"""
    pycharmm.lingo.charmm_script(header)
    pycharmm.lingo.charmm_script(pbcset)
    pycharmm.lingo.charmm_script(pbcs)
    energy.show()
    write.psf_card(f'{RESID}-{n_molecules}.psf')

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

def main(density: float, side_length: float):
    mol = read_initial_pdb(Path("initial.pdb"))
    n_molecules = determine_n_molecules_from_density(density, mol)
    run_packmol(n_molecules, side_length)
    initialize_psf()
    minimize_box()


def cli():
    """Command line interface"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--density", type=float, required=True)
    parser.add_argument("-l", "--side_length", type=float, required=True)
    args = parser.parse_args()
    main(args.density, args.side_length)


if __name__ == "__main__":
    cli()

