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

from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_RTF, CGENFF_PRM, CHARMM_HOME, CHARMM_LIB_DIR
)
from mmml.interfaces.pycharmmInterface.pycharmmCommands import (
    pbcset, pbcs
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
from mmml.interfaces.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM


# unit registry
try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False
    ureg = None
    Q_ = None
    import warnings
    warnings.warn(
        "pint is not installed. Some functions requiring unit conversions "
        "(e.g., determine_n_molecules_from_density) will not work. "
        "Install pint with: pip install pint or conda install -c conda-forge pint",
        ImportWarning
    )


cwd = Path(__file__).parent

PACKMOL_PATH = Path("~/mmml/mmml/generate/packmol/packmol").expanduser()
water_pdb_path = cwd / ".." / ".." / "data" / "charmm" / "tip3.pdb"
octanol_pdb_path = cwd / ".." / ".." / "data" / "charmm" / "ocoh.pdb"
ase_water = ase.io.read(water_pdb_path)
ase_octanol = ase.io.read(octanol_pdb_path)

def correct_names(atoms: Atoms) -> Atoms:
    """
    Corrects the names of the atoms in the atoms object
    """
    problem_symbols = ["CL", "HO"]
    e = atoms.get_chemical_symbols()
    e = [_[:1] if _.upper() in problem_symbols else _ for _ in e]
    e = [_ if _[0] != "H" else "H" for _ in e]
    an = [ase.data.chemical_symbols.index(_) for _ in e]
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


def solute_radius_from_mol(mol: Atoms, buffer: float = 1.0) -> float:
    """
    Compute the radius of a sphere that encompasses the solute.

    Uses center of mass and max distance to any atom, plus a buffer for packing.
    """
    com = mol.get_center_of_mass()
    radii = np.linalg.norm(mol.positions - com, axis=1)
    return float(np.max(radii)) + buffer


def volume_per_solvent_molecule_ang3(solvent: str) -> float:
    """
    Approximate volume per solvent molecule in Å³ from density and molecular weight.
    """
    if solvent not in solvents_ase:
        raise ValueError(f"Solvent {solvent} not found in {solvents_ase.keys()}")
    atoms = solvents_ase[solvent]
    density_kg_m3 = solvents_density[solvent]
    density_g_cm3 = density_kg_m3 / 1000.0
    mw = atoms.get_masses().sum()
    # cm³/mol -> Å³/molecule: (MW/density) / N_A * 1e24
    molar_vol_cm3 = mw / density_g_cm3
    vol_ang3 = molar_vol_cm3 / 6.022e23 * 1e24
    return vol_ang3


def outer_radius_from_n_solvent(
    n_molecules: int,
    inner_radius: float,
    solvent: str,
    buffer: float = 1.0,
) -> float:
    """
    Outer radius of solvent shell to fit n_molecules around a solute.

    Solvent occupies a spherical shell between inner_radius and outer_radius.
    Volume of shell = (4/3)*pi*(R_outer³ - R_inner³) = n * V_solvent
    """
    vol_per_mol = volume_per_solvent_molecule_ang3(solvent)
    shell_volume = n_molecules * vol_per_mol
    # (4/3)*pi*R_outer³ = shell_volume + (4/3)*pi*R_inner³
    inner_vol = (4.0 / 3.0) * np.pi * (inner_radius**3)
    outer_vol = shell_volume + inner_vol
    outer_radius = (3.0 * outer_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return outer_radius + buffer


def setup_box(mol: Atoms) -> None:
    """Sets up the box"""
    box_size = determine_box_size_from_mol(mol)
    print(f"Box size: {box_size}")


def determine_n_molecules_from_density(
    density: float, mol: Atoms, side_length: float = 35,
    solvent: str = None
) -> float:
    """
    Determine number of molecules from density.
    
    Requires pint to be installed for unit conversions.
    Install with: pip install pint or conda install -c conda-forge pint
    """
    if not _PINT_AVAILABLE:
        raise ImportError(
            "pint is required for determine_n_molecules_from_density. "
            "Please install pint with: pip install pint or conda install -c conda-forge pint"
        )
    
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


def run_packmol_solvation(
    n_molecules: int,
    side_length: float,
    solvent: str,
    solute_mol: Atoms | None = None,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    solute_buffer: float = 1.0,
    solvent_buffer: float = 1.0,
) -> None:
    """
    Pack 1 solute molecule surrounded by n_molecules of solvent in a spherical shell.

    Radii are computed from solute geometry and solvent density unless overridden.
    """
    if solvent not in solvents_ase:
        raise ValueError(f"Solvent {solvent} not found in {solvents_ase.keys()}")

    solvent_pdb = solvents_ase[solvent]
    solvent_pdb_path = f"pdb/{solvent}.pdb"
    solvent_pdb.write(solvent_pdb_path)

    center = side_length / 2
    cx, cy, cz = center, center, center

    if inner_radius is None:
        if solute_mol is None:
            solute_mol = read_initial_pdb(Path.cwd())
        inner_radius = solute_radius_from_mol(solute_mol, buffer=solute_buffer)
    if outer_radius is None:
        outer_radius = outer_radius_from_n_solvent(
            n_molecules, inner_radius, solvent, buffer=solvent_buffer
        )

    max_radius = center - 0.5
    if outer_radius > max_radius:
        print(
            f"Warning: outer_radius {outer_radius:.2f} Å exceeds box; capping to {max_radius:.2f} Å. "
            "Consider increasing side_length or reducing n_molecules."
        )
        outer_radius = max_radius

    print(f"Solvation radii: inner={inner_radius:.2f} Å, outer={outer_radius:.2f} Å")

    packmol_input = f"""

    output pdb/init-{solvent}box.pdb
    filetype pdb
    tolerance 2.0
    structure pdb/initial.pdb 
    number 1
    resnumbers 2
    chain A
    inside sphere {cx} {cy} {cz} {inner_radius}
    end structure
    structure pdb/{solvent}.pdb 
    number {n_molecules}
    resnumbers 2
    chain A
    outside sphere {cx} {cy} {cz} {inner_radius}
    inside sphere {cx} {cy} {cz} {outer_radius}
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
            [str(PACKMOL_PATH), " < ", f"packmol/packmol-{solvent}.inp"]
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
            [str(PACKMOL_PATH), " < ", "packmol/packmol.inp"]
        )
    )
    print(output)
    print("Generated initial.pdb")


def _ensure_crystal_image_str() -> None:
    """Copy crystal_image.str to cwd if missing (required by CHARMM for periodic images)."""
    dst = Path("crystal_image.str")
    if dst.exists():
        return
    src = Path(__file__).resolve().parents[2] / "data" / "charmm" / "crystal_image.str"
    if src.exists():
        shutil.copy2(src, dst)
    else:
        raise FileNotFoundError(
            f"crystal_image.str not found in cwd and source {src} does not exist. "
            "CHARMM requires this file for periodic box setup."
        )


def setup_box_generic(pdb_path, rtf=CGENFF_RTF, prm=CGENFF_PRM, side_length: float = 30, tag="", skip_energy_show: bool = False):
    """
    Sets up the box

    Args:
        skip_energy_show: If True, skip energy.show() to avoid slow CHARMM energy evaluation
            (Drude setup). Use for faster startup when validation is not needed.
    """
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet, safe_energy_show

    _ensure_crystal_image_str()
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
    pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=side_length))
    # Set nbonds with fswitch before IMAGE (in pbcs) to avoid bus error on macOS
    pycharmm.lingo.charmm_script(
        "nbonds atom cutnb 14.0 ctofnb 12.0 ctonnb 10.0 fswitch vswitch NBXMOD 5 inbfrq -1 imgfrq -1"
    )
    pycharmm.lingo.charmm_script(pbcs)
    if not skip_energy_show:
        safe_energy_show()
    write.psf_card(f"psf/system-{tag}.psf")
    write.coor_pdb(f"pdb/init-{tag}.pdb")
    print(f"wrote pdb/init-{tag}.pdb")


    pycharmm_quiet()
    atoms = ase.io.read(f"pdb/init-{tag}.pdb")
    atoms.set_cell(np.eye(3) * side_length)
    atoms.set_pbc(True)
    return atoms



def initialize_psf(resid: str, n_molecules: int, side_length: float, solvent: str = None, pdb_path: str = None):
    """
    Initializes the PSF file
    """
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
    CLEAR_CHARMM()
    if pdb_path is None:
        pdbfilename = f"pdb/init-packmol.pdb"
    else:
        pdbfilename = pdb_path

    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script("bomlev 0")

    pycharmm_quiet()
    if solvent is not None:
        resstr = " ".join([solvent.upper()]*(n_molecules-1))    
        resstr = f"{resid.upper()} {solvent.upper()}"
        pdb_path = pdbfilename
    else:
        resstr = " ".join([resid.upper()]*n_molecules)
        pdb_path = pdbfilename

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
    # pycharmm.lingo.charmm_script(pbcs)
    # print("read pbcs")
    # energy.show()
    # print("read energy")
    # pycharmm.lingo.charmm_script(write_system_psf)
    
    write.psf_card(f"psf/init.box.psf")
    write.psf_card(f"psf/init.box.psf")
    write.coor_pdb(f"pdb/init.box.pdb")
    print("wrote pdb/init.box.pdb")


def minimize_box():
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
        run_packmol_solvation(n_molecules, side_length, solvent, solute_mol=mol)
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
