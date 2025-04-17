# header = """bomlev -2
# prnlev 3
# wrnlev 1

# !#########################################
# ! Tasks
# !#########################################

# ! 0:    Do it, or
# ! Else: Do not, there is no try!
# set mini 0
# set heat 0
# set equi 0
# set ndcd 1
# ! Start Production at dcd number n
# set ndcd 0

# !#########################################
# ! Setup System
# !#########################################

# open unit 1 card read name lig.top
# read rtf card unit 1
# close unit 1

# open unit 1 form read name lig.par
# read param card unit 1
# close unit 1

# ! File name
# set name dclm

# OPEN UNIT 1 READ FORM NAME init.pdb
# READ SEQU PDB UNIT 1
# CLOSE UNIT 1
# GENERATE SOL FIRST NONE LAST NONE SETUP 

# OPEN UNIT 1 READ FORM NAME init.pdb
# READ COOR PDB UNIT 1
# CLOSE UNIT 1

# ! Generate PSF and write to a file
# write psf card name system.psf
# * My PSF file
# *
# """

write_system_psf = """write psf card name system.psf
* My PSF file
*
"""


pbcs = """!#########################################
! Setup PBC (Periodic Boundary Condition)
!#########################################

coor stat sele all end

calc xdim = int ( ( ?xmax - ?xmin + 0.0 ) ) + 1
calc ydim = int ( ( ?ymax - ?ymin + 0.0 ) ) + 1
calc zdim = int ( ( ?zmax - ?zmin + 0.0 ) ) + 1

set bsiz = 0

if @xdim .gt. @bsiz then
   set bsiz = @xdim
endif
if @ydim .gt. @bsiz then
   set bsiz = @ydim
endif
if @zdim .gt. @bsiz then
   set bsiz = @zdim
endif

open read unit 10 card name crystal_image.str
crystal defi cubic @bsiz @bsiz @bsiz 90. 90. 90.
crystal build nope 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"""

nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
nbonds atom ewald pmewald kappa 0.43  -
  fftx 32 ffty 32 fftz 32 order 4 -
  cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
  lrc vdw vswitch -
  inbfrq -1 imgfrq -1

! Constrain all X-H bonds
!shake bonh para sele all end
"""

cons = """!#########################################
! Setup PBC (Periodic Boundary Condition)
!#########################################

coor stat sele all end

calc xdim = int ( ( ?xmax - ?xmin + 0.0 ) ) + 1
calc ydim = int ( ( ?ymax - ?ymin + 0.0 ) ) + 1
calc zdim = int ( ( ?zmax - ?zmin + 0.0 ) ) + 1

set bsiz = 0

if @xdim .gt. @bsiz then
   set bsiz = @xdim
endif
if @ydim .gt. @bsiz then
   set bsiz = @ydim
endif
if @zdim .gt. @bsiz then
   set bsiz = @zdim
endif

open read unit 10 card name crystal_image.str
crystal defi cubic @bsiz @bsiz @bsiz 90. 90. 90.
crystal build nope 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"""

mini = """!#########################################
! Minimization {iseed} {NDCD}
!#########################################

mini sd nstep 1000 nprint 100

open write unit 10 card name mini.pdb
write coor unit 10 pdb

"""

pbcset = """ SET BOXTYPE  = RECT
 SET XTLTYPE  = CUBIC
 SET A = {SIDELENGTH}
 SET B = {SIDELENGTH}
 SET C = {SIDELENGTH}
 SET ALPHA = 90.0
 SET BETA  = 90.0
 SET GAMMA = 90.0
 SET IMPATCH = NO
 SET FFTX  = 40
 SET FFTY  = 40
 SET FFTZ  = 40
 SET XCEN  = 0
 SET YCEN  = 0
 SET ZCEN  = 0
"""

heat = """!#########################################
! Heating - NVT {NDCD}
!#########################################

scalar mass stat
calc pmass = int ( ?stot  /  50.0 )
calc tmass = @pmass * 10

calc tmin = 300 * 0.2 

open write unit 31 card name heat.res       ! Restart file
open write unit 32 file name heat.dcd       ! Coordinates file

dyna leap verlet start -
   timestp 0.0002 nstep 50000 -
   firstt @tmin finalt 300 tbath 300 -
   ihtfrq 1000 teminc 5 ieqfrq 0 -
   iasors 1 iasvel 1 iscvel 0 ichecw 0 -
   nprint 1000 nsavc 1000 ntrfrq 200 -
   iseed  {iseed} -
   echeck 100.0   -
   iunrea -1 iunwri 31 iuncrd 32 iunvel -1

open unit 1 write card name heat.crd
write coor card unit 1
close unit 1

open write unit 10 card name heat.pdb
write coor unit 10 pdb

"""

equi = """!#########################################
! Equilibration - NpT {NDCD}
!#########################################

open read  unit 30 card name heat.res      ! Restart file
open write unit 31 card name equi.res      ! Restart file
open write unit 32 file name equi.dcd      ! Coordinates file

dyna restart leap cpt nstep 100000 timestp 0.0002 -
  nprint 1000 nsavc 1000 ntrfrq 200 -
  iprfrq 500 inbfrq 10 imgfrq 50 ixtfrq 1000 -
  ihtfrq 0 ieqfrq 0 -
  pint pconst pref 1 pgamma 5 pmass @pmass -
   iseed  {iseed} -
  hoover reft 300 tmass @tmass firstt 300 -
  iunrea 30 iunwri 31 iuncrd 32 iunvel -1


open unit 1 write card name equi.crd
write coor card unit 1
close unit 1

open write unit 10 card name equi.pdb
write coor unit 10 pdb

close unit 30
close unit 31
close unit 32
"""

dyna = """!#########################################
! Production - NpT
!#########################################

set ndcd {NDCD}

if @ndcd .eq. 0 then
  set m @ndcd
  open read unit 33 card name equi.res        ! Restart file
  open write unit 34 card name dyna.@ndcd.res ! Restart file
  open write unit 35 file name dyna.@ndcd.dcd ! Coordinates file
else
  calc m @ndcd-1
  open read unit 33 card name dyna.@m.res
  open write unit 34 card name dyna.@ndcd.res
  open write unit 35 file name dyna.@ndcd.dcd
endif

dyna restart leap res nstep 10000 timestp 0.0002 -
  nprint 100 nsavc 10 ntrfrq 200 -
  iprfrq 1000 inbfrq -1 imgfrq 50 ixtfrq 1000 -
  ihtfrq 0 ieqfrq 0 -
  cpt pint pconst pref 1 pgamma 0 pmass @pmass -
  hoover reft 300 tmass @tmass -
   iseed  {iseed} -
  IUNREA 33 IUNWRI 34 IUNCRD 35 IUNVEL -1
  
open unit 1 write card name dyna.@ndcd.crd
write coor card unit 1
close unit 1

open write unit 10 card name dyna.@ndcd.pdb
write coor unit 10 pdb

close unit 33
close unit 34
close unit 35

"""


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
from import_pycharmm import CGENFF_RTF, CGENFF_PRM


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


from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

PACKMOL_PATH = "/pchem-data/meuwly/boittier/home/packmol/packmol"
# Constants
# packmol_input = str(Path("packmol.inp").absolute())


def read_initial_pdb(path: Path) -> Atoms:
    """Reads the initial PDB file and returns an ASE Atoms object"""
    write.coor_pdb("pdb/initial.pdb")
    mol = ase.io.read("pdb/initial.pdb")
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
    density: float, mol: Atoms, side_length: float = 35
) -> float:
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


def run_packmol(n_molecules: int, side_length: float) -> None:
    packmol_input = f"""

    output pdb/init-packmol.pdb
    filetype pdb
    tolerance 2.0
    structure pdb/initial.pdb 
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


def initialize_psf(resid: str, n_molecules: int, side_length: float):
    s = """DELETE ATOM SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    s = """DELETE PSF SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    resstr = " ".join([resid.upper()]*n_molecules)
    print(resstr)
    header = f"""bomlev -2
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

    OPEN UNIT 1 READ FORM NAME pdb/init-packmol.pdb
    READ SEQU PDB UNIT 1
    CLOSE UNIT 1
    GENERATE {resstr} FIRST NONE LAST NONE SETUP 

    OPEN UNIT 1 READ FORM NAME pdb/init-packmol.pdb
    READ COOR PDB UNIT 1
    CLOSE UNIT 1
    
    """
    pycharmm.lingo.charmm_script(header)
    pycharmm.lingo.charmm_script(pbcset.format(SIDELENGTH=side_length))
    pycharmm.lingo.charmm_script(pbcs)
    energy.show()
    pycharmm.lingo.charmm_script(write_system_psf)
    write.psf_card(f"{resid}-{n_molecules}.psf")


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


def main(density: float, side_length: float, residue: str):
    mol = read_initial_pdb(Path("initial.pdb"))
    n_molecules = determine_n_molecules_from_density(density, mol)
    run_packmol(n_molecules, side_length)
    initialize_psf(residue, n_molecules, side_length)
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
    args = parser.parse_args()
    main(args.density, args.side_length, args.residue)


if __name__ == "__main__":
    cli()
