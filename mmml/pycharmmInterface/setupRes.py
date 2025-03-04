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




def iupac_2_number(iupac):
    from mendeleev import element
    allowed = ['H','C','N','O','F','S','CL']
    number = []
    for i in iupac:
         if i[0:1] == 'CL': number.append(element(i[0:1]).atomic_number)
         elif i[0] in allowed: number.append(element(i[0]).atomic_number)
         else:
             print('Element not supported by ANI2X models: atom {}'.format(i))
             exit()
    return number


def get_residue_atoms() -> dict[str, list[list[float]]]:

    with open("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf") as f:
        lines = (f.readlines())

    """Reads the RTF file and returns a dictionary of residue atoms and the atom names of their internal coordinates"""
    resatoms = [_.strip("\n").split() for _ in lines if _.startswith("RESI") or _.startswith("IC")]
    resatoms = [_.strip("\n").split() for _ in lines if _.startswith("RESI") or _.startswith("IC")]
    x = {}
    for _ in resatoms:
        if _[0]=="RESI":
            x[_[1]] = []
            k = _[1]
        elif _[0] == "IC":
            x[k].append(_[1:4])
    resi2atoms = x



def generate_residue(resid) -> None:
    """Generates a residue from the RTF file"""
    s="""DELETE ATOM SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    read.rtf('/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_cgenff.rtf')
    bl =settings.set_bomb_level(-2)
    wl =settings.set_warn_level(-2)
    read.prm('/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36_cgenff.prm')
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    pycharmm.lingo.charmm_script('bomlev 0')
    read.sequence_string(resid)  
    gen.new_segment(seg_name=resid,
                    setup_ic=True)
    ic.prm_fill(replace_all=True)


def generate_coordinates() -> Atoms:

    # make pdb directory
    os.makedirs("pdb", exist_ok=True)
    os.makedirs("res", exist_ok=True)
    os.makedirs("dcd", exist_ok=True)

    ic.build()
    coor.show()

    xyz = coor.get_positions()
    print("positions:")
    print(xyz)

    start_energy = pycharmm.lingo.get_energy_value('ENER')
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

    end_energy = pycharmm.lingo.get_energy_value('ENER')
    print("end_energy:")
    print(end_energy)
    energy_diff = end_energy - start_energy
    print(f"Energy difference: {energy_diff}")
    if energy_diff > 0:
        print("WARNING: Energy difference is positive, something may have gone wrong")

    write.coor_pdb('pdb/initial.pdb')
    mol = ase.io.read("pdb/initial.pdb")
    e = mol.get_chemical_symbols()
    mol.set_chemical_symbols([_[:1] if _.upper() in ["HO", "CA", "CM"] else _ for _ in e])
    return mol


def mini():
    # Specify nonbonded python object called my_nbonds - this just sets it up
    # equivalant CHARMM scripting command: nbonds cutnb 18 ctonnb 13 ctofnb 17 cdie eps 1 atom vatom fswitch vfswitch
    my_nbonds = pycharmm.NonBondedScript(
        cutnb=18.0, ctonnb=13.0, ctofnb=17.0,
        eps=1.0,
        cdie=True,
        atom=True, vatom=True,
        fswitch=True, vfswitch=True)

    # Implement these non-bonded parameters by "running" them.
    my_nbonds.run()

    # equivalent CHARMM scripting command: minimize abnr nstep 1000 tole 1e-3 tolgr 1e-3
    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
    # equivalent CHARMM scripting command: energy
    energy.show()



def main(resid: str) -> None:
    """Main function"""
    resid = resid.upper()
    print("Generating residue:", resid)
    generate_residue(resid)
    print("Generating coordinates")
    generate_coordinates()
    print("Minimizing")
    mini()
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
