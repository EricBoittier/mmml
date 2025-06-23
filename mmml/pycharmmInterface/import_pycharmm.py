import os
import subprocess
from pathlib import Path
import sys

# current directory of current file
cwd = Path(__file__).parent

chmh = None
chml = None
with open(cwd / ".." / ".." / "CHARMMSETUP") as f:
    lines = f.readlines()
    for line in lines:
        if "CHARMM_HOME" in line:
            chmh = line.split("=")[1].strip()
        if "CHARMM_LIB_DIR" in line:
            chml = line.split("=")[1].strip()
if chmh is None:
    raise ValueError("CHARMM_HOME is not set")
if chml is None:
    raise ValueError("CHARMM_LIB_DIR is not set")

os.environ["CHARMM_HOME"] = chmh
os.environ["CHARMM_LIB_DIR"] = chml


CHARMM_HOME = os.environ["CHARMM_HOME"]
CHARMM_LIB_DIR = os.environ["CHARMM_LIB_DIR"]

chmhp = Path(CHARMM_HOME) / "tool" / "pycharmm"
sys.path.append(str(chmhp))

CGENFF_RTF = cwd / ".." /  "data" / "top_all36_cgenff.rtf"
CGENFF_RTF = CGENFF_RTF.resolve()
print(CGENFF_RTF)
CGENFF_PRM = cwd / ".." /   "data" / "par_all36_cgenff.prm"
CGENFF_PRM = CGENFF_PRM.resolve()
print(CGENFF_PRM)

CGENFF_RTF = str(CGENFF_RTF)
CGENFF_PRM = str(CGENFF_PRM)


print("CHARMM_HOME", CHARMM_HOME)
print("CHARMM_LIB_DIR", CHARMM_LIB_DIR)


import pycharmm

import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.select as select
import pycharmm.shake as shake

from pycharmm.lib import charmm as libcharmm
import ase
from ase.io import read as read_ase
from ase import visualize
from ase.visualize import view

def get_block(a,b):
    block = f"""BLOCK
CALL 1 SELE .NOT. (RESID {a} .OR. RESID {b}) END
CALL 2 SELE (RESID {a} .OR. RESID {b}) END
COEFF 1 1 0.0
COEFF 2 2 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0
COEFF 1 2 0.0
END
"""
    return block

def reset_block():
    block = f"""BLOCK 
        CALL 1 SELE ALL END
          COEFF 1 1 1.0 
        END
        """
    _ = pycharmm.lingo.charmm_script(block)


def reset_block_no_internal():
    block = f"""BLOCK 
        CALL 1 SELE ALL END
          COEFF 1 1 1.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 
        END
        """
    _ = pycharmm.lingo.charmm_script(block)


def view_atoms(atoms):
    return view(atoms, viewer="x3d")

def get_forces_pycharmm():
    positions = coor.get_positions()
    force_command = """coor force sele all end"""
    _ = pycharmm.lingo.charmm_script(force_command)
    forces = coor.get_positions()
    coor.set_positions(positions)
    return forces

import pandas as pd
def set_pycharmm_xyz(atom_positions):
    xyz = pd.DataFrame(atom_positions, columns=["x", "y", "z"])
    coor.set_positions(xyz)


def capture_neighbour_list():
    # Print something
    distance_command = """
    open unit 1 write form name total.dmat
    
    COOR DMAT SINGLE UNIT 1 SELE ALL END SELE ALL END
    
    close unit 1"""
    _ = pycharmm.lingo.charmm_script(distance_command)

    with open("total.dmat") as f:
        output_dmat = f.read()

    atom_number_type_dict = {}
    atom_number_resid_dict = {}

    pair_distance_dict = {}
    pair_resid_dict = {}

    for _ in output_dmat.split("\n"):
        if _.startswith("*** "):
            _, n, resid, resname, at, _ = _.split()

            n = int(n.split("=")[0]) - 1
            atom_number_type_dict[n] = at
            atom_number_resid_dict[n] = int(resid) - 1

    for _ in output_dmat.split("\n"):
        if _.startswith("  "):
            a, b, dist = _.split()
            a = int(a) - 1
            b = int(b) - 1
            dist = float(dist)
            if atom_number_resid_dict[a] < atom_number_resid_dict[b]:
                pair_distance_dict[(a, b)] = dist
                pair_resid_dict[(a, b)] = (
                    atom_number_resid_dict[a],
                    atom_number_resid_dict[b],
                )

    return {
        "atom_number_type_dict": atom_number_type_dict,
        "atom_number_resid_dict": atom_number_resid_dict,
        "pair_distance_dict": pair_distance_dict,
        "pair_resid_dict": pair_resid_dict,
    }


reset_block()
from mmml.pycharmmInterface.utils import get_Z_from_psf 

def ase_from_pycharmm_state():
    Z = get_Z_from_psf()
    R = coor.get_positions()
    return ase.Atoms(Z, R)

def view_pycharmm_state():
    return view_atoms(ase_from_pycharmm_state())


def pycharmm_quiet():
    cmd = "PRNLev 0\nWRNLev 0"
    pycharmm.lingo.charmm_script(cmd)

def pycharmm_verbose():
    cmd = "PRNLev 5\nWRNLev 5"
    pycharmm.lingo.charmm_script(cmd)

def pycharmm_loud():
    cmd = "PRNLev 9\nWRNLev 9"
    pycharmm.lingo.charmm_script(cmd)

