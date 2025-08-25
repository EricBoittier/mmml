from mmml.pycharmmInterface.import_pycharmm import *
import os
import ase
import numpy as np

def get_Z_from_psf():
    masses = psf.get_amass()
    Z = []
    for m in masses:
        mdif = (ase.data.atomic_masses_common - m)**2
        Z .append( np.argmin(mdif) )
    return Z

def set_up_directories() -> None:
    os.makedirs("pdb", exist_ok=True)
    os.makedirs("res", exist_ok=True)
    os.makedirs("dcd", exist_ok=True)
    os.makedirs("psf", exist_ok=True)
    os.makedirs("xyz", exist_ok=True)