import ase 
from ase.io import read
import numpy as np

def npzToAse(data_dict):
    R = data_dict['R']
    Z = data_dict['Z']
    N = data_dict['N']
    atoms = ase.Atoms(Z[:N[0]], R[0][:N[0]])
    if data_dict.get('cell') is not None:
        atoms.set_cell(data_dict['cell'][0])

    return atoms