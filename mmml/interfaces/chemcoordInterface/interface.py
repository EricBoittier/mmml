import chemcoord as cc
import time
import ase
from io import StringIO
import pandas as pd


def sym_to_ase(eq):
    test = ase.io.extxyz.read_xyz(StringIO(eq["sym_mol"].to_xyz()))
    test = next(test)
    return test

def to_chemcord(Z, R):
    x,y,z = R.T
    df = pd.DataFrame({"atom": Z, "x": x.flatten(), "y":y.flatten(), "z": z.flatten()})
    cart = cc.Cartesian(df)
    zmat = cart.get_zmat()
    return cart, zmat


def ase_to_chemcord(atoms):
    return to_chemcord(atoms.get_chemical_symbols(), atoms.get_positions())
