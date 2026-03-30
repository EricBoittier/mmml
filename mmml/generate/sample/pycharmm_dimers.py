#!/usr/bin/env python
# coding: utf-8


import mmml
from mmml.pycharmmInterface import import_pycharmm
from mmml.pycharmmInterface.import_pycharmm import reset_block, pycharmm_quiet, pycharmm_soft
pycharmm_quiet()
from mmml.pycharmmInterface import utils, setupRes
from mmml.pycharmmInterface.utils import view_pycharmm_state, get_Z_from_psf
import ase
import pandas as pd
import numpy as np

import pycharmm
from pycharmm import energy


# In[80]:

def make_dimer_pdb(resid):
    reset_block()
    setupRes.main(resid)
    pycharmm.read.sequence_string(resid)
    pycharmm.gen.new_segment(seg_name=resid, setup_ic=True)
    pycharmm.ic.prm_fill(replace_all=True)
    pycharmm.coor.show()

    positions = pycharmm.coor.get_positions()

    import numpy as np
    positions_np = positions.to_numpy()
    positions_np_copy = np.zeros_like(positions_np)
    positions_np_copy[:12] = positions_np[:12]
    positions_np_copy[12:] = positions_np[:12]

    positions_np_copy[12:,0] += 10
    positions_np_copy
    positions[['x', 'y', 'z']] = positions_np_copy
    pycharmm.coor.set_positions(positions)
    setupRes.mini(nbxmod=5)


    pycharmm_soft()


    setupRes.mini(nbxmod=1)


    pycharmm.energy.show()



    view_pycharmm_state()

    pycharmm.coor.show()

    pycharmm.write.coor_pdb(f"{resid.lower()}_dimer.pdb")

    atoms = ase.io.read(f"{resid.lower()}_dimer.pdb")

    R = atoms.get_positions()
    Z= get_Z_from_psf()
    atoms = ase.Atoms(Z, R)

    ase.io.write("benz_dimer.xyz", atoms)




from mmml.interfaces.chemcoordInterface import interface
from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3
patch_chemcoord_for_pandas3()
import chemcoord as cc


# In[411]:

def sample_dimer(xyz_file):
    cc_mol_xyz = cc.Cartesian.read_xyz(xyz_file)

    import sympy

    sympy.init_printing()
    ba = sympy.Symbol("ba")
    bb = sympy.Symbol("bb")
    aa = sympy.Symbol("aa")
    ab = sympy.Symbol("ab")
    da = sympy.Symbol("da")
    db = sympy.Symbol("db")

    ba_val = 5
    bb_val = 5
    aa_val = 90
    ab_val = -90
    da_val = 0
    db_val = 0

    zmat1 = fragments[0].to_zmat()
    zmat2 = zmat1.copy()

    zmat1.safe_loc[zmat1.index[0], "bond"] = ba
    zmat1.safe_loc[zmat1.index[0], "angle"] = aa
    zmat1.safe_loc[zmat1.index[0], "dihedral"] = da

    zmat2.safe_loc[zmat2.index[0], "bond"] = bb
    zmat2.safe_loc[zmat2.index[0], "angle"] = ab
    zmat2.safe_loc[zmat2.index[0], "dihedral"] = db


    ba_vals = np.arange(5, 8, 2)
    bb_vals = np.arange(7, 11, 2)
    aa_vals = np.arange(0, 90, 33)
    ab_vals = np.arange(-90, 0, 45)
    da_vals = np.arange(0, 180, 33)
    db_vals = np.arange(-181, 0, 45)


    def make_conf(ba_val, bb_val, aa_val, ab_val, da_val, db_val):


        a = zmat1.subs(
            ba, ba_val + np.random.normal()/100 ).subs(
            aa, aa_val + np.random.normal()).subs(
            da, da_val + np.random.normal()).get_cartesian()[["x", "y", "z"]].sort_index()
        a = a.to_numpy()

        b = zmat2.subs(bb, bb_val).subs(ab, ab_val).subs(db, db_val).get_cartesian()[["x", "y", "z"]].sort_index()
        b = b.to_numpy()

        combined = np.concat([a, b])
        combined += np.random.normal(size=combined.shape)/100

        XYZ = pd.DataFrame(combined, columns=["x", "y", "z"])
        return XYZ


    setupRes.mini(nbxmod=1)
    energy = pycharmm.energy.get_energy()
    energy = energy[['ENER', 'VDW', 'ELEC']]

    view_pycharmm_state()


    xyzs = []
    fragments = cc_mol_xyz.fragmentate()
    pycharmm_quiet()


    for ba_val in ba_vals:
        for bb_val in bb_vals:
            for aa_val in aa_vals:
                for ab_val in ab_vals:
                    for da_val in da_vals:
                        for db_val in db_vals:

                            XYZ = make_conf(ba_val, bb_val, aa_val, ab_val, da_val, db_val)
                            xyzs.append(XYZ)

    return xyzs


def sample_dimer_energies(xyzs):
    energies = []

    for XYZ in xyzs:
        pycharmm.coor.set_positions(XYZ)
        energy = pycharmm.energy.get_energy()
        energy = energy[['ENER', 'VDW', 'ELEC']]
        energies.append(energy)



    energies_df = pd.concat(energies)
    energies_df


    energies_df = pd.concat(energies)
    energies_df["NBOND"] = energies_df["VDW"] +  energies_df["ELEC"] 
    energies_df.index = range(len(energies_df))


    xyzs_np = np.array(xyzs)





