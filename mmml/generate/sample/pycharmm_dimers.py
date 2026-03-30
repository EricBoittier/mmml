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


from mmml.interfaces.chemcoordInterface import interface
from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3
from mmml.generate.sample import sample_cc

patch_chemcoord_for_pandas3()


def make_dimer_pdb(resid):
    reset_block()
    atoms = setupRes.main(resid)
    N = len(atoms)
    pycharmm.read.sequence_string(resid)
    pycharmm.gen.new_segment(seg_name=resid, setup_ic=True)
    pycharmm.ic.prm_fill(replace_all=True)
    pycharmm.coor.show()

    positions = pycharmm.coor.get_positions()

    positions_np = positions.to_numpy()
    positions_np_copy = np.zeros_like(positions_np)
    positions_np_copy[:N] = positions_np[:N]
    positions_np_copy[N:] = positions_np[:N]
    positions_np_copy[N:,0] += 10
    
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

    ase.io.write(f"{resid.lower()}_dimer.xyz", atoms)



def sample_dimer(xyz_file, mol_r_scale = 1.0):
    """
    Delegate to shared chemcoord-based sampler in sample_cc so that
    sampling logic and random noise are harmonized with other scripts.
    """
    return sample_cc.sample_dimer_cc(
        xyz_file,
        mol_r_scale=mol_r_scale,
    )


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




if __name__ == "__main__":
    resid = "MEOH"
    make_dimer_pdb(resid)
    reset_block()
    make_dimer_pdb(resid)
    xyzs = sample_dimer(f"{resid.lower()}_dimer.xyz", mol_r_scale=1.0)
    print(len(xyzs))
    atomic_numbers = get_Z_from_psf()
    with open(f"{resid.lower()}_dimers_sampled.xyz", "w") as f:
    # save xyzs to an xyz file
        for idx, XYZ in enumerate(xyzs):
            f.write(f"{len(XYZ)}\n")
            f.write(f"dimer {idx}\n")
            for sym, (x, y, z) in zip(atomic_numbers, XYZ[["x", "y", "z"]].to_numpy()):
                sym = ase.data.chemical_symbols[sym]
                f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
