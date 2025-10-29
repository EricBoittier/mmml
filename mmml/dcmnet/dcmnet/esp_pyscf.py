
from tqdm import tqdm
import jax

jax.config.update("jax_default_device", jax.devices("cpu")[0])


import numpy as np
from scipy.spatial import distance_matrix
import cupy
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import dist_matrix
import pyscf
import numpy as np
from pyscf import gto
from gpu4pyscf.dft import rks

from gpu4pyscf.properties import polarizability

import pandas as pd
import e3x
import os

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "None"
import pandas as pd
import dcmnet
import sys

sys.path.append("/pchem-data/meuwly/boittier/jaxeq/dcmnet")
print(sys.path)
# from dcmnet.models import DCM1, DCM2, DCM3, DCM4, dcm1_params, dcm2_params, dcm3_params, dcm4_params
from dcmnet.modules import MessagePassingModel
from dcmnet.data import prepare_datasets

import numpy as np


def atom_centered_dipole(dcm, com, q):
    dipole_out = np.zeros(3)
    for i, _ in enumerate(dcm):
        dipole_out += q[i] * (_ - com)
    # print(dipole_out*2.5417464519)
    return dipole_out, np.linalg.norm(dipole_out) * 4.80320


import jax
import jax.numpy as jnp
import pickle

import time

from dcmnet.utils import safe_mkdir
from dcmnet.training import train_model
from dcmnet.training_dipole import train_model_dipo
from pathlib import Path
from dcmnet.data import prepare_batches, prepare_datasets
from dcmnet.utils import apply_model

import optax
from dcmnet.analysis import create_model_and_params
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import jax

from pyscf import df

from dcmnet.loss import (
    esp_loss_eval,
)

devices = jax.local_devices()

'''
references:
https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.26035
'''

import numpy as np
from scipy.spatial import distance_matrix
import cupy
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import dist_matrix

from ase.data import chemical_symbols

# Van der Waals radii (in angstrom) are taken from GAMESS.
R_VDW = 1.0/radii.BOHR * np.asarray([
    -1,
    1.20, # H
    1.20, # He
    1.37, # Li
    1.45, # Be
    1.45, # B
    1.50, # C
    1.50, # N,
    1.40, # O
    1.35, # F,
    1.30, # Ne,
    1.57, # Na,
    1.36, # Mg
    1.24, # Al,
    1.17, # Si,
    1.80, # P,
    1.75, # S,
    1.70]) # Cl

def unit_surface(n):
    '''
    Generate spherical harmonics grid points on unit sphere
    The number of generated points is less than n in general.
    '''
    ux = []
    uy = []
    uz = []

    eps = 1e-10
    nequat = int(np.sqrt(np.pi*n))
    nvert = int(nequat/2)
    for i in range(nvert+1):
        fi = np.pi*i/nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat*xy+eps)
        if nhor < 1:
            nhor = 1
        
        fj = 2.0 * np.pi * np.arange(nhor) / nhor
        x = np.cos(fj) * xy
        y = np.sin(fj) * xy

        ux.append(x)
        uy.append(y)
        uz.append(z*np.ones_like(x))
    
    ux = np.concatenate(ux)
    uy = np.concatenate(uy)
    uz = np.concatenate(uz)

    return np.array([ux[:n], uy[:n], uz[:n]]).T

def vdw_surface(mol, scales=[1.0], density=4*radii.BOHR**2, rad=R_VDW):
    '''
    Generate vdw surface of molecules, in Bohr
    '''
    coords = mol.atom_coords(unit='B')
    charges = mol.atom_charges()
    atom_radii = rad[charges]

    surface_points = []
    for scale in scales:
        scaled_radii = atom_radii * scale
        for i, coord in enumerate(coords):
            r = scaled_radii[i]
            # nd is an indicator of density, not exactly the same as number of points
            nd = int(density * 4.0 * np.pi * r**2)
            points = coord + r * unit_surface(nd)
            points = points +  np.random.normal(0,0.1,np.prod(points.shape)).reshape(points.shape)
            dist = distance_matrix(points, coords) + 1e-10
            included = np.all(dist >= scaled_radii, axis=1)
            surface_points.append(points[included])
    points = np.concatenate(surface_points) 
    points = points +  np.random.normal(0,0.5,np.prod(points.shape)).reshape(points.shape)
    return points

print(devices)
print(jax.default_backend())
print(jax.devices())



def run(mol_elements, mol_positions, job_id=""):
    nonzero = np.nonzero(mol_elements)

    atoms = [
        [chemical_symbols[atom], xyz]
        for atom, xyz in zip(mol_elements[nonzero], mol_positions[nonzero])
    ]

    mol = pyscf.M(atom=atoms, unit="B", basis="def2-TZVP", charge=0, spin=0)
    # 2S, where S is the total spin quantum number
    mf = rks.RKS(mol, xc="PBE0").density_fit()
    mf.kernel()
    dm = mf.make_rdm1()  # compute one-electron density matrix
    coords = vdw_surface(mol, scales=np.arange(1.4,2.0,0.1)) #mf.grids.coords.get()
    print(coords.shape)
    fakemol = gto.fakemol_for_charges(coords)
    coords_angstrom = fakemol.atom_coords(unit="ANG")
    mol_coords_angstrom = mol.atom_coords(unit="ANG")
    charges = mol.atom_charges()
    charges = cupy.asarray(charges)
    coords = cupy.asarray(coords)
    mol_coords = cupy.asarray(mol.atom_coords(unit="B"))
    r = dist_matrix(mol_coords, coords)
    rinv = 1.0 / r
    intopt = int3c2e.VHFOpt(mol, fakemol, "int2e")
    intopt.build(1e-14, diag_block_with_triu=False, aosym=True, group_size=256)
    v_grids_e = 2.0 * int3c2e.get_j_int3c2e_pass1(intopt, dm, sort_j=False)
    v_grids_n = cupy.dot(charges, rinv)
    res = v_grids_n - v_grids_e
    dip = mf.dip_moment(unit="DEBYE", dm=dm.get())
    quad = mf.quad_moment(unit="DEBYE-ANG", dm=dm.get())

    # polarizabilities
    #polar = mf.Polarizability().polarizability()
    polar = polarizability.eval_polarizability(mf)
    #print("polar", polar)

    np.savez(
        f"{job_id}_esp_pyscf.npz",
        Z=mol_elements[nonzero],
        R=mol_coords_angstrom,
        esp=res,
        esp_grid=coords_angstrom,
        polar=polar,
        dipole=dip,
        quadrupole=quad,
    )



