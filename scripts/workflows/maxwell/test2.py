import mmml
from tqdm import tqdm

from pathlib import Path
import numpy as np
import pandas as pd

import ase

from functools import reduce
from pyscf.hessian import thermo
import numpy as np
import cupy
from pyscf.data import elements, nist
from scipy.constants import physical_constants
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf.hf import RHF
import pyscf
import numpy as np
from pyscf import gto
from gpu4pyscf.dft import rks, uks
from pyscf.geomopt.geometric_solver import optimize

import ase
from ase.io import read

import numpy as np
from scipy.spatial import distance_matrix

import cupy
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import dist_matrix

import time
from ase.data import chemical_symbols
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import ase.atoms

from gpu4pyscf.properties import ir, shielding, polarizability

import cupy

from mmml.pyscf4gpuInterface.enums import *
from mmml.pyscf4gpuInterface.helperfunctions import *
from mmml.pyscf4gpuInterface.esp_helpers import balance_array



LINDEP_THRESHOLD = 1e-7

import pyscf
from pyscf import gto, scf, tools


from mmml.pyscf4gpuInterface.calcs import setup_mol

dm_init_guess = [None]


def apply_field(E, mol):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    dm_init_guess[0] = mf.make_rdm1()
    e = mf.kernel()    
    dip_m = mf.dip_moment()
    return e, dip_m


def maxwell_eval_ir_freq_intensity(E, mol, dm_init_guess = [None]):
    '''Calculate the IR frequency and intensity of a molecule under an electric field.

    Args:
        E: electric field strength
        mol: molecule object

    Returns:
        results: a dictionary containing the following keys:
            - "Ef": the electric field strength
            - "D": the dipole moment
            - "E": the energy
            - "H": the Hessian
            - "freq": the IR frequency
            - "intensity": the IR intensity
            - "A": the polarizability
            - "dDdR": the dipole moment derivative
            - "R": the atomic coordinates
            - "Z": the atomic numbers
    '''
    # Initialize SCF calculation with electric field
    
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h = (mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
         + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    
    # Run SCF and collect basic properties
    mf = rks.RKS(mol, xc="PBE0").density_fit()  
    mf.get_hcore = lambda *args: h
    e = mf.kernel(dm_init_guess[0])  # Combine scf and kernel calls
    dip_m = mf.dip_moment(unit="AU")
    polar = polarizability.eval_polarizability(mf)
    summary = {"Ef": E, "D": dip_m, "E": e, "A": polar}

    # Calculate forces (negative gradient)
    g = mf.nuc_grad_method()
    summary['F'] = -g.kernel()
    
    # Calculate Hessian with customized settings
    hessian_obj = mf.Hessian()
    hessian_obj.auxbasis_response = 2  # Include all auxiliary basis contributions
    mf.cphf_grids.atom_grid = (50,194)  # Customize grids for CPSCF equation
    hessian = hessian_obj.kernel()
    summary['H'] = hessian

    # Setup logger for Hessian calculations
    log = logger.new_logger(hessian_obj, mf.mol.verbose)  # Use h instead of hessian_obj
    assert isinstance(mf, RHF)
    hartree_kj = nist.HARTREE2J*1e3
    unit2cm = ((hartree_kj * nist.AVOGADRO)**.5 / (nist.BOHR*1e-10)
               / (2*np.pi*nist.LIGHT_SPEED_SI) * 1e-2)
    natm = mf.mol.natm
    nao = mf.mol.nao
    dm0 = mf.make_rdm1()

    atom_charges = mf.mol.atom_charges()
    mass = cupy.array([elements.MASSES[atom_charges[i]] for i in range(natm)])
    hessian_mass = contract('ijkl,i->ijkl', cupy.array(hessian), 1/cupy.sqrt(mass))
    hessian_mass = contract('ijkl,j->ijkl', hessian_mass, 1/cupy.sqrt(mass))

    TR = thermo._get_TR(mass.get(), mf.mol.atom_coords())
    TRspace = []
    TRspace.append(TR[:3])

    rot_const = thermo.rotation_const(mass.get(), mf.mol.atom_coords())
    rotor_type = thermo._get_rotor_type(rot_const)
    if rotor_type == 'ATOM':
        pass
    elif rotor_type == 'LINEAR':  # linear molecule
        TRspace.append(TR[3:5])
    else:
        TRspace.append(TR[3:])

    if TRspace:
        TRspace = cupy.vstack(TRspace)
        q, r = cupy.linalg.qr(TRspace.T)
        P = cupy.eye(natm * 3) - q.dot(q.T)
        w, v = cupy.linalg.eigh(P)
        bvec = v[:,w > LINDEP_THRESHOLD]
        h = reduce(cupy.dot, (bvec.T, hessian_mass.transpose(0,2,1,3).reshape(3*natm,3*natm), bvec))
        e, mode = cupy.linalg.eigh(h)
        mode = bvec.dot(mode)

    c = contract('ixn,i->ixn', mode.reshape(natm, 3, -1),
                  1/np.sqrt(mass)).reshape(3*natm, -1)
    freq = cupy.sign(e)*cupy.sqrt(cupy.abs(e))*unit2cm

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = cupy.array(mo_coeff)
    mo_occ = cupy.array(mo_occ)
    mo_energy = cupy.array(mo_energy)
    mocc = mo_coeff[:, mo_occ > 0]
    mocc = cupy.array(mocc)

    atmlst = range(natm)
    h1ao = hessian_obj.make_h1(mo_coeff, mo_occ, None, atmlst)
    # TODO: compact with hessian method, which can save one time cphf solve.
    # ! Different from PySCF, mo1 is all in mo!
    fx = hessian_obj.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = hessian_obj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       fx, atmlst, hessian_obj.max_memory, log)  
    mo1 = cupy.asarray(mo1)
    mo_e1 = cupy.asarray(mo_e1)

    # Calculate dipole derivatives
    tmp = cupy.empty((3, 3, natm))  # dipole moment derivative tensor (x,y,z)
    aoslices = mf.mol.aoslice_by_atom()
    with mf.mol.with_common_orig((0, 0, 0)):
        hmuao = cupy.array(mf.mol.intor('int1e_r'))  # dipole integrals
        hmuao11 = -cupy.array(mf.mol.intor('int1e_irp').reshape(3, 3, nao, nao))

    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h11ao = cupy.zeros((3, 3, nao, nao))

        h11ao[:, :, :, p0:p1] += hmuao11[:, :, :, p0:p1]
        h11ao[:, :, p0:p1] += hmuao11[:, :, :, p0:p1].transpose(0, 1, 3, 2)

        tmp0 = contract('ypi,vi->ypv', mo1[ia], mocc)  # nabla
        dm1 = contract('ypv,up->yuv', tmp0, mo_coeff)
        tmp[:, :, ia] = -contract('xuv,yuv->xy', hmuao, dm1) * 4 #the minus means the density should be negative, but calculate it is positive.
        tmp[:, :, ia] -= contract('xyuv,vu->xy', h11ao, dm0)
        tmp[:, :, ia] += mf.mol.atom_charge(ia)*cupy.eye(3)

    # Convert to km/mol units for IR intensity
    alpha = physical_constants["fine-structure constant"][0]
    amu = physical_constants["atomic mass constant"][0]
    m_e = physical_constants["electron mass"][0]
    N_A = physical_constants["Avogadro constant"][0]
    a_0 = physical_constants["Bohr radius"][0]
    unit_kmmol = alpha**2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3

    # Calculate final IR intensities
    intensity = contract('xym,myn->xn', tmp, c.reshape(natm, 3, -1))
    intensity = contract('xn,xn->n', intensity, intensity) * unit_kmmol

    # Get molecular geometry information
    atoms = ase.atoms.Atoms(mol.elements, mol.atom_coords(unit="AU"))
    summary.update({
        "freq": freq,
        "intensity": intensity,
        "dDdR": tmp,
        "R": atoms.get_positions(),
        "Z": atoms.get_atomic_numbers()
    })
    return summary, dm0






""" SCRIPT
"""

data_files = list(Path("/scicore/home/meuwly/boitti0000/data").glob("*npz"))
print(data_files)
#data = pd.read_pickle(data_files[0])
data = np.load(data_files[0])
print(data_files[0])
print(data)

import sys



water_data_silvan = data


i = int(sys.argv[1])
Njobs = 100
start = i * Njobs # next is from 500
dm0 = None
all_output = []

for i in tqdm(range(start, start+Njobs)):
    Natoms = water_data_silvan["N"][i]
    print(Natoms)
    Natoms = 3
    atoms = ase.Atoms(water_data_silvan["Z"][i], water_data_silvan["R"][i])
    atoms
    mol_from_ase = atoms_from_ase(atoms)
    mol = pyscf.M(
    atom=mol_from_ase,                         # water molecule
    basis="def2-tzvp",                # basis set
    spin=0,
    charge=0,
    unit="B",
    # output=log_file,              # save log file
    # verbose=verbose                          # control the level of print info
    )
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    dir(mol)
    
    E = np.zeros(3)
    print(E)

    res_0, dm0 = maxwell_eval_ir_freq_intensity(E, mol, [dm0])
    # dm0 = None
    print(res_0["E"])
    all_output.append(res_0)
    print()
    
    output_res = [res_0]
    for i in range(2):
        E = np.random.normal(0, scale=0.01, size=(3,))
        print(E)
        res, dm0 = maxwell_eval_ir_freq_intensity(E, mol, [dm0])
        print(res["E"], 627.53* (res_0["E"] - res["E"]))

        all_output.append(res)


import pickle
# Save to file
with open(f'data3/data{start}.pkl', 'wb') as f:
    pickle.dump(all_output, f)
