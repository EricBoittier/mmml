import numpy as np
import pyscf
from pyscf.hessian import thermo
from gpu4pyscf.dft import rks
from pyscf import gto
 
import pyscf

import sys
import numpy as np
import pyscf
from pyscf.hessian import thermo
from gpu4pyscf.dft import rks

from .enums import *
from .helperfunctions import *

def compute_dft(mol, calcs, extra=None, xc="wB97m-v"):

    engine, mol = setup_mol(mol, xc)

    output = {"mol": mol, "calcs": calcs}

    if CALCS.ENERGY in calcs:
        # Compute Energy
        e_dft = engine.kernel()
        print(f"total energy = {e_dft}")       
        output['energy'] = e_dft

    if CALCS.GRADIENT in calcs:
        # Compute Gradient
        g = engine.nuc_grad_method()
        g_dft = g.kernel()
        output['gradient'] = g_dft

    if CALCS.HESSIAN in calcs:
        # Compute Hessian
        h = engine.Hessian()
        h.auxbasis_response = 2                # 0: no aux contribution, 1: some contributions, 2: all
        engine.cphf_grids.atom_grid = (50,194) # customize grids for solving CPSCF equation, SG1 by default
        h_dft = h.kernel()
        output['hessian'] = h_dft

    if CALCS.HARMONIC in calcs:
        # harmonic analysis
        results = thermo.harmonic_analysis(mol, h_dft)
        thermo.dump_normal_mode(mol, results)
        output['harmonic'] = results

    if CALCS.THERMO in calcs:
        results = thermo.thermo(
            engine,                            # GPU4PySCF object
            results['freq_au'],
            298.15,                            # room temperature
            101325)                            # standard atmosphere

        thermo.dump_thermo(mol, results)
        output['thermo'] = results

    if CALCS.HESSIAN in calcs:
        # force translational symmetry
        natm = mol.natm
        h_dft = h_dft.transpose([0,2,1,3]).reshape(3*natm,3*natm)
        h_diag = h_dft.sum(axis=0)
        h_dft -= np.diag(h_diag)
        output['hessian'] = h_dft

    if CALCS.INTERACTION in calcs and extra is None:
        raise ValueError("Interaction energy requires extra arguments (monomer_a, monomer_b)")
        
    if CALCS.INTERACTION in calcs:
        # Compute Interaction Energy
        e_interaction = compute_interaction_energy(mol, extra)
        output['interaction'] = e_interaction

    return output


def create_mol(atom, basis, spin, charge, log_file="./pyscf.log", verbose=True):
    M =  pyscf.M(
        atom=atom,                         # water molecule
        basis=basis,                # basis set
        spin=spin,
        charge=charge,
        output=log_file,              # save log file
        verbose=verbose                          # control the level of print info
        )
    M.build()
    return M

def setup_mol(atom, basis, xc, log_file='./pyscf.log', 
    verbose=6, 
    lebedev_grids=(99,590),
    scf_tol=1e-10,
    scf_max_cycle=50,
    cpscf_tol=1e-3,
    conv_tol=1e-10,
    conv_tol_cpscf=1e-3,
              spin=0,
              charge=1,
    ):
    if type(atom) == str:
        mol = create_mol(atom, basis, spin=spin, charge=charge)
    else:
        mol = atom

    print(mol)
    
    mf_GPU = rks.RKS(                      # restricted Kohn-Sham DFT
        mol,                               # pyscf.gto.object
        xc=xc                         # xc funtionals, such as pbe0, wb97m-v, tpss,
        ).density_fit()                    # density fitting

    mf_GPU.grids.atom_grid = lebedev_grids      # (99,590) lebedev grids, (75,302) is often enough
    mf_GPU.conv_tol = scf_tol                   # controls SCF convergence tolerance
    mf_GPU.max_cycle = scf_max_cycle            # controls max iterations of SCF
    mf_GPU.conv_tol_cpscf = conv_tol_cpscf      # controls max iterations of CPSCF (for hessian)



def get_erefs(basis, xc):
    # Flat arrays for Nα and Nβ values
    elements = [
        'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 
        'Oxygen', 'Fluorine', 'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 
        'Phosphorus', 'Sulfur', 'Chlorine'
    ]
    
    Nalpha = [1, 0, 1, 0, 1, 1, 2, 2, 2, 0, 1, 0, 1, 1, 2, 2, 2]
    Nbeta  = [0, 0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 0, 1, 1, 2, 3]
    spins = np.array(Nalpha) - np.array(Nbeta)
    zs = range(1,18)
    Eref = np.zeros([20], dtype=float)
    for z, s in zip(zs, spins):
        # create two atoms at a large distance apart to get over limitations in gpu4pyscf
        engine, mol = setup_mol(f"{z} -1000. -1000. -1000.; {z} 1000. 1000. 1000. ", basis, xc, spin = 2*s, charge=0)
        e_dft = engine.kernel()/2
        Eref[z] = e_dft
    
    
    return Eref


def save_ref(basis, xc):
    Erefs = get_erefs(basis, xc)
    np.savez("Eref-{xc}-{basis}", Eref = Eref)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=str, default="def2-tzvp")
    parser.add_argument("--xc", type=str, default="wB97m-v")
    args = parser.parse_args()
    save_ref(args.basis, args.xc)
