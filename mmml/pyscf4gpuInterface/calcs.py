import numpy as np
import pyscf
from pyscf.hessian import thermo
from gpu4pyscf.dft import rks

from enums import *
from helperfunctions import *


def setup_mol(atoms, basis, xc, log_file='./pyscf.log', 
    verbose=6, 
    lebedev_grids=(99,590),
    scf_tol=1e-10,
    scf_max_cycle=50,
    cpscf_tol=1e-3,
    conv_tol=1e-10,
    conv_tol_cpscf=1e-3,
    ):
    if type(atoms) == str:
        mol = pyscf.M(
        atom=atoms,                         # water molecule
        basis=basis,                # basis set
        output=log_file,              # save log file
        verbose=verbose                          # control the level of print info
        )

    else:
        mol = atoms
        
    mf_GPU = rks.RKS(                      # restricted Kohn-Sham DFT
        mol,                               # pyscf.gto.object
        xc=xc                         # xc funtionals, such as pbe0, wb97m-v, tpss,
        ).density_fit()                    # density fitting

    mf_GPU.grids.atom_grid = lebedev_grids      # (99,590) lebedev grids, (75,302) is often enough
    mf_GPU.conv_tol = scf_tol                   # controls SCF convergence tolerance
    mf_GPU.max_cycle = scf_max_cycle            # controls max iterations of SCF
    mf_GPU.conv_tol_cpscf = conv_tol_cpscf      # controls max iterations of CPSCF (for hessian)

    return mf_GPU, mol



def compute_dft(mol, calcs, extra=None, basis='def2-tzvpp', xc="wB97m-v"):

    engine, mol = setup_mol(mol, basis, xc)

    print(mol)

    opt_callback = None

    if CALCS.OPTIMIZE in calcs:
        print("-"*100)
        print("Optimizing geometry")
        print("-"*100)
        from pyscf.geomopt.geometric_solver import optimize
        import time

        
        gradients = []
        energies = []
        coords = []
        def callback(envs):
            print(list(envs.keys()))
            gradients.append(envs['gradients'])
            energies.append(envs['energy'])
            coords.append(envs['coords'])

        start_time = time.time()
        mol = optimize(engine, maxsteps=20, callback=callback)
        print("Optimized coordinate:")
        print(mol.atom_coords())
        print('Geometry optimization took', time.time() - start_time, 's')
        opt_callback = {
            'gradients': gradients,
            'energies': energies,
            'coords': coords
        }

    output = {"mol": mol, "calcs": calcs, "opt_callback": opt_callback}

    if CALCS.ENERGY in calcs:
        print("-"*100)
        print("Computing Energy")
        print("-"*100)
        # Compute Energy
        e_dft = engine.kernel()
        print(f"total energy = {e_dft}")       # -76.46668196729536
        output['energy'] = e_dft

    if CALCS.GRADIENT in calcs:
        print("-"*100)
        print("Computing Gradient")
        print("-"*100)
        # Compute Gradient
        g = engine.nuc_grad_method()
        g_dft = g.kernel()
        output['gradient'] = g_dft

    if CALCS.HESSIAN in calcs:
        print("-"*100)
        print("Computing Hessian")
        print("-"*100)
        # Compute Hessian
        h = engine.Hessian()
        h.auxbasis_response = 2                # 0: no aux contribution, 1: some contributions, 2: all
        engine.cphf_grids.atom_grid = (50,194) # customize grids for solving CPSCF equation, SG1 by default
        h_dft = h.kernel()
        output['hessian'] = h_dft

    if CALCS.HARMONIC in calcs:
        print("-"*100)
        print("Computing Harmonic Analysis")
        print("-"*100)
        # harmonic analysis
        results = thermo.harmonic_analysis(mol, h_dft)
        thermo.dump_normal_mode(mol, results)
        output['harmonic'] = results

    if CALCS.THERMO in calcs:
        print("-"*100)
        print("Computing Thermodynamics")
        print("-"*100)
        results = thermo.thermo(
            engine,                            # GPU4PySCF object
            results['freq_au'],
            298.15,                            # room temperature
            101325)                            # standard atmosphere

        thermo.dump_thermo(mol, results)
        output['thermo'] = results

    if CALCS.HESSIAN in calcs:
        print("-"*100)
        print("Computing Hessian")
        print("-"*100)
        # force translational symmetry
        natm = mol.natm
        h_dft = h_dft.transpose([0,2,1,3]).reshape(3*natm,3*natm)
        h_diag = h_dft.sum(axis=0)
        h_dft -= np.diag(h_diag)
        output['hessian'] = h_dft

    if CALCS.INTERACTION in calcs:
        print("-"*100)
        print("Computing Interaction Energy")
        print("-"*100)
        # Compute Interaction Energy
        e_interaction = compute_interaction_energy(mol, extra)
        output['interaction'] = e_interaction

    return output




def compute_interaction_energy(monomer_a, monomer_b, basis='cc-pVDZ', xc='wB97m-v'):
    # Convert string geometries to PySCF atom format
    def parse_xyz(xyz_str):
        atoms = []
        for line in xyz_str.strip().split('\n'):
            if not line.strip(): continue
            symbol, *coords = line.split()
            coords = tuple(float(x) for x in coords)
            atoms.append((symbol, coords))
        return atoms

    atom_A = parse_xyz(monomer_a)
    atom_B = parse_xyz(monomer_b)
    atom_AB = atom_A + atom_B

    # Build molecular objects
    mol_A = pyscf.M(atom=atom_A, basis=basis).build()
    mol_B = pyscf.M(atom=atom_B, basis=basis).build()
    mol_AB = pyscf.M(atom=atom_AB, basis=basis).build()

    # Create ghost-atom systems for BSSE correction
    mol_A_ghost = mol_A.copy()
    ghost_atoms_B = mol_B.atom
    mol_A_ghost.atom.extend([('X-' + atom[0], atom[1]) for atom in ghost_atoms_B])
    mol_A_ghost.build()

    mol_B_ghost = mol_B.copy()
    ghost_atoms_A = mol_A.atom
    mol_B_ghost.atom.extend([('X-' + atom[0], atom[1]) for atom in ghost_atoms_A])
    mol_B_ghost.build()

    regular_calcs = [CALCS.ENERGY, CALCS.GRADIENT, CALCS.HESSIAN, CALCS.HARMONIC, CALCS.THERMO]
    ghost_calcs = [CALCS.ENERGY]
    
    output_AB = compute_dft(mol_AB, regular_calcs)
    E_AB = output_AB["energy"]

    output_A = compute_dft(mol_A, regular_calcs)
    E_A = output_A["energy"]

    output_B = compute_dft(mol_B, regular_calcs)
    E_B = output_B["energy"]    

    output_A_ghost = compute_dft(mol_A_ghost, ghost_calcs)
    E_A_ghost = output_A_ghost["energy"]    

    output_B_ghost = compute_dft(mol_B_ghost, ghost_calcs)
    E_B_ghost = output_B_ghost["energy"]
    

    # Calculate interaction energies
    IE_no_bsse = E_AB - (E_A + E_B)
    IE_energy_bsse = E_AB - (E_A_ghost + E_B_ghost)

    intE = {
        'E_AB': E_AB,
        'E_A': E_A,
        'E_B': E_B,
        'E_A_ghost': E_A_ghost,
        'E_B_ghost': E_B_ghost,
        'IE_no_bsse': IE_no_bsse,
        'IE_energy_bsse': IE_energy_bsse,
    }
    output = {
        'results_AB': output_AB,
        'results_A': output_A,
        'results_B': output_B,
        'results_A_ghost': output_A_ghost,
        'results_B_ghost': output_B_ghost,
        'intE_results': intE
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    # molecule
    parser.add_argument("--mol", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.pkl")
    parser.add_argument("--log_file", type=str, default="pyscf.log")
    parser.add_argument("--monomer_a", type=str, default="")
    parser.add_argument("--monomer_b", type=str, default="")
    parser.add_argument("--basis", type=str, default="def2-tzvp")
    parser.add_argument("--xc", type=str, default="wB97m-v")
    # flags to do certain calcs
    parser.add_argument("--energy", default=True, action="store_true")
    parser.add_argument("--optimize", default=False, action="store_true")
    parser.add_argument("--gradient", default=True, action="store_true")
    parser.add_argument("--hessian", default=False, action="store_true")
    parser.add_argument("--harmonic", default=False, action="store_true")
    parser.add_argument("--thermo", default=False, action="store_true")
    parser.add_argument("--interaction", default=False, action="store_true")
    parser.add_argument("--dens_esp", default=False, action="store_true")
    args = parser.parse_args()
    return args


def process_calcs(args):
    calcs = []
    extra = None

    if args.optimize:
        calcs.append(CALCS.OPTIMIZE)

    if args.energy:
        calcs.append(CALCS.ENERGY)
    if args.gradient:
        calcs.append(CALCS.GRADIENT)
    if args.hessian:
        calcs.append(CALCS.HESSIAN)
    if args.harmonic:
        calcs.append(CALCS.HARMONIC)
    if args.thermo:
        calcs.append(CALCS.THERMO)

    if args.dens_esp:
        calcs.append(CALCS.DENS_ESP)
    if args.interaction:
        calcs.append(CALCS.INTERACTION)
        extra = (args.monomer_a, args.monomer_b)

    return calcs, extra

if __name__ == "__main__":
    import argparse
    import json
    args = parse_args()
    calcs, extra = process_calcs(args)
    print(calcs, extra)
    # mol = setup_mol(args.mol, args.basis, args.xc)
    output = compute_dft(args.mol, calcs, extra, args.basis, args.xc)
    print(output)
    import pickle
    # save output to pickle
    with open(args.output, "wb") as f:
        pickle.dump(output, f)
