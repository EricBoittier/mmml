import numpy as np
import pyscf
from pyscf.hessian import thermo
from pyscf import gto
from pyscf.data import radii
# from gpu4pyscf.dft import int3c2e
from gpu4pyscf.lib.cupy_helper import dist_matrix
from gpu4pyscf.dft import rks
from gpu4pyscf.properties import ir, shielding, polarizability

import cupy

from mmml.pyscf4gpuInterface.enums import *
from mmml.pyscf4gpuInterface.helperfunctions import *
from mmml.pyscf4gpuInterface.esp_helpers import balance_array

def setup_mol(atoms, basis, xc, spin, charge, log_file='./pyscf.log', 
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
        spin=spin,
        charge=charge,
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



def compute_dft(args, calcs, extra=None):

    engine, mol = setup_mol(args.mol, args.basis, args.xc, args.spin, args.charge)

    print(mol)
    from mmml.pyscf4gpuInterface.helperfunctions import print_basis
    print_basis(mol)

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
        print(f"total energy = {e_dft}")       
        output['energy'] = e_dft

    if CALCS.DENS_ESP in calcs:
        print("-"*100)
        print("Computing Density ESP")
        print("-"*100)
        print('------------------ Density ----------------------------')
        dm = engine.make_rdm1()
        grids = engine.grids
        grid_coords = grids.coords.get()
        density = engine._numint.get_rho(mol, dm, grids).get()
        print('------------------ Selecting points ----------------------------')
        grid_indices = np.where(np.less(density, 0.001) & np.greater_equal(density, 0.0001))[0]
        print(grid_indices)
        # grid_positions_a = grid_coords[cupy.where(density < 0.001)[0]]
        grid_positions_a = grid_coords[grid_indices]
        print(grid_positions_a.shape)
        print(grid_positions_a.min(), grid_positions_a.max())
        print('------------------ ESP ----------------------------')
        dm = engine.make_rdm1()  # compute one-electron density matrix
        coords = grid_positions_a
        print(coords.shape)
        fakemol = gto.fakemol_for_charges(coords)
        coords_bohr = fakemol.atom_coords(unit="B")
        mol_coords_angstrom = mol.atom_coords(unit="ANG")
        charges = mol.atom_charges()
        charges = cupy.asarray(charges)
        coords = cupy.asarray(coords)
        mol_coords = cupy.asarray(mol.atom_coords(unit="ANG"))
        print("distance matrix")
        r = dist_matrix(mol_coords, coords_bohr)
        rinv = 1.0 / r
        intopt = int3c2e.VHFOpt(mol, fakemol, "int2e")
        intopt.build(1e-14, diag_block_with_triu=False, aosym=True, group_size=256)
        # electronic grids
        print("electronic grids")
        v_grids_e = 2.0 * int3c2e.get_j_int3c2e_pass1(intopt, dm, sort_j=False)
        # nuclear grids
        print("nuclear grids")
        v_grids_n = cupy.dot(charges, rinv)
        res = v_grids_n - v_grids_e
        res = res.get()
        output['esp'] = res
        output['esp_grid'] = coords.get()
        output['R'] = mol_coords_angstrom
        output['Z'] = mol.atom_charges()
        output['D'] = engine.dip_moment(unit="DEBYE", dm=dm )
        output['Q'] = engine.quad_moment(unit="DEBYE-ANG", dm=dm )
        output['density'] = density
        output['density_grid'] = grid_coords


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
        harmonic_results = thermo.harmonic_analysis(mol, h_dft)
        thermo.dump_normal_mode(mol, harmonic_results)
        output['harmonic'] = harmonic_results

    if CALCS.IR in calcs:
        assert CALCS.HESSIAN in calcs, "Hessian must be computed for IR"
        print("-"*100)
        print("Computing IR")
        print("-"*100)
        freq, intensity = ir.eval_ir_freq_intensity(engine, h)
        output['freq'] = freq
        output['intensity'] = intensity

    if CALCS.SHIELDING in calcs:
        assert CALCS.ENERGY in calcs, "Energy must be computed for shielding"
        print("-"*100)
        print("Computing Shielding")
        print("-"*100)
        msc_d, msc_p = shielding.eval_shielding(engine)
        msc = (msc_d + msc_p).get()
        output['shielding'] = msc

    if CALCS.POLARIZABILITY in calcs:
        assert CALCS.ENERGY in calcs, "Energy must be computed for polarizability"
        print("-"*100)
        print("Computing Polarizability")
        print("-"*100)
        polar = polarizability.eval_polarizability(engine)
        output['polarizability'] = polar

    if CALCS.THERMO in calcs:
        assert CALCS.HARMONIC in calcs, "Harmonic must be computed for thermodynamics"
        print("-"*100)
        print("Computing Thermodynamics")
        print("-"*100)
        results = thermo.thermo(
            engine,                            # GPU4PySCF object
            harmonic_results['freq_au'],
            298.15,                            # room temperature
            101325)                            # standard atmosphere

        thermo.dump_thermo(mol, results)
        output['thermo'] = results


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
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)
    # flags to do certain calcs
    parser.add_argument("--energy", default=False, action="store_true")
    parser.add_argument("--optimize", default=False, action="store_true")
    parser.add_argument("--gradient", default=False, action="store_true")
    parser.add_argument("--hessian", default=False, action="store_true")
    parser.add_argument("--harmonic", default=False, action="store_true")
    parser.add_argument("--thermo", default=False, action="store_true")
    parser.add_argument("--interaction", default=False, action="store_true")
    parser.add_argument("--dens_esp", default=False, action="store_true")
    parser.add_argument("--ir", default=False, action="store_true")
    parser.add_argument("--shielding", default=False, action="store_true")
    parser.add_argument("--polarizability", default=False, action="store_true")
    parser.add_argument("--save_option", type=str, default="pkl")
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

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

    if args.ir:
        calcs.append(CALCS.IR)

    if args.shielding:
        calcs.append(CALCS.SHIELDING)

    if args.polarizability:
        calcs.append(CALCS.POLARIZABILITY)

    if args.interaction:
        calcs.append(CALCS.INTERACTION)
        extra = (args.monomer_a, args.monomer_b)

    return calcs, extra

def get_dummy_args(mol: str, calcs: list[CALCS]):
    # instead of parsing the args, trick python into thinking we have parsed the args
    class Args:
        def __init__(self):
            self.mol = mol
            self.output = "output.pkl"
            self.log_file = "pyscf.log"
            self.monomer_a = ""
            self.monomer_b = ""
            self.basis = "def2-tzvp"
            self.xc = "wB97m-v"
            self.spin = 0
            self.charge = 0 
            self.energy = CALCS.ENERGY in calcs
            self.optimize = CALCS.OPTIMIZE in calcs
            self.gradient = CALCS.GRADIENT in calcs
            self.hessian = CALCS.HESSIAN in calcs
            self.harmonic = CALCS.HARMONIC in calcs
            self.thermo = CALCS.THERMO in calcs
            self.dens_esp = CALCS.DENS_ESP in calcs
            self.ir = CALCS.IR in calcs
            self.shielding = CALCS.SHIELDING in calcs
            self.polarizability = CALCS.POLARIZABILITY in calcs
            self.interaction = CALCS.INTERACTION in calcs
            self.save_option = "pkl"

    return Args()
            



def _to_numpy(value):
    """Best-effort conversion of arrays to numpy, pass through scalars/lists, drop unsupported."""
    try:
        import cupy as _cp
    except Exception:
        _cp = None

    if isinstance(value, np.ndarray):
        return value
    if _cp is not None and isinstance(value, _cp.ndarray):
        return value.get()
    if isinstance(value, (int, float, np.number)):
        return np.array(value)
    if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, np.number)) for x in value):
        return np.asarray(value)
    return None


def save_output(output_path: str, data: dict, save_option: str = "pkl") -> None:
    """
    Save a dictionary of results. Intended for dictionaries whose values are primarily
    numpy/cupy arrays.

    Supported formats:
    - "pkl": Full pickle of the dict (allows arbitrary Python objects)
    - "npz": Only array-like entries are saved as separate arrays in a .npz
    - "hdf5": Only array-like entries saved under datasets by key

    Parquet/Feather are not supported here because they require tabular structures.
    """
    import os as _os
    import pickle as _pickle

    if _os.path.dirname(output_path) != "":
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)

    if save_option == "pkl":
        with open(output_path, "wb") as f:
            _pickle.dump(data, f, protocol=_pickle.HIGHEST_PROTOCOL)
        return

    arrays_only: dict[str, np.ndarray] = {}
    for k, v in data.items():
        arr = _to_numpy(v)
        if arr is not None:
            arrays_only[k] = arr

    if save_option == "npz":
        if len(arrays_only) == 0:
            with open(output_path, "wb") as f:
                _pickle.dump(data, f, protocol=_pickle.HIGHEST_PROTOCOL)
            return
        np.savez_compressed(output_path, **arrays_only)
        return

    if save_option == "hdf5":
        import h5py as _h5py
        with _h5py.File(output_path, "w") as h5f:
            for k, arr in arrays_only.items():
                h5f.create_dataset(str(k), data=arr)
        return

    if save_option in {"parquet", "feather"}:
        raise ValueError(
            "Parquet/Feather are not supported for arbitrary dicts of arrays. Use 'npz' or 'hdf5', or 'pkl' for full objects."
        )

    raise ValueError(f"Invalid save option: {save_option}")


if __name__ == "__main__":
    import argparse
    import json
    args = parse_args()
    calcs, extra = process_calcs(args)
    print(calcs, extra)
    output = compute_dft(args, calcs, extra)
    print(output)
    save_output(args.output, output, args.save_option)