#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.mport numpy

from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr, Debye
# from pyscf.prop.polarizability.uhf import polarizability, Polarizability
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import jsonpickle


import pyscf
import time
import argparse
from pyscf import lib
from gpu4pyscf import dft
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
# from pyscf4gpuInterface.aseInterface import PYSCF, parameters


def run_dft(ATOMS, BASIS, AUGBASIS, SOLVENT, XC, do_forces=True):
    """Run DFT calculation with optional solvent model.

    Args:
        ATOMS: Atomic coordinates and species
        BASIS: Basis set specification
        SOLVENT: Whether to use solvent model
        XC: Exchange-correlation functional

    Returns:
        tuple: Total energy and nuclear gradients
    """
    mol = pyscf.M(
        atom=ATOMS,
        basis=BASIS,
        max_memory=32000,
        verbose=1,  # Set verbose >= 6 for debugging timer
    )

    mf_df = dft.RKS(mol, xc=XC).density_fit(auxbasis=AUGBASIS)
    mf_df.verbose = 1 

    if SOLVENT:
        mf_df = mf_df.PCM()
        mf_df.with_solvent.lebedev_order = 29
        mf_df.with_solvent.method = args.solvent
        mf_df.with_solvent.eps = 78.3553

    # Set grid parameters
    mf_df.grids.atom_grid = (99, 590)
    if mf_df._numint.libxc.is_nlc(mf_df.xc):
        mf_df.nlcgrids.atom_grid = (50, 194)

    # Set convergence parameters
    mf_df.direct_scf_tol = 1e-14
    mf_df.conv_tol = 1e-10
    mf_df.conv_tol_cpscf = 1e-6
    mf_df.chkfile = None

    # Run SCF
    e_tot = mf_df.kernel()

    forces = None
    if do_forces:  
        # Calculate nuclear gradients
        g = mf_df.nuc_grad_method()
        g.auxbasis_response = True
        forces = g.kernel()

    return e_tot, forces


class parameters:
    # holds the calculation mode and user-chosen attributes of post-HF objects
    def __init__(self):
        self.mode = "hf"
        self.basis = "cc-pVDZ"
        self.xc = None  # For DFT calculations
        self.auxbasis = None  # For density fitting
        self.solvent = None  # For PCM calculations
        self.verbose = 1 

    def show(self):
        print("------------------------")
        print("calculation-specific parameters set by the user")
        print("------------------------")
        for v in vars(self):
            print("{}:  {}".format(v, vars(self)[v]))
        print("\n\n")


def todict(x):
    return jsonpickle.encode(x, unpicklable=True)


def init_geo(mf, atoms):
    # convert ASE structural information to PySCF information
    if atoms.pbc.any():
        cell = mf.cell.copy()
        cell.atom = atoms_from_ase(atoms)
        cell.a = atoms.cell.copy()
        cell.build()
        mf.reset(cell=cell.copy())
    else:
        mol = mf.mol.copy()
        mol.atom = atoms_from_ase(atoms)
        mol.build()
        mf.reset(mol=mol.copy())


class PYSCF(Calculator):
    # based on PySCF ASE calculator by Jakob Kraus
    # units:  ase         -> units [eV,Angstroem,eV/Angstroem,e*A,A**3]
    #         pyscf       -> units [Ha,Bohr,Ha/Bohr,Debye,Bohr**3]

    implemented_properties = [
        "energy",
        #"forces",
        #"dipole",
        #'polarizability'
    ]

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=False,
        label="PySCF",
        atoms=None,
        directory=".",
        **kwargs,
    ):
        # constructor
        Calculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, directory, **kwargs
        )
        self.initialize(**kwargs)

    def initialize(self, mf=None, p=None):
        # attach the mf object to the calculator
        # add the todict functionality to enable ASE trajectories:
        # https://github.com/pyscf/pyscf/issues/624
        self.mf = mf
        self.p = p if p is not None else parameters()
        self.mf.todict = lambda: todict(self.mf)
        self.p.todict = lambda: todict(self.p)

    def set(self, **kwargs):
        # allow for a calculator reset
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def get_polarizability(self, atoms=None):
        return self.get_property("polarizability", atoms)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        if self.p.mode.lower() == "dft":
            # Use GPU4PySCF DFT
            e_tot, forces = run_dft(
                atoms_from_ase(atoms), self.p.basis, self.p.auxbasis, bool(self.p.solvent), self.p.xc
            )
            self.results["energy"] = e_tot * Ha
            if "forces" in properties:
                self.results["forces"] = -forces * (Ha / Bohr)

            # # Create mf object for other properties
            # mol = pyscf.M(atom=atoms_from_ase(atoms), basis=self.p.basis)
            # self.mf = dft.RKS(mol, xc=self.p.xc)
            # self.mf.kernel()
        else:
            # Regular PySCF calculation
            init_geo(self.mf, atoms)

            if hasattr(self.mf, "_scf"):
                self.mf._scf.kernel()
                self.mf.__init__(self.mf._scf)
                for v in vars(self.p):
                    if v != "mode":
                        setattr(self.mf, v, vars(self.p)[v])
            self.mf.kernel()
            e = self.mf.e_tot

            if self.p.mode.lower() == "ccsd(t)":
                e += self.mf.ccsd_t()

            self.results["energy"] = e * Ha

            if "forces" in properties:
                gf = self.mf.nuc_grad_method()
                gf.verbose = self.mf.verbose
                forces = -1.0 * gf.kernel() * (Ha / Bohr)
                self.results["forces"] = forces

        # Calculate dipole and polarizability for both cases
        if "dipole" in properties:
            if hasattr(self.mf, "_scf"):
                self.results["dipole"] = (
                    self.mf._scf.dip_moment(verbose=self.mf._scf.verbose) * Debye
                )
            else:
                self.results["dipole"] = (
                    self.mf.dip_moment(verbose=self.mf.verbose) * Debye
                )

        if "polarizability" in properties:
            if hasattr(self.mf, "_scf"):
                self.results["polarizability"] = Polarizability(
                    self.mf._scf
                ).polarizability() * (Bohr**3)
            else:
                self.results["polarizability"] = Polarizability(
                    self.mf
                ).polarizability() * (Bohr**3)


def main():
    parser = argparse.ArgumentParser(description='Run PySCF calculations on structures from an ASE trajectory')
    parser.add_argument('trajectory', help='Path to ASE trajectory file')
    parser.add_argument('--output', default='output.traj',
                       help='Output trajectory file (default: output.traj)')
    parser.add_argument('--method', choices=['dft', 'hf'], default='dft',
                       help='Calculation method (default: dft)')
    parser.add_argument('--basis', default='cc-pVTZ',
                       help='Basis set (default: cc-pVTZ)')
    parser.add_argument('--xc', default='wB97m-v',
                       help='Exchange-correlation functional for DFT (default: wB97m-v)')
    parser.add_argument('--auxbasis', default='def2-tzvp-jkfit',
                       help='Auxiliary basis set for density fitting (default: def2-tzvp-jkfit)')
    parser.add_argument('--solvent', choices=['cosmo', 'pcm'], 
                       help='Solvent model (optional)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    args = parser.parse_args()

    # Read trajectory
    from ase.io import read, write
    atoms_list = read(args.trajectory, ':')
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    # Process each structure and write to trajectory
    print(f"\nProcessing {len(atoms_list)} structures using {args.method.upper()}/{args.basis}")
    
    for i, atoms in enumerate(atoms_list):
        # Create fresh calculator for each structure
        if args.method == 'dft':
            mol = pyscf.M(atom=atoms_from_ase(atoms), 
                         basis=args.basis, 
                         spin=0, 
                         charge=0)
            mf = dft.RKS(mol, xc=args.xc)
        else:
            mol = pyscf.M(atom=atoms_from_ase(atoms),
                         basis=args.basis,
                         spin=0,
                         charge=0)
            mf = mol.HF()
            mf.verbose = args.verbose

        # Set up parameters
        p = parameters()
        p.mode = args.method
        p.basis = args.basis
        p.verbose = args.verbose
        if args.method == 'dft':
            p.xc = args.xc
            p.auxbasis = args.auxbasis
            p.solvent = args.solvent

        calc = PYSCF(mf=mf, p=p)
        atoms.calc = calc
        
        # Run calculation
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(f"\nStructure {i+1}:")
        print(f"Energy: {energy:.6f} eV")
        print(f"Forces (eV/Å):\n{forces}")
        
        # Write structure to trajectory
        if i == 0:
            print(f"Writing structure {i+1} to trajectory")
            write(args.output, atoms, format='traj')
        else:
            print(f"Appending structure {i+1} to trajectory")
            write(args.output, atoms, format='traj', append=True)
            
        # Force garbage collection
        import gc
        gc.collect()

    print(f"\nCalculation results written to: {args.output}")

if __name__ == '__main__':
    main()
