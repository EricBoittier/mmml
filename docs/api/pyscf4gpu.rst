PySCF GPU Interface API
=======================

GPU-accelerated PySCF interface for quantum chemistry calculations.

Main Functions
--------------

.. function:: mmml.pyscf4gpuInterface.calcs.compute_dft(args, calcs, extra=None)

   Compute DFT properties using GPU4PySCF.

   :param args: Parsed command line arguments or dummy args object
   :param calcs: List of calculation types to perform
   :param extra: Extra arguments for specific calculations
   :returns: Dictionary containing calculation results

.. function:: mmml.pyscf4gpuInterface.calcs.setup_mol(atoms, basis, xc, spin, charge, log_file='./pyscf.log', verbose=6, lebedev_grids=(99,590), scf_tol=1e-10, scf_max_cycle=50, cpscf_tol=1e-3, conv_tol=1e-10, conv_tol_cpscf=1e-3)

   Set up molecular system and GPU4PySCF engine.

   :param atoms: Molecular geometry (string or PySCF mol object)
   :param basis: Basis set name
   :param xc: Exchange-correlation functional
   :param spin: Spin multiplicity
   :param charge: Total charge
   :param log_file: Log file path
   :param verbose: Verbosity level
   :param lebedev_grids: Grid settings (nrad, nleb)
   :param scf_tol: SCF convergence tolerance
   :param scf_max_cycle: Maximum SCF iterations
   :param cpscf_tol: CPSCF convergence tolerance
   :param conv_tol: General convergence tolerance
   :param conv_tol_cpscf: CPSCF convergence tolerance
   :returns: Tuple of (engine, mol)

Calculation Types
-----------------

Available calculation types (from mmml.pyscf4gpuInterface.enums.CALCS):

- ENERGY: Compute total energy
- OPTIMIZE: Geometry optimization
- GRADIENT: Nuclear gradients
- HESSIAN: Second derivatives
- HARMONIC: Harmonic vibrational analysis
- THERMO: Thermodynamic properties
- DENS_ESP: Density and ESP on grid
- IR: Infrared frequencies and intensities
- SHIELDING: NMR shielding tensors
- POLARIZABILITY: Polarizability tensor
- INTERACTION: Interaction energies

Command Line Interface
-----------------------

.. code-block:: bash

   python -m mmml.pyscf4gpuInterface.calcs \
     --mol "O 0.000 0.000 0.000; H 0.000 0.757 0.586; H 0.000 -0.757 0.586" \
     --basis def2-tzvp \
     --xc wB97m-v \
     --spin 0 \
     --charge 0 \
     --energy \
     --dens_esp \
     --output results.pkl

Output Format
-------------

The compute_dft function returns a dictionary containing:

- mol: PySCF molecular object
- calcs: List of performed calculations
- energy: Total energy (if ENERGY in calcs)
- esp: ESP values on grid (if DENS_ESP in calcs)
- esp_grid: Grid coordinates (if DENS_ESP in calcs)
- R: Nuclear coordinates in Angstrom
- Z: Nuclear charges
- gradient: Nuclear gradients (if GRADIENT in calcs)
- hessian: Hessian matrix (if HESSIAN in calcs)
- freq: Vibrational frequencies (if HARMONIC in calcs)
- thermo: Thermodynamic properties (if THERMO in calcs)

Prerequisites
-------------

- CUDA-compatible GPU
- gpu4pyscf package
- PySCF
- Optional: ASE for geometry optimization
