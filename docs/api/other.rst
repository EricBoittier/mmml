Other Interfaces API
====================

Additional interfaces and utilities in mmml.

ASE Interface
-------------

.. function:: mmml.aseInterface.mmml_ase.MMMLCalculator(atoms, model_params, cutoff=4.0)

   ASE calculator for MMML models.

   :param atoms: ASE Atoms object
   :param model_params: Trained model parameters
   :param cutoff: Distance cutoff
   :returns: ASE calculator instance

PyCharmm Interface
------------------

.. note::
   These interfaces require the optional ``charmm-interface`` extra and a locally built ``pycharmm`` installation from ``setup/charmm/tool/pycharmm``.

.. function:: mmml.pycharmmInterface.dyna.run_dynamics(psf_file, pdb_file, res_file, output_prefix, num_steps=1000, dt=0.001, temperature=300.0)

   Run molecular dynamics with PyCharmm.

   :param psf_file: PSF topology file
   :param pdb_file: PDB coordinate file
   :param res_file: RES parameter file
   :param output_prefix: Output file prefix
   :param num_steps: Number of MD steps
   :param dt: Time step in ps
   :param temperature: Temperature in K
   :returns: None

.. function:: mmml.pycharmmInterface.setupBox.setup_box(psf_file, pdb_file, res_file, output_prefix, box_size=50.0, water_model='TIP3')

   Set up simulation box with solvent.

   :param psf_file: PSF topology file
   :param pdb_file: PDB coordinate file
   :param res_file: RES parameter file
   :param output_prefix: Output file prefix
   :param box_size: Box size in Angstrom
   :param water_model: Water model name
   :returns: None

OpenMM Interface
----------------

.. function:: mmml.openmmInterface.interface.setup_openmm_system(pdb_file, forcefield='amber14-all.xml', water_model='tip3p')

   Set up OpenMM system.

   :param pdb_file: PDB coordinate file
   :param forcefield: Force field XML file
   :param water_model: Water model name
   :returns: OpenMM system object

JAX-MD Interface
----------------

.. function:: mmml.jaxmdInterface.jaxmdInterface.run_jaxmd_simulation(positions, masses, forces, num_steps=1000, dt=0.001, temperature=300.0)

   Run molecular dynamics with JAX-MD.

   :param positions: Initial positions
   :param masses: Atomic masses
   :param forces: Force function
   :param num_steps: Number of MD steps
   :param dt: Time step
   :param temperature: Temperature
   :returns: Trajectory array

Data Processing
----------------

.. function:: mmml.io.parseCharmmOutput.parse_dcd(dcd_file)

   Parse CHARMM DCD trajectory file.

   :param dcd_file: DCD file path
   :returns: Trajectory data

.. function:: mmml.io.parseOpenMMOutput.parse_dcd(dcd_file)

   Parse OpenMM DCD trajectory file.

   :param dcd_file: DCD file path
   :returns: Trajectory data

Visualization
-------------

.. function:: mmml.visualize.ase_x3d.write_x3d(atoms, filename, trajectory=None)

   Write ASE atoms to X3D format.

   :param atoms: ASE Atoms object
   :param filename: Output file path
   :param trajectory: Optional trajectory data
   :returns: None

.. function:: mmml.plotting.esp.plot_esp_surface(esp_data, grid_coords, output_file)

   Plot ESP on molecular surface.

   :param esp_data: ESP values
   :param grid_coords: Grid coordinates
   :param output_file: Output file path
   :returns: None

Utilities
---------

.. function:: mmml.transformations.pca.apply_pca(data, n_components=2)

   Apply PCA dimensionality reduction.

   :param data: Input data array
   :param n_components: Number of components
   :returns: Transformed data

PyCharmm MMML Calculator
------------------------

High-level calculator that couples ML and MM terms with smooth switching for monomer/dimer systems.

.. class:: mmml.pycharmmInterface.mmml_calculator.CutoffParameters(ml_cutoff=2.0, mm_switch_on=5.0, mm_cutoff=1.0)

   Parameters controlling ML/MM switching distances.

   :param ml_cutoff: Distance where ML potential is cut off
   :param mm_switch_on: Distance where MM potential starts switching on
   :param mm_cutoff: Final cutoff for MM potential

.. class:: mmml.pycharmmInterface.mmml_calculator.ModelOutput(energy, forces, dH, internal_E, internal_F, mm_E, mm_F, ml_2b_E, ml_2b_F)

   Structured output for energies and forces.

   :ivar energy: Total energy (kcal/mol)
   :ivar forces: Forces (kcal/mol/Å)
   :ivar dH: Interaction energy
   :ivar internal_E: Sum of monomer energies
   :ivar internal_F: Monomer forces
   :ivar mm_E: Classical MM energy
   :ivar mm_F: Classical MM forces
   :ivar ml_2b_E: ML two-body interaction energy
   :ivar ml_2b_F: ML two-body interaction forces

.. function:: mmml.pycharmmInterface.mmml_calculator.prepare_batches_md(data, batch_size, data_keys=None, num_atoms=60, dst_idx=None, src_idx=None, include_id=False, debug_mode=False)

   Prepare batched inputs for the underlying JAX model with precomputed indices and masks.

   :param data: Dataset with keys like 'R', 'Z', 'N', optionally 'F', 'E', etc.
   :param batch_size: Batch size
   :param data_keys: Keys to include; defaults to all
   :param num_atoms: Max atoms per system
   :param dst_idx: Optional destination indices for pairs
   :param src_idx: Optional source indices for pairs
   :param include_id: Include 'id' if present
   :param debug_mode: Extra checks/assertions
   :returns: List of batch dictionaries

.. function:: mmml.pycharmmInterface.mmml_calculator.setup_calculator(ATOMS_PER_MONOMER, N_MONOMERS=2, ml_cutoff_distance=2.0, mm_switch_on=5.0, mm_cutoff=1.0, doML=True, doMM=True, doML_dimer=True, debug=False, ep_scale=None, sig_scale=None, model_restart_path=None, MAX_ATOMS_PER_SYSTEM=100)

   Build a configured calculator factory that computes energies and forces combining ML and MM with switching.

   :param ATOMS_PER_MONOMER: Number of atoms in a monomer
   :param N_MONOMERS: Number of monomers in the system
   :param ml_cutoff_distance: ML cutoff distance
   :param mm_switch_on: Distance where MM switches on
   :param mm_cutoff: MM cutoff distance
   :param doML: Include ML term
   :param doMM: Include MM term
   :param doML_dimer: Include ML two-body interactions
   :param debug: Enable verbose debug
   :param ep_scale: Optional epsilon scaling per atom type
   :param sig_scale: Optional sigma scaling per atom type
   :param model_restart_path: Path to trained model checkpoint
   :param MAX_ATOMS_PER_SYSTEM: Padding size for batching
   :returns: A factory function to create an ASE calculator and the core compute function

.. function:: mmml.transformations.tsne.apply_tsne(data, n_components=2, perplexity=30.0)

   Apply t-SNE dimensionality reduction.

   :param data: Input data array
   :param n_components: Number of components
   :param perplexity: Perplexity parameter
   :returns: Transformed data
