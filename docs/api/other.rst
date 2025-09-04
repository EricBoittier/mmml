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

.. function:: mmml.transformations.tsne.apply_tsne(data, n_components=2, perplexity=30.0)

   Apply t-SNE dimensionality reduction.

   :param data: Input data array
   :param n_components: Number of components
   :param perplexity: Perplexity parameter
   :returns: Transformed data
