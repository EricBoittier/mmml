Command Line Interface (CLI)
============================

This page documents the main CLI entry points under ``mmml/cli`` with example
invocations, minimal input files, and Slurm submission scripts.

Prerequisites
-------------

- Python virtual environment with mmml installed and model/data dependencies available
- Access to GPU/CPU as required by the chosen command
- For Slurm examples, an HPC partition with CUDA modules or suitable CPU nodes


make_res.py
-----------

Purpose
  Create a residue (or small molecule) template and minimal inputs for subsequent steps.

Usage
  .. code-block:: bash

     python -m mmml.cli.make_res \
       --resname WAT \
       --pdb water.pdb \
       --out water_res

Inputs
  - ``--resname``: residue name (e.g., WAT, ETH, ACE)
  - ``--pdb``: input PDB containing the residue

Outputs
  - Directory ``water_res`` with processed residue files


make_box.py
-----------

Purpose
  Build boxes from residues and write PDB/PSF (or equivalent) for simulation setup.

Usage
  .. code-block:: bash

     python -m mmml.cli.make_box \
       --residue water_res \
       --count 1000 \
       --box 30 \
       --out water_box

Inputs
  - ``--residue``: residue directory from ``make_res``
  - ``--count``: number of molecules
  - ``--box``: box edge length (Å)

Outputs
  - Directory ``water_box`` with PDB (and auxiliary) files


make_training.py
----------------

Purpose
  Prepare and/or run training for a PhysNetJAX (or compatible) model.

Common flags
  - ``--data``: path to dataset (npz)
  - ``--tag``: run name tag
  - ``--model``: model definition (JSON/INP); if omitted, a default EF model is created
  - ``--n_train`` / ``--n_valid``: split sizes
  - ``--num_epochs``: number of epochs
  - ``--batch_size``: batch size
  - ``--learning_rate``: optimizer learning rate
  - ``--num_atoms``: number of atoms per structure (auto-detected from data if not specified)
  - ``--ckpt_dir``: checkpoints directory

Usage (basic - num_atoms auto-detected)
  .. code-block:: bash

     python -m mmml.cli.make_training \
       --data data/dimers.npz \
       --tag physnet_run1 \
       --num_epochs 5 \
       --batch_size 4 \
       --learning_rate 1e-3 \
       --ckpt_dir checkpoints/physnet_run1

Usage (explicit num_atoms)
  .. code-block:: bash

     python -m mmml.cli.make_training \
       --data data/dimers.npz \
       --tag physnet_run1 \
       --num_atoms 60 \
       --num_epochs 5 \
       --batch_size 4 \
       --learning_rate 1e-3 \
       --ckpt_dir checkpoints/physnet_run1

Outputs
  - Checkpoints in ``checkpoints/physnet_run1``
  - Parameter snapshots ``paramsYYYY-mm-dd_HH-MM-SS.json``

Notes
  - The ``--num_atoms`` parameter is now auto-detected from the dataset (R.shape[1])
  - You only need to specify it explicitly if auto-detection fails or you want to override it


run_sim.py
----------

Purpose
  Run a short ASE+MM/ML hybrid simulation (or energy/force evaluation) using a trained model.

Common flags
  - ``--pdbfile``: input PDB to load
  - ``--checkpoint``: path to trained model checkpoint directory
  - ``--n-monomers`` / ``--n-atoms-monomer``: topology assumptions for ML partitions
  - ``--temperature``: target temperature (K) for MD
  - ``--num-steps`` / ``--timestep``: MD length and integration step (fs)
  - ``--output-prefix``: prefix for trajectory/outputs

Usage
  .. code-block:: bash

     python -m mmml.cli.run_sim \
       --pdbfile water_box/water.pdb \
       --checkpoint checkpoints/physnet_run1 \
       --n-monomers 1000 \
       --n-atoms-monomer 3 \
       --temperature 100 \
       --timestep 0.1 \
       --num-steps 10000 \
       --output-prefix md_simulation

Outputs
  - Trajectory ``md_simulation_trajectory_100K_10000steps.traj``
  - Console logs of energy/temperature


calculator.py
-------------

Purpose
  Provides a generic ASE calculator for trained MMML models. Can be used as a Python module or from command line.

Common flags
  - ``--checkpoint``: path to checkpoint file or directory
  - ``--cutoff``: neighbor list cutoff distance (Angstroms)
  - ``--use-dcmnet-dipole``: use DCMNet dipole if available
  - ``--test-molecule``: test with predefined molecule (CO2, H2O, CH4, NH3)

Usage as module
  .. code-block:: python

     from mmml.cli.calculator import MMMLCalculator
     from ase import Atoms
     
     calc = MMMLCalculator.from_checkpoint('checkpoints/my_model')
     atoms = Atoms('CO2', positions=[[0,0,0], [1.16,0,0], [-1.16,0,0]])
     atoms.calc = calc
     
     energy = atoms.get_potential_energy()
     forces = atoms.get_forces()
     dipole = atoms.get_dipole_moment()

Usage from command line
  .. code-block:: bash

     python -m mmml.cli.calculator \
       --checkpoint checkpoints/my_model \
       --test-molecule CO2

Outputs
  - Energy, forces, dipole moment, and atomic charges for test molecule


clean_data.py
-------------

Purpose
  Clean and validate NPZ datasets by removing structures with quality issues and keeping only essential training fields.

Common flags
  - ``input``: input NPZ file to clean
  - ``-o, --output``: output NPZ file (cleaned)
  - ``--max-force``: maximum allowed force magnitude (eV/Å), default: 10.0
  - ``--min-distance``: minimum allowed interatomic distance (Å), default: 0.4
  - ``--no-check-distances``: skip distance checks (faster, recommended)
  - ``--quiet``: suppress output

Essential fields kept
  - E, F, R, Z, N: Required for energy/force training
  - D, Dxyz: Optional dipole data
  - All other fields (cube_*, orbital_*, metadata) are removed

Usage (recommended - fast, keeps 99%+ data)
  .. code-block:: bash

     python -m mmml.cli.clean_data input.npz -o cleaned.npz --no-check-distances

Usage (stricter - removes overlapping atoms)
  .. code-block:: bash

     python -m mmml.cli.clean_data input.npz -o cleaned.npz \
       --max-force 10.0 --min-distance 0.4

Custom thresholds
  .. code-block:: bash

     python -m mmml.cli.clean_data input.npz -o cleaned.npz \
       --max-force 5.0 --min-distance 0.3

Outputs
  - Cleaned NPZ file with only essential fields (E, F, R, Z, N, D, Dxyz)
  - Invalid structures removed
  - Statistics about removed structures and failure reasons

Notes
  - Use ``--no-check-distances`` for faster cleaning and higher data retention (recommended)
  - Only removes clear SCF failures, keeping good training data
  - Automatically strips unnecessary QM fields (orbital energies, cube data, etc.)


inspect_checkpoint.py
---------------------

Purpose
  Inspect model checkpoints and infer configuration from parameter structure.

Common flags
  - ``--checkpoint``: path to checkpoint file or directory
  - ``--save-config``: save inferred configuration to JSON file
  - ``--quiet``: suppress detailed output

Usage
  .. code-block:: bash

     python -m mmml.cli.inspect_checkpoint --checkpoint model/best_params.pkl

Save configuration
  .. code-block:: bash

     python -m mmml.cli.inspect_checkpoint --checkpoint model/ \\
       --save-config inferred_config.json

Outputs
  - Total parameter count
  - Parameter structure breakdown by component
  - Inferred model configuration (features, iterations, etc.)
  - Optionally saves configuration to JSON


convert_npz_traj.py
-------------------

Purpose
  Convert NPZ datasets to ASE trajectory format for visualization.

Common flags
  - ``input``: input NPZ file
  - ``-o, --output``: output trajectory file (.traj, .xyz, .pdb, etc.)
  - ``--max-structures``: maximum number of structures to convert
  - ``--stride``: use every Nth structure
  - ``--quiet``: suppress output

Usage
  .. code-block:: bash

     python -m mmml.cli.convert_npz_traj data.npz -o trajectory.traj

Convert subset
  .. code-block:: bash

     python -m mmml.cli.convert_npz_traj data.npz -o traj.traj \\
       --max-structures 100 --stride 10

To XYZ format
  .. code-block:: bash

     python -m mmml.cli.convert_npz_traj data.npz -o structures.xyz

Outputs
  - ASE trajectory file (can be viewed with ``ase gui``)
  - Removes padding automatically
  - Includes energies and forces if available


evaluate_model.py
-----------------

Purpose
  Evaluate trained models on datasets with detailed metrics (under development).

Common flags
  - ``--checkpoint``: model checkpoint directory or file
  - ``--data``: single dataset to evaluate
  - ``--train, --valid, --test``: evaluate on multiple splits
  - ``--detailed``: compute per-structure breakdown
  - ``--plots``: generate correlation and error distribution plots
  - ``--output-dir``: output directory for results

Usage
  .. code-block:: bash

     python -m mmml.cli.evaluate_model --checkpoint model/ --data test.npz

Multiple splits
  .. code-block:: bash

     python -m mmml.cli.evaluate_model --checkpoint model/ \\
       --train train.npz --valid valid.npz --test test.npz \\
       --output-dir evaluation

Outputs
  - Error metrics (MAE, RMSE, R²) for energy, forces, dipoles
  - Correlation plots (if --plots specified)
  - Per-structure analysis (if --detailed specified)


dynamics.py
-----------

Purpose
  Molecular dynamics and vibrational analysis with multiple framework support (ASE, JAX MD).

Common flags
  - ``--checkpoint``: model checkpoint directory or file
  - ``--molecule``: predefined molecule (CO2, H2O, CH4, NH3)
  - ``--structure``: load structure from file (XYZ, PDB, etc.)
  - ``--optimize``: optimize geometry
  - ``--frequencies``: calculate vibrational frequencies
  - ``--ir-spectra``: calculate IR spectrum (requires --frequencies)
  - ``--md``: run molecular dynamics
  - ``--framework``: MD framework (ase or jaxmd)
  - ``--ensemble``: MD ensemble (nve, nvt, npt)
  - ``--temperature``: temperature (K)
  - ``--timestep``: MD timestep (fs)
  - ``--nsteps``: number of MD steps
  - ``--output-dir``: output directory

Usage - Optimization
  .. code-block:: bash

     python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \\
       --optimize --output-dir co2_opt

Usage - Vibrational analysis
  .. code-block:: bash

     python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \\
       --frequencies --ir-spectra --output-dir co2_vib

Usage - Molecular dynamics (ASE)
  .. code-block:: bash

     python -m mmml.cli.dynamics --checkpoint model/ --molecule CO2 \\
       --md --framework ase --ensemble nvt --temperature 300 --nsteps 10000 \\
       --output-dir co2_md

Usage - Full workflow
  .. code-block:: bash

     python -m mmml.cli.dynamics --checkpoint model/ --structure molecule.xyz \\
       --optimize --frequencies --ir-spectra --md --nsteps 5000 \\
       --output-dir full_analysis

Outputs
  - Optimized geometries (XYZ format)
  - Vibrational frequencies and normal modes
  - IR spectra (plots and data)
  - MD trajectories (ASE trajectory format)
  - Analysis results and statistics


plot_training.py
----------------

Purpose
  Visualize training history and analyze model parameters from saved checkpoints.

Common flags
  - ``history_files``: one or more training history JSON files
  - ``--compare``: compare two training runs (requires 2 history files)
  - ``--params``: parameter pickle file(s) for analysis
  - ``--analyze-params``: analyze and plot parameter structure
  - ``--output-dir``: output directory for plots
  - ``--dpi``: DPI for output images (default: 150)
  - ``--format``: output format (png, pdf, svg, jpg)
  - ``--smoothing``: exponential smoothing factor (0-1, 0=none)
  - ``--summary-only``: only print text summary, no plots

Usage single model
  .. code-block:: bash

     python -m mmml.cli.plot_training \
       checkpoints/my_model/history.json \
       --output-dir plots --dpi 300

Usage comparison
  .. code-block:: bash

     python -m mmml.cli.plot_training \
       model1/history.json model2/history.json \
       --compare --names "Model A" "Model B" \
       --smoothing 0.9

With parameter analysis
  .. code-block:: bash

     python -m mmml.cli.plot_training history.json \
       --params best_params.pkl \
       --analyze-params

Outputs
  - Training history plots showing loss curves and metrics
  - Parameter analysis plots (if requested)
  - Text summary of training performance


Minimal example files
---------------------

Model args (EF) JSON (if constructing a default model)::

  {
    "features": 64,
    "max_degree": 0,
    "num_basis_functions": 32,
    "num_iterations": 2,
    "n_res": 2,
    "cutoff": 8.0,
    "max_atomic_number": 28,
    "zbl": false,
    "efa": false
  }

Dataset layout
  - Single ``npz`` file with arrays at least: ``R`` (positions), ``Z`` (atomic numbers), ``E`` (energies), ``F`` (forces)

Minimal Slurm scripts
---------------------

Training (1 GPU)
  .. code-block:: bash

     #!/bin/bash
     #SBATCH -J mmml-train
     #SBATCH -A your_account
     #SBATCH -p gpu
     #SBATCH -N 1
     #SBATCH -c 8
     #SBATCH --gres=gpu:1
     #SBATCH -t 12:00:00
     #SBATCH -o slurm-%j.out

     module load cuda/12.1  # if needed
     source /path/to/venv/bin/activate

     srun python -m mmml.cli.make_training \
       --data /path/to/data.npz \
       --tag physnet_run1 \
       --num_epochs 20 \
       --batch_size 8 \
       --learning_rate 1e-3 \
       --ckpt_dir /scratch/$USER/mmml_checkpoints/physnet_run1

MD run (CPU or GPU)
  .. code-block:: bash

     #!/bin/bash
     #SBATCH -J mmml-md
     #SBATCH -A your_account
     #SBATCH -p gpu
     #SBATCH -N 1
     #SBATCH -c 8
     #SBATCH --gres=gpu:1
     #SBATCH -t 02:00:00
     #SBATCH -o slurm-%j.out

     module load cuda/12.1  # if needed
     source /path/to/venv/bin/activate

     srun python -m mmml.cli.run_sim \
       --pdbfile /path/to/box.pdb \
       --checkpoint /scratch/$USER/mmml_checkpoints/physnet_run1 \
       --n-monomers 1000 \
       --n-atoms-monomer 3 \
       --temperature 100 \
       --timestep 0.1 \
       --num-steps 10000 \
       --output-prefix md_simulation

Debug (short) job
  .. code-block:: bash

     #!/bin/bash
     #SBATCH -J mmml-debug
     #SBATCH -A your_account
     #SBATCH -p debug
     #SBATCH -N 1
     #SBATCH -c 4
     #SBATCH -t 00:10:00
     #SBATCH -o slurm-%j.out

     source /path/to/venv/bin/activate
     srun python -m mmml.cli.make_res --resname WAT --pdb water.pdb --out water_res


Notes
-----

- For reproducible results, set seeds where provided by flags.
- Ensure the box size in ``run_sim.py`` is physically reasonable for your system.
- If running on CPU-only nodes, remove CUDA module loads.
- The ``calculator.py`` module provides a generic interface that automatically detects model types.
- Use ``plot_training.py`` to visualize and compare training runs from JSON history files.
- All CLI tools support ``--help`` for detailed usage information.


