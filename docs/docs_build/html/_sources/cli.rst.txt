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
  - ``--ckpt_dir``: checkpoints directory

Usage
  .. code-block:: bash

     python -m mmml.cli.make_training \
       --data data/dimers.npz \
       --tag physnet_run1 \
       --num_epochs 5 \
       --batch_size 4 \
       --learning_rate 1e-3 \
       --ckpt_dir checkpoints/physnet_run1

Outputs
  - Checkpoints in ``checkpoints/physnet_run1``
  - Parameter snapshots ``paramsYYYY-mm-dd_HH-MM-SS.json``


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


