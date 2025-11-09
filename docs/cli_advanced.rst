Advanced Training Tools
======================

This document covers advanced training CLI tools for specialized use cases.

Overview
--------

MMML provides several advanced training tools beyond the basic ``make_training.py``:

- **train_joint.py** - Joint PhysNet+DCMNet for ESP prediction
- **train_memmap.py** - Training with memory-mapped datasets (large-scale)
- **train_charge_spin.py** - Multi-state training with charge/spin conditioning
- **test_deps.py** - Test optional dependencies

train_joint.py
--------------

Joint PhysNet+DCMNet training for electrostatic potential (ESP) prediction.

**Purpose:** Train combined PhysNet+DCMNet models for accurate ESP prediction

**Architecture Options:**

1. **DCMNet (Equivariant, Default)**
   
   - PhysNet → Atomic Charges → DCMNet → Distributed Multipoles → ESP
   - Fully rotationally equivariant
   - Uses spherical harmonics

2. **Non-Equivariant**
   
   - PhysNet → Atomic Charges → MLP → Cartesian Displacements → ESP
   - Faster but breaks rotational equivariance
   - Use ``--use-noneq-model``

**Key Features:**

- Multiple optimizer support (Adam, AdamW, RMSprop, Muon)
- Automatic hyperparameter recommendations
- Flexible loss configuration (JSON/YAML)
- Learnable charge orientation mixing
- Exponential moving average (EMA)
- Comprehensive validation plots

**Basic Usage:**

.. code-block:: bash

    # Train DCMNet (equivariant)
    python -m mmml.cli.train_joint \
        --train-efd train_efd.npz \
        --train-esp train_esp.npz \
        --valid-efd valid_efd.npz \
        --valid-esp valid_esp.npz \
        --epochs 100 \
        --batch-size 4

**Advanced Usage:**

.. code-block:: bash

    # Non-equivariant model with Muon optimizer
    python -m mmml.cli.train_joint \
        --train-efd train_efd.npz \
        --train-esp train_esp.npz \
        --valid-efd valid_efd.npz \
        --valid-esp valid_esp.npz \
        --use-noneq-model \
        --optimizer muon \
        --use-recommended-hparams \
        --epochs 100

**Data Format:**

Training requires two NPZ files per split:

1. **EFD file** (energies/forces/dipoles):
   
   - ``E``: Energies (eV)
   - ``F``: Forces (eV/Å)
   - ``Dxyz`` or ``D``: Dipoles (e·Å)

2. **ESP file** (electrostatic grids):
   
   - ``R``: Atom positions (Å)
   - ``Z``: Atomic numbers
   - ``N``: Number of atoms per structure
   - ``esp``: ESP values (Hartree/e)
   - ``vdw_surface``: Grid points (Å)

**Loss Configuration:**

Custom loss configuration via JSON/YAML:

.. code-block:: json

    {
      "dipole": [
        {"source": "physnet", "weight": 25.0, "metric": "l2"},
        {"source": "mixed", "weight": 10.0, "metric": "mae"}
      ],
      "esp": [
        {"source": "dcmnet", "weight": 10000.0, "metric": "l2"}
      ]
    }

.. code-block:: bash

    python -m mmml.cli.train_joint \
        --train-efd train_efd.npz \
        --train-esp train_esp.npz \
        --valid-efd valid_efd.npz \
        --valid-esp valid_esp.npz \
        --loss-config custom_loss.json

**Important Flags:**

.. code-block:: text

    Model Architecture:
      --use-noneq-model              Use non-equivariant model (faster, not equivariant)
      --physnet-features INT         PhysNet features (default: 64)
      --physnet-iterations INT       PhysNet iterations (default: 3)
      --dcmnet-features INT          DCMNet features (default: 128)
      --dcmnet-iterations INT        DCMNet iterations (default: 2)
      --n-dcm INT                    Distributed charges per atom (default: 3)
      --max-degree INT               Spherical harmonic degree (default: 2)

    Optimization:
      --optimizer {adam,adamw,rmsprop,muon}  Optimizer choice (default: adamw)
      --learning-rate FLOAT          Learning rate (auto-selected if not specified)
      --weight-decay FLOAT           Weight decay (auto-selected if not specified)
      --use-recommended-hparams      Auto-tune hyperparameters based on dataset

    Loss Configuration:
      --energy-weight FLOAT          Energy loss weight (default: 10.0)
      --forces-weight FLOAT          Forces loss weight (default: 50.0)
      --dipole-weight FLOAT          Dipole loss weight (default: 25.0)
      --esp-weight FLOAT             ESP loss weight (default: 10000.0)
      --mono-weight FLOAT            Monopole constraint weight (default: 100.0)
      --loss-config PATH             JSON/YAML loss configuration file
      --esp-min-distance FLOAT       Min distance from atoms for ESP points (Å)
      --esp-max-value FLOAT          Max |ESP| to include (Ha/e)

    Training:
      --batch-size INT               Batch size (default: 1)
      --epochs INT                   Number of epochs (default: 100)
      --grad-clip-norm FLOAT         Gradient clipping (default: 1.0)
      --restart PATH                 Restart from checkpoint

    Visualization:
      --plot-freq INT                Plot validation every N epochs (default: 10)
      --plot-results                 Create final validation plots
      --plot-samples INT             Number of samples to plot (default: 100)
      --plot-esp-examples INT        Number of ESP examples to plot (default: 2)

**Outputs:**

- ``checkpoints/<name>/best_params.pkl`` - Best model parameters (EMA)
- ``checkpoints/<name>/model_config.pkl`` - Model configuration
- ``checkpoints/<name>/history.json`` - Training history
- ``checkpoints/<name>/plots/`` - Validation plots (if enabled)

**Example Training Workflow:**

.. code-block:: bash

    # 1. Train with default settings
    python -m mmml.cli.train_joint \
        --train-efd train_efd.npz \
        --train-esp train_esp.npz \
        --valid-efd valid_efd.npz \
        --valid-esp valid_esp.npz \
        --name co2_esp_model \
        --epochs 100

    # 2. Check plots in checkpoints/co2_esp_model/plots/

    # 3. Continue training if needed
    python -m mmml.cli.train_joint \
        --train-efd train_efd.npz \
        --train-esp train_esp.npz \
        --valid-efd valid_efd.npz \
        --valid-esp valid_esp.npz \
        --name co2_esp_model \
        --restart checkpoints/co2_esp_model \
        --epochs 200

train_memmap.py
---------------

Train PhysNet on large-scale memory-mapped datasets (e.g., OpenQDC).

**Purpose:** Efficiently train on datasets too large to fit in RAM

**Features:**

- Memory-mapped data loading (no RAM limits)
- Bucketed batching (minimizes padding)
- Automatic train/validation split
- Compatible with OpenQDC packed format

**Usage:**

.. code-block:: bash

    python -m mmml.cli.train_memmap \
        --data_path openqdc_packed_memmap \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 0.001 \
        --num_atoms 60

**Data Format:**

Memory-mapped directory with:

- ``offsets.npy`` - Molecule start indices
- ``n_atoms.npy`` - Atoms per molecule
- ``Z_pack.int32`` - Atomic numbers (packed)
- ``R_pack.f32`` - Positions (packed, Å)
- ``F_pack.f32`` - Forces (packed, eV/Å)
- ``E.f64`` - Energies (eV)
- ``Qtot.f64`` - Total charges

**Important Flags:**

.. code-block:: text

    --data_path PATH          Path to packed memmap directory
    --valid_split FLOAT       Validation fraction (default: 0.1)
    --batch_size INT          Batch size (default: 32)
    --num_epochs INT          Number of epochs (default: 100)
    --learning_rate FLOAT     Learning rate (default: 0.001)
    --num_atoms INT           Max atoms per molecule (default: 60)
    --bucket_size INT         Bucket size for sorting (default: 8192)
    --features INT            Hidden features (default: 128)
    --num_iterations INT      Message passing iterations (default: 3)
    --cutoff FLOAT            Cutoff distance in Å (default: 5.0)

**Outputs:**

- ``checkpoints/<name>/epoch-<N>/`` - Orbax checkpoints
- Saves best model based on validation loss

train_charge_spin.py
--------------------

Train PhysNet with charge and spin state conditioning for multi-state predictions.

**Purpose:** Train models that can predict properties for different charge/spin states

**Features:**

- Accepts total charge and spin multiplicity as inputs
- Enables prediction for ions and excited states
- Memory-mapped data support
- Charge embedding (default: 16-dim)
- Spin embedding (default: 16-dim)

**Usage:**

.. code-block:: bash

    python -m mmml.cli.train_charge_spin \
        --data_path openqdc_packed_memmap \
        --batch_size 32 \
        --num_epochs 100 \
        --charge_min -2 \
        --charge_max 2 \
        --spin_min 1 \
        --spin_max 5

**Data Requirements:**

Same as ``train_memmap.py`` but should include:

- ``total_charge`` field (if available in data)
- ``total_spin`` field (if available in data)

If not present, defaults to:
- Charge = 0 (neutral)
- Spin = 1 (singlet)

**Important Flags:**

.. code-block:: text

    --charge_min INT          Minimum charge to support (default: -5)
    --charge_max INT          Maximum charge to support (default: 5)
    --spin_min INT            Minimum spin multiplicity (default: 1, singlet)
    --spin_max INT            Maximum spin multiplicity (default: 7, septet)
    --charge_embed_dim INT    Charge embedding dimension (default: 16)
    --spin_embed_dim INT      Spin embedding dimension (default: 16)

**Example Workflow:**

.. code-block:: bash

    # Train for neutral and ±1 charge states, singlet to triplet
    python -m mmml.cli.train_charge_spin \
        --data_path my_data_memmap \
        --batch_size 16 \
        --num_epochs 50 \
        --charge_min -1 \
        --charge_max 1 \
        --spin_min 1 \
        --spin_max 3 \
        --name molecule_multistates

**Outputs:**

- ``checkpoints/<name>/epoch-<N>/`` - Orbax checkpoints with charge/spin model

test_deps.py
------------

Test optional dependencies and verify MMML installation.

**Purpose:** Check which optional features are available

**Usage:**

.. code-block:: bash

    python -m mmml.cli.test_deps

**Output:**

.. code-block:: text

    ======================================================================
    TESTING CORE IMPORTS
    ======================================================================
    ✅ Model creation works
    ✅ Restart utilities work
    ✅ Data loading utilities work
    ✅ Training utilities work

    ======================================================================
    CHECKING OPTIONAL DEPENDENCIES
    ======================================================================
    ✅ asciichartpy: True
    ✅ polars: True
    ✅ tensorboard: True
    ✅ tensorflow: True

    ======================================================================
    SUMMARY
    ======================================================================

    Core Functionality: 4/4 tests passed
    ✅ All core functionality working!

    Optional Dependencies:
      ✅ Plotting support (asciichartpy + polars)
      ✅ TensorBoard support (tensorboard + tensorflow + polars)

    ✨ All optional features available!

**Recommendations:**

If optional dependencies are missing, the tool suggests:

.. code-block:: bash

    # For plotting support
    pip install -e '.[plotting]'
    
    # For TensorBoard support
    pip install -e '.[tensorboard]'
    
    # For all optional features
    pip install -e '.[plotting,tensorboard]'

Comparison Table
----------------

+-------------------+-------------------+-------------------+-------------------+
| Feature           | train_joint       | train_memmap      | train_charge_spin |
+===================+===================+===================+===================+
| ESP prediction    | ✅                | ❌                | ❌                |
+-------------------+-------------------+-------------------+-------------------+
| Memory-mapped     | ❌                | ✅                | ✅                |
+-------------------+-------------------+-------------------+-------------------+
| Multi-state       | ❌                | ❌                | ✅                |
+-------------------+-------------------+-------------------+-------------------+
| Large datasets    | ❌ (RAM limited)  | ✅                | ✅                |
+-------------------+-------------------+-------------------+-------------------+
| Equivariance      | ✅ (optional)     | ✅                | ✅                |
+-------------------+-------------------+-------------------+-------------------+
| Visualization     | ✅ Comprehensive  | ❌                | ❌                |
+-------------------+-------------------+-------------------+-------------------+

When to Use Each Tool
---------------------

**Use train_joint.py when:**

- You need ESP prediction (electrostatics)
- You have E/F/D/ESP data available
- Dataset fits in RAM (~10k structures)
- You want comprehensive validation plots
- You're studying electrostatics, solvation, docking

**Use train_memmap.py when:**

- Dataset is too large for RAM (>100k structures)
- You only need E/F prediction (no ESP/dipoles)
- You have OpenQDC or similar packed format
- Training on HPC clusters with limited memory

**Use train_charge_spin.py when:**

- You need multi-state predictions (ions, excited states)
- Dataset includes charge/spin information
- Studying charged species or spin states
- Dataset is memory-mapped

**Use make_training.py when:**

- Standard E/F/D training
- Dataset fits in RAM
- Simple NPZ format
- Quick prototyping

See Also
--------

- **cli.rst** - Basic CLI tools documentation
- **examples/co2/** - CO2 training examples with ESP
- **examples/glycol/** - Standard training workflow

