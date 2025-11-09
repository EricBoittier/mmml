"""
Example notebook code for using training_helpers.py

Copy-paste this into a Jupyter notebook cell to use the training helpers.
"""

# ============================================================================
# Setup: Import required modules
# ============================================================================
import sys
from pathlib import Path

# Add project to path if needed
PROJECT_ROOT = Path("/pchem-data/meuwly/boittier/home/mmml")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmml.cli.training_helpers import TrainingConfig, prepare_training, run_training

# ============================================================================
# Example 1: Basic usage with TrainingConfig dataclass
# ============================================================================

# Create configuration
config = TrainingConfig(
    # Data paths
    data="/scicore/home/meuwly/boitti0000/data/vibml/md17_data/hono_mp2_avtz_6406.npz",
    n_train=1000,
    n_valid=100,
    num_atoms=20,
    
    # Model architecture
    features=64,
    max_degree=0,
    num_basis_functions=32,
    num_iterations=2,
    n_res=2,
    cutoff=8.0,
    max_atomic_number=28,
    
    # Training hyperparameters
    seed=42,
    batch_size=1,
    num_epochs=10,
    learning_rate=0.001,
    energy_weight=1.0,
    forces_weight=52.91,
    
    # Output settings
    tag="hono_mp2_experiment",
    ckpt_dir=Path("checkpoints"),
    verbose=True,
)

# Prepare training setup (loads data, creates model)
setup = prepare_training(config)

# Inspect what we got
print("Training data keys:", list(setup['train_data'].keys()))
print("Validation samples:", len(setup['valid_data']['E']))
print("Model attributes:", setup['model_attrs'])

# Run training
results = run_training(config, setup=setup)

print("\nTraining complete!")
print(f"Final parameters saved to: {results.get('params_file', 'N/A')}")

# ============================================================================
# Example 2: Using a dictionary instead of TrainingConfig
# ============================================================================

config_dict = {
    "data": "/scicore/home/meuwly/boitti0000/data/vibml/md17_data/ch2o_mp2_avtz_3601.npz",
    "n_train": 500,
    "n_valid": 50,
    "num_atoms": 15,
    "num_epochs": 5,
    "tag": "ch2o_quick_test",
    "verbose": True,
}

# This works the same way
setup2 = prepare_training(config_dict)
results2 = run_training(config_dict, setup=setup2)

# ============================================================================
# Example 3: Step-by-step workflow (more control)
# ============================================================================

# Step 1: Prepare everything
config3 = TrainingConfig(
    data="/scicore/home/meuwly/boitti0000/data/vibml/md17_data/hcooh_mp2_avtz_5401.npz",
    n_train=800,
    n_valid=100,
    num_atoms=18,
    tag="hcooh_stepwise",
)

setup3 = prepare_training(config3)

# Step 2: Inspect data before training
train_data = setup3['train_data']
valid_data = setup3['valid_data']
model = setup3['model']

print(f"Train samples: {len(train_data['E'])}")
print(f"Valid samples: {len(valid_data['E'])}")
print(f"Model features: {model.features}")

# Step 3: Modify config if needed
config3.num_epochs = 20  # Change epochs
config3.learning_rate = 0.0005  # Change learning rate

# Step 4: Run training with modified config
results3 = run_training(config3, setup=setup3)

# ============================================================================
# Example 4: Load existing model and continue training
# ============================================================================

# First, save a model (from a previous run)
# setup4 = prepare_training(config)
# model_file = "saved_model.json"
# with open(model_file, 'w') as f:
#     import json
#     json.dump(setup4['model_attrs'], f, default=lambda x: str(x))

# Then load it later
config4 = TrainingConfig(
    data="/scicore/home/meuwly/boitti0000/data/vibml/md17_data/ch3oh_mp2_avtz_7201.npz",
    model="saved_model.json",  # Load existing model
    n_train=1000,
    n_valid=100,
    num_atoms=20,
    num_epochs=10,
    tag="ch3oh_continued",
)

setup4 = prepare_training(config4)
results4 = run_training(config4, setup=setup4)

# ============================================================================
# Example 5: Override specific parameters when running
# ============================================================================

config5 = TrainingConfig(
    data="/scicore/home/meuwly/boitti0000/data/vibml/md17_data/ch3cho_mp2_avtz_10073.npz",
    n_train=1000,
    n_valid=100,
    num_epochs=5,  # Base value
)

setup5 = prepare_training(config5)

# Override epochs and learning rate just for this run
results5 = run_training(
    config5,
    setup=setup5,
    num_epochs=20,  # Override
    learning_rate=0.0005,  # Override
)

# ============================================================================
# Example 6: Multiple datasets
# ============================================================================

config6 = TrainingConfig(
    data=[
        "/scicore/home/meuwly/boitti0000/data/vibml/md17_data/hono_mp2_avtz_6406.npz",
        "/scicore/home/meuwly/boitti0000/data/vibml/md17_data/ch2o_mp2_avtz_3601.npz",
    ],
    n_train=2000,  # Total across both datasets
    n_valid=200,
    num_atoms=20,
    tag="multi_dataset",
)

setup6 = prepare_training(config6)
results6 = run_training(config6, setup=setup6)

