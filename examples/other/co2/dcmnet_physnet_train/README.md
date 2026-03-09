# DCMNet PhysNet Training Examples

This directory contains examples for training and evaluating joint PhysNet-DCMNet models.

## Main Scripts

### Training
- **trainer.py** - Main training script with DCMNet/NonEquivariant options

### Analysis (Now in mmml/cli!)
The following tools have been generalized and moved to `mmml/cli`:

- ✅ **plot_training_history.py** → `mmml.cli.plot_training`
- ✅ **simple_calculator.py** → `mmml.cli.calculator`
- ✅ **inspect_checkpoint.py** → `mmml.cli.inspect_checkpoint`
- ✅ **convert_npz_to_traj.py** → `mmml.cli.convert_npz_traj`

Use them from anywhere with `python -m mmml.cli.*`!

### Specialized Scripts (Still Here)
- **compare_models.py** - Compare DCMNet vs NonEquivariant
- **compare_downstream_tasks.py** - Downstream task comparison
- **dynamics_calculator.py** - Advanced dynamics with ASE
- **jaxmd_dynamics.py** - Ultra-fast JAX MD simulations
- **raman_calculator.py** - Raman spectroscopy calculations
- **active_learning_manager.py** - Active learning workflow
- **gas_phase_calculator.py** - Multi-molecule gas phase simulations

### Visualization
- **povray_visualization.py** - High-quality POV-Ray renders
- **matplotlib_3d_viz.py** - Interactive 3D visualizations
- **visualize_*.py** - Various visualization tools

### Data Preparation
- **prepare_qm_inputs.py** - Prepare QM calculation inputs
- **convert_molpro_to_training.py** - Convert Molpro outputs
- **create_molpro_jobs_from_al.py** - Active learning job creation

## Documentation

Old documentation files have been moved to `_old_docs/` for reference.

For current documentation, see:
- `../../docs/cli.rst` - CLI tools reference
- `../../AI/CLI_TOOLS_ADDED.md` - New tool documentation

## Quick Start

```bash
# Use generalized CLI tools (recommended)
python -m mmml.cli.plot_training history.json
python -m mmml.cli.calculator --checkpoint ./ckpts/model
python -m mmml.cli.inspect_checkpoint --checkpoint ./ckpts/model
python -m mmml.cli.convert_npz_traj data.npz -o trajectory.traj

# Use specialized scripts (for specific DCMNet tasks)
python trainer.py --train-efd train.npz --valid-efd valid.npz
python compare_models.py --train-efd train.npz --valid-efd valid.npz
```

## Data Files

- `energies_forces_dipoles_train.npz` - Training dataset
- `energies_forces_dipoles_valid.npz` - Validation dataset
- `energies_forces_dipoles_test.npz` - Test dataset
- `split_indices.npz` - Train/valid/test split indices

## Organized Structure

```
dcmnet_physnet_train/
├── README.md (this file)
├── trainer.py (main training script)
├── *_calculator.py (specialized calculators)
├── compare_*.py (comparison tools)
├── visualize_*.py (visualization tools)
├── _old_docs/ (archived documentation)
└── data files (*.npz)
```

## Migration to CLI

Many tools from this directory have been generalized and are now available as:
- `mmml/cli/plot_training.py`
- `mmml/cli/calculator.py`
- `mmml/cli/clean_data.py`
- `mmml/cli/dynamics.py`
- `mmml/cli/inspect_checkpoint.py`
- `mmml/cli/convert_npz_traj.py`
- `mmml/cli/evaluate_model.py`

Use these instead of the local versions!

