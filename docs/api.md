# API Reference

This reference is generated from source modules and includes functions/classes documented in each namespace.

The previous version of this page only listed a minimal subset while docs generation was being stabilized. This page now covers the main KARML modules.

## Top-Level Package

::: karml

## Data

### Units

::: karml.data.units

### Atomic References

::: karml.data.atomic_references

### XML Conversion

::: karml.data.xml_to_npz

## Utilities

### Electrostatics

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/utils/electrostatics.py`](https://github.com/EricBoittier/karml/blob/main/karml/utils/electrostatics.py).

### Simulation Utilities

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/utils/simulation_utils.py`](https://github.com/EricBoittier/karml/blob/main/karml/utils/simulation_utils.py).

### HDF5 Reporter

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/utils/hdf5_reporter.py`](https://github.com/EricBoittier/karml/blob/main/karml/utils/hdf5_reporter.py).

### Model Checkpoint Utilities

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/utils/model_checkpoint.py`](https://github.com/EricBoittier/karml/blob/main/karml/utils/model_checkpoint.py).

## Interfaces

### OpenMM Interface

The OpenMM integration provides helpers to set up and run CHARMM/OpenMM simulations (PSF/PDB, parameter sets, integrators, and schedules). It depends on the optional [OpenMM](https://openmm.org/) Python package (`pip install openmm`).

Source: [`karml/interfaces/openmmInterface/interface.py`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/openmmInterface/interface.py).

### PyCHARMM Setup Box

This module currently requires a local PyCHARMM installation at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/interfaces/pycharmmInterface/setupBox.py`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/pycharmmInterface/setupBox.py).

### PyCHARMM Setup Residue

This module currently requires a local PyCHARMM installation at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/interfaces/pycharmmInterface/setupRes.py`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/pycharmmInterface/setupRes.py).

### PyCHARMM Commands

This module currently requires a local PyCHARMM installation at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/interfaces/pycharmmInterface/pycharmmCommands.py`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/pycharmmInterface/pycharmmCommands.py).

### PySCF4GPU Calculations

This module requires optional PySCF dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`karml/interfaces/pyscf4gpuInterface/calcs.py`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/pyscf4gpuInterface/calcs.py).

## Models

### External electric-field PhysNet (`EFieldPhysNet`)

Field-dependent energy/force model for Raman/IR and related spectroscopy. Formerly under `karml/models/EF/` (deprecated import path).

Source: [`karml/models/efield/training.py`](https://github.com/EricBoittier/karml/blob/main/karml/models/efield/training.py).

### E-field training CLI

Canonical command: `karml efield-train` (replaces deprecated `ef-train`).

Source: [`karml/models/efield/training.py`](https://github.com/EricBoittier/karml/blob/main/karml/models/efield/training.py).

### E-field evaluation CLI

Canonical command: `karml efield-evaluate` (replaces deprecated `ef-evaluate`).

Source: [`karml/models/efield/evaluate.py`](https://github.com/EricBoittier/karml/blob/main/karml/models/efield/evaluate.py).

### Unified energy/forces providers

ML checkpoints (PhysNet, joint PhysNet+DCMNet, E-field) and QC backends (PySCF, ORCA, xTB, Molpro) share :class:`~karml.interfaces.energy_forces.EnergyForcesProvider`.

Source: [`karml/interfaces/energy_forces/`](https://github.com/EricBoittier/karml/blob/main/karml/interfaces/energy_forces/__init__.py).

Hybrid CHARMM monomer/dimer MLpot requires ``supports_decomposed_ml`` (PhysNet family only); use ``build_provider`` for single-structure inference and cross-check.

## CLI

### Entry Point

::: karml.cli.__main__

### Shared CLI Utilities

::: karml.cli.base
