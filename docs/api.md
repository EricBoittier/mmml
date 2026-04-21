# API Reference

This reference is generated from source modules and includes functions/classes documented in each namespace.

The previous version of this page only listed a minimal subset while docs generation was being stabilized. This page now covers the main MMML modules.

## Top-Level Package

::: mmml

## Data

### Units

::: mmml.data.units

### Atomic References

::: mmml.data.atomic_references

### XML Conversion

::: mmml.data.xml_to_npz

## Utilities

### Electrostatics

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/utils/electrostatics.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/utils/electrostatics.py).

### Simulation Utilities

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/utils/simulation_utils.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/utils/simulation_utils.py).

### HDF5 Reporter

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/utils/hdf5_reporter.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/utils/hdf5_reporter.py).

### Model Checkpoint Utilities

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/utils/model_checkpoint.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/utils/model_checkpoint.py).

## Interfaces

### OpenMM Interface

The OpenMM integration provides helpers to set up and run CHARMM/OpenMM simulations (PSF/PDB, parameter sets, integrators, and schedules). It depends on the optional [OpenMM](https://openmm.org/) Python package (`pip install openmm`).

Source: [`mmml/interfaces/openmmInterface/interface.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/openmmInterface/interface.py).

### PyCHARMM Setup Box

This module currently requires a local PyCHARMM installation at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/interfaces/pycharmmInterface/setupBox.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/setupBox.py).

### PyCHARMM Setup Residue

This module currently requires a local PyCHARMM installation at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/interfaces/pycharmmInterface/setupRes.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/setupRes.py).

### PyCHARMM Commands

This module currently requires a local PyCHARMM installation at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/interfaces/pycharmmInterface/pycharmmCommands.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/pycharmmCommands.py).

### PySCF4GPU Calculations

This module requires optional PySCF dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/interfaces/pyscf4gpuInterface/calcs.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pyscf4gpuInterface/calcs.py).

## Models

### EF Model

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/models/EF/model.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/EF/model.py).

### EF Training

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/models/EF/training.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/EF/training.py).

### EF Evaluation

This module requires optional JAX dependencies at import time, so it is not auto-rendered by mkdocstrings in the default docs build environment.

Source: [`mmml/models/EF/evaluate.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/EF/evaluate.py).

## CLI

### Entry Point

::: mmml.cli.__main__

### Shared CLI Utilities

::: mmml.cli.base
