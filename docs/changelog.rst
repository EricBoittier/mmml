Changelog
=========

Recent Additions (November 2025)
---------------------------------

Charge-Spin Conditioned PhysNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added new PhysNet model variant that accepts total molecular charge and spin multiplicity as inputs.

- **Module**: ``mmml.physnetjax.physnetjax.models.model_charge_spin.EF_ChargeSpinConditioned``
- **Training script**: ``train_physnet_charge_spin.py``
- **Examples**: ``examples/train_charge_spin_simple.py``, ``examples/predict_options_demo.py``
- **Documentation**: :doc:`physnet_charge_spin`

Key features:

- Multi-state predictions from single model
- Learnable charge and spin embeddings
- Support for ionization energies, singlet-triplet gaps
- Configurable prediction modes (energy only, forces only, both)

Packed Memmap Data Loader
^^^^^^^^^^^^^^^^^^^^^^^^^^

Added efficient data loader for large molecular datasets using memory-mapped files.

- **Module**: ``mmml.data.packed_memmap_loader.PackedMemmapLoader``
- **Training script**: ``train_physnet_memmap.py``
- **Converter**: ``scripts/convert_npz_to_packed_memmap.py``
- **Examples**: ``examples/train_memmap_simple.py``
- **Documentation**: :doc:`packed_memmap_loader`

Key features:

- Train on datasets larger than RAM
- Fast startup (seconds vs minutes)
- Bucketed batching minimizes padding
- Space-efficient packed storage format

