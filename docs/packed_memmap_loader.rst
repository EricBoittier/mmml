Packed Memmap Data Loader
==========================

Efficient loading of large molecular datasets via memory-mapped files.

Overview
--------

The ``PackedMemmapLoader`` enables training on datasets larger than RAM by:

- **Memory mapping**: Data stays on disk, loaded only when needed
- **Packed storage**: Variable-size molecules stored without wasted space
- **Bucketed batching**: Molecules sorted by size to minimize padding

Typical speedups: 10-100x faster startup than loading full NPZ into memory.

Quick Start
-----------

.. code-block:: python

    from mmml.data.packed_memmap_loader import PackedMemmapLoader, split_loader

    # Load data
    loader = PackedMemmapLoader(
        path="data_directory",
        batch_size=32,
        shuffle=True,
    )

    # Split into train/validation
    train_loader, valid_loader = split_loader(loader, train_fraction=0.9)

    # Generate batches
    for batch in train_loader.batches(num_atoms=60):
        # batch: dict with Z, R, F, E, N, dst_idx, src_idx, batch_segments
        # Ready for PhysNet training
        pass

Data Format
-----------

Your data directory should contain:

.. code-block:: text

    data_path/
    ├── offsets.npy       # (N+1,) atom offsets
    ├── n_atoms.npy       # (N,) atom counts
    ├── Z_pack.int32      # (sum_atoms,) atomic numbers
    ├── R_pack.f32        # (sum_atoms, 3) positions
    ├── F_pack.f32        # (sum_atoms, 3) forces
    ├── E.f64             # (N,) energies
    └── Qtot.f64          # (N,) total charges (optional)

Converting from NPZ
-------------------

.. code-block:: bash

    python scripts/convert_npz_to_packed_memmap.py \
        --input dataset.npz \
        --output packed_dataset

Parameters
----------

- ``path``: Directory containing packed memmap files
- ``batch_size``: Number of molecules per batch
- ``shuffle``: Whether to shuffle data (default: True)
- ``bucket_size``: Size of bucketing groups (default: 8192)
- ``seed``: Random seed

Batch Format
------------

Batches are dictionaries with:

- ``Z``: (batch_size, num_atoms) atomic numbers
- ``R``: (batch_size, num_atoms, 3) positions
- ``F``: (batch_size, num_atoms, 3) forces
- ``E``: (batch_size,) energies
- ``N``: (batch_size,) atom counts
- ``dst_idx``: Pair indices for message passing
- ``src_idx``: Pair indices for message passing
- ``batch_segments``: Molecule index for each atom

Training Example
----------------

.. code-block:: python

    from mmml.data.packed_memmap_loader import PackedMemmapLoader
    from mmml.physnetjax.physnetjax.models.model import EF

    # Load data
    loader = PackedMemmapLoader("large_dataset", batch_size=32)
    
    # Create model
    model = EF(features=128, num_iterations=3, natoms=60)
    
    # Train
    for epoch in range(num_epochs):
        for batch in loader.batches(num_atoms=60):
            # Train step
            outputs = model.apply(params, batch["Z"], batch["R"], ...)

Examples
--------

See ``examples/train_memmap_simple.py`` for a complete working example.

See ``train_physnet_memmap.py`` for full training script.

API Reference
-------------

.. autoclass:: mmml.data.packed_memmap_loader.PackedMemmapLoader
   :members:
   :undoc-members:

.. autofunction:: mmml.data.packed_memmap_loader.split_loader

