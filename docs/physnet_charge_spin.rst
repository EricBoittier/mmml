Charge-Spin Conditioned PhysNet
=================================

Multi-state energy and force predictions with charge and spin conditioning.

Overview
--------

The charge-spin conditioned PhysNet extends the standard PhysNet model to accept:

- **Total molecular charge** (Q): e.g., 0, +1, -1
- **Total spin multiplicity** (S): e.g., 1 (singlet), 2 (doublet), 3 (triplet)

This enables predictions across different electronic states from a single model.

Quick Start
-----------

.. code-block:: python

    from mmml.physnetjax.physnetjax.models.model_charge_spin import EF_ChargeSpinConditioned
    import jax.numpy as jnp
    import e3x

    # Create model
    model = EF_ChargeSpinConditioned(
        features=128,
        num_iterations=3,
        natoms=60,
        charge_range=(-2, 2),    # Support charges -2 to +2
        spin_range=(1, 5),       # Support singlet to quintet
    )

    # Initialize
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    params = model.init(
        key, Z, R, dst_idx, src_idx,
        total_charges=jnp.array([0.0]),
        total_spins=jnp.array([1.0]),
    )

    # Predict
    outputs = model.apply(
        params, Z, R, dst_idx, src_idx,
        total_charges=jnp.array([0, 1, -1]),  # Neutral, cation, anion
        total_spins=jnp.array([1, 2, 2]),     # Singlet, doublet, doublet
    )

    # outputs["energy"]: (3,) - energies for each state
    # outputs["forces"]: (num_atoms, 3) - forces

Use Cases
---------

Ionization Energies
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Neutral vs cation
    Q = jnp.array([0, 1])
    S = jnp.array([1, 2])
    
    E = model.apply(params, Z, R, ..., Q, S)["energy"]
    IE = E[1] - E[0]  # Ionization energy

Singlet-Triplet Gaps
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Compare spin states
    Q = jnp.array([0, 0])
    S = jnp.array([1, 3])  # Singlet vs triplet
    
    E = model.apply(params, Z, R, ..., Q, S)["energy"]
    gap = E[1] - E[0]  # S-T gap

Prediction Options
------------------

Control what gets computed:

.. code-block:: python

    # Energy only (faster)
    outputs = model.apply(
        params, Z, R, ..., Q, S,
        predict_energy=True,
        predict_forces=False,
    )

    # Forces only
    outputs = model.apply(
        params, Z, R, ..., Q, S,
        predict_energy=False,
        predict_forces=True,
    )

    # Both (default)
    outputs = model.apply(
        params, Z, R, ..., Q, S,
        predict_energy=True,
        predict_forces=True,
    )

Model Parameters
----------------

- ``charge_range``: Tuple of (min_charge, max_charge) to support
- ``spin_range``: Tuple of (min_spin, max_spin) multiplicities
- ``charge_embed_dim``: Dimension of charge embedding (default: 16)
- ``spin_embed_dim``: Dimension of spin embedding (default: 16)

All other parameters same as standard PhysNet (features, num_iterations, etc.)

Examples
--------

See ``examples/train_charge_spin_simple.py`` for a complete working example.

See ``examples/predict_options_demo.py`` for prediction mode examples.

API Reference
-------------

.. autoclass:: mmml.physnetjax.physnetjax.models.model_charge_spin.EF_ChargeSpinConditioned
   :members:
   :undoc-members:

