Fitting DCMNet to ESP data
==========================

This guide explains how to prepare ESP datasets and train DCMNet using the provided training utilities in ``mmml.dcmnet.dcmnet``.

Overview
--------

DCMNet predicts monopoles and distributed dipoles that reproduce ESP on a molecular surface. Training minimizes ESP and charge-related losses using batches prepared from NPZ datasets.

Data format
-----------

Training expects NPZ files containing keys assembled by ``prepare_multiple_datasets``:

- ``R``: positions shaped ``(num, natoms, 3)``
- ``Z``: atomic numbers shaped ``(num, natoms)``
- ``N``: number of atoms per system shaped ``(num, 1)``
- ``mono``: reference atomic monopoles shaped ``(num, natoms)`` (optional if only ESP loss is used)
- ``esp``: target ESP values (flattened over grid points)
- ``vdw_surface``: grid coordinates of the ESP evaluation points
- ``n_grid``: number of grid points for each structure
- Optional: ``D``, ``Dxyz``, ``esp_grid``, ``com``, ``quadrupole``, etc.

See ``mmml.dcmnet.dcmnet.data`` for details and batching logic.

Quick start: training via CLI
-----------------------------

The entrypoint ``mmml.dcmnet.dcmnet.main`` wraps dataset preparation, model creation, and training.

.. code-block:: bash

   python -m mmml.dcmnet.dcmnet.main \
     --data_dir /path/to/data_root \
     --data_files /path/to/esp_data_0.npz /path/to/esp_data_1.npz \
     --n_train 80000 --n_valid 2000 \
     --num_epochs 5000 --batch_size 1 \
     --learning_rate 1e-3 \
     --esp_w 10000.0 \
     --n_dcm 1 \
     --type default \
     --log_dir ./runs/dcmnet_example \
     --checkpoint_dir ./checkpoints/dcmnet_example

Key options
~~~~~~~~~~~

- ``--type``: ``default`` uses ESP+monopole loss, ``dipole`` adds dipole loss terms
- ``--n_dcm``: number of distributed multipoles per atom
- ``--esp_w``: ESP term weight in the total loss
- ``--n_feat``, ``--n_basis``, ``--max_degree``, ``--n_mp``: model size/depth

Programmatic training
---------------------

You can directly use the training functions for custom workflows:

.. code-block:: python

   import jax
   from tensorboardX import SummaryWriter
   from mmml.dcmnet.dcmnet.data import prepare_datasets
   from mmml.dcmnet.dcmnet.modules import MessagePassingModel
   from mmml.dcmnet.dcmnet.training import train_model, train_model_dipo

   key = jax.random.PRNGKey(0)
   train_data, valid_data = prepare_datasets(
       key, num_train=80000, num_valid=2000,
       filename=["/path/to/esp_data_0.npz", "/path/to/esp_data_1.npz"],
       clean=True,
   )

   model = MessagePassingModel(
       features=16, max_degree=2, num_iterations=2,
       num_basis_functions=16, cutoff=4.0, n_dcm=1,
       include_pseudotensors=False,
   )

   writer = SummaryWriter("./runs/dcmnet_programmatic")

   params, valid_loss = train_model(
       key=key, model=model,
       train_data=train_data, valid_data=valid_data,
       num_epochs=5000, learning_rate=1e-3, batch_size=1,
       writer=writer, ndcm=model.n_dcm, esp_w=10000.0,
   )

Dipole-augmented training
-------------------------

To include dipole targets use ``train_model_dipo`` and ensure the dataset contains dipole-related keys (``Dxyz``, ``com``, ``espMask``). See the function docstrings in ``mmml.dcmnet.dcmnet.training`` for details.

Monitoring and checkpoints
--------------------------

- Logs are written via TensorBoardX to ``--log_dir``; track ``Loss/train`` and ``Loss/valid``.
- Best parameters (by validation loss) are saved as ``best_<esp_w>_params.pkl`` in the log directory.

Troubleshooting
---------------

- If you encounter "datasets only contains X points" errors, reduce ``--n_train``/``--n_valid`` or check that your NPZ files contain enough entries.
- If memory is tight, lower ``--batch_size`` or model sizes, or clip ESP with ``clip_esp=True`` in ``prepare_datasets``.


