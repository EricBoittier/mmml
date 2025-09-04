Tutorials
=========

This page collects practical, copy-pasteable examples for common tasks.

Compute ESP and export a cube
-----------------------------

.. code-block:: bash

   python -m mmml.pyscf4gpuInterface.calcs \
     --mol "O 0.000 0.000 0.000; H 0.000 0.757 0.586; H 0.000 -0.757 0.586" \
     --basis def2-tzvp --xc wB97m-v --spin 0 --charge 0 \
     --dens_esp --energy --output water_esp.pkl

.. code-block:: python

   # Map sampled ESP to rectilinear grid and write water.cube
   # (See esp_cube_generation page for a full script.)

Train DCMNet on ESP datasets (CLI)
----------------------------------

.. code-block:: bash

   python -m mmml.dcmnet.dcmnet.main \
     --data_files /path/set1.npz /path/set2.npz \
     --n_train 80000 --n_valid 2000 \
     --num_epochs 5000 --batch_size 1 --learning_rate 1e-3 \
     --esp_w 10000 --n_dcm 1 --type default \
     --log_dir ./runs/example --checkpoint_dir ./ckpts/example

Train DCMNet (programmatic)
---------------------------

.. code-block:: python

   import jax
   from tensorboardX import SummaryWriter
   from mmml.dcmnet.dcmnet.data import prepare_datasets
   from mmml.dcmnet.dcmnet.modules import MessagePassingModel
   from mmml.dcmnet.dcmnet.training import train_model

   key = jax.random.PRNGKey(0)
   train_data, valid_data = prepare_datasets(
       key, 80000, 2000, ["/path/set1.npz", "/path/set2.npz"], clean=True
   )
   model = MessagePassingModel(16, 2, 2, 16, cutoff=4.0, n_dcm=1)
   writer = SummaryWriter("./runs/programmatic")
   train_model(key, model, train_data, valid_data, 5000, 1e-3, 1, writer, ndcm=1, esp_w=10000)

Dipole-augmented training
-------------------------

.. code-block:: python

   from mmml.dcmnet.dcmnet.training import train_model_dipo
   # Requires dataset with Dxyz, com, espMask
   train_model_dipo(key, model, train_data, valid_data, 3000, 5e-4, 1, writer, ndcm=1, esp_w=10000)

Prepare a minimal NPZ from an ESP pickle
----------------------------------------

.. code-block:: python

   import numpy as np, pickle
   with open("water_esp.pkl", "rb") as f:
       d = pickle.load(f)
   R = d["R"][None, ...]
   Z = d["Z"][None, ...]
   N = np.array([[len(d["Z"])]]);
   mono = np.zeros((1, len(d["Z"])));
   esp = d["esp"][None, ...]
   vdw_surface = d["esp_grid"][None, ...]
   n_grid = np.array([len(d["esp"])])
   np.savez("water_esp_train.npz", R=R, Z=Z, N=N, mono=mono,
            esp=esp, vdw_surface=vdw_surface, n_grid=n_grid)

Tips
----

- Reduce memory by lowering batch size and model size.
- Use TensorBoard to monitor ``Loss/train`` and ``Loss/valid``.


