Tutorial: From ESP to DCMNet
============================

This step-by-step tutorial walks you through computing an ESP for a molecule, exporting a cube, preparing a dataset, and training DCMNet.

Step 1: Compute ESP with GPU4PySCF
----------------------------------

.. code-block:: bash

   python -m mmml.pyscf4gpuInterface.calcs \
     --mol "O 0.000 0.000 0.000; H 0.000 0.757 0.586; H 0.000 -0.757 0.586" \
     --basis def2-tzvp --xc wB97m-v --spin 0 --charge 0 \
     --dens_esp --energy --output water_esp.pkl

Step 2: Export a Gaussian cube
-------------------------------

Map the sampled ESP to a rectilinear grid and write ``water.cube``.

.. code-block:: python

   # See esp_cube_generation.rst for a complete script
   # Adapt paths and grid resolution if needed

Step 3: Prepare a training NPZ
------------------------------

Create the keys expected by DCMNet. For illustration, we fake ``mono`` and minimal metadata; for real data, use your reference charges and surface definition.

.. code-block:: python

   import numpy as np, pickle
   with open("water_esp.pkl", "rb") as f:
       d = pickle.load(f)
   # One-sample dataset (natoms = len(d["Z"]))
   R = d["R"][None, ...]
   Z = d["Z"][None, ...]
   N = np.array([[len(d["Z"])]])
   esp = d["esp"][None, ...]
   vdw_surface = d["esp_grid"][None, ...]  # using ESP grid as surface points
   n_grid = np.array([len(d["esp"])])
   mono = np.zeros((1, len(d["Z"])))  # placeholder monopoles
   np.savez("water_esp_train.npz", R=R, Z=Z, N=N, mono=mono,
            esp=esp, vdw_surface=vdw_surface, n_grid=n_grid)

Step 4: Train DCMNet
---------------------

.. code-block:: bash

   python -m mmml.dcmnet.dcmnet.main \
     --data_files ./water_esp_train.npz \
     --n_train 1 --n_valid 0 \
     --num_epochs 200 --batch_size 1 --learning_rate 1e-3 \
     --esp_w 1000 --n_dcm 1 --type default \
     --log_dir ./runs/water_demo --checkpoint_dir ./checkpoints/water_demo

Notes
-----

- For meaningful results, train on many molecules and surfaces; the single-sample demo is for mechanics only.
- Replace the placeholder monopoles with a reference scheme (e.g., RESP, MBIS) if you use the monopole loss.


