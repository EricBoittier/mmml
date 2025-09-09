# import pandas as pd
from .modules import MessagePassingModel
import numpy as np
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

models = []
for i in range(1,8):

    
    models.append(MessagePassingModel(
        features=32, max_degree=2, num_iterations=2,
        num_basis_functions=32, cutoff=10.0, n_dcm=i,
        include_pseudotensors=False,
    )
    )

model_params = []
for i in range(1,8):
    model_params.append(np.load(os.path.join(this_dir, "..", f"modelA{i}.npy"),
     allow_pickle=True).tolist())


DCM1,DCM2, DCM3, DCM4, DCM5, DCM6, DCM7 = models
dcm1_params, dcm2_params, dcm3_params, dcm4_params, dcm5_params, dcm6_params, dcm7_params = model_params

"""
Pre-trained DCMNet models for distributed multipole prediction.

This module provides pre-trained DCMNet models with different numbers
of distributed multipoles per atom (DCM1-DCM4). These models were
trained on QM9 ESP data and can be used for inference without
additional training.

Models
------
DCM1 : MessagePassingModel
    Model with 1 distributed multipole per atom
DCM2 : MessagePassingModel  
    Model with 2 distributed multipoles per atom
DCM3 : MessagePassingModel
    Model with 3 distributed multipoles per atom
DCM4 : MessagePassingModel
    Model with 4 distributed multipoles per atom

Parameters
----------
dcm1_params : dict
    Trained parameters for DCM1 model
dcm2_params : dict
    Trained parameters for DCM2 model
dcm3_params : dict
    Trained parameters for DCM3 model
dcm4_params : dict
    Trained parameters for DCM4 model

Hyperparameters
--------------
features : int
    Number of features per atom (16)
max_degree : int
    Maximum spherical harmonic degree (2)
num_iterations : int
    Number of message passing iterations (2)
num_basis_functions : int
    Number of radial basis functions (8)
cutoff : float
    Distance cutoff for interactions (4.0 Å)

Usage
-----
To use a pre-trained model:

```python
from dcmnet.models import DCM2, dcm2_params

# Apply model to get predictions
mono, dipo = DCM2.apply(dcm2_params, atomic_numbers, positions, dst_idx, src_idx)
```

Notes
-----
- Models were trained with ESP weight of 10000.0 except DCM1 (0.0)
- All models use the same hyperparameters except n_dcm
- Parameters are loaded from pickle files on module import
"""
