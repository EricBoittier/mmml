# %%
import mmml
import matplotlib.pyplot as plt
import patchworklib as pw
import os
from pathlib import Path
import numpy as np
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

# %%
from mmml import dcmnet

# %%
import jax

from mmml.dcmnet.dcmnet.data import prepare_datasets
from mmml.dcmnet.dcmnet.modules import MessagePassingModel
from mmml.dcmnet.dcmnet.training import train_model, train_model_dipo

key = jax.random.PRNGKey(0)


# %%
NDCM = 4
model = MessagePassingModel(
    features=128, max_degree=1, num_iterations=2,
    num_basis_functions=64, cutoff=8.0, n_dcm=NDCM,
    include_pseudotensors=False,
)


# %%
index = 30
data_path_resolved = Path('test.npz') 
data_loaded = np.load(data_path_resolved, 
allow_pickle=True)
data_path_resolved

# %%
# factorize a number
def factorize(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

factorize(data_loaded["esp"].shape[1])

for k in data_loaded.keys():
    
    print(k)
    shape = data_loaded[k].shape
    print(shape
    )
    if len(shape) < 3:
        try:
            d = data_loaded[k]
            d = d.flatten()
            plt.hist(d)
            title = f"{k}: {d.min()} - {d.max()}"
            plt.title(title)
            plt.show()
        except:
            pass

n_sample = 1000  # Number of points to keep
data_key = jax.random.PRNGKey(0)

train_data, valid_data = prepare_datasets(
    data_key, num_train=1000, num_valid=100,
    filename=[data_path_resolved],
    clean=False, esp_mask=False,
    natoms=18,
    clip_esp=False,
)




def random_sample_esp(esp, esp_grid, n_sample, seed=42):
    np.random.seed(seed)
    sampled_esp = []
    sampled_grid = []
    
    for i in range(len(esp)):
        lessthan = esp[i] < 2
        morethan = esp[i] > -2
        not_0 = esp[i] != 0.0
        condmask = lessthan*morethan*not_0
        _shape = esp[i][condmask].shape[0]
        print(_shape)
        indices = np.random.choice(_shape, n_sample, replace=False)
        indices = np.sort(indices) 
        sampled_esp.append(esp[i][condmask][indices])
        sampled_grid.append(esp_grid[i][condmask][indices])
    
    return np.array(sampled_esp), np.array(sampled_grid)

train_data["esp"], train_data["esp_grid"] = random_sample_esp(
    train_data["esp"] , train_data["esp_grid"], n_sample
)
valid_data["esp"], valid_data["esp_grid"] = random_sample_esp(
    valid_data["esp"] , valid_data["esp_grid"], n_sample
)


valid_data["esp"] = 0.0016 * valid_data["esp"]
train_data["esp"] = 0.0016 * train_data["esp"]

train_data["vdw_surface"] = train_data["esp_grid"] 
valid_data["vdw_surface"] = valid_data["esp_grid"] 
train_data["n_grid"] = np.full(len(train_data["vdw_surface"]), n_sample)
valid_data["n_grid"] = np.full(len(valid_data["vdw_surface"]), n_sample)


train_data["vdw_surface"] = train_data["esp_grid"]
valid_data["vdw_surface"] = valid_data["esp_grid"]

Hs_train = train_data["Z"] == 1.0
Os_train = train_data["Z"] == 8.0
Hs_valid = valid_data["Z"] == 1.0
Os_valid = valid_data["Z"] == 8.0

train_data["mono"] = Hs_train * 0.1 + Os_train * -0.2
valid_data["mono"] = Hs_valid * 0.1 + Os_valid * -0.2

# Fix n_grid shape
train_data["n_grid"] = np.full(train_data["Z"].shape[0], n_sample)
valid_data["n_grid"] = np.full(valid_data["Z"].shape[0], n_sample)

# Fix N shape  
train_data["N"] = np.count_nonzero(train_data["Z"], axis=1)
valid_data["N"] = np.count_nonzero(valid_data["Z"], axis=1)


_ = plt.hist(valid_data["esp"][1])


_ = plt.hist(valid_data["esp"][0])

# %%
# valid_data["esp"][0][lessthan * morethan ].shape

# %%
# Check current batch shapes
print("After fixes:")
batch = {k: v[0:1] if len(v.shape) > 0 else v for k, v in train_data.items()}
for key in ['mono', 'esp', 'vdw_surface', 'n_grid', 'N', 'R', 'Z']:
    if key in batch:
        print(f"{key}: {batch[key].shape}")

# Also check the specific values
print(f"\nmono values: {batch['mono']}")
print(f"N values: {batch['N']}")
print(f"n_grid values: {batch['n_grid']}")

# %%
esp_data = train_data["esp"]


# %%
params, valid_loss = train_model(
    key=data_key, model=model,
    writer=None,
    train_data=train_data, valid_data=valid_data,
    num_epochs=100, learning_rate=1e-4, batch_size=1,
    restart_params=params if params is None else params,
    ndcm=model.n_dcm, esp_w=1.0, chg_w=0.0, use_grad_clip=True, grad_clip_norm=1.0,
)
new_params = params.copy()



from mmml.dcmnet.dcmnet.analysis import dcmnet_analysis, prepare_batch

from mmml.dcmnet.dcmnet.data import prepare_batches
from mmml.dcmnet.dcmnet.analysis import dcmnet_analysis

def prepare_batch_for_analysis(data, index=0):
    """Prepare a single batch correctly for dcmnet_analysis."""
    # Extract single item but keep batch dimension
    _dict = {k: np.array(v[[index]]) for k, v in data.items()}
    
    # Use prepare_batches with include_id=True
    batch = prepare_batches(jax.random.PRNGKey(0), _dict, batch_size=1, include_id=False, num_atoms =18)[0]
    batch["com"] = np.array([0,0,0])
    batch["Dxyz"] = np.array([0,0,0])
    return batch

batch = prepare_batch_for_analysis(train_data, index=0)
output = dcmnet_analysis(params, model, batch, 18)
print(f"RMSE: {output['rmse_model']:.6f}")
print(f"RMSE (masked): {output['rmse_model_masked']:.6f}")

import sys
sys.exit()


# %%
from mmml.dcmnet.dcmnet.utils import apply_model
NATOMS = 18
_ = apply_model(model, params, batch, 1, NATOMS)

# %%
batch["R"][:6]

# %%
6*4

# %%
m,d = _
# d.reshape(1, 18 * 4, 3)[0,:6*model.n_dcm]
d[:6*model.n_dcm]

# %%
dcmnet_analysis

# %%
batch

# %%


# %%
import ase
from ase.visualize import view
atoms = ase.Atoms(batch["Z"][:int(batch["N"])], 
batch["R"][:int(batch["N"])])
view(atoms, viewer="x3d")


# %%
output.keys()
output["mono"].sum(axis=-1)

# %%
output["dipo"][:24]

# %%
plt.scatter(batch["esp"], output['esp_pred'])

# %%
import patchworklib as pw
VMAX = 0.001
xy_ax = pw.Brick()
xy_ax.scatter(batch["esp"], output['esp_pred'], s=1)
max_val = np.sqrt(max(np.max(batch["esp"]**2), np.max(output['esp_pred']**2)))
xy_ax.plot(np.linspace(-max_val, max_val, 100), np.linspace(-max_val, max_val, 100))
xy_ax.set_aspect('equal')

ax_true = pw.Brick()
Npoints = 1000
vdw_surface_min = np.min(batch["vdw_surface"][0], axis=-1)
vdw_surface_max = np.max(batch["vdw_surface"][0], axis=-1)

ax_true.scatter(
    batch["vdw_surface"][0][:Npoints,0], 
batch["vdw_surface"][0][:Npoints,1], 
c=batch["esp"][0][:Npoints],
s=0.01,
    vmin=-VMAX, vmax=VMAX
)
max_val = np.sqrt(max(np.max(batch["esp"]**2), np.max(output['esp_pred']**2)))
# ax.plot(np.linspace(-max_val, max_val, 100), np.linspace(-max_val, max_val, 100))
ax_true.set_aspect('equal')

ax_pred = pw.Brick()

ax_pred.scatter(
    batch["vdw_surface"][0][:Npoints,0], 
batch["vdw_surface"][0][:Npoints,1], 
c=output['esp_pred'][:Npoints],
s=0.01,
    vmin=-VMAX, vmax=VMAX
)
max_val = np.sqrt(max(np.max(batch["esp"]**2), np.max(output['esp_pred']**2)))
# ax.plot(np.linspace(-max_val, max_val, 100), np.linspace(-max_val, max_val, 100))
ax_pred.set_aspect('equal')


ax_diff = pw.Brick()
ax_diff.scatter(
    batch["vdw_surface"][0][:Npoints,0], 
batch["vdw_surface"][0][:Npoints,1], 
c=batch["esp"][0][:Npoints] - output['esp_pred'][:Npoints],
s=0.01,
    vmin=-VMAX, vmax=VMAX
)
ax_diff.set_aspect('equal')

for _ in [ax_pred, ax_true, ax_diff]:
    _.set_xlim(vdw_surface_min[0], -vdw_surface_min[0])
    _.set_ylim(vdw_surface_min[1], -vdw_surface_min[0])


xy_ax | ax_pred | ax_true | ax_diff



output["mono"][0][:int(batch["N"])].sum(axis=-1)

output["mono"][0][:int(batch["N"])*model.n_dcm]

charge_ax = pw.Brick()
charge_ax.matshow(output["mono"][0][:int(batch["N"])],vmin=-1,vmax=1)
scharge_ax = pw.Brick()
scharge_ax.matshow(output["mono"][0][:int(batch["N"])].sum(axis=-1)[:, None],vmin=-1,vmax=1)
# scharge_ax.add_colorbar(vmin=-1,vmax=1)
scharge_ax.axis("off")
f = (scharge_ax | charge_ax)
f.add_colorbar(vmin=-1,vmax=1)

R = output["dipo"][:int(batch["N"])*NDCM]
Z = np.array([1 if _ > 0 else 1 for _ in output["mono"][0][:int(batch["N"])].flatten()])
R.shape, Z.shape
dcm_atoms = ase.Atoms(Z, R)
view(dcm_atoms,  viewer="x3d")

new_params, valid_loss = train_model(
    key=jax.random.PRNGKey(0), model=model,
    writer=None,
    train_data=train_data, valid_data=valid_data,
    num_epochs=100, learning_rate=5e-4, batch_size=1,
    ndcm=model.n_dcm, esp_w=1000.0,
    restart_params=params if new_params is None else new_params,
)



