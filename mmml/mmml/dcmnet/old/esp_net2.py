import functools
import os
import pandas as pd
import e3x
import flax.linen as nn
import jax
import numpy as np
import optax
# Disable future warnings.
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt

plt.set_cmap('bwr')

import jax.numpy as jnp
from jax import jit, grad
from jax import vmap
NATOMS = 60

def prepare_datasets(key, num_train, num_valid, filename="esp2000.npz"):
    # Load the dataset.
    dataset = np.load(filename)

    for k, v in dataset.items():
        print(k, v.shape)

    dataR = dataset['R']
    dataZ = dataset['Z']
    dataMono = dataset['mono']
    dataEsp = dataset["esp"]
    dataVDW = dataset["vdw_surface"]
    dataNgrid = dataset["n_grid"]

    # Make sure that the dataset contains enough entries.
    num_data = len(dataR)
    print(num_data)
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f'datasets only contains {num_data} points, '
            f'requested num_train={num_train}, num_valid={num_valid}')

    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]

    atomic_numbers = dataZ

    # Collect and return train and validation sets.
    train_data = dict(
        atomic_numbers=jnp.asarray(atomic_numbers[train_choice]),
        ngrid=jnp.array(dataNgrid[train_choice]),
        positions=jnp.asarray(dataR[train_choice]),
        mono=jnp.asarray(dataMono[train_choice]),
        esp=jnp.asarray(dataEsp[train_choice]),
        vdw_surface=jnp.asarray(dataVDW[train_choice]),
    )
    valid_data = dict(
        atomic_numbers=jnp.asarray(atomic_numbers[valid_choice]),
        positions=jnp.asarray(dataR[valid_choice]),
        mono=jnp.asarray(dataMono[valid_choice]),
        ngrid=jnp.array(dataNgrid[valid_choice]),
        esp=jnp.asarray(dataEsp[valid_choice]),
        vdw_surface=jnp.asarray(dataVDW[valid_choice]),
    )
    print("...")
    print("...")
    for k, v in train_data.items():
        print(k, v.shape)
    print("...")
    for k, v in valid_data.items():
        print(k, v.shape)

    return train_data, valid_data


def prepare_batches(key, data, batch_size):
    # Determine the number of training steps per epoch.
    data_size = len(data['mono'])
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[:steps_per_epoch * batch_size]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    num_atoms = len(data['atomic_numbers'][0])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)
    # Assemble and return batches.
    return [
        dict(
            mono=data["mono"][perm].reshape(-1),
            ngrid=data["ngrid"][perm].reshape(-1),
            esp=data["esp"][perm],  # .reshape(-1),
            vdw_surface=data["vdw_surface"][perm],  # .reshape(-1, 3),
            atomic_numbers=data["atomic_numbers"][perm].reshape(-1),
            positions=data['positions'][perm].reshape(-1, 3),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
        )
        for perm in perms
    ]


NATOMS = 60
class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 17
    n_dcm: int = 4

    def mono(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments,
             batch_size):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(
            # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )

        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1,
                         features=self.features)(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                y = e3x.nn.MessagePass(max_degree=max_degree,
                                       include_pseudotensors=False
                                       )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)
            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features)(y)
            # Residual connection.
            x = e3x.nn.add(x, y)

        x = e3x.nn.TensorDense(
            features=n_dcm,
            max_degree=1,
            include_pseudotensors=False,
        )(x)

        atomic_mono = e3x.nn.change_max_degree_or_type(x,
                                                       max_degree=0,
                                                       include_pseudotensors=False)
        element_bias = self.param('element_bias',
                                  lambda rng, shape: jnp.zeros(shape),
                                  (self.max_atomic_number + 1))
        atomic_mono = nn.Dense(n_dcm, use_bias=False,
                               )(atomic_mono)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono += element_bias[atomic_numbers][:, None]
        
        atomic_dipo = x[:, 1, 1:4, :]
        atomic_dipo = e3x.nn.hard_tanh(atomic_dipo) * 0.173
        atomic_dipo += positions[:, :, None]

        return atomic_mono, atomic_dipo

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx,
                 batch_segments=None, batch_size=None):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(atomic_numbers, positions, dst_idx, src_idx, batch_segments,
                         batch_size)

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx,
                 batch_segments=None, batch_size=None):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(atomic_numbers, positions, dst_idx, src_idx, batch_segments,
                         batch_size)


@functools.partial(jax.jit)
def nan_safe_coulomb_potential(q, r):
    # potential = jnp.where(jnp.isnan(r) | (r == 0.0), 0.0, q / (r * 1.88973))
    return q / (r * 1.88973)
    # return potential

@functools.partial(jax.jit, static_argnames=('grid_positions'))
def calc_esp(charge_positions, charge_values, grid_positions):
    # chg_mask = jnp.where(mono != 0, 1.0, 0.0)
    # Expand the grid positions and charge positions to compute all pairwise differences
    diff = grid_positions[:, None, :] - charge_positions[None, :, :]
    # Compute the Euclidean distance between each grid point and each charge
    r = jnp.linalg.norm(diff, axis=-1)
    avg_chg = jnp.sum(charge_values) / charge_values.shape[0]
    new_charge_values = charge_values - avg_chg
    C = nan_safe_coulomb_potential(new_charge_values[None, :], r)
    V = jnp.sum(C, axis=-1)
    return V

# @functools.partial(jax.jit)
# def calc_esp(charge_positions, charge_values, grid_positions, mono):
#     chg_mask = jnp.where(mono != 0, 1.0, 0.0)
#     # Expand the grid positions and charge positions to compute all pairwise differences
#     diff = grid_positions[:, None, :] - charge_positions[None, :, :]
#     # Compute the Euclidean distance between each grid point and each charge
#     r = jnp.linalg.norm(diff, axis=-1)
#     avg_chg = jnp.sum(chg_mask * charge_values) / jnp.sum(chg_mask)
#     new_charge_values = charge_values - avg_chg
#     C = nan_safe_coulomb_potential((chg_mask * new_charge_values)[None, :], r)
#     V = jnp.sum(C, axis=-1)
#     return V


def esp_loss_eval(pred, target, ngrid):
    target = target.flatten()
    esp_non_zero = np.nonzero(target)
    l2_loss = optax.l2_loss(pred[esp_non_zero], target[esp_non_zero])
    esp_loss = np.mean(l2_loss)
    esp_loss = esp_loss * 1  
    return esp_loss
    
batched_electrostatic_potential = vmap(calc_esp, in_axes=(0, 0, 0), out_axes=0)


def clip_colors(c):
    return np.clip(c, -0.015, 0.015)


@functools.partial(jax.jit, static_argnames=('batch_size', 'esp_w'))
def esp_mono_loss(dipo_prediction, mono_prediction, esp_target,
                  vdw_surface, mono, batch_size, esp_w):
    """
    """
    l2_loss_mono = optax.l2_loss(mono_prediction.sum(axis=-1), mono)
    mono_loss = jnp.mean(l2_loss_mono)
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, NATOMS * n_dcm, 3)
    mono = jnp.repeat(mono.reshape(batch_size, NATOMS), n_dcm, axis=-1)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface) 
    l2_loss = optax.l2_loss(batched_pred, esp_target)
    esp_loss = jnp.mean(l2_loss)
    return esp_loss*esp_w + mono_loss


def esp_mono_loss_pots(dipo_prediction, mono_prediction, esp_target,
                       vdw_surface, mono, batch_size):
    """
    """
    d = dipo_prediction.reshape(batch_size, NATOMS, 3, n_dcm)
    d = jnp.moveaxis(d, -1, -2)
    d = d.reshape(batch_size, NATOMS * n_dcm, 3)
    mono = jnp.repeat(mono.reshape(batch_size, NATOMS), n_dcm, axis=-1)
    m = mono_prediction.reshape(batch_size, NATOMS * n_dcm)

    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def esp_loss_pots(dipo_prediction, mono_prediction,
                  esp_target, vdw_surface, mono, batch_size):
    d = dipo_prediction.reshape(batch_size, NATOMS, 3)
    mono = mono.reshape(batch_size, NATOMS)
    m = mono_prediction.reshape(batch_size, NATOMS)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def mean_absolute_error(prediction, target, batch_size):
    nonzero = jnp.nonzero(target, size=batch_size * 60)
    return jnp.mean(jnp.abs(prediction[nonzero] - target[nonzero]))

@functools.partial(jax.jit,
                   static_argnames=(
                   'model_apply', 'optimizer_update', 'batch_size', 'esp_w'))
def train_step(model_apply, optimizer_update, batch,
               batch_size, opt_state, params, esp_w):
    def loss_fn(params):
        mono, dipo = model_apply(
            params,
            atomic_numbers=batch['atomic_numbers'],
            positions=batch['positions'],
            dst_idx=batch['dst_idx'],
            src_idx=batch['src_idx'],
            batch_segments=batch['batch_segments'],
        )
        loss = esp_mono_loss(
            dipo_prediction=dipo,
            mono_prediction=mono,
            vdw_surface=batch['vdw_surface'],
            esp_target=batch['esp'],
            mono=batch['mono'],
            batch_size=batch_size,
            esp_w=esp_w
        )
        return loss, (mono, dipo)

    (loss, (mono, dipo)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size', 'esp_w'))
def eval_step(model_apply, batch, batch_size, params, esp_w):
    mono, dipo = model_apply(
        params,
        atomic_numbers=batch['atomic_numbers'],
        positions=batch['positions'],
        dst_idx=batch['dst_idx'],
        src_idx=batch['src_idx'],
        batch_segments=batch['batch_segments'],
        batch_size=batch_size
    )
    loss = esp_mono_loss(
        dipo_prediction=dipo,
        mono_prediction=mono,
        vdw_surface=batch['vdw_surface'],
        esp_target=batch['esp'],
        mono=batch['mono'],
        batch_size=batch_size,
        esp_w=esp_w
    )
    return loss


def train_model(key, model, train_data, valid_data,
                num_epochs, learning_rate, batch_size,
                esp_w=1.0,
                restart_params=None):
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(
        len(train_data['atomic_numbers'][0]))
    params = model.init(init_key,
                        atomic_numbers=train_data['atomic_numbers'][0],
                        positions=train_data['positions'][0],
                        dst_idx=dst_idx,
                        src_idx=src_idx,
                        )
    if restart_params is not None:
        params = restart_params

    opt_state = optimizer.init(params)

    print("Preparing batches")
    print("..................")
    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)
        # Loop over train batches.
        train_loss = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w
            )
            train_loss += (loss - train_loss) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        for i, batch in enumerate(valid_batches):
            loss = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=params,
                esp_w=esp_w
            )
            valid_loss += (loss - valid_loss) / (i + 1)

        # Print progress.
        print(f"epoch: {epoch: 3d}      train:   valid:")
        print(f"    loss [a.u.]             {train_loss : 8.3e} {valid_loss : 8.3e}")

    # Return final model parameters.
    return params, valid_loss

def safe_mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("--data_dir", type=str, default="data")
    args.add_argument("--model_dir", type=str, default="model")
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--learning_rate", type=float, default=0.0001)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--esp_w", type=float, default=10.0)
    args.add_argument("--num_epics", type=int, default=100)
    args.add_argument("--restart", type=str, default=None)
    args.add_argument("--random_seed", type=int, default=0)
    args.add_argument("--n_dcm", type=int, default=1)    
    args.add_argument("--n_gpu", type=str, default="0")
    args = args.parse_args()
    print("args:")
    for k, v in vars(args).items():
        print(f"{k} = {v}")
 
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.n_gpu
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())

    NATOMS = 60
    data_key, train_key = jax.random.split(jax.random.PRNGKey(
        args.random_seed
    ), 2)

    # Model hyperparameters.
    features = 16
    max_degree = 2
    num_iterations = 3
    num_basis_functions = 16
    cutoff = 4.0

    n_dcm = args.n_dcm
    # Training hyperparameters.
    learning_rate = args.learning_rate
    batch_size = 16
    esp_w = args.esp_w
    restart_params = args.restart
    if restart_params is not None:
        restart_params = pd.read_pickle(restart_params)
    params = restart_params    
    num_epochs = args.num_epochs

    train_data, valid_data = prepare_datasets(
        data_key,
        2 ** 14,
        2 ** 11,
        filename="data/qm9-esp20000.npz")

    # Create and train model.
    message_passing_model = MessagePassingModel(
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        n_dcm=n_dcm,
    )

    # make checkpoint directory
    safe_mkdir(f'checkpoints2/dcm{n_dcm}-{esp_w}')

    for epic in range(20):
        print(f"epic {epic}")
        params, val = train_model(
            key=train_key,
            model=message_passing_model,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            restart_params=params,
            esp_w=esp_w*epic
        )

        # open a file, where you want to store the data
        with open(f'checkpoints2/dcm{n_dcm}-{esp_w}/{epic}-{val}-esp_params.pkl',
                  'wb') as file:
            pickle.dump(params, file)


