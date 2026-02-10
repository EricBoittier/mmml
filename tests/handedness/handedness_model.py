import functools
import e3x
from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# Disable future warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", type=str, default="handedness.npz")
  parser.add_argument("--features", type=int, default=128)
  parser.add_argument("--max_degree", type=int, default=4)
  parser.add_argument("--num_iterations", type=int, default=3)
  parser.add_argument("--num_basis_functions", type=int, default=32)
  parser.add_argument("--cutoff", type=float, default=10.0)
  parser.add_argument("--num_train", type=int, default=800)
  parser.add_argument("--num_valid", type=int, default=200)
  parser.add_argument("--num_epochs", type=int, default=100)
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--seed", type=int, default=42)




class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 55  
    include_pseudotensors: bool = True

    def handedness(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        num_atoms = positions.shape[1]
    
        # Flatten batch
        positions_flat = positions.reshape(-1, 3)       # (batch_size*num_atoms, 3)
        atomic_numbers_flat = atomic_numbers.reshape(-1)  # (batch_size*num_atoms,)
    
        # Adjust indices for batching
        offsets = jnp.arange(batch_size) * num_atoms
        dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
    
        # Compute displacements
        displacements = positions_flat[src_idx_flat] - positions_flat[dst_idx_flat]
    
        # Compute basis
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )
    
        # Embed atomic numbers
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1,
                         features=self.features)(atomic_numbers_flat)
        # x.shape == (num_atoms_flat, features) → correct for MessagePass
    
        # Message-passing
        for i in range(self.num_iterations):
            y = e3x.nn.MessagePass(include_pseudotensors=self.include_pseudotensors,max_degree=self.max_degree if i < self.num_iterations-1 else 0)(
                x, basis, dst_idx=dst_idx_flat, src_idx=src_idx_flat
            )
    
            x = e3x.nn.add(x, y)
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(x)
            x = e3x.nn.add(x, y)
    
        # Atomic contributions
        # element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1,))
        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=-1)
        # atomic_energies += element_bias[atomic_numbers_flat]
    
        # Sum per batch
        # energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)
        # energy = energy.reshape((batch_size, 1, 1))
        final = e3x.nn.activations.hard_tanh(atomic_energies)


        # jax.debug.print("{x} {y}", x=final, y=final.sum())
        
        return final.sum()


    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
        if batch_segments is None:
            batch_segments = jnp.zeros(atomic_numbers.shape[:1], dtype=jnp.int32)
            batch_size = 1
        return self.handedness(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)


def prepare_datasets(key, num_train, num_valid, dataset):

  # Make sure that the dataset contains enough entries.
  num_data = len(dataset['R'])
  num_draw = num_train + num_valid
  if num_draw > num_data:
    raise RuntimeError(
      f'datasets only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}')

  # Randomly draw train and validation sets from dataset.
  choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
  train_choice = choice[:num_train]
  valid_choice = choice[num_train:]


  # Collect and return train and validation sets.
  train_data = dict(
    handedness=jnp.asarray([-1.0 if _ == "R" else 1.0 for _ in dataset["handedness"]],dtype=jnp.float32)[train_choice],
    atomic_numbers=jnp.asarray(dataset['Z'], dtype=jnp.int32)[train_choice],
    positions=jnp.asarray(dataset['R'], dtype=jnp.float32)[train_choice],
  )
  valid_data = dict(
    handedness=jnp.asarray([-1 if _ == "R" else 1 for _ in dataset["handedness"]], dtype=jnp.float32)[valid_choice],
    atomic_numbers=jnp.asarray(dataset['Z'], dtype=jnp.int32)[valid_choice],
    positions=jnp.asarray(dataset['R'], dtype=jnp.float32)[valid_choice],
  )
  return train_data, valid_data


def mean_squared_loss(handedness_prediction,handedness_target ):
  return jnp.mean(optax.l2_loss(handedness_prediction, handedness_target))


def mean_absolute_error(prediction, target):
  return jnp.mean(jnp.abs(prediction - target))



def prepare_batches(key, data, batch_size):
  # Determine the number of training steps per epoch.
  data_size = len(data['handedness'])
  steps_per_epoch = data_size//batch_size

  # Draw random permutations for fetching batches from the train data.
  perms = jax.random.permutation(key, data_size)
  perms = perms[:steps_per_epoch * batch_size]  # Skip the last batch (if incomplete).
  perms = perms.reshape((steps_per_epoch, batch_size))

  # Prepare entries that are identical for each batch.
  num_atoms = 60
  batch_segments = jnp.zeros(60, dtype=jnp.int32)
  atomic_numbers = jnp.tile(data['atomic_numbers'], batch_size)
  offsets = jnp.arange(batch_size) * num_atoms
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
  dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
  src_idx = (src_idx + offsets[:, None]).reshape(-1)

  # Assemble and return batches.
  return [
    dict(
        handedness=np.atleast_2d(data['handedness'][perm]),
        atomic_numbers=atomic_numbers,
        positions=data['positions'][perm].reshape(-1, 3),
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments = batch_segments,
    )
    for perm in perms
  ]



@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update', 'batch_size'))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params):
  def loss_fn(params):
    handedness = model_apply(
      params,
      atomic_numbers=batch['atomic_numbers'],
      positions=batch['positions'],
      dst_idx=batch['dst_idx'],
      src_idx=batch['src_idx'],
      batch_segments=batch['batch_segments'],
      batch_size=batch_size
    )
    loss = mean_squared_loss(
      handedness_prediction=handedness.flatten(),
      handedness_target=batch['handedness'].flatten(),

    )
    return loss, handedness
  (loss, handedness), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
  updates, opt_state = optimizer_update(grad, opt_state, params)
  params = optax.apply_updates(params, updates)
  handedness_mae = mean_absolute_error(handedness, batch['handedness'])
  
  return params, opt_state, loss, handedness_mae


@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size'))
def eval_step(model_apply, batch, batch_size, params):
  handedness = model_apply(
    params,
    atomic_numbers=batch['atomic_numbers'],
    positions=batch['positions'],
    dst_idx=batch['dst_idx'],
    src_idx=batch['src_idx'],
    batch_segments=batch['batch_segments'],
    batch_size=batch_size
  )
  loss = mean_squared_loss(
    handedness_prediction=handedness.flatten(),
    handedness_target=batch['handedness'].flatten(),
    
  )
  handedness_mae = mean_absolute_error(handedness, batch['handedness'])
  return loss, handedness_mae


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size):
  # Initialize model parameters and optimizer state.
  key, init_key = jax.random.split(key)
  optimizer = optax.adam(learning_rate)
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(60)
  params = model.init(init_key,
    atomic_numbers=train_data['atomic_numbers'][0],
    positions=train_data['positions'][0],
    dst_idx=dst_idx,
    src_idx=src_idx,
  )
  opt_state = optimizer.init(params)

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
    train_handedness_mae = 0.0

    for i, batch in enumerate(train_batches):
      params, opt_state, loss, handedness_mae = train_step(
        model_apply=model.apply,
        optimizer_update=optimizer.update,
        batch=batch,
        batch_size=batch_size,
        opt_state=opt_state,
        params=params
      )
      train_loss += (loss - train_loss)/(i+1)
      train_handedness_mae += (handedness_mae - train_handedness_mae)/(i+1)

    # Evaluate on validation set.
    valid_loss = 0.0
    valid_handedness_mae = 0.0
    for i, batch in enumerate(valid_batches):
      loss, handedness_mae = eval_step(
        model_apply=model.apply,
        batch=batch,
        batch_size=batch_size,
        params=params
      )
      valid_loss += (loss - valid_loss)/(i+1)
      valid_handedness_mae += (handedness_mae - valid_handedness_mae)/(i+1)

    # Print progress.
    print(f"epoch: {epoch: 3d}                    train:   valid:")
    print(f"    loss [a.u.]             {train_loss : 8.3f} {valid_loss : 8.3f}")
    print(f"    handedness mae [kcal/mol]   {train_handedness_mae: 8.3f} {valid_handedness_mae: 8.3f}")


  # Return final model parameters.
  return params



def main(args):
  key = jax.random.PRNGKey(args.seed)
  dataset = np.load(args.data)
  train_data, valid_data = prepare_datasets(key, args.num_train, args.num_valid, dataset)
  model = MessagePassingModel(
    features=args.features, 
    max_degree=args.max_degree, 
    num_iterations=args.num_iterations,
     num_basis_functions=args.num_basis_functions, 
     cutoff=args.cutoff)

  params = train_model(
    key, model, train_data, valid_data,
     args.num_epochs, 
     args.learning_rate, 
     args.batch_size)
     
  return params


if __name__ == "__main__":
  args = get_args()
  params = main(args)
  print(params)