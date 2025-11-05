"""
Sets up a training set for the ML model.

Args: 
    -d, --data the npz file to use for training
    -t, --tag the name of the run
    -m, --model the model to use for training, as .inp file
    -n, --n_train the number of training samples to use
    -v, --n_valid the number of validation samples to use
    -s, --seed the seed for the random number generator
    -b, --batch_size the batch size for training
    -e, --num_epochs the number of epochs to train for
    -l, --learning_rate the learning rate for training
    -w, --energy_weight the weight for the energy loss
    -o, --objective the objective function to optimize
    -r, --restart the restart file to use for training
    -c, --ckpt_dir the directory to save the checkpoints to
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from datetime import datetime
# Check JAX configuration
import jax
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

import mmml
import ase
import os
from pathlib import Path

# from mmml.physnetjax.physnetjax.models import model as model
from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.training import train_model
from mmml.physnetjax.physnetjax.data.data import prepare_datasets
# from mmml.physnetjax.physnetjax.data.batches import prepare_batches_jit

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str)  # Will be converted to absolute Path later

    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--energy_weight", type=float, default=1)
    parser.add_argument("--objective", type=str, default="valid_loss")
    parser.add_argument("--restart", type=str, default=None)
    
    parser.add_argument("--num_atoms", type=int, default=None,
                        help="Number of atoms per structure (auto-detected if not specified)")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--max_degree", type=int, default=0)
    parser.add_argument("--num_basis_functions", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--n_res", type=int, default=2)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_atomic_number", type=int, default=28)
    return parser.parse_args()



def to_jsonable(obj: Any):
    """Recursively convert JAX/NumPy objects to JSON-serializable types."""
    # Handle JAX arrays (ArrayImpl) and NumPy arrays
    try:
        import jax
        jax_array_cls = getattr(jax, "Array", None)
        jax_array_types = (jax_array_cls,) if jax_array_cls is not None else tuple()
    except Exception:
        jax_array_types = tuple()

    if isinstance(obj, (np.ndarray,)) or (jax_array_types and isinstance(obj, jax_array_types)):
        return np.asarray(obj).tolist()
    # NumPy scalar types
    if isinstance(obj, np.generic):
        return obj.item()
    # Basic containers
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    # Paths
    if isinstance(obj, Path):
        return str(obj)
    # Fallback: return as-is; json will handle primitives
    return obj

def load_model(model_file):
    # load the model from the file
    with open(model_file, 'r') as f:
        model = EF(**json.load(f))
    return model


def main_loop(args):
    seed = args.seed
    data_key, train_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    files = [args.data]
    train_size = args.n_train
    valid_size = args.n_valid
    
    # Convert ckpt_dir to absolute path (required by Orbax)
    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir).resolve()
        print(f"Checkpoint directory (absolute): {ckpt_dir}")
    else:
        ckpt_dir = None
    
    # Auto-detect num_atoms from dataset if not provided
    if args.num_atoms is None:
        print("Auto-detecting number of atoms from dataset...")
        data = np.load(args.data)
        
        # Prefer R shape over N since R includes padding (which is what we need for training)
        if 'R' in data and len(data['R'].shape) >= 2:
            natoms = int(data['R'].shape[1])
            print(f"  ✅ Detected num_atoms = {natoms} from R.shape (includes padding)")
            
            # Also report actual molecule size if N exists
            if 'N' in data:
                max_n = int(np.max(data['N']))
                if max_n != natoms:
                    print(f"     (Note: Actual molecule size = {max_n}, padded to {natoms})")
        elif 'N' in data:
            natoms = int(np.max(data['N']))
            print(f"  ✅ Detected num_atoms = {natoms} from max(N)")
        else:
            raise ValueError("Could not auto-detect num_atoms from dataset. Please specify --num_atoms explicitly.")
    else:
        natoms = args.num_atoms
        print(f"Using specified num_atoms = {natoms}")
    
    train_data, valid_data = prepare_datasets(data_key, train_size, valid_size, files, natoms=natoms)
    
    if args.model is not None:
        model = load_model(args.model)
    else:
        model = EF(
            features=args.features,
            max_degree=args.max_degree,
            num_basis_functions=args.num_basis_functions,
            num_iterations=args.num_iterations,
            n_res=args.n_res,
            cutoff=args.cutoff,
            max_atomic_number=args.max_atomic_number,
            zbl=False, # TODO: add zbl
            efa=False, # TODO: add efa
        )
        try:
            # save the model to a file
            with open("args.model.json", 'w') as f:
                print("Saving model to args.model.json")
                print(model.return_attributes())
                json.dump(model.return_attributes(), f, default=to_jsonable)
        except Exception as e:
            print(e)
            pass
    
    params_out = train_model(
        train_key,
        model,
        train_data,
        valid_data, 
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_atoms=natoms,
        energy_weight=args.energy_weight,
        restart=args.restart,
        conversion={'energy': 1, 'forces': 1},
        print_freq=1,
        name=args.tag,
        best=False,
        optimizer=None,
        transform=None,
        schedule_fn=None,
        objective=args.objective,
        ckpt_dir=ckpt_dir,  # Use absolute path
        log_tb=False,
        batch_method="default",
        batch_args_dict=None,
        data_keys=('R', 'Z', 'F', "N", 'E', 'D', 'batch_segments'),
        num_epochs=args.num_epochs,
    )

    # save the parameters named with the date-time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save as a directory with the name of the run
    # save the parameter dictionary
    with open(f"params{now}.json", 'w') as f:
        print("Saving parameters to params.json")
        print(params_out)
        json.dump(params_out, f, default=to_jsonable)
    


def main():
    args = parse_args()
    print(args)
    main_loop(args)

if __name__ == "__main__":
    main()