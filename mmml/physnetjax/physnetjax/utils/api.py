import os

from physnetjax.data import prepare_datasets

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

from physnetjax.models.model import EF


# Check JAX configuration
def print_jax_config():
    devices = jax.local_devices()
    print("JAX devices:", devices)
    print("JAX default backend:", jax.default_backend())
    print("JAX devices:", jax.devices())


print_jax_config()
import argparse

import jax
from model import EF

from training import train_model

"""
Usage example:
python api.py --data <path_to_data> --ntrain 5000 --nvalid 500 --features 32 --max_degree 3 --num_iterations 2 --num_basis_functions 16 --cutoff 6.0 --max_atomic_number 118 --n_res 3
"""

DEFAULT_DATA_KEYS = ["Z", "R", "D", "E", "N", "F"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a molecular dynamics model")
    # Add all arguments here
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--restart", type=str, default=False, help="Restart training from a checkpoint"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--train_key", type=str, default=None)
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--ntrain", type=int, default=500)
    parser.add_argument("--nvalid", type=int, default=500)
    parser.add_argument("--name", type=str, default="diox-q3.2")
    parser.add_argument("--natoms", type=int, default=8)
    parser.add_argument("--forces_w", type=float, default=52.917721)
    parser.add_argument("--dipole_w", type=float, default=27.211386)
    parser.add_argument("--charges_w", type=float, default=14.399645)
    parser.add_argument("--energy_w", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="amsgrad")
    parser.add_argument("--schedule", type=str, default="constant")
    parser.add_argument("--transform", type=str, default=None)
    # Model parameters
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_basis_functions", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max_atomic_number", type=int, default=9)
    parser.add_argument("--n_res", type=int, default=3)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--total_charge", default=0)

    return parser.parse_args()


if __name__ == "__main__":

    def main():
        """"""
        args = parse_arguments()
        print("Parsed arguments:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

        data_key, train_key = jax.random.split(jax.random.PRNGKey(43), 2)

        # Load data
        files = [args.data]
        train_data, valid_data = prepare_datasets(
            data_key,
            args.ntrain,
            args.nvalid,
            files,
            clip_esp=False,
            natoms=args.natoms,
            clean=False,
        )
        ntest = len(valid_data["E"]) // 2
        test_data = {k: v[ntest:] for k, v in valid_data.items()}
        valid_data = {k: v[:ntest] for k, v in valid_data.items()}

        print(f"Train data: {len(train_data['E'])}")
        print(f"Valid data: {len(valid_data['E'])}")
        print(f"Test data: {len(test_data['E'])}")

        # Create model
        model = EF(
            features=args.features,
            max_degree=args.max_degree,
            num_iterations=args.num_iterations,
            num_basis_functions=args.num_basis_functions,
            cutoff=args.cutoff,
            max_atomic_number=args.max_atomic_number,
            charges=True,
            zbl=False,
            natoms=args.natoms,
            total_charge=args.total_charge,
            n_res=args.n_res,
            debug=args.debug,
        )
        print(model)
        print("Training model:")
        _ = train_model(
            train_key,
            model,
            train_data,
            valid_data,
            num_epochs=args.nepochs,
            learning_rate=args.lr,
            forces_weight=args.forces_w,
            charges_weight=args.charges_w,
            dipole_weight=args.dipole_w,
            batch_size=args.batch_size,
            num_atoms=args.natoms,
            data_keys=DEFAULT_DATA_KEYS,
            restart=args.restart,
            name=args.name,
            optimizer=args.optimizer,
            schedule_fn=args.schedule,
            transform=args.transform,
        )

    main()
