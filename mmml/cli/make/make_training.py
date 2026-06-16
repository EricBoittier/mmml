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
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from datetime import datetime
import jax

import yaml

# from mmml.models.physnetjax.physnetjax.models import model as model
from mmml.models.physnetjax.physnetjax.models.model import EF
from mmml.models.physnetjax.physnetjax.training.training import train_model
from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets
# from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit

import numpy as np

# YAML / config aliases (e.g. train -> data, output -> ckpt_dir)
CONFIG_ALIASES: Dict[str, str] = {
    "train": "data",
    "train_file": "data",
    "valid": "valid_data",
    "valid_file": "valid_data",
    "output": "ckpt_dir",
    "output_dir": "ckpt_dir",
    "max_epochs": "num_epochs",
    "epochs": "num_epochs",
    "model_file": "model",
    "restart_file": "restart",
}


def _normalize_config_key(key: str) -> str:
    return CONFIG_ALIASES.get(key, key).replace("-", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PhysNetJAX EF model from NPZ data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mmml physnet-train \\
      --data output/energies_forces_dipoles_train.npz \\
      --ckpt-dir ./ckpts/ama_mp2 \\
      --tag ama_mp2 \\
      --n-train 24000 --n-valid 3000 \\
      --batch-size 32 --num-epochs 2000 \\
      --max-atomic-number 35

  mmml physnet-train --config train.yaml

YAML keys match CLI flags (with optional aliases: train, output, max_epochs).
See mmml/cli/misc/physnet_train.example.yaml for a template.
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file with training options (CLI flags override file values)",
    )
    parser.add_argument("--data", type=str, default=None, help="Training NPZ file")
    parser.add_argument(
        "--valid-data",
        type=str,
        default=None,
        dest="valid_data",
        help="Optional validation NPZ (use full files; no random re-split)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        dest="ckpt_dir",
        help="Checkpoint directory (absolute path used for Orbax)",
    )

    parser.add_argument("--tag", type=str, default="run", help="Run name for checkpoints")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model JSON to load instead of creating a new EF model",
    )
    parser.add_argument("--n-train", type=int, default=1000, dest="n_train")
    parser.add_argument("--n-valid", type=int, default=100, dest="n_valid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    parser.add_argument("--num-epochs", type=int, default=100, dest="num_epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, dest="learning_rate")
    parser.add_argument("--energy-weight", type=float, default=1.0, dest="energy_weight")
    parser.add_argument("--forces-weight", type=float, default=52.91, dest="forces_weight")
    parser.add_argument("--dipole-weight", type=float, default=27.21, dest="dipole_weight")
    parser.add_argument("--charges-weight", type=float, default=14.39, dest="charges_weight")
    parser.add_argument("--objective", type=str, default="valid_loss")
    parser.add_argument("--restart", type=str, default=None, help="Checkpoint path to restart from")

    parser.add_argument(
        "--num-atoms",
        type=int,
        default=None,
        dest="num_atoms",
        help="Atoms per structure (auto-detected from N/R if omitted)",
    )
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--max-degree", type=int, default=0, dest="max_degree")
    parser.add_argument("--num-basis-functions", type=int, default=32, dest="num_basis_functions")
    parser.add_argument("--num-iterations", type=int, default=2, dest="num_iterations")
    parser.add_argument("--n-res", type=int, default=2, dest="n_res")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-atomic-number", type=int, default=28, dest="max_atomic_number")
    parser.add_argument("--zbl", action="store_true", default=False)
    parser.add_argument("--use-pbc", action="store_true", default=False, dest="use_pbc")
    parser.add_argument(
        "--no-energy-bias",
        action="store_false",
        dest="use_energy_bias",
        help="Disable per-element energy bias in the model",
    )
    parser.set_defaults(use_energy_bias=True)
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Write resolved training options to YAML and exit",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress JAX device summary",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")
    return raw


def apply_mapping_to_namespace(
    args: argparse.Namespace,
    mapping: Mapping[str, Any],
    *,
    source: str,
) -> None:
    unknown = []
    for raw_key, value in mapping.items():
        key = _normalize_config_key(str(raw_key))
        if not hasattr(args, key):
            unknown.append(str(raw_key))
            continue
        setattr(args, key, value)
    if unknown:
        raise ValueError(
            f"Unknown {source} key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys include: {', '.join(sorted(k for k in vars(args) if not k.startswith('_')))}"
        )


def namespace_from_yaml(path: str | Path) -> argparse.Namespace:
    args = parse_args([])
    apply_mapping_to_namespace(args, load_yaml_config(path), source=f"config '{path}'")
    return args


def parse_train_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args, optionally seeded from --config YAML."""
    parser = build_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)

    defaults = vars(parse_args([]))
    if pre_args.config:
        file_args = vars(namespace_from_yaml(pre_args.config))
        defaults.update(file_args)
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def save_train_config(args: argparse.Namespace, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in sorted(vars(args).items()) if k != "save_config"}
    with out.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def validate_train_args(args: argparse.Namespace) -> None:
    if not args.data:
        raise ValueError("--data is required (or set 'data' / 'train' in --config)")
    if args.valid_data:
        if args.n_train > 0 or args.n_valid > 0:
            raise ValueError(
                "With --valid-data, do not set --n-train/--n-valid (full files are used)"
            )
        return
    if args.n_train < 0 or args.n_valid < 0:
        raise ValueError("--n-train and --n-valid must be >= 0")
    if args.n_train + args.n_valid <= 0:
        raise ValueError("At least one of --n-train or --n-valid must be > 0")


def args_from_kwargs(**overrides) -> argparse.Namespace:
    """Create an argparse-like Namespace using defaults, then apply overrides.

    Handy for notebook use: import this module and call `run_notebook(...)`
    with keyword arguments instead of constructing CLI strings.
    """
    args = parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown argument: {key}")
        setattr(args, key, value)
    return args


def log_jax_devices():
    """Print a short JAX device summary."""
    devices = jax.local_devices()
    print(devices)
    print(jax.default_backend())
    print(jax.devices())



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


def _maybe_unpad_dataset(data_path: str, natoms: Optional[int]) -> tuple[str, int]:
    """Return (path, natoms), optionally writing an unpadded NPZ."""
    data = np.load(data_path, allow_pickle=True)
    if natoms is not None:
        return data_path, natoms

    if "N" in data:
        max_n = int(np.max(data["N"]))
        if "R" in data and len(data["R"].shape) >= 2:
            padded_atoms = int(data["R"].shape[1])
            if padded_atoms > max_n:
                print(f"  ⚠️  Data is PADDED: {padded_atoms} atoms in array (padding: {padded_atoms - max_n})")
                print("  🔧 Auto-removing padding to train efficiently...")
                data_unpadded = {}
                for key, value in data.items():
                    arr = np.asarray(value)
                    if key == "R" and arr.ndim == 3:
                        data_unpadded[key] = arr[:, :max_n, :]
                    elif key == "Z" and arr.ndim == 2:
                        data_unpadded[key] = arr[:, :max_n]
                    elif key == "F" and arr.ndim == 3:
                        data_unpadded[key] = arr[:, :max_n, :]
                    else:
                        data_unpadded[key] = value
                unpadded_path = Path(data_path).parent / f"{Path(data_path).stem}_unpadded.npz"
                np.savez_compressed(unpadded_path, **data_unpadded)
                print(f"  ✅ Saved unpadded data to: {unpadded_path}")
                return str(unpadded_path), max_n
            return data_path, padded_atoms
        return data_path, max_n
    if "R" in data and len(data["R"].shape) >= 2:
        return data_path, int(data["R"].shape[1])
    raise ValueError("Could not auto-detect num_atoms from dataset. Please specify --num-atoms.")


def _load_physnet_npz_dict(path: str, natoms: int) -> dict:
    """Load one NPZ split into the dict format expected by train_model."""
    from mmml.models.physnetjax.physnetjax.data.data import make_dicts, prepare_multiple_datasets

    data, keys, _, _ = prepare_multiple_datasets(
        jax.random.PRNGKey(0),
        train_size=0,
        valid_size=0,
        filename=[path],
        natoms=natoms,
        verbose=False,
    )
    n_samples = data[keys.index("R")].shape[0]
    all_idx = np.arange(n_samples)
    train_data, _ = make_dicts(data, keys, all_idx, np.array([], dtype=np.int64))
    return train_data


def main_loop(args):
    seed = args.seed
    data_key, train_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir).resolve()
        print(f"Checkpoint directory (absolute): {ckpt_dir}")
    else:
        ckpt_dir = None

    if args.num_atoms is None:
        print("Auto-detecting number of atoms from dataset...")
        data_path, natoms = _maybe_unpad_dataset(args.data, None)
        args.data = data_path
    else:
        natoms = args.num_atoms
        print(f"Using specified num_atoms = {natoms}")
        data_path, natoms = _maybe_unpad_dataset(args.data, natoms)
        args.data = data_path

    if args.valid_data:
        valid_path, _ = _maybe_unpad_dataset(args.valid_data, natoms)
        args.valid_data = valid_path
        print(f"Using fixed splits:\n  train: {args.data}\n  valid: {args.valid_data}")
        train_data = _load_physnet_npz_dict(args.data, natoms)
        valid_data = _load_physnet_npz_dict(args.valid_data, natoms)
    else:
        files = [args.data]
        train_data, valid_data = prepare_datasets(
            data_key, args.n_train, args.n_valid, files, natoms=natoms
        )
    
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
            zbl=args.zbl, # TODO: add zbl
            efa=False, # TODO: add efa
            use_pbc=args.use_pbc,
            use_energy_bias=args.use_energy_bias,
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
        forces_weight=args.forces_weight,
        dipole_weight=args.dipole_weight,
        charges_weight=args.charges_weight,
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
    params_path = (ckpt_dir / f"params_{args.tag}_{now}.json") if ckpt_dir else Path(f"params_{args.tag}_{now}.json")
    if ckpt_dir:
        params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, 'w') as f:
        print(f"Saving parameters to {params_path}")
        json.dump(params_out, f, default=to_jsonable)

    return params_out, params_path
    


def run(args):
    print(args)
    return main_loop(args)


def run_notebook(**kwargs):
    """Convenience entrypoint for notebooks.

    Example:
        from mmml.cli import make_training

        params, params_path = make_training.run_notebook(
            data="train.npz",
            ckpt_dir="/tmp/ckpts",
            tag="run",
            model=None,
            n_train=1000,
            n_valid=100,
            seed=42,
            batch_size=1,
            num_epochs=2,
            learning_rate=0.001,
            energy_weight=1,
            objective="valid_loss",
            restart=None,
            num_atoms=None,
            features=64,
            max_degree=0,
            num_basis_functions=32,
            num_iterations=2,
            n_res=2,
            cutoff=8.0,
            max_atomic_number=28,
        )
    """
    args = args_from_kwargs(**kwargs)
    return run(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import sys

    try:
        args = parse_train_args(argv)
        if args.save_config:
            save_train_config(args, args.save_config)
            print(f"Wrote training config to {args.save_config}")
            return 0
        validate_train_args(args)
        if not args.quiet:
            log_jax_devices()
        run(args)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cli_main():
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()