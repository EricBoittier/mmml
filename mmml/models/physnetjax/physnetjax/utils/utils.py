import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp

DTYPE = jnp.float32

def parse_non_int(s):
    return "".join([_ for _ in s if _.isalpha()]).lower().capitalize()

    
def create_checkpoint_dir(name: str, base: Path) -> Path:
    """Create a unique checkpoint directory path.

    Args:
        name: Base name for the checkpoint directory

    Returns:
        Path object for the checkpoint directory
    """
    uuid_ = str(uuid.uuid4())
    return base / f"/{name}-{uuid_}/"


def get_epoch_weights(epoch: int) -> Tuple[float, float]:
    """Calculate energy and forces weights based on epoch number.

    Args:
        epoch: Current training epoch

    Returns:
        Tuple of (energy_weight, forces_weight)
    """
    if epoch < 500:
        return 1.0, 1000.0
    elif epoch < 1000:
        return 1000.0, 1.0
    else:
        return 1.0, 50.0


def sort_names_safe(x) -> int:
    """"""
    if ("/" in x) and ("-" in x):
        _ = str(x).split("/")[-1].split("-")[-1]

    if _.isdigit():
        return int(_)

    return -1

def get_files(path: str) -> List[Path]:
    """Get sorted epoch checkpoint directories, ignoring orbax-internal dirs."""
    dirs = list(Path(path).glob("epoch-*/"))
    dirs = [d for d in dirs if "tfevent" not in str(d) and "tmp" not in str(d)]
    dirs.sort(key=lambda x: sort_names_safe(str(x)))
    return dirs


def _process_model_attributes(
    attrs: Dict[str, Any], natoms: int = None
) -> Dict[str, Any]:
    """Process model attributes from checkpoint."""
    kwargs = attrs.copy()

    int_fields = [
        "features",
        "max_degree",
        "num_iterations",
        "num_basis_functions",
        "natoms",
        "n_res",
        "max_atomic_number",
    ]
    float_fields = ["cutoff", "total_charge"]
    bool_fields = ["charges", "zbl", "efa", "charges"]

    import re

    non_decimal = re.compile(r"^-?[0-9]\d*(\.\d+)?$")
    for field in int_fields:
        _ = kwargs[field]
        print(field, _)
        if isinstance(_, str):
            kwargs[field] = int(non_decimal.sub("", _))
        elif isinstance(_, int):
            kwargs[field] = int(_)
        elif _ is None:
            pass
        else:
            raise ValueError(f"Field {field} is not an int or string")
    for field in float_fields:
        _ = kwargs[field]
        print(field, _)
        if isinstance(_, str):
            try:
                kwargs[field] = float(non_decimal.sub("", _))
            except ValueError:
                print(f"Could not convert {field} to float")
                print(f"Setting {field} to 0.0")
                kwargs[field] = 0.0
        elif isinstance(_, float):
            kwargs[field] = float(_)
        elif _ is None:
            pass
        else:
            raise ValueError(f"Field {field} is not a float or string")
    for field in bool_fields:
        if field in kwargs.keys():
            kwargs[field] = bool(str(kwargs[field]))
        else:
            print(f"Field {field} not found in model attributes")
            kwargs[field] = False

    kwargs["debug"] = []
    if natoms is not None:
        kwargs["natoms"] = natoms

    return kwargs
