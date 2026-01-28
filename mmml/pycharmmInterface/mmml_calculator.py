"""MM/ML calculator helpers.

This module historically pulled in a large collection of heavy optional
dependencies (ASE, PyCHARMM, JAX, ...) at import time. That behaviour made the
module unusable in documentation builds or test environments where the optional
packages were not installed. The fallout was a cascade of import errors for
code that only needed light-weight helpers such as the ``ev2kcalmol`` constant.

To make the module robust we now gate optional imports and provide small
shims that fail lazily when the associated functionality is requested. The
physics functionality is unchanged when the third-party libraries are present.
"""

from __future__ import annotations

from functools import partial
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

# In your module that defines spherical_cutoff_calculator
import jax.numpy as jnp
from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper


# CHARMM force-field definitions are optional.  During documentation builds we
# often do not have a functional PyCHARMM installation, so fall back to ``None``
# when the import fails for any reason (missing module or missing shared libs).
try:
    from mmml.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF
except Exception:  # pragma: no cover - exercised in lightweight envs
    CGENFF_PRM = CGENFF_RTF = None
try:
    from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
except ModuleNotFoundError:  # pragma: no cover - helper requires ASE

    def get_ase_calc(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("ase is required for get_ase_calc")
try:
    from mmml.physnetjax.physnetjax.data.batches import (
        _prepare_batches as prepare_batches,
    )
    from mmml.physnetjax.physnetjax.data.data import prepare_datasets
    from mmml.physnetjax.physnetjax.models.model import EF
    from mmml.physnetjax.physnetjax.restart.restart import get_files, get_last, get_params_model
    from mmml.physnetjax.physnetjax.training.loss import dipole_calc
    # Skip training import that requires lovely_jax
    # from mmml.physnetjax.physnetjax.training.training import train_model
    def train_model(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("lovely_jax is required for train_model")
except ModuleNotFoundError:  # pragma: no cover - ML stack optional for docs

    def prepare_batches(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("e3x and jax are required for prepare_batches")

    def prepare_datasets(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("e3x and jax are required for prepare_datasets")

    def EF(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("jax is required for EF model")

    def get_files(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("jax is required for restart helpers")

    def get_last(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("jax is required for restart helpers")

    def get_params_model(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("jax is required for restart helpers")

    def dipole_calc(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("jax is required for dipole calculations")

    def train_model(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("jax and optax are required for training")

# Optional imports ---------------------------------------------------------

try:  # JAX is required for the actual calculator but optional for docs/tests
    import jax
    from jax import Array, jit
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - exercised when JAX is absent
    jax = None  # type: ignore[assignment]
    Array = Any  # type: ignore[misc,assignment]

    def jit(fn: Callable) -> Callable:  # type: ignore[override]
        raise ModuleNotFoundError("jax is required for mmml_calculator functionality")

    jnp = None  # type: ignore[assignment]


try:  # matplotlib is only used for optional plotting utilities
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - plotting disabled without matplotlib
    plt = None  # type: ignore[assignment]


try:  # ASE is optional for documentation/tests that only need constants
    import ase  # type: ignore[import-not-found]
    import ase.calculators.calculator as ase_calc
    from ase import visualize
    from ase.io import read as read_ase
    from ase.visualize import view
    _HAVE_ASE = True
except ModuleNotFoundError:  # pragma: no cover - exercised during doc builds
    ase = None  # type: ignore[assignment]
    ase_calc = None  # type: ignore[assignment]
    visualize = None  # type: ignore[assignment]
    read_ase = None  # type: ignore[assignment]

    def view(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[override]
        raise ModuleNotFoundError("ase is required for visualisation support")

    _HAVE_ASE = False


try:  # PyCHARMM is optional and only required for the MM plumbing
    import pycharmm  # type: ignore[import-not-found]
    import pycharmm.cons_fix as cons_fix
    import pycharmm.cons_harm as cons_harm
    import pycharmm.coor as coor
    import pycharmm.crystal as crystal
    import pycharmm.dynamics as dyn
    import pycharmm.energy as energy
    import pycharmm.generate as gen
    import pycharmm.ic as ic
    import pycharmm.image as image
    import pycharmm.minimize as minimize
    import pycharmm.nbonds as nbonds
    import pycharmm.psf as psf
    import pycharmm.read as read
    import pycharmm.select as select
    import pycharmm.settings as settings
    import pycharmm.shake as shake
    import pycharmm.write as write
    from pycharmm.lib import charmm as libcharmm
    _HAVE_PYCHARMM = True
except Exception:  # pragma: no cover - exercised when PyCHARMM missing
    pycharmm = None  # type: ignore[assignment]
    cons_fix = None  # type: ignore[assignment]
    cons_harm = None  # type: ignore[assignment]
    coor = None  # type: ignore[assignment]
    crystal = None  # type: ignore[assignment]
    dyn = None  # type: ignore[assignment]
    energy = None  # type: ignore[assignment]
    gen = None  # type: ignore[assignment]
    ic = None  # type: ignore[assignment]
    image = None  # type: ignore[assignment]
    minimize = None  # type: ignore[assignment]
    nbonds = None  # type: ignore[assignment]
    psf = None  # type: ignore[assignment]
    read = None  # type: ignore[assignment]
    select = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    shake = None  # type: ignore[assignment]
    write = None  # type: ignore[assignment]
    libcharmm = None  # type: ignore[assignment]
    _HAVE_PYCHARMM = False


try:  # Additional optional scientific libraries used for ML/MM coupling
    import e3x  # type: ignore[import-not-found]
    _HAVE_E3X = True
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    e3x = None  # type: ignore[assignment]
    _HAVE_E3X = False


try:
    import optax  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    optax = None  # type: ignore[assignment]


# Public constants ---------------------------------------------------------

# Energy conversion (1 eV -> kcal / mol). Use ASE when available for
# consistency; otherwise fall back to the known constant so documentation tests
# can still import this module.
if _HAVE_ASE:
    ev2kcalmol = 1 / (ase.units.kcal / ase.units.mol)  # type: ignore[attr-defined]
else:
    ev2kcalmol = 23.060548867
kcalmol2ev = 1.0 / ev2kcalmol


# Module-level configuration ------------------------------------------------

SPATIAL_DIMS: int = 3  # Number of spatial dimensions (x, y, z)
GAMMA_ON = 1.0
GAMMA_OFF = 3.0

if jax is not None:  # pragma: no branch - keeps default behaviour when JAX present
    # If you want to perform simulations in float64 you have to call this before
    # any JAX computation.
    # jax.config.update('jax_enable_x64', True)
    data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)

    # Debug helpers that used to be printed on import are now toggled via
    # explicit logging to avoid side effects during import.
    try:
        devices = jax.local_devices()
        default_backend = jax.default_backend()
    except Exception:  # pragma: no cover - only hit when runtime misconfigured
        devices = []
        default_backend = "unknown"
else:  # pragma: no cover - executed in docs/test environments without JAX
    data_key = train_key = None
    devices = []
    default_backend = "unavailable"


def parse_non_int(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha()).lower().capitalize()


def dimer_permutations(n_mol: int) -> List[Tuple[int, int]]:
    return list(combinations(range(n_mol), 2))
# -----------------------------------------------------------------------------
# JAX-native smooth switching utilities (avoid traced-python conditionals)
# -----------------------------------------------------------------------------
def _safe_den(x: float | Array) -> Array:
    return jnp.maximum(x, 1e-6)

def jax_smooth_switch_linear(r: Array, x0: float, x1: float) -> Array:
    t = (r - x0) / _safe_den(x1 - x0)
    return jnp.clip(t, 0.0, 1.0)

def jax_smooth_cutoff_cosine(r: Array, cutoff: float) -> Array:
    t = r / _safe_den(cutoff)
    val = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    return jnp.where(r < cutoff, val, 0.0)

def ml_switch_simple(r: Array, ml_cutoff: float, mm_switch_on: float) -> Array:
    """ML active at short range, tapers 1→0 over [mm_switch_on - ml_cutoff, mm_switch_on].
    
    Args:
        r: distance
        ml_cutoff: width of ML taper region
        mm_switch_on: distance where ML reaches 0 and MM starts
    
    Returns:
        ML scale factor: 1.0 at short range, smooth taper to 0.0 at mm_switch_on
    """
    taper_start = mm_switch_on - ml_cutoff
    # Use cosine taper for smoothness
    t = (r - taper_start) / _safe_den(ml_cutoff)
    cosine_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    # Return 1.0 below taper_start, cosine taper in [taper_start, mm_switch_on], 0.0 above
    return jnp.where(r < taper_start, 1.0, jnp.where(r < mm_switch_on, cosine_taper, 0.0))

def mm_switch_simple(r: Array, mm_switch_on: float, mm_cutoff: float) -> Array:
    """MM off at short range, ramps 0→1 over [mm_switch_on, mm_switch_on + mm_cutoff].
    
    Args:
        r: distance
        mm_switch_on: distance where MM starts ramping on
        mm_cutoff: width of MM ramp region
    
    Returns:
        MM scale factor: 0.0 at short range, smooth ramp to 1.0 at mm_switch_on + mm_cutoff
    """
    ramp_end = mm_switch_on + mm_cutoff
    # Use cosine ramp for smoothness (inverse of ML taper)
    t = (r - mm_switch_on) / _safe_den(mm_cutoff)
    cosine_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(t, 0.0, 1.0)))
    # Return 0.0 below mm_switch_on, cosine ramp in [mm_switch_on, ramp_end], 1.0 above
    return jnp.where(r < mm_switch_on, 0.0, jnp.where(r < ramp_end, cosine_ramp, 1.0))




def set_pycharmm_xyz(atom_positions):
    xyz = pd.DataFrame(atom_positions, columns=["x", "y", "z"])
    coor.set_positions(xyz)


def capture_neighbour_list():
    # Print something
    distance_command = """
    open unit 1 write form name total.dmat
    
    COOR DMAT SINGLE UNIT 1 SELE ALL END SELE ALL END
    
    close unit 1"""
    _ = pycharmm.lingo.charmm_script(distance_command)

    with open("total.dmat") as f:
        output_dmat = f.read()

    atom_number_type_dict = {}
    atom_number_resid_dict = {}

    pair_distance_dict = {}
    pair_resid_dict = {}

    for _ in output_dmat.split("\n"):
        if _.startswith("*** "):
            _, n, resid, resname, at, _ = _.split()

            n = int(n.split("=")[0]) - 1
            atom_number_type_dict[n] = at
            atom_number_resid_dict[n] = int(resid) - 1

    for _ in output_dmat.split("\n"):
        if _.startswith("  "):
            a, b, dist = _.split()
            a = int(a) - 1
            b = int(b) - 1
            dist = float(dist)
            if atom_number_resid_dict[a] < atom_number_resid_dict[b]:
                pair_distance_dict[(a, b)] = dist
                pair_resid_dict[(a, b)] = (
                    atom_number_resid_dict[a],
                    atom_number_resid_dict[b],
                )

    return {
        "atom_number_type_dict": atom_number_type_dict,
        "atom_number_resid_dict": atom_number_resid_dict,
        "pair_distance_dict": pair_distance_dict,
        "pair_resid_dict": pair_resid_dict,
    }


def get_forces_pycharmm():
    positions = coor.get_positions()
    force_command = """coor force sele all end"""
    _ = pycharmm.lingo.charmm_script(force_command)
    forces = coor.get_positions()
    coor.set_positions(positions)
    return forces


def view_atoms(atoms):
    return view(atoms, viewer="x3d")


Eref = np.zeros([20], dtype=float)
Eref[1] = -0.498232909223
Eref[6] = -37.731440432799
Eref[8] = -74.878159582108
Eref[17] = -459.549260062932



def debug_print(debug: bool, msg: str, *args, **kwargs):
    """Helper function for conditional debug printing"""
    if debug:
        print(msg)
        for arg in args:
            # jax.debug.print(f"{msg}\n{{x}}", x=arg)
            pass
        try:
            for name, value in kwargs.items():
                print(f"{name}: {value.shape}")
        except:
            pass


def prepare_batches_md(
    data,
    batch_size: int,
    data_keys = None,
    num_atoms: int = 60,
    dst_idx = None,
    src_idx= None,
    include_id: bool = False,
    debug_mode: bool = False,
) :
    """
    Efficiently prepare batches for training.

    Args:
        key: JAX random key for shuffling.
        data (dict): Dictionary containing the dataset.
            Expected keys: 'R', 'N', 'Z', 'F', 'E', and optionally others.
        batch_size (int): Size of each batch.
        data_keys (list, optional): List of keys to include in the output.
            If None, all keys in `data` are included.
        num_atoms (int, optional): Number of atoms per example. Default is 60.
        dst_idx (jax.numpy.ndarray, optional): Precomputed destination indices for atom pairs.
        src_idx (jax.numpy.ndarray, optional): Precomputed source indices for atom pairs.
        include_id (bool, optional): Whether to include 'id' key if present in data.
        debug_mode (bool, optional): If True, run assertions and extra checks.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """

    # -------------------------------------------------------------------------
    # Validation and Setup
    # -------------------------------------------------------------------------

    # Check for mandatory keys
    required_keys = ["R", "N", "Z"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    # Default to all keys in data if none provided
    if data_keys is None:
        data_keys = list(data.keys())

    # Verify data sizes
    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    # -------------------------------------------------------------------------
    # Compute Random Permutation for Batches
    # -------------------------------------------------------------------------
    # perms = jax.random.permutation(key, data_size)
    perms = jnp.arange(0, data_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # -------------------------------------------------------------------------
    # Precompute Batch Segments and Indices
    # -------------------------------------------------------------------------
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms

    # Compute pairwise indices only if not provided
    # E3x: e3x.ops.sparse_pairwise_indices(num_atoms) -> returns (dst_idx, src_idx)
    if dst_idx is None or src_idx is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # Adjust indices for batching
    dst_idx = dst_idx + offsets[:, None]
    src_idx = src_idx + offsets[:, None]

    # Centralize reshape logic
    # For keys not listed here, we default to their original shape after indexing.
    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "D": (batch_size,3),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    output = []

    # -------------------------------------------------------------------------
    # Batch Preparation Loop
    # -------------------------------------------------------------------------
    for perm in perms:
        # Build the batch dictionary
        batch = {}
        for k in data_keys:
            if k not in data:
                continue
            v = data[k][jnp.array(perm)]
            new_shape = reshape_rules.get(k, None)
            if new_shape is not None:
                batch[k] = v.reshape(new_shape)
            else:
                batch[k] = v

        # Optionally include 'id' if requested and present
        if include_id and "id" in data and "id" in data_keys:
            batch["id"] = data["id"][jnp.array(perm)]

        # Compute good_indices (mask for valid atom pairs)
        # Vectorized approach: We know N is shape (batch_size,)
        # Expand N to compare with dst_idx/src_idx
        # dst_idx[i], src_idx[i] range over atom pairs within the ith example
        # Condition: (dst_idx[i] < N[i]+i*num_atoms) & (src_idx[i] < N[i]+i*num_atoms)
        # We'll compute this for all i and concatenate.
        N = batch["N"]
        # Expand N and offsets for comparison
        expanded_n = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        good_pairs = (valid_dst & valid_src).astype(jnp.int32)
        good_indices = good_pairs.reshape(-1)

        # Add metadata to the batch
        atom_mask = jnp.where(batch["Z"] > 0, 1, 0)
        batch.update(
            {
                "dst_idx": dst_idx.flatten(),
                "src_idx": src_idx.flatten(),
                "batch_mask": good_indices,
                "batch_segments": batch_segments,
                "atom_mask": atom_mask,
            }
        )

        # Debug checks
        if debug_mode:
            # Check expected shapes
            assert batch["R"].shape == (
                batch_size * num_atoms,
                3,
            ), f"R shape mismatch: {batch['R'].shape}"
            assert batch["F"].shape == (
                batch_size * num_atoms,
                3,
            ), f"F shape mismatch: {batch['F'].shape}"
            assert batch["E"].shape == (
                batch_size,
                1,
            ), f"E shape mismatch: {batch['E'].shape}"
            assert batch["Z"].shape == (
                batch_size * num_atoms,
            ), f"Z shape mismatch: {batch['Z'].shape}"
            assert batch["N"].shape == (
                batch_size,
            ), f"N shape mismatch: {batch['N'].shape}"
            # Optional: print or log if needed

        output.append(batch)

    return output



epsilon = 10 ** (-6)

def indices_of_pairs(a, b, n_atoms=5, n_mol=20):
    assert a < b, "by convention, res a must have a smaller index than res b"
    assert a >= 1, "res indices can't start from 1"
    assert b >= 1, "res indices can't start from 1"
    assert a != b, "pairs can't contain same residue"
    return np.concatenate(
        [
            np.arange(0, n_atoms, 1) + (a - 1) * n_atoms,
            np.arange(0, n_atoms, 1) + (b - 1) * n_atoms,
        ]
    )


def indices_of_monomer(a, n_atoms=5, n_mol=20):
    assert a < (n_mol + 1), "monomer index outside total n molecules"
    return np.arange(0, n_atoms, 1) + (a - 1) * n_atoms




def get_loss_terms(fns, MM_CUTON=6.0, MM_CUTOFF=10.0, BUFFER=0.01, MM_lambda=1.0, ML_lambda=0.0, DO_MM=True, DO_ML=True):
    import time

    start = time.time()
    err_mmml_list = []
    err_charmm_list = []
    for fn in fns:
        results_dict = compare_energies(fn, df, DO_MM=DO_MM, DO_ML=DO_ML, MM_CUTON=MM_CUTON, MM_CUTOFF=MM_CUTOFF, BUFFER=BUFFER)
        err_mmml_list.append(results_dict["err_mmml"])
        err_charmm_list.append(results_dict["err_charmm"])
        print(
            "{} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
                fn.stem,
                results_dict["ref_energy"],
                results_dict["mmml_energy"],
                results_dict["charmm"],
                results_dict["err_mmml"],
                results_dict["err_charmm"],
            )
        )

    end = time.time()
    print("Finished")
    print("Time taken", end - start)
    print("--------------------------------")
    err_mmml_list = np.array(err_mmml_list)
    err_charmm_list = np.array(err_charmm_list)

    print("RMSE MMML", np.sqrt(np.mean(err_mmml_list**2)))
    print("MAE MMML", np.mean(np.abs(err_mmml_list)))
    print("RMSE Charmm", np.sqrt(np.mean(err_charmm_list**2)))
    print("MAE Charmm", np.mean(np.abs(err_charmm_list)))

    loss = MM_lambda * np.mean(err_mmml_list**2) + ML_lambda * np.mean(err_charmm_list**2)
    return loss, err_mmml_list, err_charmm_list

def get_loss_fn(train_filenames, DO_ML=True, DO_MM=True, NTRAIN=20, MM_CUTON=6.0, MM_lambda=1.0, ML_lambda=0.0):
    def loss_fn(x0):
        print("Starting")
        # random_indices = np.random.randint(0, len(train_filenames),6)
        fns = [train_filenames[i] for i in range(NTRAIN)]
        CG321EP, CG321RM, CLGA1EP, CLGA1RM = x0[:4]
        set_param_card(CG321EP, CG321RM, CLGA1EP, CLGA1RM)
        loss, _, _ = get_loss_terms(fns, MM_CUTON=MM_CUTON, MM_lambda=MM_lambda, ML_lambda=ML_lambda, DO_MM=DO_MM, DO_ML=DO_ML)
        print("Loss", loss)
        return loss
    return loss_fn


def ep_scale_loss(x0):
    print("Starting")
    random_indices = np.random.randint(0, len(train_filenames), 4)
    fns = [train_filenames[i] for i in random_indices]
    ep_scale = float(x0)
    set_param_card(CG321EP * ep_scale, CG321RM, CLGA1EP * ep_scale, CLGA1RM)
    loss, _, _ = get_loss_terms(fns)
    print("Loss", loss)
    return loss

def create_initial_simplex(x0, delta=0.0001):
    initial_simplex = np.zeros((len(x0) + 1, len(x0)))
    initial_simplex[0] = x0  # First point is x0
    for i in range(len(x0)):
        initial_simplex[i + 1] = x0.copy()
        initial_simplex[i + 1, i] += delta  # Add small step in dimension i
    return initial_simplex


def optimize_params_simplex(x0, bounds, 
loss, method="Nelder-Mead", maxiter=100, xatol=0.0001, fatol=0.0001):
    initial_simplex = create_initial_simplex(x0)
    res = scipy_minimize(
        loss,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "xatol": 0.0001,  # Absolute tolerance on x
            "fatol": 0.0001,  # Absolute tolerance on function value
            "initial_simplex": initial_simplex,
            "maxiter": 100,
        },
    )  # Initial simplex with steps of 0.0001

    print(res)
    return res
    
def get_bounds(x0, scale=0.1):
    b= [(x0[i] * (1-scale), x0[i] * (1+scale)) if x0[i] > 0 else (x0[i] * (1+scale), x0[i] * (1-scale)) 
    for i in range(len(x0)) ]
    return b

def _smoothstep01(s):
    return s * s * (3.0 - 2.0 * s)


def _sharpstep(r, x0, x1, gamma=GAMMA_ON):
    s = jnp.clip((r - x0) / _safe_den(x1 - x0), 0.0, 1.0)
    s = s ** gamma
    return _smoothstep01(s)


from mmml.pycharmmInterface.cutoffs import CutoffParameters

class ModelOutput(NamedTuple):
    energy: Array  # Shape: (,), total energy in eV
    forces: Array  # Shape: (n_atoms, 3), forces in eV/Å
    dH: Array # Shape: (,), total interaction energy in eV
    internal_E: Array # Shape: (,) total internal energy in eV
    internal_F: Array
    mm_E: Array
    mm_F: Array
    ml_2b_E: Array
    ml_2b_F: Array

def setup_calculator(
    ATOMS_PER_MONOMER,
    N_MONOMERS: int = 2,
    ml_cutoff_distance: float = 2.0,
    mm_switch_on: float = 5.0,
    mm_cutoff: float = 1.0,
    complementary_handoff: bool = True,
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    debug: bool = False,
    ep_scale=None,
    sig_scale=None,
    model_restart_path=None,
    MAX_ATOMS_PER_SYSTEM: int = 20,
    ml_energy_conversion_factor: float = 1.0,
    ml_force_conversion_factor: float = 1.0,
    cell=False,
    verbose: bool = False,
    ml_reorder_indices=None,
    at_codes_override=None,
):
    """Create hybrid ML/MM calculator with outputs in eV/eV-A.

    ML energies/forces are assumed to be in eV already. MM energies/forces
    (kcal/mol, kcal/mol/Å) are converted to eV/eV-Å internally before summing.
    """
    if model_restart_path is None:
        raise ValueError("model_restart_path must be provided")
        # model_restart_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-7c36e6f9-6f10-4d21-bf6d-693df9b8cd40"
    n_monomers = N_MONOMERS

    cutoffparameters = CutoffParameters(
        ml_cutoff_distance, mm_switch_on, mm_cutoff, complementary_handoff=complementary_handoff
    )
    print(
        "[setup_calculator] Cutoff inputs -> ml_cutoff=%.4f, mm_switch_on=%.4f, mm_cutoff=%.4f, complementary_handoff=%s"
        % (ml_cutoff_distance, mm_switch_on, mm_cutoff, complementary_handoff)
    )
    
    all_dimer_idxs = []
    for a, b in dimer_permutations(n_monomers):
        all_dimer_idxs.append(indices_of_pairs(a + 1, b + 1, n_atoms=ATOMS_PER_MONOMER))

    all_monomer_idxs = []
    for a in range(1, n_monomers + 1):
        all_monomer_idxs.append(indices_of_monomer(a, n_atoms=ATOMS_PER_MONOMER, n_mol=n_monomers))
    # print("all_monomer_idxs", all_monomer_idxs)
    # print("all_dimer_idxs", all_dimer_idxs)
    dimer_perms = dimer_permutations(n_monomers)
    # Print all dimer pairs for verification
    for a, b in dimer_perms:
        print(a, b)

    print("len(dimer_perms)", len(dimer_perms))

    N_MONOMERS = n_monomers
    # Batch processing constants
    BATCH_SIZE: int = N_MONOMERS + len(dimer_perms)  # Number of systems per batch
    # print(BATCH_SIZE)
    restart_path = Path(model_restart_path) if type(model_restart_path) == str else model_restart_path
    
    # Check if this is a JSON checkpoint (has params.json file)
    json_params_path = restart_path / "params.json"
    is_json_checkpoint = json_params_path.exists()
    
    if is_json_checkpoint:
        # This is a JSON checkpoint - use it directly
        restart = restart_path
        # Load using JSON loader
        try:
            from mmml.utils.model_checkpoint import load_model_checkpoint
            checkpoint = load_model_checkpoint(restart, use_orbax=False, load_params=True, load_config=True)
            params = checkpoint.get('params')
            config = checkpoint.get('config', {})
            
            if params is None:
                raise FileNotFoundError(f"params not found in JSON checkpoint at {restart_path}")
            if not config:
                raise FileNotFoundError(f"model_config not found in JSON checkpoint at {restart_path}")
            
            # Reconstruct model from config
            from mmml.physnetjax.physnetjax.models.model import EF
            
            # Convert JSON arrays back to JAX arrays for model config
            def json_to_jax_config(obj):
                """Convert JSON config values back to appropriate types."""
                if isinstance(obj, dict):
                    return {k: json_to_jax_config(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    # Lists in config are usually parameters, keep as lists
                    return obj
                else:
                    return obj
            
            model_config = json_to_jax_config(config)
            model_config['natoms'] = MAX_ATOMS_PER_SYSTEM
            MODEL = EF(**model_config)
            MODEL.natoms = MAX_ATOMS_PER_SYSTEM
            
            # Convert JSON params back to JAX arrays
            def json_to_jax_params(obj):
                """Recursively convert JSON lists to JAX arrays."""
                if isinstance(obj, dict):
                    return {k: json_to_jax_params(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    # Check if it's a nested list (array)
                    if len(obj) > 0 and isinstance(obj[0], (list, int, float)):
                        return jnp.array(obj)
                    else:
                        return [json_to_jax_params(item) for item in obj]
                elif isinstance(obj, (int, float)):
                    return obj
                else:
                    return obj
            
            params = json_to_jax_params(params)
            
            # Flax models expect params to be wrapped in {'params': {...}}
            # Check if params is already in the correct format
            # get_params_model from Orbax returns {'params': {...}}, but JSON loading
            # might return just the raw params dict, so we need to wrap it
            if isinstance(params, dict):
                # Check if it's already in Flax format (has 'params' key at top level)
                if 'params' in params:
                    # Already wrapped, but check if it's the right structure
                    # Flax expects {'params': {...}} where {...} is the actual params
                    pass  # Already in correct format
                else:
                    # Not wrapped, need to wrap it
                    params = {'params': params}
            else:
                # Params is not a dict (unlikely but handle it)
                params = {'params': params}
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load JSON checkpoint from {restart_path}. "
                f"Error: {e}. "
                f"Make sure params.json and model_config.json exist."
            ) from e
    else:
        # This is an orbax checkpoint - use get_last to find the latest epoch
        try:
            restart = get_last(restart_path)
        except (IndexError, FileNotFoundError) as e:
            raise FileNotFoundError(
                f"Checkpoint directory is empty or invalid: {restart_path}. "
                f"Available files: {list(restart_path.glob('*')) if restart_path.exists() else 'Directory does not exist'}. "
                f"If this is a JSON checkpoint, make sure params.json exists in the directory."
            ) from e
        # Setup monomer model using orbax
        params, MODEL = get_params_model(restart)
    MODEL.natoms = MAX_ATOMS_PER_SYSTEM 

    if cell:
        cell = float(cell)
        # somewhere in your factory / calculator init
        from mmml.pycharmmInterface.pbc_prep_factory import make_pbc_mapper
        # turn the length into a 3x3 matrix for a cubic cell
        cell = jnp.asarray([[cell, 0, 0], [0, cell, 0], [0, 0, cell]])
        mol_id = None
        try:
            # for now just use a simple array of integers for the molecule id
            # in order e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] for n_atoms_monomer = 3
            mol_id_np = jnp.asarray([i * jnp.ones(ATOMS_PER_MONOMER,
             dtype=jnp.int32) for i in np.arange(n_monomers)], dtype=jnp.int32)
            mol_id = jnp.asarray(mol_id_np, dtype=jnp.int32)
        except Exception:
            print("No mol_id provided")
            mol_id = None
        do_pbc_map = True
        pbc_map = make_pbc_mapper(cell=cell, mol_id=mol_id, n_monomers=n_monomers)
    else:
        pbc_map = do_pbc_map = False



    @partial(jax.jit, static_argnames=['ml_cutoff', 'mm_switch_on', 'ATOMS_PER_MONOMER'])
    def switch_ML(X,
        ml_energy,
        ml_cutoff=ml_cutoff_distance,
        mm_switch_on=mm_switch_on,
        ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
    ):
        # COM–COM distance (used for ML taper; must match debug "dimer COM distance")
        com1 = jnp.mean(X[:ATOMS_PER_MONOMER], axis=0)
        com2 = jnp.mean(X[ATOMS_PER_MONOMER:2*ATOMS_PER_MONOMER], axis=0)
        r = jnp.linalg.norm(com2 - com1)
    
        # ML: 1 -> 0 over [mm_switch_on - ml_cutoff, mm_switch_on]
        ml_scale = 1.0 - _sharpstep(r, mm_switch_on - ml_cutoff, mm_switch_on, gamma=GAMMA_ON)
      
        return ml_scale * ml_energy

    switch_ML_grad = jax.grad(switch_ML)

    
    def get_MM_energy_forces_fns(
        R,
        ATOMS_PER_MONOMER,
        N_MONOMERS=N_MONOMERS,
        ml_cutoff_distance=ml_cutoff_distance,
        mm_switch_on=mm_switch_on,
        mm_cutoff=mm_cutoff,
        complementary_handoff=True,
        sig_scale=sig_scale,
        ep_scale=ep_scale,
    ):
        """Creates functions for calculating MM energies and forces with switching."""
        # koading from pycharmm (for consistency), will consider moving 
        # this out of the calculator set-up function eventually
        from mmml.pycharmmInterface.import_pycharmm import reset_block
        reset_block()
        read.rtf(CGENFF_RTF)
        bl =settings.set_bomb_level(-2)
        wl =settings.set_warn_level(-2)
        read.prm(CGENFF_PRM)
        settings.set_bomb_level(bl)
        settings.set_warn_level(wl)
        pycharmm.lingo.charmm_script('bomlev 0')
        cgenff_rtf = open(CGENFF_RTF).readlines()
        atc = pycharmm.param.get_atc()
        cgenff_params_dict_q = {}
        atom_name_to_param = {k: [] for k in atc}
        
        for _ in cgenff_rtf:
            if _.startswith("ATOM"):
                _, atomname, at, q = _.split()[:4]
                try:
                    cgenff_params_dict_q[at] = float(q)
                except:
                    cgenff_params_dict_q[at] = float(q.split("!")[0])
                atom_name_to_param[atomname] = at

        cgenff_params_dict = {}
        cgenff_params_dict_list = []
        for p in open(CGENFF_PRM).readlines():
            if len(p) > 5 and len(p.split()) > 4 and p.split()[1] == "0.0" and p[0] != "!":
                res, _, ep, sig = p.split()[:4]
                cgenff_params_dict[res] = (float(ep), float(sig))
                cgenff_params_dict_list.append((float(ep), float(sig)))

        params = list(range(len(atc)))
        
        atc_epsilons = [cgenff_params_dict[_][0] if _ in cgenff_params_dict.keys() else 0.0 for _ in atc ]
        atc_rmins = [cgenff_params_dict[_][1] if _ in cgenff_params_dict.keys() else 0.0 for _ in atc ]
        atc_qs = [cgenff_params_dict_q[_] if _ in cgenff_params_dict_q.keys() else 0.0 for _ in atc  ]

        if ep_scale is None:
            ep_scale = np.ones_like(np.array(atc_epsilons))
        if sig_scale is None:
            sig_scale = np.ones_like(np.array(atc_epsilons))
        
        at_ep = -1 * abs( np.array(atc_epsilons)) * ep_scale
        at_rm = np.array(atc_rmins) * sig_scale
        at_q = np.array(atc_qs)


        at_flat_q = np.array(at_q)
        at_flat_ep =  np.array(at_ep)
        at_flat_rm =  np.array(at_rm)
        
        pair_idxs_product = jnp.array([(a,b) for a,b in list(product(np.arange(ATOMS_PER_MONOMER), repeat=2))])
        dimer_perms = jnp.array(dimer_permutations(N_MONOMERS))
        
        pair_idxs_np = dimer_perms * ATOMS_PER_MONOMER
        pair_idx_atom_atom = pair_idxs_np[:, None, :] + pair_idxs_product[None,...]
        pair_idx_atom_atom = pair_idx_atom_atom.reshape(-1, 2)
        
        displacements = R[pair_idx_atom_atom[:,0]] - R[pair_idx_atom_atom[:,1]]
        distances = jnp.linalg.norm(displacements, axis=1)
        at_perms = [_ for _ in list(product(params, repeat=2)) if _[0] <= _[1]]
        
        charges = np.array(psf.get_charges())[:N_MONOMERS*ATOMS_PER_MONOMER]
        masses = np.array(psf.get_amass())[:N_MONOMERS*ATOMS_PER_MONOMER]
        at_codes = np.array(psf.get_iac())[:N_MONOMERS*ATOMS_PER_MONOMER]
        if at_codes_override is not None:
            at_codes_override_arr = np.array(at_codes_override)
            if at_codes_override_arr.shape[0] != at_codes.shape[0]:
                raise ValueError(
                    f"at_codes_override length {at_codes_override_arr.shape[0]} "
                    f"does not match expected {at_codes.shape[0]}"
                )
            at_codes = at_codes_override_arr
        atomtype_codes = np.array(psf.get_atype())[:N_MONOMERS*ATOMS_PER_MONOMER]

        if debug:
            atc_eps_arr = np.array(atc_epsilons)
            missing_eps_codes = np.where(atc_eps_arr == 0.0)[0]
            used_missing = np.unique(at_codes[np.isin(at_codes, missing_eps_codes)])
            if used_missing.size > 0:
                missing_names = [atc[idx] for idx in used_missing if idx < len(atc)]
                print(
                    "WARNING: Missing LJ params for atom types in PSF:",
                    missing_names,
                    "(epsilon=0 -> zero MM forces possible)",
                )

        rmins_per_system = jnp.take(at_flat_rm, at_codes) 
        epsilons_per_system = jnp.take(at_flat_ep, at_codes)

        rs = distances
        q_per_system = jnp.take(at_flat_q, at_codes)
        q_per_system = charges
        # # make sure the system is charge neutral
        # if jnp.sum(q_per_system) != 0:
        #     raise ValueError(
        #         "System is not charge neutral. Please check the charges in the PSF file."
        #     )

        q_a = jnp.take(q_per_system, pair_idx_atom_atom[:, 0])
        q_b = jnp.take(q_per_system, pair_idx_atom_atom[:, 1])
        
        rm_a = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 0])
        rm_b = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 1])
        
        ep_a = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 0])
        ep_b = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 1])

        pair_qq = q_a * q_b
        pair_rm = (rm_a + rm_b)
        pair_ep = (ep_a * ep_b)**0.5

        
        def lennard_jones(r, sig, ep):
            """
            rmin = 2^(1/6) * sigma
                https://de.wikipedia.org/wiki/Lennard-Jones-Potential
            Lennard-Jones potential for a pair of atoms
            """
            a = 6
            b = 2
            # Add epsilon to prevent division by zero when r is very small
            lj_epsilon = 1e-10
            r_safe = jnp.maximum(r, lj_epsilon)
            # sig = sig / (2 ** (1 / 6))
            r6 = (sig / r_safe) ** a
            return ep * (r6 ** b - 2 * r6)
        
        coulombs_constant = 3.32063711e2 #Coulomb's constant kappa = 1/(4*pi*e0) in kcal-Angstroms/e^2.
        # Small epsilon to prevent division by zero in coulomb interactions
        coulomb_epsilon = 1e-10
        def coulomb(r, qq, constant = coulombs_constant, eps = coulomb_epsilon):
            # Add epsilon to prevent division by zero (r can be very small for bonded atoms)
            r_safe = jnp.maximum(r, eps)
            return constant * qq / r_safe
        

        def get_switching_function(
            ml_cutoff_distance: float = ml_cutoff_distance,
            mm_switch_on: float = mm_switch_on,
            mm_cutoff: float = mm_cutoff,
            complementary_handoff: bool = complementary_handoff,
        ):
            """MM scale: complementary (1-s_ML) over handoff, then taper to 0 at mm_switch_on+mm_cutoff; or legacy."""
            @jax.jit
            def apply_switching_function(positions: Array, pair_energies: Array) -> Array:
                com1 = jnp.mean(positions[:ATOMS_PER_MONOMER], axis=0)
                com2 = jnp.mean(positions[ATOMS_PER_MONOMER:2*ATOMS_PER_MONOMER], axis=0)
                r = jnp.linalg.norm(com2 - com1)
                if complementary_handoff:
                    # handoff: s_MM = 1 - s_ML over [mm_switch_on - ml_cutoff, mm_switch_on]
                    handoff = _sharpstep(r, mm_switch_on - ml_cutoff_distance, mm_switch_on, gamma=GAMMA_ON)
                    # taper to 0 at mm_switch_on + mm_cutoff so energies/forces go to 0
                    mm_taper = 1.0 - _sharpstep(
                        r, mm_switch_on, mm_switch_on + mm_cutoff, gamma=GAMMA_OFF
                    )
                    mm_scale = handoff * mm_taper
                else:
                    mm_on = _sharpstep(r, mm_switch_on, mm_switch_on + mm_cutoff, gamma=GAMMA_ON)
                    mm_off = _sharpstep(r, mm_switch_on + mm_cutoff, mm_switch_on + 2.0 * mm_cutoff, gamma=GAMMA_OFF)
                    mm_scale = mm_on * (1.0 - mm_off)
                return (pair_energies * mm_scale).sum()
            return apply_switching_function

        apply_switching_function = get_switching_function(
            ml_cutoff_distance=ml_cutoff_distance,
            mm_switch_on=mm_switch_on,
            mm_cutoff=mm_cutoff,
            complementary_handoff=complementary_handoff,
        )

        @jax.jit
        def calculate_mm_energy(positions: Array) -> Array:
            """Calculates MM energies including both VDW and electrostatic terms."""
            # Calculate pairwise distances
            displacements = positions[pair_idx_atom_atom[:,0]] - positions[pair_idx_atom_atom[:,1]]
            distances = jnp.linalg.norm(displacements, axis=1)
            
            # Only include interactions between unique pairs
            pair_mask = 1 #(pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1])

            # Calculate VDW (Lennard-Jones) energies
            vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
            vdw_total = vdw_energies.sum()
            
            # Calculate electrostatic energies
            electrostatic_energies = coulomb(distances, pair_qq) * pair_mask    
            electrostatic_total = electrostatic_energies.sum()
                
            return vdw_total + electrostatic_total

        @jax.jit
        def calculate_mm_pair_energies(positions: Array) -> Array:
            """Calculates per-pair MM energies for switching calculations."""
            displacements = positions[pair_idx_atom_atom[:,0]] - positions[pair_idx_atom_atom[:,1]]
            distances = jnp.linalg.norm(displacements, axis=1)
            pair_mask = (pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1])
            
            vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
            electrostatic_energies = coulomb(distances, pair_qq) * pair_mask    
                
            return vdw_energies + electrostatic_energies
        
        def switched_mm_energy(positions: Array) -> Array:
            """MM energy with switching applied (differentiable)."""
            pair_energies = calculate_mm_pair_energies(positions)
            return apply_switching_function(positions, pair_energies)

        switched_mm_grad = jax.grad(switched_mm_energy)

        @jax.jit 
        def calculate_mm_energy_and_forces(
            positions: Array,  # Shape: (n_atoms, 3)
        ) -> Tuple[Array, Array]:
            """Calculates MM energy and forces with switching."""
            # Calculate base MM energies
            pair_energies = calculate_mm_pair_energies(positions)
            
            # Apply switching function
            switched_energy = apply_switching_function(positions, pair_energies)
            
            # Calculate forces with switching (include energy dependence on positions)
            forces = -1.0 * switched_mm_grad(positions)
            # Check for NaN/Inf in forces and replace with zeros
            forces = jnp.where(jnp.isfinite(forces), forces, 0.0)

            return switched_energy, forces

        return calculate_mm_energy_and_forces


    
    @partial(jax.jit, static_argnames=['n_monomers', 'cutoff_params', 'doML', 'doMM', 'doML_dimer', 'debug',])
    def spherical_cutoff_calculator(
        positions: Array,  # Shape: (n_atoms, 3)
        atomic_numbers: Array,  # Shape: (n_atoms,)
        n_monomers: int,
        cutoff_params: CutoffParameters,
        doML: bool = True,
        doMM: bool = True,
        doML_dimer: bool = True,
        debug: bool = False,
    ) -> ModelOutput:
        """Calculates energy and forces using combined ML/MM potential.
        
        Args:
            positions: Atomic positions in Angstroms
            atomic_numbers: Atomic numbers of each atom
            n_monomers: Number of monomers in system
            cutoff_params: Parameters for cutoffs and switching
            doML: Whether to include ML potential
            doMM: Whether to include MM potential
            doML_dimer: Whether to include ML dimer interactions
            debug: Whether to enable debug output
            
        Returns:
            ModelOutput containing total energy and forces
        """
        n_dimers = len(dimer_permutations(n_monomers))
        n_atoms = positions.shape[0]
        
        # Optional: reorder to model order for ML, then remap back
        ml_perm = None
        ml_inv_perm = None
        if ml_reorder_indices is not None:
            ml_perm = jnp.array(ml_reorder_indices, dtype=jnp.int32)
            # Ensure permutation length matches atom count
            if ml_perm.shape[0] == n_atoms:
                ml_inv_perm = jnp.empty_like(ml_perm)
                ml_inv_perm = ml_inv_perm.at[ml_perm].set(jnp.arange(n_atoms))
            else:
                ml_perm = None
                ml_inv_perm = None
        
        # Initialize force arrays with correct shape for all atoms
        outputs = {
            "out_E": 0,
            "out_F": jnp.zeros((n_atoms, 3)),
            "dH": 0,
            "internal_E": 0, 
            "internal_F": jnp.zeros((n_atoms, 3)),
            "ml_2b_E": 0,
            "ml_2b_F": jnp.zeros((n_atoms, 3))
        }

        if doML:
            if ml_perm is not None:
                positions_ml = positions[ml_perm]
                atomic_numbers_ml = atomic_numbers[ml_perm]
            else:
                positions_ml = positions
                atomic_numbers_ml = atomic_numbers

            ml_out = calculate_ml_contributions(
                positions_ml, atomic_numbers_ml, n_dimers, n_monomers,
                cutoff_params=cutoff_params,
                doML_dimer=doML_dimer,
                debug=debug,
                ml_energy_conversion_factor=ml_energy_conversion_factor,
                ml_force_conversion_factor=ml_force_conversion_factor
            )
            # Get ML forces from calculate_ml_contributions
            # CRITICAL: These forces are ALREADY correctly mapped to atoms 0 to (n_monomers * ATOMS_PER_MONOMER - 1)
            # via segment_sum in process_monomer_forces and process_dimer_forces
            # They should be in the same order as the atom positions (atoms 0, 1, 2, ..., n_atoms-1)
            ml_forces_raw = ml_out.get("out_F", jnp.zeros((n_monomers * ATOMS_PER_MONOMER, 3)))
            ml_internal_F_raw = ml_out.get("internal_F", jnp.zeros((n_monomers * ATOMS_PER_MONOMER, 3)))
            ml_2b_F_raw = ml_out.get("ml_2b_F", jnp.zeros((n_monomers * ATOMS_PER_MONOMER, 3)))

            # Remap ML outputs back to PyCHARMM ordering if a permutation was applied
            def _remap(arr):
                if ml_inv_perm is None or arr.shape[0] != n_atoms:
                    return arr
                return arr[ml_inv_perm]
            ml_forces_raw = _remap(ml_forces_raw)
            ml_internal_F_raw = _remap(ml_internal_F_raw)
            ml_2b_F_raw = _remap(ml_2b_F_raw)
            
            # IMMEDIATELY check for NaN/Inf and replace with zeros
            # This is critical - NaN values will corrupt all subsequent calculations
            ml_forces = jnp.where(jnp.isfinite(ml_forces_raw), ml_forces_raw, 0.0)
            ml_internal_F = jnp.where(jnp.isfinite(ml_internal_F_raw), ml_internal_F_raw, 0.0)
            ml_2b_F = jnp.where(jnp.isfinite(ml_2b_F_raw), ml_2b_F_raw, 0.0)
            
            # Debug: Check for NaN in raw forces (jax.debug.print handles conditional execution)
            nan_count_raw = jnp.sum(~jnp.isfinite(ml_forces_raw))
            # jax.debug.print("CRITICAL: Found {n} NaN/Inf in ml_forces from calculate_ml_contributions!",
            # n=nan_count_raw, ordered=False)
            
            # Ensure ML forces have the correct shape (should match n_atoms)
            expected_n_ml_atoms = n_monomers * ATOMS_PER_MONOMER
            if ml_forces.shape[0] != expected_n_ml_atoms:
                if ml_forces.shape[0] < expected_n_ml_atoms:
                    padding = jnp.zeros((expected_n_ml_atoms - ml_forces.shape[0], 3))
                    ml_forces = jnp.concatenate([ml_forces, padding], axis=0)
                    ml_internal_F = jnp.concatenate([ml_internal_F, jnp.zeros((expected_n_ml_atoms - ml_internal_F.shape[0], 3))], axis=0)
                    ml_2b_F = jnp.concatenate([ml_2b_F, jnp.zeros((expected_n_ml_atoms - ml_2b_F.shape[0], 3))], axis=0)
                else:
                    ml_forces = ml_forces[:expected_n_ml_atoms]
                    ml_internal_F = ml_internal_F[:expected_n_ml_atoms]
                    ml_2b_F = ml_2b_F[:expected_n_ml_atoms]
            
            # Ensure ML forces match system size (in case n_atoms != expected_n_ml_atoms)
            if ml_forces.shape[0] > n_atoms:
                ml_forces = ml_forces[:n_atoms]
                ml_internal_F = ml_internal_F[:n_atoms]
                ml_2b_F = ml_2b_F[:n_atoms]
            elif ml_forces.shape[0] < n_atoms:
                padding = jnp.zeros((n_atoms - ml_forces.shape[0], 3))
                ml_forces = jnp.concatenate([ml_forces, padding], axis=0)
                ml_internal_F = jnp.concatenate([ml_internal_F, padding], axis=0)
                ml_2b_F = jnp.concatenate([ml_2b_F, padding], axis=0)
            
            # CRITICAL: Ensure shapes match exactly before adding
            if ml_forces.shape[0] != n_atoms:
                if ml_forces.shape[0] < n_atoms:
                    padding = jnp.zeros((n_atoms - ml_forces.shape[0], 3))
                    ml_forces = jnp.concatenate([ml_forces, padding], axis=0)
                    ml_internal_F = jnp.concatenate([ml_internal_F, padding], axis=0)
                    ml_2b_F = jnp.concatenate([ml_2b_F, padding], axis=0)
                else:
                    ml_forces = ml_forces[:n_atoms]
                    ml_internal_F = ml_internal_F[:n_atoms]
                    ml_2b_F = ml_2b_F[:n_atoms]

            nan_count = jnp.sum(~jnp.isfinite(ml_forces))
           
            outputs["out_F"] = outputs["out_F"] + ml_forces
            outputs["internal_F"] = outputs["internal_F"] + ml_internal_F
            
            # Final verification: check for any atoms that have zero forces when they shouldn't
            # (This is a debug check - not for fixing, just for warning)
            force_magnitudes = jnp.linalg.norm(outputs["out_F"], axis=1)
            near_zero_atoms = jnp.sum(force_magnitudes < 1e-8)

            
            # ml_2b_F is stored separately for analysis but is already included in ml_forces
            outputs["ml_2b_F"] = outputs["ml_2b_F"] + ml_2b_F
                   
            # Update energy terms (these are scalars, so no shape issue)
            outputs["out_E"] = ml_out.get("out_E", 0)
            outputs["dH"] = ml_out.get("dH", 0)
            outputs["internal_E"] = ml_out.get("internal_E", 0)
            outputs["ml_2b_E"] = ml_out.get("ml_2b_E", 0)

        if doMM:
            mm_out = calculate_mm_contributions(
                positions,
                cutoff_params=cutoff_params,
                debug=debug
            )
            # Preserve separate MM terms and add to totals instead of overwriting
            mm_E = mm_out.get("mm_E", 0)
            mm_F = mm_out.get("mm_F", 0)
            
            # Final NaN check on MM contributions
            mm_E = jnp.where(jnp.isfinite(mm_E), mm_E, 0.0)
            mm_F = jnp.where(jnp.isfinite(mm_F), mm_F, 0.0)
            
            # Ensure MM forces match system size
            if mm_F.shape[0] != n_atoms:
                if mm_F.shape[0] < n_atoms:
                    padding = jnp.zeros((n_atoms - mm_F.shape[0], 3))
                    mm_F = jnp.concatenate([mm_F, padding], axis=0)
                else:
                    mm_F = mm_F[:n_atoms]
            
            outputs["mm_E"] = mm_E
            outputs["mm_F"] = mm_F
            outputs["out_E"] = outputs.get("out_E", 0) + outputs["mm_E"]
            outputs["out_F"] = outputs.get("out_F", 0) + outputs["mm_F"]

        # Final validation: check for NaN/Inf in final forces
        final_forces = outputs["out_F"]
        
        # # Debug: Check force accumulation (jax.debug.print handles conditional execution)
        ml_force_norm = jnp.linalg.norm(outputs.get("out_F", jnp.zeros((n_atoms, 3)))) if "out_F" in outputs else 0.0
        mm_force_norm = jnp.linalg.norm(outputs.get("mm_F", jnp.zeros((n_atoms, 3)))) if "mm_F" in outputs else 0.0
        # jax.debug.print("Before final check - ML force norm: {ml}, MM force norm: {mm}, n_atoms: {n}",
        # ml=ml_force_norm, mm=mm_force_norm, n=n_atoms,
        # ordered=False)
        ml_non_zero = jnp.sum(jnp.any(jnp.abs(outputs.get("out_F", jnp.zeros((n_atoms, 3)))) > 1e-10, axis=1)) if "out_F" in outputs else 0
        # jax.debug.print("ML forces non-zero atoms: {c} / {t}", c=ml_non_zero, t=n_atoms, ordered=False)
        
        # Debug: Check which specific atoms have zero forces BEFORE final NaN check
        final_force_mags_before = jnp.linalg.norm(final_forces, axis=1)
        zero_count_before = jnp.sum(final_force_mags_before < 1e-10)


        
        final_forces = jnp.where(jnp.isfinite(final_forces), final_forces, 0.0)

        # Total energy: combined ML (monomer+dimer switched) + MM
        final_energy = outputs["out_E"]
        if isinstance(final_energy, (int, float)):
            final_energy = jnp.array(final_energy)
        final_energy = jnp.where(jnp.isfinite(final_energy), final_energy, 0.0)
        
        # Compute energy sum safely
        if hasattr(final_energy, 'sum'):
            energy_sum = final_energy.sum()
        else:
            energy_sum = final_energy
        
        return ModelOutput(
            energy=energy_sum,
            forces=final_forces,
            dH=outputs["dH"],
            ml_2b_E=outputs["ml_2b_E"],
            ml_2b_F=outputs["ml_2b_F"],
            internal_E=outputs["internal_E"],
            internal_F=outputs["internal_F"],
            mm_E=outputs.get("mm_E", 0),
            mm_F=outputs.get("mm_F", 0)
        )

    def get_ML_energy_fn(
        atomic_numbers: Array,  # Shape: (n_atoms,)
        positions: Array,  # Shape: (n_atoms, 3)
        BATCH_SIZE,
    ) -> Tuple[Any, Dict[str, Array]]:
        """Prepares the ML model and batching for energy calculations.
        
        Args:
            atomic_numbers: Array of atomic numbers
            positions: Atomic positions in Angstroms
            
        Returns:
            Tuple of (model_apply_fn, batched_inputs)
        """
        batch_data: Dict[str, Array] = {}
        
        # Prepare monomer data
        # n_monomers = len(all_monomer_idxs)
        # Use the maximum number of atoms for consistent array shapes
        max_atoms = max(ATOMS_PER_MONOMER, 2 * ATOMS_PER_MONOMER)
        
        # Position of the atoms in the monomer
        monomer_positions = jnp.zeros((n_monomers, max_atoms, SPATIAL_DIMS))
        monomer_positions = monomer_positions.at[:, :ATOMS_PER_MONOMER].set(
            positions[jnp.array(all_monomer_idxs)]
        )
        # Atomic numbers of the atoms in the monomer
        monomer_atomic = jnp.zeros((n_monomers, max_atoms), dtype=jnp.int32)
        monomer_atomic = monomer_atomic.at[:, :ATOMS_PER_MONOMER].set(
            atomic_numbers[jnp.array(all_monomer_idxs)]
        )
        
        # Prepare dimer data
        n_dimers = len(all_dimer_idxs)
        # Position of the atoms in the dimer
        dimer_positions = jnp.zeros((n_dimers, max_atoms, SPATIAL_DIMS))
        dimer_positions = dimer_positions.at[:, :2 * ATOMS_PER_MONOMER].set(
            positions[jnp.array(all_dimer_idxs)]
        )
        # Atomic numbers of the atoms in the dimer
        dimer_atomic = jnp.zeros((n_dimers, max_atoms), dtype=jnp.int32)
        dimer_atomic = dimer_atomic.at[:, :2 * ATOMS_PER_MONOMER].set(
            atomic_numbers[jnp.array(all_dimer_idxs)]
        )
        
        # Combine monomer and dimer data
        batch_data["R"] = jnp.concatenate([monomer_positions, dimer_positions])
        batch_data["Z"] = jnp.concatenate([monomer_atomic, dimer_atomic])
        batch_data["N"] = jnp.concatenate([
            jnp.full((n_monomers,), ATOMS_PER_MONOMER),
            jnp.full((n_dimers,), 2 * ATOMS_PER_MONOMER)
        ])
        BATCH_SIZE = n_monomers + n_dimers
        batches = prepare_batches_md(batch_data, batch_size=BATCH_SIZE, num_atoms=max_atoms)[0]
        
        @jax.jit
        def apply_model(
            atomic_numbers: Array,  # Shape: (batch_size * num_atoms,)
            positions: Array,  # Shape: (batch_size * num_atoms, 3)
        ) -> Dict[str, Array]:
            """Applies the ML model to batched inputs.
            
            Args:
                atomic_numbers: Batched atomic numbers
                positions: Batched atomic positions
                
            Returns:
                Dictionary containing 'energy' and 'forces'
            """
            return MODEL.apply(
                params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                dst_idx=batches["dst_idx"],
                src_idx=batches["src_idx"],
                batch_segments=batches["batch_segments"],
                batch_size=BATCH_SIZE,
                batch_mask=batches["batch_mask"],
                atom_mask=batches["atom_mask"]
            )
    
        return apply_model, batches

    def calculate_ml_contributions(
        positions: Array,
        atomic_numbers: Array, 
        n_dimers: int,
        n_monomers: int,
        cutoff_params: CutoffParameters,
        doML_dimer: bool = True,
        debug: bool = False,
        ml_energy_conversion_factor: float = 1.0,
        ml_force_conversion_factor: float = 1.0
    ) -> Dict[str, Array]:
        """Calculate ML energy and force contributions"""
        # Calculate max atoms for consistent array shapes
        max_atoms = max(ATOMS_PER_MONOMER, 2 * ATOMS_PER_MONOMER)
        
        # Get model predictions
        apply_model, batches = get_ML_energy_fn(atomic_numbers, positions, n_dimers+n_monomers)
        output = apply_model(batches["Z"], batches["R"])

        f = output["forces"] * ml_force_conversion_factor 
        e = output["energy"] * ml_energy_conversion_factor

        # Calculate monomer contributions
        monomer_contribs = calculate_monomer_contributions(e, f, n_monomers, max_atoms, debug)
        
        if not doML_dimer:
            # Return same keys as full path so caller always sees ml_2b_* (as zeros when skipped)
            return {
                **monomer_contribs,
                "ml_2b_E": 0,
                "ml_2b_F": jnp.zeros((n_monomers * ATOMS_PER_MONOMER, 3)),
            }

        # Calculate dimer contributions
        dimer_contribs = calculate_dimer_contributions(
            positions, e, f, n_dimers, 
            monomer_contribs["monomer_energy"],
            cutoff_params,
            debug
        )

        print(f"DEBUG dimer_contribs: {dimer_contribs}")
        
        # Combine contributions
        # Ensure both force arrays are finite before combining
        monomer_forces_safe = jnp.where(jnp.isfinite(monomer_contribs["out_F"]), monomer_contribs["out_F"], 0.0)
        dimer_forces_safe = jnp.where(jnp.isfinite(dimer_contribs["out_F"]), dimer_contribs["out_F"], 0.0)
        
        # Ensure shapes match - both should be (n_monomers * ATOMS_PER_MONOMER, 3)
        # If they don't match, we need to pad/truncate to the expected size, not min_size
        expected_force_size = n_monomers * ATOMS_PER_MONOMER
        if monomer_forces_safe.shape[0] != dimer_forces_safe.shape[0]:
            # jax.debug.print("ERROR: Shape mismatch in combine! monomer: {m}, dimer: {d}, expected: {e}",
            # m=monomer_forces_safe.shape, d=dimer_forces_safe.shape, e=expected_force_size,
            # ordered=False)
            # Fix both to expected size (not min_size, as that could cut off atoms incorrectly)
            # Fix monomer_forces_safe
            if monomer_forces_safe.shape[0] < expected_force_size:
                padding = jnp.zeros((expected_force_size - monomer_forces_safe.shape[0], 3))
                monomer_forces_safe = jnp.concatenate([monomer_forces_safe, padding], axis=0)
            elif monomer_forces_safe.shape[0] > expected_force_size:
                monomer_forces_safe = monomer_forces_safe[:expected_force_size]
            # Fix dimer_forces_safe
            if dimer_forces_safe.shape[0] < expected_force_size:
                padding = jnp.zeros((expected_force_size - dimer_forces_safe.shape[0], 3))
                dimer_forces_safe = jnp.concatenate([dimer_forces_safe, padding], axis=0)
            elif dimer_forces_safe.shape[0] > expected_force_size:
                dimer_forces_safe = dimer_forces_safe[:expected_force_size]
        
        combined_forces = monomer_forces_safe + dimer_forces_safe
        
        # Ensure combined forces are finite
        combined_forces = jnp.where(jnp.isfinite(combined_forces), combined_forces, 0.0)
               
        return {
            "out_E": monomer_contribs["out_E"] + dimer_contribs["out_E"],
            "out_F": combined_forces,
            "dH": dimer_contribs["dH"],
            "internal_E": monomer_contribs["internal_E"],
            "internal_F": monomer_contribs["internal_F"],
            "ml_2b_E": dimer_contribs["ml_2b_E"],
            "ml_2b_F": dimer_contribs["ml_2b_F"]
        }

    def calculate_monomer_contributions(
        e: Array, 
        f: Array,
        n_monomers: int,
        max_atoms: int,
        debug: bool
    ) -> Dict[str, Array]:
        """Calculate energy and force contributions from monomers"""
        ml_monomer_energy = jnp.array(e[:n_monomers]).flatten()

        ml_monomer_forces = f[:max_atoms*n_monomers]
        
        # Calculate segment indices for force summation
        # These indices map each force to its corresponding atom in the system
        # For 2 monomers with 10 atoms each: [0,1,2,...,9, 10,11,12,...,19]
        monomer_segment_idxs = jnp.concatenate([
            jnp.arange(ATOMS_PER_MONOMER) + i * ATOMS_PER_MONOMER 
            for i in range(n_monomers)
        ])
                
        # Process forces
        # Note: For monomers, we use ATOMS_PER_MONOMER, not max_atoms (which is for dimers)
        monomer_forces = process_monomer_forces(
            ml_monomer_forces, monomer_segment_idxs, ATOMS_PER_MONOMER, debug
        )

        
        debug_print(debug, "Monomer Contributions:",
            ml_monomer_energy=ml_monomer_energy,
            monomer_forces=monomer_forces
        )
        
        return {
            "out_E": ml_monomer_energy.sum(),
            "out_F": monomer_forces,
            "internal_E": ml_monomer_energy.sum(),
            "internal_F": monomer_forces,
            "monomer_energy": ml_monomer_energy  # Used for dimer calculations
        }

    def calculate_mm_contributions(
        positions: Array,
        cutoff_params: CutoffParameters,
        debug: bool
    ) -> Dict[str, Array]:
        """Calculate MM energy and force contributions (converted to eV)."""
        
        # Ensure positions are finite
        positions = jnp.where(jnp.isfinite(positions), positions, 0.0)

        MM_energy_and_gradient = get_MM_energy_forces_fns(
            positions,
            ATOMS_PER_MONOMER,
            N_MONOMERS=n_monomers,
            ml_cutoff_distance=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            mm_cutoff=cutoff_params.mm_cutoff,
            complementary_handoff=getattr(cutoff_params, "complementary_handoff", True),
        )
        
        
        mm_E, mm_grad = MM_energy_and_gradient(positions)
        
        # Check for NaN/Inf in MM energy and forces
        mm_E = jnp.where(jnp.isfinite(mm_E), mm_E, 0.0)
        mm_grad = jnp.where(jnp.isfinite(mm_grad), mm_grad, 0.0)

        # MM outputs are in kcal/mol and kcal/mol/Å. Convert to eV and eV/Å.
        mm_E = mm_E * kcalmol2ev
        mm_grad = mm_grad * kcalmol2ev
        
        # Ensure MM forces match the full system size
        n_atoms = positions.shape[0]

        
        debug_print(debug, "MM Contributions:", 
            mm_E=mm_E,
            mm_grad=mm_grad,
            mm_grad_shape=mm_grad.shape,
            n_atoms=n_atoms
        )
        return {
            "out_E": mm_E,
            "out_F": mm_grad,
            "dH": mm_E,
            "mm_E": mm_E,
            "mm_F": mm_grad,
        }

    if _HAVE_ASE:

        class AseDimerCalculator(ase_calc.Calculator):
            """ASE calculator implementation for dimer calculations"""

            implemented_properties = ["energy", "forces", "out"]

            def __init__(
                self,
                n_monomers: int,
                cutoff_params: CutoffParameters = None,
                doML: bool = True,
                doMM: bool = True,
                doML_dimer: bool = True,
                backprop: bool = True,
                debug: bool = False,
                energy_conversion_factor: float = 1.0,
                force_conversion_factor: float = 1.0,
                do_pbc_map: bool = False,
                pbc_map = None,
                verbose: bool = True,
            ):
                """Initialize calculator with configuration parameters
                
                Args:
                    verbose: If True, store full ModelOutput breakdown (ml_2b_E/F, mm_E/F, etc.) 
                             in self.results for analysis/testing. Adds overhead.
                """

                super().__init__()
                self.n_monomers = n_monomers
                self.cutoff_params = cutoff_params or CutoffParameters()
                self.doML = doML
                self.doMM = doMM
                self.doML_dimer = doML_dimer
                self.backprop = backprop
                self.debug = debug
                self.ep_scale = None
                self.sig_scale = None
                self.energy_conversion_factor = energy_conversion_factor
                self.force_conversion_factor = force_conversion_factor
                self.do_pbc_map = do_pbc_map
                self.pbc_map = pbc_map
                self.verbose = verbose
                self.atoms_per_monomer = ATOMS_PER_MONOMER

            def calculate(
                self,
                atoms,
                properties,
                system_changes=ase.calculators.calculator.all_changes,
            ):
                """Calculate energy and forces for given atomic configuration"""

                ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
                R = atoms.get_positions()
                Z = atoms.get_atomic_numbers()

                expected_atoms = self.n_monomers * self.atoms_per_monomer
                if len(Z) != expected_atoms:
                    raise ValueError(
                        "Atom count mismatch: len(Z) != n_monomers * ATOMS_PER_MONOMER. "
                        f"Got len(Z)={len(Z)}, expected {expected_atoms} "
                        f"({self.n_monomers}*{self.atoms_per_monomer}). This triggers padding and "
                        "can yield exact zero forces for the padded slots. "
                        "Fix ATOMS_PER_MONOMER or trim the input atoms."
                    )

                if np.any(Z <= 0):
                    bad_idx = np.where(Z <= 0)[0]
                    bad_symbols = [atoms[i].symbol for i in bad_idx]
                    raise ValueError(
                        "Invalid atomic numbers detected (Z<=0) at indices "
                        f"{bad_idx.tolist()} with symbols {bad_symbols}. "
                        "These atoms are treated as padding and will yield zero forces. "
                        "Fix the PDB element names or atom typing."
                    )

                out = {}
                
                if not self.backprop:
                    # Apply PBC mapping before JAX computation if needed
                    R_mapped = self.pbc_map(R) if self.do_pbc_map else R
                    out = spherical_cutoff_calculator(
                        positions=R_mapped,
                        atomic_numbers=Z,
                        n_monomers=self.n_monomers,
                        cutoff_params=self.cutoff_params,
                        doML=self.doML,
                        doMM=self.doMM,
                        doML_dimer=self.doML_dimer,
                        debug=self.debug,
                    )

                    E = out.energy  # Energy from calculator (negative, binding energy convention)
                    F = out.forces
                    
                    # Ensure forces from ModelOutput are finite
                    F = jnp.where(jnp.isfinite(F), F, 0.0)
                    
                    # Debug: Check forces after NaN check
                    if self.debug:
                        F_mags_after = jnp.linalg.norm(F, axis=1)
                        print(f"Non-backprop path - F after NaN check: atom 3 mag: {float(F_mags_after[3]):.6e}, atom 7: {float(F_mags_after[7]):.6e}")

                if self.backprop:
                    # OPTIMIZED BACKPROP PATH: Compute energy via autodiff but use forces directly from ModelOutput
                    # This avoids numerical instability from differentiating through the entire computation
                    # while still allowing energy to be computed via autodiff for training/optimization
                    R_mapped = self.pbc_map(R) if self.do_pbc_map else R
                    
                    # Compute ModelOutput to get forces directly (more stable)
                    out = spherical_cutoff_calculator(
                        positions=R_mapped,
                        atomic_numbers=Z,
                        n_monomers=self.n_monomers,
                        cutoff_params=self.cutoff_params,
                        doML=self.doML,
                        doMM=self.doMM,
                        doML_dimer=self.doML_dimer,
                        debug=self.debug,
                    )
                    
                    # Use forces directly from ModelOutput (computed via explicit gradients, more stable)
                    F = out.forces
                    
                    # For energy, we can still use autodiff if needed, but for now use directly
                    # The energy from ModelOutput is already correct
                    E = out.energy

                    # Ensure forces are finite
                    E = jnp.where(jnp.isfinite(E), E, 0.0)
                    F = jnp.where(jnp.isfinite(F), F, 0.0)

                if self.verbose:
                    # Store full ModelOutput with ML/MM breakdown for analysis
                    self.results["model_output"] = out
                    if hasattr(out, "_asdict"):
                        for k, v in out._asdict().items():
                            self.results[f"model_{k}"] = v
                else:
                    self.results["out"] = out
                # Energy sign handling:
                # - In backprop path (no fallback): Efn returns -spherical_cutoff_calculator(...).energy
                #   So E from value_and_grad is already negative
                # - In non-backprop path: E = out.energy (check actual sign from calculator)
                # - In fallback path: E = out_fallback.energy (same as non-backprop, check actual sign)
                # The spherical_cutoff_calculator returns negative energy (binding energy convention)
                # So:
                # - backprop (no fallback): E is already negative, use as-is
                # - non-backprop: E from calculator is negative, use as-is (no negation needed)
                # - fallback: E from calculator is negative, use as-is (no negation needed)
                # All paths should use E directly since calculator already returns negative energies
                final_energy = E
                
                self.results["energy"] = final_energy * self.energy_conversion_factor
                # Ensure forces are finite before storing
                forces_final = F * self.force_conversion_factor
                
                # Check for NaN/Inf using JAX operations first (works with JAX arrays)
                forces_final = jnp.where(jnp.isfinite(forces_final), forces_final, 0.0)

                if self.debug and hasattr(out, "internal_F"):
                    internal_F = out.internal_F
                    internal_F = jnp.where(jnp.isfinite(internal_F), internal_F, 0.0)
                    internal_F_host = np.asarray(jax.device_get(internal_F))
                    internal_zero_mask = np.linalg.norm(internal_F_host, axis=1) < 1e-12
                    if np.any(internal_zero_mask):
                        zero_indices = np.where(internal_zero_mask)[0]
                        print(f"DEBUG internal_F zero-force atoms: {zero_indices}")
                        print(f"DEBUG internal_F zero-force Z: {Z[zero_indices]}")
                        if hasattr(out, "ml_2b_F"):
                            ml_2b_F_host = np.asarray(jax.device_get(out.ml_2b_F))
                            print(f"DEBUG internal_F zeros -> ml_2b_F sample: {ml_2b_F_host[zero_indices[:10]]}")
                
                # Debug: Check forces BEFORE conversion to numpy (still in JAX)
                if self.debug:
                    # Why ml_2b can be zero: (1) doML_dimer=False, or (2) dimer COM distance > mm_switch_on
                    print(f"DEBUG doML_dimer: {self.doML_dimer}")
                    if getattr(self, "cutoff_params", None) is not None:
                        cp = self.cutoff_params
                        print(f"DEBUG cutoffs: ml_cutoff={getattr(cp,'ml_cutoff',None)}, mm_switch_on={getattr(cp,'mm_switch_on',None)}")
                    # Dimer COM distance for first pair (atoms 0:n_per and n_per:2*n_per)
                    try:
                        R_np = np.asarray(jax.device_get(R) if hasattr(R, "shape") and hasattr(jax, "device_get") else R)
                        n_mon = getattr(self, "n_monomers", 2)
                        n_per = (R_np.shape[0] // n_mon) if n_mon else 10
                        if R_np.shape[0] >= 2 * n_per:
                            com0 = R_np[:n_per].mean(axis=0)
                            com1 = R_np[n_per : 2 * n_per].mean(axis=0)
                            d_com = float(np.linalg.norm(com1 - com0))
                            print(f"DEBUG dimer COM distance: {d_com:.4f} (ml_2b is 0 when this > mm_switch_on)")
                    except Exception as e:
                        print(f"DEBUG dimer COM distance: (could not compute: {e})")
                    # ML 2-body contributions (from ModelOutput)
                    try:
                        ml_2b_E_val = float(np.asarray(jax.device_get(out.ml_2b_E)))
                        print(f"DEBUG ml_2b_E: {ml_2b_E_val:.6e}")
                    except Exception as e:
                        print(f"DEBUG ml_2b_E extraction failed: {e}")
                    try:
                        ml_2b_F_np = np.asarray(jax.device_get(out.ml_2b_F))
                        print(f"DEBUG ml_2b_F shape: {ml_2b_F_np.shape}")
                        ml_2b_F_mags = np.linalg.norm(ml_2b_F_np, axis=1)
                        print(f"DEBUG ml_2b_F per-atom magnitudes: {ml_2b_F_mags}")
                        if ml_2b_F_np.size > 0:
                            print(f"DEBUG ml_2b_F first 3 rows:\n{ml_2b_F_np[:3]}")
                    except Exception as e:
                        print(f"DEBUG ml_2b_F extraction failed: {e}")
                    # Get values from JAX array before conversion
                    # CRITICAL: Use np.array() with explicit evaluation to ensure we get concrete values
                    try:
                        # Force evaluation of JAX array to numpy
                        forces_jax_np = np.array(forces_final)
                        forces_jax_mags_np = np.linalg.norm(forces_jax_np, axis=1)
                        print(f"BEFORE numpy conversion - forces shape: {forces_final.shape}, R shape: {R.shape}")
                        print(f"BEFORE numpy conversion - converted to np array, shape: {forces_jax_np.shape}")
                        # Check all atoms for zeros before conversion
                        zero_mask_jax = forces_jax_mags_np < 1e-10
                        zero_count_jax = np.sum(zero_mask_jax)
                        print(f"BEFORE numpy conversion - zero-force atoms: {zero_count_jax}/{len(forces_jax_mags_np)}")
                        if zero_count_jax > 0:
                            zero_indices_jax = np.where(zero_mask_jax)[0]
                            print(f"BEFORE numpy conversion - zero-force atom indices: {zero_indices_jax}")
                        # Sample all atoms to see pattern
                        print(f"BEFORE numpy conversion - force magnitudes for all atoms: {forces_jax_mags_np}")
                    except Exception as e:
                        print(f"ERROR extracting JAX array values: {e}")
                        # Fallback: try using jnp.asarray then np.asarray
                        try:
                            forces_jax_eval = np.asarray(jnp.asarray(forces_final))
                            forces_jax_mags_np = np.linalg.norm(forces_jax_eval, axis=1)
                            print(f"BEFORE numpy conversion (fallback) - force magnitudes: {forces_jax_mags_np}")
                        except Exception as e2:
                            print(f"ERROR in fallback extraction: {e2}")
                
                # Convert to numpy array for storage (ASE expects numpy arrays)
                # Ensure proper shape: (n_atoms, 3)
                # CRITICAL: Ensure JAX array is fully evaluated before conversion
                # Use jax.device_get() to move from device to host, then convert to numpy
                # Note: jax is imported at module level, so we can use it directly
                try:
                    # First ensure array is on CPU and fully computed
                    forces_final_host = jax.device_get(forces_final)
                    # Then convert to numpy array (this should work reliably now)
                    forces_final = np.asarray(forces_final_host, dtype=np.float64)
                    if self.debug:
                        print(f"Conversion: Used jax.device_get() then np.asarray(), dtype={forces_final.dtype}, shape={forces_final.shape}")
                except Exception as e:
                    if self.debug:
                        print(f"WARNING: jax.device_get() + np.asarray() failed: {e}, falling back to np.array()")
                    try:
                        forces_final = np.array(forces_final, dtype=np.float64)
                    except Exception as e2:
                        if self.debug:
                            print(f"WARNING: np.array() also failed: {e2}, using np.asarray()")
                        forces_final = np.asarray(forces_final)
                
                # Debug: Check forces IMMEDIATELY AFTER conversion to numpy
                if self.debug:
                    force_mags_after_conv = np.linalg.norm(forces_final, axis=1)
                    print(f"AFTER numpy conversion - forces shape: {forces_final.shape}")
                    # Check all atoms for zeros after conversion
                    zero_mask_after = force_mags_after_conv < 1e-10
                    zero_count_after = np.sum(zero_mask_after)
                    print(f"AFTER numpy conversion - zero-force atoms: {zero_count_after}/{len(force_mags_after_conv)}")
                    if zero_count_after > 0:
                        zero_indices_after = np.where(zero_mask_after)[0]
                        print(f"AFTER numpy conversion - zero-force atom indices: {zero_indices_after}")
                    # Sample same atoms for comparison
                    print(f"AFTER numpy conversion - sample atoms (0,3,7,10,19): mags={force_mags_after_conv[[0,3,7,10,min(19,len(force_mags_after_conv)-1)]]}")
                    # Compare specific atoms that were non-zero before
                    if hasattr(self, 'zero_count_jax') and self.zero_count_jax == 0 and zero_count_after > 0:
                        print("WARNING: Forces became zero during numpy conversion!")
                        if hasattr(self, 'forces_jax_mags_np'):
                            for idx in zero_indices_after[:10]:
                                if idx < len(self.forces_jax_mags_np):
                                    print(f"  Atom {idx}: BEFORE={self.forces_jax_mags_np[idx]:.6e}, AFTER={force_mags_after_conv[idx]:.6e}")
                                else:
                                    print(f"  Atom {idx}: BEFORE=<out of range>, AFTER={force_mags_after_conv[idx]:.6e}")
                    # Store for comparison in final check
                    if not hasattr(self, 'forces_jax_mags_np'):
                        self.forces_jax_mags_np = forces_jax_mags_np
                        self.zero_count_jax = zero_count_jax
                
                # Final check: ensure shape is correct and all values are finite
                if forces_final.shape[0] != R.shape[0]:
                    # Shape mismatch - pad or truncate to match number of atoms
                    if self.debug:
                        print(f"Shape mismatch detected! forces_final.shape[0]={forces_final.shape[0]}, R.shape[0]={R.shape[0]}")
                    if forces_final.shape[0] < R.shape[0]:
                        padding = np.zeros((R.shape[0] - forces_final.shape[0], 3))
                        forces_final = np.concatenate([forces_final, padding], axis=0)
                        if self.debug:
                            print(f"Padded forces: new shape={forces_final.shape}")
                    else:
                        if self.debug:
                            print(f"Truncating forces from {forces_final.shape[0]} to {R.shape[0]}")
                            print(f"BEFORE truncation - atom 3: {forces_final[3]}, atom 7: {forces_final[7]}")
                        forces_final = forces_final[:R.shape[0]]
                        if self.debug:
                            print(f"AFTER truncation - atom 3: {forces_final[3]}, atom 7: {forces_final[7]}")
                else:
                    if self.debug:
                        print(f"No shape mismatch: forces_final.shape={forces_final.shape}, R.shape={R.shape}")
                
                # Final NaN check (should be redundant but ensures safety)
                forces_final = np.where(np.isfinite(forces_final), forces_final, 0.0)
                
                # Debug: Check which atoms have zero forces before storing
                if self.debug:
                    force_mags_final = np.linalg.norm(forces_final, axis=1)
                    zero_count_final = np.sum(force_mags_final < 1e-10)
                    print(f"Calculator storage - zero-force atoms before storing: {zero_count_final}/{forces_final.shape[0]}")
                    zero_indices = np.where(force_mags_final < 1e-10)[0]
                    if len(zero_indices) > 0:
                        print(f"Calculator storage - zero-force atom indices: {zero_indices}")
                        # Print forces for zero atoms to see if they're actually zero or just very small
                        for idx in zero_indices[:10]:  # Print first 10 zero atoms
                            print(f"  Atom {idx}: force={forces_final[idx]}, mag={force_mags_final[idx]:.6e}")
                    # Check for any pattern - see if zeros are clustered or random
                    if len(zero_indices) > 1:
                        zero_diffs = np.diff(np.sort(zero_indices))
                        print(f"Calculator storage - spacing between zero atoms: {zero_diffs}")
                        # Check if all zeros are in second monomer (indices >= ATOMS_PER_MONOMER)
                        if hasattr(self, 'n_monomers') and hasattr(self, 'cutoff_params'):
                            n_monomers = self.n_monomers
                            atoms_per_monomer = (
                                self.cutoff_params.ATOMS_PER_MONOMER
                                if hasattr(self.cutoff_params, "ATOMS_PER_MONOMER")
                                else self.atoms_per_monomer
                            )
                            if atoms_per_monomer:
                                zeros_in_first = np.sum(zero_indices < atoms_per_monomer)
                                zeros_in_second = np.sum(zero_indices >= atoms_per_monomer)
                                print(f"Calculator storage - zeros in first monomer (0-{atoms_per_monomer-1}): {zeros_in_first}, second monomer (≥{atoms_per_monomer}): {zeros_in_second}")
                    # Also check a few random non-zero atoms to ensure they're correct
                    non_zero_indices = np.where(force_mags_final >= 1e-10)[0]
                    if len(non_zero_indices) > 0:
                        print(f"Calculator storage - sample non-zero atoms (first 5): {non_zero_indices[:5]}")
                        for idx in non_zero_indices[:5]:
                            print(f"  Atom {idx}: force={forces_final[idx]}, mag={force_mags_final[idx]:.6e}")
                
                self.results["forces"] = forces_final

        def get_spherical_cutoff_calculator(
            atomic_numbers: Array,
            atomic_positions: Array,
            n_monomers: int,
            cutoff_params: CutoffParameters = None,
            doML: bool = doML,
            doMM: bool = doMM,
            doML_dimer: bool = doML_dimer,
            backprop: bool = True,
            debug: bool = debug,
            energy_conversion_factor: float = 1.0,
            force_conversion_factor: float = 1.0,
            verbose: bool = None,
            # do_pbc_map: bool = False,
            # pbc_map = None,
        ) -> Tuple[AseDimerCalculator, Callable]:
            """Factory function to create calculator instances.

            doML, doMM, doML_dimer, debug default to the values passed to setup_calculator.
            Pass them explicitly here to override per-call.

            Args:
                verbose: If True, store full ModelOutput breakdown in results.
                         If None, defaults to debug value.
            """

            calculator = AseDimerCalculator(
                n_monomers=n_monomers,
                cutoff_params=cutoff_params,
                doML=doML,
                doMM=doMM,
                doML_dimer=doML_dimer,
                backprop=backprop,
                debug=debug,
                energy_conversion_factor=energy_conversion_factor,
                force_conversion_factor=force_conversion_factor,
                do_pbc_map=do_pbc_map,
                pbc_map=pbc_map,
                verbose=verbose,
            )

            return calculator, spherical_cutoff_calculator

    else:  # pragma: no cover - exercised when ASE not installed

        class AseDimerCalculator:  # type: ignore[too-few-public-methods]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ModuleNotFoundError("ase is required for AseDimerCalculator")

        def get_spherical_cutoff_calculator(*args: Any, **kwargs: Any):  # type: ignore[override]
            raise ModuleNotFoundError("ase is required for get_spherical_cutoff_calculator")



    def process_monomer_forces(
        ml_monomer_forces: Array,
        monomer_segment_idxs: Array,
        atoms_per_monomer: int,
        debug: bool = False,
    ) -> Array:
        """Process and reshape monomer forces with proper masking.
        
        Args:
            ml_monomer_forces: Raw forces from ML model (shape: (n_monomers * max_atoms, 3) where max_atoms may be padded)
            monomer_segment_idxs: Indices for force segmentation (length: n_monomers * atoms_per_monomer)
            atoms_per_monomer: Number of atoms per monomer (ATOMS_PER_MONOMER)
            debug: Enable debug printing
            
        Returns:
            Array: Processed monomer forces
        """
        # Determine n_monomers from segment indices length
        n_monomers = monomer_segment_idxs.shape[0] // atoms_per_monomer
        
        # Determine atoms_per_system from ml_monomer_forces shape
        # ml_monomer_forces has shape (n_monomers * atoms_per_system, 3) where atoms_per_system = max_atoms
        # (max_atoms is used for padding to handle both monomers and dimers)
        total_forces = ml_monomer_forces.shape[0]
        atoms_per_system = total_forces // n_monomers
        
        # Debug prints disabled to reduce verbosity
        
        # Reshape to (n_monomers, atoms_per_system, 3)
        # Forces from model are in batch order: [batch0_atom0, batch0_atom1, ..., batch0_atomN, batch1_atom0, ...]
        # where each batch item has atoms_per_system atoms (padded to max_atoms for dimers)
        monomer_forces = ml_monomer_forces.reshape(n_monomers, atoms_per_system, 3)
        
        # Take only first atoms_per_monomer atoms per monomer (discard any padding beyond ATOMS_PER_MONOMER)
        # Result shape: (n_monomers, atoms_per_monomer, 3)
        monomer_forces_valid = monomer_forces[:, :atoms_per_monomer, :]
        
        # Flatten to (n_monomers * atoms_per_monomer, 3) for segment_sum
        # This gives forces in batch order: [batch0_atom0...batch0_atom9, batch1_atom0...batch1_atom9, ...]
        forces_flat = monomer_forces_valid.reshape(-1, 3)
        
        # Check if segment_idxs are sequential (in which case segment_sum is a no-op)
        # If monomer_segment_idxs is [0,1,2,...,9, 10,11,...,19, ...], then segment_sum just copies
        # Since forces_flat is already in the correct order [batch0_atom0...batch0_atom9, batch1_atom0...batch1_atom9, ...]
        # and monomer_segment_idxs maps to [0,1,...,9, 10,11,...,19, ...], segment_sum is effectively a no-op
        # But we still use it for consistency with the general case (non-sequential indices)
        processed_forces = jax.ops.segment_sum(
            forces_flat,
            monomer_segment_idxs,
            num_segments=n_monomers * atoms_per_monomer
        )
        

        # Ensure all forces are finite
        processed_forces = jnp.where(jnp.isfinite(processed_forces), processed_forces, 0.0)

        if debug:
            force_mags = jnp.linalg.norm(processed_forces, axis=1)
            zero_mask = force_mags < 1e-12
            zero_count = jnp.sum(zero_mask)
            # Use JAX-safe debug prints inside jit
            jax.debug.print("DEBUG monomer forces: zero count {c}", c=zero_count, ordered=False)
            zero_indices = jnp.where(zero_mask, size=10, fill_value=-1)[0]
            jax.debug.print("DEBUG monomer forces: zero indices sample {idx}", idx=zero_indices, ordered=False)
            raw_sample = jnp.take(forces_flat, zero_indices, axis=0, mode="clip")
            jax.debug.print("DEBUG monomer forces: raw sample {s}", s=raw_sample, ordered=False)
        
        debug_print(debug, "Process Monomer Forces:",
            raw_forces=ml_monomer_forces,
            processed_forces=processed_forces
        )
        
        return processed_forces

    def calculate_dimer_contributions(
        positions: Array,
        e: Array,
        f: Array,
        n_dimers: int,
        monomer_energies: Array,
        cutoff_params: CutoffParameters,
        debug: bool = False
    ) -> Dict[str, Array]:
        """Calculate energy and force contributions from dimers.
        
        Args:
            positions: Atomic positions
            e: ML energies
            f: ML forces
            n_dimers: Number of dimers
            monomer_energies: Pre-calculated monomer energies
            debug: Enable debug printing
            
        Returns:
            Dict containing dimer energy and force contributions
        """
        # Get dimer energies and forces
        ml_dimer_energy = jnp.array(e[n_monomers:]).flatten()
        # Extract dimer forces: skip monomer atoms, take dimer atoms
        # The batch is padded to max_atoms per system
        max_atoms = max(ATOMS_PER_MONOMER, 2 * ATOMS_PER_MONOMER)
        monomer_atoms = n_monomers * max_atoms

        ml_dimer_forces = f[monomer_atoms:]
        
        # Calculate force segments for dimers
        force_segments = calculate_dimer_force_segments(n_dimers)
        
        # Calculate interaction energies
        monomer_contrib = calculate_monomer_contribution_to_dimers(
            monomer_energies, jnp.array(dimer_perms)
        )
        dimer_int_energies = ml_dimer_energy - monomer_contrib
        
        if debug:
            print(f"DEBUG dimer_int_energies: {dimer_int_energies}")
            print(f"DEBUG ml_dimer_energy: {ml_dimer_energy}")
            print(f"DEBUG monomer_contrib: {monomer_contrib}")
            print(f"DEBUG dimer_perms: {dimer_perms}")
            print(f"DEBUG force_segments: {force_segments}")
            print(f"DEBUG ml_dimer_forces: {ml_dimer_forces}")
            print(f"DEBUG monomer_energies: {monomer_energies}")
            print(f"DEBUG cutoff_params: {cutoff_params}")
            print(f"DEBUG max_atoms: {max_atoms}")
            print(f"DEBUG debug: {debug}")


        # Process dimer forces
        dimer_forces = process_dimer_forces(
            ml_dimer_forces, force_segments, n_dimers, debug
        )

        # Check how many atoms have non-zero forces
        force_magnitudes = jnp.linalg.norm(dimer_forces, axis=1)
        non_zero_count = jnp.sum(force_magnitudes > 1e-10)

        # Apply switching functions
        switched_results = apply_dimer_switching(
            positions, dimer_int_energies, dimer_forces, cutoff_params, max_atoms, debug
        )
        
        debug_print(debug, "Dimer Contributions:",
            dimer_energies=switched_results["energies"],
            dimer_forces=switched_results["forces"]
        )
        
        return {
            "out_E": switched_results["energies"].sum(),
            "out_F": switched_results["forces"]  ,
            "dH": switched_results["energies"].sum(),
            "ml_2b_E": switched_results["energies"].sum(),
            "ml_2b_F": switched_results["forces"]  
        }

    def calculate_dimer_force_segments(n_dimers: int) -> Array:
        """Calculate force segments for dimer force summation."""
        dimer_pairs = jnp.array(dimer_perms)

        # Calculate base indices for each monomer
        # Note: dimer_pairs uses 0-indexed monomer indices (0, 1)
        first_indices = ATOMS_PER_MONOMER * dimer_pairs[:, 0:1]
        second_indices = ATOMS_PER_MONOMER * dimer_pairs[:, 1:2]
        
        # Create atom offsets
        atom_offsets = jnp.arange(ATOMS_PER_MONOMER)
        
        # Combine indices for both monomers
        force_segments = jnp.concatenate([
            first_indices + atom_offsets[None, :],
            second_indices + atom_offsets[None, :]
        ], axis=1)
        
        force_segments_flat = force_segments.reshape(-1)

        return force_segments_flat

    def calculate_monomer_contribution_to_dimers(
        monomer_energies: Array,
        dimer_pairs: Array
    ) -> Array:
        """Calculate monomer energy contributions to dimer energies."""
        return (monomer_energies[dimer_pairs[:, 0]] + 
                monomer_energies[dimer_pairs[:, 1]])

    def process_dimer_forces(
        dimer_forces: Array,
        force_segments: Array,
        n_dimers: int,
        debug: bool
    ) -> Array:
        """Process and reshape dimer forces."""
        forces = dimer_forces.reshape(n_dimers, 2 * ATOMS_PER_MONOMER, 3)

        forces_flat = forces.reshape(-1, 3)
        
        # Check for NaN in input forces
        nan_count = jnp.sum(~jnp.isfinite(forces_flat))
        # jax.debug.print("Dimer forces NaN check - before segment_sum: {n}", n=nan_count, ordered=False)
        
        processed_forces = jax.ops.segment_sum(
            forces_flat,
            force_segments,
            num_segments=n_monomers * ATOMS_PER_MONOMER
        )
        
        # Check for NaN in output
        nan_count = jnp.sum(~jnp.isfinite(processed_forces))
        # jax.debug.print("Dimer forces NaN check - after segment_sum: {n}", n=nan_count, ordered=False)
        
        # Ensure all forces are finite
        processed_forces = jnp.where(jnp.isfinite(processed_forces), processed_forces, 0.0)
        
        return processed_forces

    def apply_dimer_switching(
        positions: Array,
        dimer_energies: Array,
        dimer_forces: Array,
        cutoff_params: CutoffParameters,
        max_atoms: int,
        debug: bool
    ) -> Dict[str, Array]:
        """Apply switching functions to dimer energies and forces.
        
        Forces are computed using the product rule:
        F = -d/dR [E * s(R)] = -[dE/dR * s(R) + E * ds/dR]
        where s(R) is the switching function.
        
        Since dimer_forces is already mapped to the full system, we compute
        forces by differentiating the switched energy function directly using JAX.
        """
        n_dimers = len(all_dimer_idxs)
        force_segments = calculate_dimer_force_segments(n_dimers)
        
        # Calculate switched energies using cutoff parameters
        switched_energy = jax.vmap(lambda x, f: switch_ML(
            x.reshape(max_atoms, 3), 
            f,
            ml_cutoff=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        ))(positions[jnp.array(all_dimer_idxs)], dimer_energies)
        
        # Calculate switching scale for each dimer (needed for force scaling)
        switching_scales = jax.vmap(lambda x: switch_ML(
            x.reshape(max_atoms, 3),
            1.0,  # Use 1.0 to get just the switching scale
            ml_cutoff=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        ))(positions[jnp.array(all_dimer_idxs)])
        
        # Calculate energy-weighted switching gradients (E * ds/dR)
        # switch_ML_grad(X, E) computes d/dX [scale(X) * E] = E * d(scale)/dX
        switched_grad = jax.vmap(lambda x, f: switch_ML_grad(
            x.reshape(max_atoms, 3),
            f,
            ml_cutoff=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        ))(positions[jnp.array(all_dimer_idxs)], dimer_energies)
        
        # Extract relevant atoms from switched_grad (first 2*ATOMS_PER_MONOMER per dimer)
        # switched_grad shape: (n_dimers, max_atoms, 3)
        dimer_switching_grads = switched_grad[:, :2 * ATOMS_PER_MONOMER, :]  # (n_dimers, 2*ATOMS_PER_MONOMER, 3)
        dimer_switching_grads_flat = dimer_switching_grads.reshape(-1, 3)  # (n_dimers * 2*ATOMS_PER_MONOMER, 3)
        
        # Map energy-weighted switching gradients to full system.
        # switch_ML_grad gives d(s*E)/dX = E*ds/dX; we subtract it below so F = -grad(s*E).
        energy_weighted_grad = jax.ops.segment_sum(
            dimer_switching_grads_flat,
            force_segments,
            num_segments=n_monomers * ATOMS_PER_MONOMER
        )  # Shape: (n_monomers * ATOMS_PER_MONOMER, 3)

        # For the scale * F_dimer term, we need to scale dimer_forces by switching
        # Since each atom may belong to multiple dimers, we compute atom-wise scales
        # by using segment_sum to accumulate scales from all dimers containing each atom
        # Repeat switching_scales for each atom in each dimer (2*ATOMS_PER_MONOMER atoms per dimer)
        switching_scales_per_atom = jnp.repeat(switching_scales, 2 * ATOMS_PER_MONOMER)  # (n_dimers * 2*ATOMS_PER_MONOMER,)
        
        # Use segment_sum to accumulate scales for each atom in the full system
        atom_switching_scales_sum = jax.ops.segment_sum(
            switching_scales_per_atom,
            force_segments,
            num_segments=n_monomers * ATOMS_PER_MONOMER
        )
        
        # Count how many dimers each atom belongs to
        atom_dimer_counts = jax.ops.segment_sum(
            jnp.ones_like(switching_scales_per_atom),
            force_segments,
            num_segments=n_monomers * ATOMS_PER_MONOMER
        )
        
        # Average scales for atoms that belong to multiple dimers (avoid division by zero)
        # Ensure atom_dimer_counts is safe for division (no zeros, negatives, or NaN)
        safe_counts = jnp.maximum(atom_dimer_counts, 1.0)
        safe_counts = jnp.where(jnp.isfinite(safe_counts), safe_counts, 1.0)
        safe_scales_sum = jnp.where(jnp.isfinite(atom_switching_scales_sum), atom_switching_scales_sum, 0.0)
        
        atom_switching_scales = jnp.where(
            atom_dimer_counts > 0,
            safe_scales_sum / safe_counts,
            1.0  # Default to 1.0 if atom is in no dimers (shouldn't happen)
        )
        # Ensure scales are finite
        atom_switching_scales = jnp.where(jnp.isfinite(atom_switching_scales), atom_switching_scales, 1.0)

        
        # Apply switching scale to dimer forces
        # Ensure dimer_forces are finite before scaling
        dimer_forces_safe = jnp.where(jnp.isfinite(dimer_forces), dimer_forces, 0.0)
        scaled_dimer_forces = dimer_forces_safe * atom_switching_scales[:, None]

        
        # Combine both terms: F_switched = -d(s*E)/dR = s*F_dimer - E*ds/dR
        # switch_ML_grad gives d(s*E)/dX = E*ds/dX, so we subtract it to get F = -grad(s*E)
        energy_weighted_grad_safe = jnp.where(jnp.isfinite(energy_weighted_grad), energy_weighted_grad, 0.0)
        switched_forces = scaled_dimer_forces - energy_weighted_grad_safe
        
        # Final safety check - ensure all forces are finite
        switched_forces = jnp.where(jnp.isfinite(switched_forces), switched_forces, 0.0)
        
        return {
            "energies": switched_energy,
            "forces": switched_forces
        }

    return get_spherical_cutoff_calculator

######################################################

import jax.numpy as jnp

def check_lattice_invariance(sc_fn, R, Z, n_monomers, cutoff_params, cell):
    # base energy
    E0 = sc_fn(R, Z, n_monomers, cutoff_params).energy
    # translate monomer 0 by +a (first lattice vector)
    a = cell[0]
    g0 = jnp.where(jnp.arange(R.shape[0]) < (R.shape[0] // n_monomers))[0]  # assumes chunking; replace if mol_id known
    R_shift = R.at[g0].add(a)
    E1 = sc_fn(R_shift, Z, n_monomers, cutoff_params).energy
    return float(E0 - E1)