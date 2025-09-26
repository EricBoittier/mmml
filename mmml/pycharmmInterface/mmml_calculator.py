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
from pbc_prep_factory import make_pbc_mapper


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
    from mmml.physnetjax.physnetjax.training.training import train_model
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
    from e3x.nn import smooth_switch, smooth_cutoff  # type: ignore[import-not-found]
    _HAVE_E3X = True
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    e3x = None  # type: ignore[assignment]
    _HAVE_E3X = False

    def smooth_switch(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("e3x is required for smooth_switch")

    def smooth_cutoff(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("e3x is required for smooth_cutoff")


try:
    import optax  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    optax = None  # type: ignore[assignment]


# Public constants ---------------------------------------------------------

# Energy conversion (1 eV -> kcal / mol).  Use ASE when available for
# consistency; otherwise fall back to the known constant so documentation tests
# can still import this module.
if _HAVE_ASE:
    ev2kcalmol = 1 / (ase.units.kcal / ase.units.mol)  # type: ignore[attr-defined]
else:
    ev2kcalmol = 23.060548867


# Module-level configuration ------------------------------------------------

SPATIAL_DIMS: int = 3  # Number of spatial dimensions (x, y, z)


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
            jax.debug.print(f"{msg}\n{{x}}", x=arg)
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




class CutoffParameters:
    """Parameters for ML and MM cutoffs and switching functions"""
    def __init__(
        self,
        ml_cutoff: float = 2.0,
        mm_switch_on: float = 5.0,
        mm_cutoff: float = 1.0
    ):
        """
        Args:
            ml_cutoff: Distance where ML potential is cut off
            mm_switch_on: Distance where MM potential starts switching on
            mm_cutoff: Final cutoff for MM potential
        """
        self.ml_cutoff = ml_cutoff
        self.mm_switch_on = mm_switch_on
        self.mm_cutoff = mm_cutoff


class ModelOutput(NamedTuple):
    energy: Array  # Shape: (,), total energy in kcal/mol
    forces: Array  # Shape: (n_atoms, 3), forces in kcal/mol/Å
    dH: Array # Shape: (,), total interaction energy in kcal/mol
    internal_E: Array # Shape: (,) total internal energy in kcal/mol
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
    doML: bool = True,
    doMM: bool = True,
    doML_dimer: bool = True,
    debug: bool = False,
    ep_scale = None,
    sig_scale = None,
    model_restart_path = None,
    MAX_ATOMS_PER_SYSTEM = 100,
    ml_energy_conversion_factor: float = ev2kcalmol,
    ml_force_conversion_factor: float = ev2kcalmol,
    cell = False,
):
    if model_restart_path is None:
        # raise ValueError("model_restart_path must be provided")
        model_restart_path = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-7c36e6f9-6f10-4d21-bf6d-693df9b8cd40"
    n_monomers = N_MONOMERS

    cutoffparameters = CutoffParameters(ml_cutoff_distance, mm_switch_on, mm_cutoff)
    
    all_dimer_idxs = []
    for a, b in dimer_permutations(n_monomers):
        all_dimer_idxs.append(indices_of_pairs(a + 1, b + 1, n_atoms=ATOMS_PER_MONOMER))

    all_monomer_idxs = []
    for a in range(1, n_monomers + 1):
        all_monomer_idxs.append(indices_of_monomer(a, n_atoms=ATOMS_PER_MONOMER, n_mol=n_monomers))
    # print("all_monomer_idxs", all_monomer_idxs)
    # print("all_dimer_idxs", all_dimer_idxs)
    unique_res_ids = []
    collect_monomers = []
    dimer_perms = dimer_permutations(n_monomers)
    for i, _ in enumerate(dimer_perms):
        a,b = _
        if a not in unique_res_ids and b not in unique_res_ids:
            unique_res_ids.append(a)
            unique_res_ids.append(b)
            collect_monomers.append(1)
            print(a,b)
        else:
            collect_monomers.append(0)

    print("unique_res_ids", unique_res_ids)
    # print("collect_monomers", collect_monomers)
    print("len(dimer_perms)", len(dimer_perms))

    N_MONOMERS = n_monomers
    # Batch processing constants
    BATCH_SIZE: int = N_MONOMERS + len(dimer_perms)  # Number of systems per batch
    # print(BATCH_SIZE)
    restart_path = Path(model_restart_path) if type(model_restart_path) == str else model_restart_path
    restart = get_last(restart_path)
    # Setup monomer model
    params, MODEL = get_params_model(restart)
    MODEL.natoms = MAX_ATOMS_PER_SYSTEM 

    if cell:
        # somewhere in your factory / calculator init
        from pbc_prep_factory import make_pbc_mapper

        cell = jnp.asarray(atoms.get_cell(complete=True).array)
        mol_id = None
        try:
            mol_id_np = atoms.get_array('molecule_id')
            mol_id = jnp.asarray(mol_id_np, dtype=jnp.int32)
        except Exception:
            mol_id = None
        do_pbc_map = True
        pbc_map = make_pbc_mapper(cell=cell, mol_id=mol_id, n_monomers=n_monomers)
    else:
        pbc_map = do_pbc_map = False

    from functools import partial
    @partial(jax.jit, static_argnames=['dif', 'ml_cutoff', 'mm_switch_on', 'debug', 'ATOMS_PER_MONOMER'])
    def switch_ML(X,
        ml_energy,
        dif=10 ** (-6),
        ml_cutoff=0.01,
        mm_switch_on=5.0,
        debug=False,    
        ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
    ):
        # Calculate center-of-mass distance between monomers
        r = jnp.linalg.norm(X[:ATOMS_PER_MONOMER].T.mean(axis=1) - X[ATOMS_PER_MONOMER:2*ATOMS_PER_MONOMER].T.mean(axis=1))
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        r = r + eps

        ml_cutoff_region = mm_switch_on - ml_cutoff   
        ml_cutoff_region = mm_switch_on - ml_cutoff
        ml_cutoff_fn = 1 - smooth_cutoff(r+3, cutoff=ml_cutoff_region)
        switch_off_ml = 1 - smooth_switch(r, x0=mm_switch_on-0.01, x1=mm_switch_on)
        ml_scale = switch_off_ml
        
        # Ensure scale is between 0 and 1
        ml_scale = jnp.clip(ml_scale, 0.0, 1.0)
        
        ml_contrib = ml_scale * ml_energy
        return ml_contrib

    switch_ML_grad = jax.grad(switch_ML)




    
    def get_MM_energy_forces_fns(
        R, 
        ATOMS_PER_MONOMER, 
        N_MONOMERS=N_MONOMERS, 
        ml_cutoff_distance=ml_cutoff_distance, 
        mm_switch_on=mm_switch_on, 
        mm_cutoff=mm_cutoff,
        sig_scale = sig_scale,
        ep_scale = ep_scale
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
        atomtype_codes = np.array(psf.get_atype())[:N_MONOMERS*ATOMS_PER_MONOMER]

        rmins_per_system = jnp.take(at_flat_rm, at_codes) 
        epsilons_per_system = jnp.take(at_flat_ep, at_codes)

        rs = distances
        q_per_system = jnp.take(at_flat_q, at_codes)


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
            # sig = sig / (2 ** (1 / 6))
            r6 = (sig / r) ** a
            return ep * (r6 ** b - 2 * r6)
        
        coulombs_constant = 3.32063711e2 #Coulomb's constant kappa = 1/(4*pi*e0) in kcal-Angstroms/e^2.
        def coulomb(r, qq, constant = coulombs_constant):
            return -constant * qq/r
        

        def get_switching_function(
            ml_cutoff_distance: float = 2.0,
            mm_switch_on: float = 5.0,
            mm_cutoff: float = 1.0,
        ):
            @jax.jit
            def apply_switching_function(
                positions: Array,  # Shape: (n_atoms, 3)
                pair_energies: Array,  # Shape: (n_pairs,)
            ) -> Array:
                """Applies smooth switching function to MM energies based on distances.
                
                Args:
                    positions: Atomic positions
                    pair_energies: Per-pair MM energies to be scaled
                    
                Returns:
                    Array: Scaled MM energies after applying switching function
                """
                # Calculate pairwise distances
                # r = jnp.linalg.norm(X[:5].T.mean(axis=1) - X[5:10].T.mean(axis=1))
                pair_positions = positions[pair_idx_atom_atom[:,0]].T.mean(axis=1) - positions[pair_idx_atom_atom[:,1]].T.mean(axis=1)
                distances = jnp.linalg.norm(pair_positions)
                
                # Calculate switching functions
                ml_cutoff = mm_switch_on - ml_cutoff_distance
                switch_on = smooth_switch(distances, x0=ml_cutoff, x1=mm_switch_on)
                switch_off = 1 - smooth_switch(distances - mm_cutoff - mm_switch_on, 
                                            x0=ml_cutoff, 
                                            x1=mm_switch_on)
                cutoff = 1 - smooth_cutoff(distances, cutoff=ml_cutoff_distance)
                
                # Combine switching functions and apply to energies
                switching_factor = switch_on * switch_off * cutoff
                scaled_energies = pair_energies * switching_factor
                
                return scaled_energies.sum()
                
            return apply_switching_function

        # Create the switching function with specified parameters
        apply_switching_function = get_switching_function(
            ml_cutoff_distance=ml_cutoff_distance,
            mm_switch_on=mm_switch_on,
            mm_cutoff=mm_cutoff
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
        
        # Calculate gradients
        mm_energy_grad = jax.grad(calculate_mm_energy)
        switching_grad = jax.grad(apply_switching_function)

        @jax.jit 
        def calculate_mm_energy_and_forces(
            positions: Array,  # Shape: (n_atoms, 3)
        ) -> Tuple[Array, Array]:
            """Calculates MM energy and forces with switching."""
            # Calculate base MM energies
            pair_energies = calculate_mm_pair_energies(positions)
            
            # Apply switching function
            switched_energy = apply_switching_function(positions, pair_energies)
            
            # Calculate forces with switching
            forces = -(mm_energy_grad(positions) + 
                    switching_grad(positions, pair_energies))

            return switched_energy, forces

        return calculate_mm_energy_and_forces


    from functools import partial
    @partial(jax.jit, static_argnames=['n_monomers', 'cutoff_params', 'doML', 'doMM', 'doML_dimer', 'debug', 'do_pbc_map',])
    def spherical_cutoff_calculator(
        positions: Array,  # Shape: (n_atoms, 3)
        atomic_numbers: Array,  # Shape: (n_atoms,)
        n_monomers: int,
        cutoff_params: CutoffParameters,
        doML: bool = True,
        doMM: bool = True,
        doML_dimer: bool = True,
        debug: bool = False,
        do_pbc_map: bool = False,
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

        if do_pbc_map:
            positions = pbc_map(positions)
        
        outputs = {
            "out_E": 0,
            "out_F": 0,
            "dH": 0,
            "internal_E": 0, 
            "internal_F": 0,
            "ml_2b_E": 0,
            "ml_2b_F": 0
        }

        if doML:
            outputs.update(calculate_ml_contributions(
                positions, atomic_numbers, n_dimers, n_monomers,
                cutoff_params=cutoff_params,
                debug=debug,
                ml_energy_conversion_factor=ml_energy_conversion_factor,
                ml_force_conversion_factor=ml_force_conversion_factor
            ))

        if doMM:
            outputs.update(calculate_mm_contributions(
                positions,
                cutoff_params=cutoff_params,
                debug=debug
            ))

        return ModelOutput(
            energy=(outputs["out_E"].sum() + outputs["internal_E"] - outputs.get("mm_E", 0)),
            forces=outputs["out_F"],
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
        
        # Convert units
        f = output["forces"] * ml_force_conversion_factor 
        e = output["energy"] * ml_energy_conversion_factor
        
        # Calculate monomer contributions
        monomer_contribs = calculate_monomer_contributions(e, f, n_monomers, max_atoms, debug)
        
        if not doML_dimer:
            return monomer_contribs
            
        # Calculate dimer contributions
        dimer_contribs = calculate_dimer_contributions(
            positions, e, f, n_dimers, 
            monomer_contribs["monomer_energy"],
            debug
        )
        
        # Combine contributions
        return {
            "out_E": monomer_contribs["out_E"] + dimer_contribs["out_E"],
            "out_F": monomer_contribs["out_F"] + dimer_contribs["out_F"],
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
        
        monomer_idx_max = max_atoms * n_monomers
        ml_monomer_forces = f[:monomer_idx_max]
        
        # Calculate segment indices for force summation
        monomer_segment_idxs = jnp.concatenate([
            jnp.arange(ATOMS_PER_MONOMER) + i * ATOMS_PER_MONOMER 
            for i in range(n_monomers)
        ])
        
        # Process forces
        monomer_forces = process_monomer_forces(
            ml_monomer_forces, monomer_segment_idxs, max_atoms, debug
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
        """Calculate MM energy and force contributions"""

        MM_energy_and_gradient = get_MM_energy_forces_fns(
            positions, 
            ATOMS_PER_MONOMER , 
            N_MONOMERS=n_monomers, 
            ml_cutoff_distance=cutoff_params.ml_cutoff, 
            mm_switch_on=cutoff_params.mm_switch_on, 
            mm_cutoff=cutoff_params.mm_cutoff
        )
        
        
        mm_E, mm_grad = MM_energy_and_gradient(positions)
        
        debug_print(debug, "MM Contributions:", 
            mm_E=mm_E,
            mm_grad=mm_grad
        )
        kcal2ev = 1/23.0605
        return {
            "out_E": mm_E * kcal2ev,
            "out_F": mm_grad * kcal2ev,
            "dH": mm_E * kcal2ev,
            "mm_E": mm_E * kcal2ev,
            "mm_F": mm_grad * kcal2ev
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
            ):
                """Initialize calculator with configuration parameters"""

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

                out = {}
                if not self.backprop:
                    out = spherical_cutoff_calculator(
                        positions=R,
                        atomic_numbers=Z,
                        n_monomers=self.n_monomers,
                        cutoff_params=self.cutoff_params,
                        doML=self.doML,
                        doMM=self.doMM,
                        doML_dimer=self.doML_dimer,
                        debug=self.debug,
                    )

                    E = out.energy
                    F = out.forces

                if self.backprop:

                    def Efn(R):
                        return -spherical_cutoff_calculator(
                            positions=R,
                            atomic_numbers=Z,
                            n_monomers=self.n_monomers,
                            cutoff_params=self.cutoff_params,
                            doML=self.doML,
                            doMM=self.doMM,
                            doML_dimer=self.doML_dimer,
                            debug=self.debug,
                        ).energy

                    E, F = jax.value_and_grad(Efn)(R)
                    F = F

                self.results["out"] = out
                self.results["energy"] = -E * self.energy_conversion_factor
                self.results["forces"] = F * self.force_conversion_factor

        def get_spherical_cutoff_calculator(
            atomic_numbers: Array,
            atomic_positions: Array,
            n_monomers: int,
            cutoff_params: CutoffParameters = None,
            doML: bool = True,
            doMM: bool = True,
            doML_dimer: bool = True,
            backprop: bool = False,
            debug: bool = False,
            energy_conversion_factor: float = 1.0,
            force_conversion_factor: float = 1.0,
        ) -> Tuple[AseDimerCalculator, Callable]:
            """Factory function to create calculator instances"""

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
        max_atoms: int,
        debug: bool = False,
    ) -> Array:
        """Process and reshape monomer forces with proper masking.
        
        Args:
            ml_monomer_forces: Raw forces from ML model
            monomer_segment_idxs: Indices for force segmentation
            debug: Enable debug printing
            
        Returns:
            Array: Processed monomer forces
        """
        # Reshape forces to (n_monomers, atoms_per_system, 3)
        monomer_forces = ml_monomer_forces.reshape(-1, max_atoms, 3)
        
        # Create mask for valid atoms
        atom_mask = jnp.arange(max_atoms)[None, :] < ATOMS_PER_MONOMER
        
        # Apply mask
        monomer_forces = jnp.where(
            atom_mask[..., None],
            monomer_forces,
            0.0
        )
        
        # Sum forces for valid atoms
        processed_forces = jax.ops.segment_sum(
            monomer_forces[:, :ATOMS_PER_MONOMER].reshape(-1, 3),
            monomer_segment_idxs,
            num_segments=n_monomers * ATOMS_PER_MONOMER
        )
        
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
        if debug:
            print(f"Debug dimer forces: f.shape={f.shape}, monomer_atoms={monomer_atoms}, max_atoms={max_atoms}")
            print(f"ml_dimer_forces shape: {f[monomer_atoms:].shape}")
        ml_dimer_forces = f[monomer_atoms:]
        
        # Calculate force segments for dimers
        force_segments = calculate_dimer_force_segments(n_dimers)
        
        # Calculate interaction energies
        monomer_contrib = calculate_monomer_contribution_to_dimers(
            monomer_energies, jnp.array(dimer_perms)
        )
        dimer_int_energies = ml_dimer_energy - monomer_contrib
        
        # Process dimer forces
        dimer_forces = process_dimer_forces(
            ml_dimer_forces, force_segments, n_dimers, debug
        )
        
        # Apply switching functions
        switched_results = apply_dimer_switching(
            positions, dimer_int_energies, dimer_forces, cutoffparameters, max_atoms, debug
        )
        
        debug_print(debug, "Dimer Contributions:",
            dimer_energies=switched_results["energies"],
            dimer_forces=switched_results["forces"]
        )
        
        return {
            "out_E": switched_results["energies"].sum(),
            "out_F": switched_results["forces"],
            "dH": switched_results["energies"].sum(),
            "ml_2b_E": switched_results["energies"].sum(),
            "ml_2b_F": switched_results["forces"]
        }

    def calculate_dimer_force_segments(n_dimers: int) -> Array:
        """Calculate force segments for dimer force summation."""
        dimer_pairs = jnp.array(dimer_perms)
        
        # Calculate base indices for each monomer
        first_indices = ATOMS_PER_MONOMER * dimer_pairs[:, 0:1]
        second_indices = ATOMS_PER_MONOMER * dimer_pairs[:, 1:2]
        
        # Create atom offsets
        atom_offsets = jnp.arange(ATOMS_PER_MONOMER)
        
        # Combine indices for both monomers
        force_segments = jnp.concatenate([
            first_indices + atom_offsets[None, :],
            second_indices + atom_offsets[None, :]
        ], axis=1)
        
        return force_segments.reshape(-1)

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
        
        return jax.ops.segment_sum(
            forces.reshape(-1, 3),
            force_segments,
            num_segments=n_monomers * ATOMS_PER_MONOMER
        )

    def apply_dimer_switching(
        positions: Array,
        dimer_energies: Array,
        dimer_forces: Array,
        cutoff_params: CutoffParameters,
        max_atoms: int,
        debug: bool
    ) -> Dict[str, Array]:
        """Apply switching functions to dimer energies and forces."""
        # Calculate switched energies using cutoff parameters
        switched_energy = jax.vmap(lambda x, f: switch_ML(
            x.reshape(max_atoms, 3), 
            f,
            ml_cutoff=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        ))(positions[jnp.array(all_dimer_idxs)], dimer_energies)
        
        # Calculate switched energy gradients
        switched_grad = jax.vmap(lambda x, f: switch_ML_grad(
            x.reshape(max_atoms, 3),
            f,
            ml_cutoff=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        ))(positions[jnp.array(all_dimer_idxs)], dimer_energies)
        
        # Combine forces using product rule
        dudx_v = dimer_energies.sum() * switched_grad
        
        # Reshape dimer_forces to match switched_grad shape
        n_dimers = len(all_dimer_idxs)
        # dimer_forces_reshaped = dimer_forces.reshape(n_dimers, 2 * ATOMS_PER_MONOMER, 3)
        # dvdx_u = dimer_forces_reshaped / switched_energy.sum()
        
        # combined_forces =  -1 * (dudx_v + dvdx_u)
        
        # Convert forces back to flat format
        # forces_flat = combined_forces.reshape(-1, 3)
        
        return {
            "energies": -switched_energy,
            "forces": 0
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