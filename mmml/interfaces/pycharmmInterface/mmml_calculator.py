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

import warnings
from functools import partial
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

# In your module that defines spherical_cutoff_calculator
import jax.numpy as jnp
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
    mic_displacement,
    mic_displacement_smooth,
    mic_displacements_batched,
    mic_displacements_batched_smooth,
)
from mmml.interfaces.pycharmmInterface.calculator_utils import (
    ModelOutput,
    debug_print,
    dimer_permutations,
    epsilon,
    indices_of_monomer,
    indices_of_pairs,
    ml_switch_simple,
    mm_switch_simple,
    parse_non_int,
    _sharpstep,
)
from mmml.interfaces.pycharmmInterface.ml_batching import prepare_batches_md
from mmml.interfaces.pycharmmInterface.mm_energy_forces import build_mm_energy_forces_fn


# CHARMM force-field definitions are optional.  During documentation builds we
# often do not have a functional PyCHARMM installation, so fall back to ``None``
# when the import fails for any reason (missing module or missing shared libs).
try:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF
except Exception:  # pragma: no cover - exercised in lightweight envs
    CGENFF_PRM = CGENFF_RTF = None
try:
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
except ModuleNotFoundError:  # pragma: no cover - helper requires ASE

    def get_ase_calc(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise ModuleNotFoundError("ase is required for get_ase_calc")
try:
    from mmml.models.physnetjax.physnetjax.data.batches import (
        _prepare_batches as prepare_batches,
    )
    from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets
    from mmml.models.physnetjax.physnetjax.models.model import EF
    from mmml.models.physnetjax.physnetjax.restart.restart import get_files, get_last, get_params_model
    from mmml.models.physnetjax.physnetjax.training.loss import dipole_calc
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

    # Keep restart helpers available whenever possible, even when model imports
    # fail due to optional extras such as e3x.
    try:
        from mmml.models.physnetjax.physnetjax.restart.restart import get_files, get_last, get_params_model
    except ModuleNotFoundError as exc_restart:
        _restart_import_error = exc_restart

        def _raise_restart_import_error() -> None:
            missing = getattr(_restart_import_error, "name", None)
            if missing:
                raise ModuleNotFoundError(
                    f"restart helpers unavailable; missing dependency '{missing}'"
                ) from _restart_import_error
            raise ModuleNotFoundError(
                f"restart helpers unavailable: {_restart_import_error}"
            ) from _restart_import_error

        def get_files(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
            _raise_restart_import_error()

        def get_last(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
            _raise_restart_import_error()

        def get_params_model(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
            _raise_restart_import_error()

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
from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters, GAMMA_OFF, GAMMA_ON

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



def setup_calculator(
    ATOMS_PER_MONOMER: Union[int, List[int], Sequence[int]],
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
    lambda_monomer: Optional[np.ndarray] = None,
    cell_list_safety_factor: float = 2.5,
    max_pairs: Optional[int] = None,
    flat_bottom_radius: float | None = None,
    flat_bottom_force_const: float = 1.0,
    use_smooth_mic: Optional[bool] = None,
    ensemble: str = "nve",
):
    """Create hybrid ML/MM calculator with outputs in eV/eV-A.

    ML energies/forces are assumed to be in eV already. MM energies/forces
    (kcal/mol, kcal/mol/Å) are converted to eV/eV-Å internally before summing.

    Args:
        ATOMS_PER_MONOMER: Either a single int (all monomers same size) or a
            list/sequence of ints giving the atom count for each monomer.  When
            a list is provided its length is used as *N_MONOMERS*.
        lambda_monomer: Optional array of shape ``(N_MONOMERS,)`` with values
            in [0, 1].  Scales inter-monomer contributions (ML dimer, MM
            non-bonded) for each monomer.  Internal monomer energy is NOT
            scaled (decouple only inter-monomer interactions in FEP/TI).
            ``None`` (default) is equivalent to all ones.
        cell_list_safety_factor: Multiplicative safety margin for cell-list
            pair count estimation (PBC).  Increase if you see
            "Truncating. Increase max_pairs" warnings.  Default 2.5.
        max_pairs: If set, use this value directly for cell-list max_pairs
            instead of estimating.  Use when safety_factor is insufficient.
    """
    if model_restart_path is None:
        raise ValueError("model_restart_path must be provided")

    # --- Normalise ATOMS_PER_MONOMER into a per-monomer list ----------------
    if isinstance(ATOMS_PER_MONOMER, (list, tuple, np.ndarray)):
        atoms_per_monomer_list: List[int] = [int(x) for x in ATOMS_PER_MONOMER]
        N_MONOMERS = len(atoms_per_monomer_list)
    else:
        atoms_per_monomer_list = [int(ATOMS_PER_MONOMER)] * N_MONOMERS

    n_monomers = N_MONOMERS

    # Cumulative atom offsets: monomer_offsets[i] is the global index of the
    # first atom of monomer i.  E.g. [0, 3, 6, 12] for [3, 3, 6, ...].
    monomer_offsets = np.zeros(n_monomers + 1, dtype=int)
    for i, n in enumerate(atoms_per_monomer_list):
        monomer_offsets[i + 1] = monomer_offsets[i] + n
    total_atoms = int(monomer_offsets[-1])

    # Convenience: keep a uniform value when all monomers are the same size.
    _all_same_size = len(set(atoms_per_monomer_list)) == 1
    ATOMS_PER_MONOMER_UNIFORM: Optional[int] = atoms_per_monomer_list[0] if _all_same_size else None

    # --- Lambda array -------------------------------------------------------
    if lambda_monomer is None:
        lambda_monomer = jnp.ones(n_monomers)
    else:
        lambda_monomer = jnp.asarray(lambda_monomer, dtype=jnp.float32)
        if lambda_monomer.shape != (n_monomers,):
            raise ValueError(
                f"lambda_monomer must have shape ({n_monomers},), "
                f"got {lambda_monomer.shape}"
            )

    cutoffparameters = CutoffParameters(
        ml_cutoff_distance, mm_switch_on, mm_cutoff, complementary_handoff=complementary_handoff
    )
    print(
        "[setup_calculator] Cutoff inputs -> ml_cutoff=%.4f, mm_switch_on=%.4f, mm_cutoff=%.4f, complementary_handoff=%s"
        % (ml_cutoff_distance, mm_switch_on, mm_cutoff, complementary_handoff)
    )
    print(f"[setup_calculator] atoms_per_monomer={atoms_per_monomer_list}, total_atoms={total_atoms}")
    print(f"[setup_calculator] lambda_monomer={lambda_monomer}")

    # --- Build monomer / dimer index arrays ---------------------------------
    all_dimer_idxs = []
    for a, b in dimer_permutations(n_monomers):
        all_dimer_idxs.append(indices_of_pairs(
            a + 1, b + 1,
            monomer_offsets=monomer_offsets,
            atoms_per_monomer_list=atoms_per_monomer_list,
        ))

    all_monomer_idxs = []
    for a in range(1, n_monomers + 1):
        all_monomer_idxs.append(indices_of_monomer(
            a, n_mol=n_monomers,
            monomer_offsets=monomer_offsets,
            atoms_per_monomer_list=atoms_per_monomer_list,
        ))

    dimer_perms = dimer_permutations(n_monomers)

    print("len(dimer_perms)", len(dimer_perms))

    N_MONOMERS = n_monomers
    # Batch processing constants
    BATCH_SIZE: int = N_MONOMERS + len(dimer_perms)  # Number of systems per batch
    # print(BATCH_SIZE)
    restart_path = Path(model_restart_path) if type(model_restart_path) == str else model_restart_path

    # Check if this is a JSON checkpoint (params.json in dir, or path to .json file)
    is_json_checkpoint = (
        (restart_path.is_file() and restart_path.suffix == ".json")
        or ((restart_path / "params.json").exists())
    )
    
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
            # Use default PhysNet EF config when config is missing (params-only checkpoint)
            if not config:
                config = {
                    "features": 32,
                    "max_degree": 3,
                    "num_iterations": 2,
                    "num_basis_functions": 16,
                    "cutoff": 6.0,
                    "max_atomic_number": 118,
                    "charges": False,
                    "natoms": MAX_ATOMS_PER_SYSTEM,
                    "total_charge": 0,
                    "n_res": 3,
                    "zbl": True,
                    "debug": False,
                    "efa": False,
                    "use_energy_bias": False,
                    "use_pbc": bool(cell),
                }

            # Reconstruct model from config
            from mmml.physnetjax.physnetjax.models.model import EF as StandardEF
            from mmml.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF

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
            if cell:
                model_config['use_pbc'] = True
            is_spooky_model = (
                str(config.get("model_type", "")).lower() == "spooky"
                or "spooky" in str(restart_path).lower()
            )
            model_cls = SpookyEF if is_spooky_model else StandardEF
            MODEL = model_cls(**model_config)
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
    is_spooky_model = "spooky_model" in type(MODEL).__module__

    if use_smooth_mic is None:
        use_smooth_mic = bool(cell)

    if cell:
        MODEL.use_pbc = True
        cell_arr = jnp.asarray(cell)
        if cell_arr.ndim == 0:
            cell = jnp.asarray([[float(cell), 0, 0], [0, float(cell), 0], [0, 0, float(cell)]])
        elif cell_arr.shape == (3,):
            a, b, c = float(cell_arr[0]), float(cell_arr[1]), float(cell_arr[2])
            cell = jnp.asarray([[a, 0, 0], [0, b, 0], [0, 0, c]])
        elif cell_arr.shape == (3, 3):
            cell = jnp.asarray(cell_arr, dtype=jnp.float64)
        else:
            raise ValueError(f"cell must be scalar, (3,), or (3,3); got {cell_arr.shape}")
        # MIC-only PBC: cell list uses wrap-by-molecule for binning (in cell_list.py).
        pbc_cell = cell
        do_pbc_map = False
        pbc_map = None
    else:
        pbc_cell = None
        pbc_map = do_pbc_map = False



    @partial(jax.jit, static_argnames=['ml_cutoff', 'mm_switch_on', 'n_atoms_a', 'n_atoms_b'])
    def switch_ML(X,
        ml_energy,
        ml_cutoff=ml_cutoff_distance,
        mm_switch_on=mm_switch_on,
        n_atoms_a: int = atoms_per_monomer_list[0],
        n_atoms_b: int = atoms_per_monomer_list[-1],
        pbc_cell=None,
    ):
        """Apply ML switching based on COM-COM distance of the two monomers in a dimer.

        Args:
            X: Dimer positions, shape ``(n_atoms_a + n_atoms_b, 3)`` (possibly padded).
            n_atoms_a: Number of atoms in the first monomer.
            n_atoms_b: Number of atoms in the second monomer.
        """
        # COM–COM distance (used for ML taper; must match debug "dimer COM distance")
        com1 = jnp.mean(X[:n_atoms_a], axis=0)
        com2 = jnp.mean(X[n_atoms_a:n_atoms_a + n_atoms_b], axis=0)
        if pbc_cell is not None:
            mic_fn = mic_displacement_smooth if use_smooth_mic else mic_displacement
            d = mic_fn(com1, com2, pbc_cell)
            r = jnp.linalg.norm(d)
        else:
            r = jnp.linalg.norm(com2 - com1)
    
        # ML: 1 -> 0 over [mm_switch_on - ml_cutoff, mm_switch_on]
        ml_scale = 1.0 - _sharpstep(r, mm_switch_on - ml_cutoff, mm_switch_on, gamma=GAMMA_ON)
      
        return ml_scale * ml_energy

    switch_ML_grad = jax.grad(switch_ML)

    
    _fractional_coordinates = ensemble == "npt"

    def get_MM_energy_forces_fns(
        R,
        ATOMS_PER_MONOMER_ARG=None,
        N_MONOMERS=N_MONOMERS,
        ml_cutoff_distance=ml_cutoff_distance,
        mm_switch_on=mm_switch_on,
        mm_cutoff=mm_cutoff,
        complementary_handoff=True,
        sig_scale=sig_scale,
        ep_scale=ep_scale,
        pbc_cell_override=None,
    ):
        """Creates functions for calculating MM energies and forces with switching.

        Supports heterogeneous monomer sizes via the outer ``atoms_per_monomer_list``
        and ``monomer_offsets``. When pbc_cell_override is provided (e.g. for NPT),
        use it instead of the closure's pbc_cell.
        """
        cell_for_build = pbc_cell_override if pbc_cell_override is not None else pbc_cell
        result_jaxmd = build_mm_energy_forces_fn(
            R,
            total_atoms=total_atoms,
            n_monomers=N_MONOMERS,
            monomer_offsets=monomer_offsets,
            atoms_per_monomer_list=atoms_per_monomer_list,
            lambda_monomer=lambda_monomer,
            ml_cutoff_distance=ml_cutoff_distance,
            mm_switch_on=mm_switch_on,
            mm_cutoff=mm_cutoff,
            complementary_handoff=complementary_handoff,
            ep_scale=ep_scale,
            sig_scale=sig_scale,
            at_codes_override=at_codes_override,
            pbc_cell=cell_for_build,
            max_pairs=max_pairs,
            cell_list_safety_factor=cell_list_safety_factor,
            use_smooth_mic=use_smooth_mic,
            use_jax_md_neighbor_list=True,
            fractional_coordinates=_fractional_coordinates,
            debug=debug,
        )
        if isinstance(result_jaxmd, tuple):
            mm_fn_jaxmd, update_fn = result_jaxmd
            mm_fn_cell = build_mm_energy_forces_fn(
                R,
                total_atoms=total_atoms,
                n_monomers=N_MONOMERS,
                monomer_offsets=monomer_offsets,
                atoms_per_monomer_list=atoms_per_monomer_list,
                lambda_monomer=lambda_monomer,
                ml_cutoff_distance=ml_cutoff_distance,
                mm_switch_on=mm_switch_on,
                mm_cutoff=mm_cutoff,
                complementary_handoff=complementary_handoff,
                ep_scale=ep_scale,
                sig_scale=sig_scale,
            at_codes_override=at_codes_override,
            pbc_cell=cell_for_build,
            max_pairs=max_pairs,
            cell_list_safety_factor=cell_list_safety_factor,
            use_smooth_mic=use_smooth_mic,
            use_jax_md_neighbor_list=False,
            debug=debug,
            )
            return (mm_fn_jaxmd, mm_fn_cell), update_fn
        return result_jaxmd, None

    # Lazy cache for the pre-computed MM energy/force function.
    # Must be built once with *concrete* positions (outside JIT) so that
    # the cell-list path can call NumPy without hitting TracerArrayConversion.
    _cached_mm_fn = [None]          # [0] = calculate_mm_energy_and_forces | None
    _cached_update_mm_pairs = [None]  # [0] = update_mm_pairs | None (jax_md path)
    _cached_mm_cutoff_key = [None]  # hashable key to detect cutoff-param changes

    def _ensure_mm_fn(positions_concrete, cutoff_params):
        """Build the MM energy/force function if not yet cached (or if cutoffs changed)."""
        key = (cutoff_params.ml_cutoff, cutoff_params.mm_switch_on, cutoff_params.mm_cutoff,
               getattr(cutoff_params, "complementary_handoff", True))
        if _cached_mm_fn[0] is None or _cached_mm_cutoff_key[0] != key:
            mm_result, update_fn = get_MM_energy_forces_fns(
                positions_concrete,
                N_MONOMERS=n_monomers,
                ml_cutoff_distance=cutoff_params.ml_cutoff,
                mm_switch_on=cutoff_params.mm_switch_on,
                mm_cutoff=cutoff_params.mm_cutoff,
                complementary_handoff=getattr(cutoff_params, "complementary_handoff", True),
            )
            _cached_mm_fn[0] = mm_result
            _cached_update_mm_pairs[0] = update_fn
            _cached_mm_cutoff_key[0] = key

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
        mm_pair_idx: Optional[Array] = None,
        mm_pair_mask: Optional[Array] = None,
        box: Optional[Array] = None,
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
            # CRITICAL: These forces are ALREADY correctly mapped to atoms 0 to (total_atoms - 1)
            # via segment_sum in process_monomer_forces and process_dimer_forces
            # They should be in the same order as the atom positions (atoms 0, 1, 2, ..., n_atoms-1)
            ml_forces_raw = ml_out.get("out_F", jnp.zeros((total_atoms, 3)))
            ml_internal_F_raw = ml_out.get("internal_F", jnp.zeros((total_atoms, 3)))
            ml_2b_F_raw = ml_out.get("ml_2b_F", jnp.zeros((total_atoms, 3)))

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
            expected_n_ml_atoms = total_atoms
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
                debug=debug,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box,
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
        
        final_forces = jnp.where(jnp.isfinite(final_forces), final_forces, 0.0)

        # Total energy: combined ML (monomer+dimer switched) + MM
        final_energy = outputs["out_E"]
        if isinstance(final_energy, (int, float)):
            final_energy = jnp.array(final_energy)
        final_energy = jnp.where(jnp.isfinite(final_energy), final_energy, 0.0)

        # Flat bottom potential: constrain COM to center (e.g. box center for PBC)
        # V = 0 when |d| <= R, else V = k * (|d| - R)^2
        _pbc_cell = pbc_cell
        if flat_bottom_radius is not None and flat_bottom_radius > 0:
            from ase.data import atomic_masses as ase_atomic_masses
            masses = jnp.take(jnp.array(ase_atomic_masses), atomic_numbers)
            M = jnp.sum(masses)
            com = jnp.sum(positions * masses[:, None], axis=0) / M
            if _pbc_cell is not None:
                center = (_pbc_cell[0] + _pbc_cell[1] + _pbc_cell[2]) / 2.0
                mic_fn = mic_displacement_smooth if use_smooth_mic else mic_displacement
                d = mic_fn(center, com, _pbc_cell)
            else:
                center = jnp.zeros(3)
                d = com - center
            dist = jnp.linalg.norm(d) + 1e-12
            excess = jnp.maximum(0.0, dist - flat_bottom_radius)
            flat_E = flat_bottom_force_const * excess ** 2
            unit_d = d / dist
            F_com = -flat_bottom_force_const * 2.0 * excess * unit_d
            flat_F = (masses[:, None] / M) * F_com[None, :]
            final_energy = final_energy + flat_E
            final_forces = final_forces + flat_F
        
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

        Supports heterogeneous monomer sizes: each monomer/dimer system is
        padded individually to ``max_atoms`` (the largest dimer atom count).

        Args:
            atomic_numbers: Array of atomic numbers
            positions: Atomic positions in Angstroms

        Returns:
            Tuple of (model_apply_fn, batched_inputs)
        """
        batch_data: Dict[str, Array] = {}

        # max_atoms must accommodate the largest single system (dimer).
        _dimer_atom_counts = [
            atoms_per_monomer_list[a] + atoms_per_monomer_list[b]
            for a, b in dimer_perms
        ]
        max_monomer_atoms = max(atoms_per_monomer_list)
        max_dimer_atoms = max(_dimer_atom_counts) if _dimer_atom_counts else 2 * max_monomer_atoms
        max_atoms = max(max_monomer_atoms, max_dimer_atoms)

        # --- Monomer data (variable sizes, padded to max_atoms) ---
        monomer_positions = jnp.zeros((n_monomers, max_atoms, SPATIAL_DIMS))
        monomer_atomic = jnp.zeros((n_monomers, max_atoms), dtype=jnp.int32)
        for mi in range(n_monomers):
            idxs = all_monomer_idxs[mi]
            n_i = len(idxs)
            monomer_positions = monomer_positions.at[mi, :n_i].set(positions[jnp.array(idxs)])
            monomer_atomic = monomer_atomic.at[mi, :n_i].set(atomic_numbers[jnp.array(idxs)])

        # --- Dimer data (variable sizes, padded to max_atoms) ---
        n_dimers = len(all_dimer_idxs)
        dimer_positions = jnp.zeros((n_dimers, max_atoms, SPATIAL_DIMS))
        dimer_atomic = jnp.zeros((n_dimers, max_atoms), dtype=jnp.int32)
        for di in range(n_dimers):
            idxs = all_dimer_idxs[di]
            n_d = len(idxs)
            dimer_positions = dimer_positions.at[di, :n_d].set(positions[jnp.array(idxs)])
            dimer_atomic = dimer_atomic.at[di, :n_d].set(atomic_numbers[jnp.array(idxs)])

        # Combine monomer and dimer data
        batch_data["R"] = jnp.concatenate([monomer_positions, dimer_positions])
        batch_data["Z"] = jnp.concatenate([monomer_atomic, dimer_atomic])
        # N: actual atom count per system (variable)
        monomer_N = jnp.array([atoms_per_monomer_list[i] for i in range(n_monomers)])
        dimer_N = jnp.array(_dimer_atom_counts) if _dimer_atom_counts else jnp.zeros((0,), dtype=jnp.int32)
        batch_data["N"] = jnp.concatenate([monomer_N, dimer_N])
        BATCH_SIZE = n_monomers + n_dimers
        batches = prepare_batches_md(batch_data, batch_size=BATCH_SIZE, num_atoms=max_atoms)[0]

        @jax.jit
        def apply_model(
            atomic_numbers: Array,  # Shape: (batch_size * num_atoms,)
            positions: Array,  # Shape: (batch_size * num_atoms, 3)
        ) -> Dict[str, Array]:
            """Applies the ML model to batched inputs."""
            if is_spooky_model:
                atom_mask = batches["atom_mask"].astype(jnp.float32)
                q_atoms = jnp.zeros((atom_mask.shape[0], 1), dtype=jnp.float32)
                s_atoms = atom_mask.reshape(-1, 1)  # neutral singlet: multiplicity 1 on real atoms
                return MODEL.apply(
                    params,
                    atomic_numbers=atomic_numbers,
                    charges=q_atoms,
                    spins=s_atoms,
                    positions=positions,
                    dst_idx=batches["dst_idx"],
                    src_idx=batches["src_idx"],
                    batch_segments=batches["batch_segments"],
                    batch_size=BATCH_SIZE,
                    batch_mask=batches["batch_mask"],
                    atom_mask=atom_mask,
                    cell=pbc_cell,
                )
            return MODEL.apply(
                params,
                atomic_numbers=atomic_numbers,
                positions=positions,
                dst_idx=batches["dst_idx"],
                src_idx=batches["src_idx"],
                batch_segments=batches["batch_segments"],
                batch_size=BATCH_SIZE,
                batch_mask=batches["batch_mask"],
                atom_mask=batches["atom_mask"],
                cell=pbc_cell,
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
        """Calculate ML energy and force contributions (heterogeneous-safe)."""
        # Calculate max atoms for consistent array shapes (heterogeneous)
        _dimer_atom_counts = [
            atoms_per_monomer_list[a] + atoms_per_monomer_list[b]
            for a, b in dimer_perms
        ]
        max_monomer_atoms = max(atoms_per_monomer_list)
        max_dimer_atoms = max(_dimer_atom_counts) if _dimer_atom_counts else 2 * max_monomer_atoms
        max_atoms = max(max_monomer_atoms, max_dimer_atoms)

        # Get model predictions
        apply_model, batches = get_ML_energy_fn(atomic_numbers, positions, n_dimers + n_monomers)
        output = apply_model(batches["Z"], batches["R"])

        f = output["forces"] * ml_force_conversion_factor
        e = output["energy"] * ml_energy_conversion_factor

        # Calculate monomer contributions (flatten variable-size monomer indices)
        monomer_atomic_numbers_flat = jnp.concatenate([
            atomic_numbers[jnp.array(all_monomer_idxs[i])] for i in range(n_monomers)
        ])
        monomer_positions_flat = jnp.concatenate([
            positions[jnp.array(all_monomer_idxs[i])] for i in range(n_monomers)
        ])
        # Build atom mask for valid monomer atoms (variable sizes)
        monomer_atom_mask_parts = []
        for mi in range(n_monomers):
            n_i = atoms_per_monomer_list[mi]
            monomer_atom_mask_parts.append(jnp.ones(n_i, dtype=jnp.int32))
        monomer_atom_mask_flat = jnp.concatenate(monomer_atom_mask_parts)

        monomer_contribs = calculate_monomer_contributions(
            e,
            f,
            n_monomers,
            max_atoms,
            debug,
            monomer_atomic_numbers_flat=monomer_atomic_numbers_flat,
            monomer_positions_flat=monomer_positions_flat,
            monomer_atom_mask_flat=monomer_atom_mask_flat,
        )
        
        if not doML_dimer:
            return {
                **monomer_contribs,
                "ml_2b_E": 0,
                "ml_2b_F": jnp.zeros((total_atoms, 3)),
            }

        # Calculate dimer contributions
        dimer_contribs = calculate_dimer_contributions(
            positions, e, f, n_dimers, 
            monomer_contribs["monomer_energy"],
            cutoff_params,
            debug
        )

        debug_print(debug, f"DEBUG dimer_contribs: {dimer_contribs}")
        
        # Combine contributions
        monomer_forces_safe = jnp.where(jnp.isfinite(monomer_contribs["out_F"]), monomer_contribs["out_F"], 0.0)
        dimer_forces_safe = jnp.where(jnp.isfinite(dimer_contribs["out_F"]), dimer_contribs["out_F"], 0.0)
        
        # Ensure shapes match -- both should be (total_atoms, 3)
        expected_force_size = total_atoms
        for name, arr in [("monomer", monomer_forces_safe), ("dimer", dimer_forces_safe)]:
            if arr.shape[0] < expected_force_size:
                pad = jnp.zeros((expected_force_size - arr.shape[0], 3))
                arr = jnp.concatenate([arr, pad], axis=0)
            elif arr.shape[0] > expected_force_size:
                arr = arr[:expected_force_size]
            if name == "monomer":
                monomer_forces_safe = arr
            else:
                dimer_forces_safe = arr
        
        combined_forces = monomer_forces_safe + dimer_forces_safe
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
        debug: bool,
        monomer_atomic_numbers_flat: Array | None = None,
        monomer_positions_flat: Array | None = None,
        monomer_atom_mask_flat: Array | None = None,
    ) -> Dict[str, Array]:
        """Calculate energy and force contributions from monomers.

        Supports heterogeneous monomer sizes via ``atoms_per_monomer_list``
        and ``monomer_offsets``.  Internal energy is NOT scaled by lambda
        (only inter-monomer interactions are decoupled in FEP/TI).
        """
        ml_monomer_energy = jnp.array(e[:n_monomers]).flatten()
        # Internal monomer energy is NOT scaled by lambda (decouple only inter-monomer)

        ml_monomer_forces = f[:max_atoms * n_monomers]
        
        # Segment indices mapping each force entry back to the global atom index.
        # Heterogeneous: monomer 0 has atoms [0..n0-1], monomer 1 has [n0..n0+n1-1], ...
        monomer_segment_idxs = jnp.concatenate([
            jnp.arange(atoms_per_monomer_list[i]) + int(monomer_offsets[i])
            for i in range(n_monomers)
        ])

        # Process forces
        monomer_forces = process_monomer_forces(
            ml_monomer_forces,
            monomer_segment_idxs,
            max_atoms,  # atoms_per_system in the batch (padded)
            monomer_atomic_numbers_flat=monomer_atomic_numbers_flat,
            monomer_positions_flat=monomer_positions_flat,
            monomer_atom_mask_flat=monomer_atom_mask_flat,
            debug=debug,
        )

        # Internal monomer forces are NOT scaled by lambda (decouple only inter-monomer)

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
        debug: bool,
        mm_pair_idx: Optional[Array] = None,
        mm_pair_mask: Optional[Array] = None,
        box: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Calculate MM energy and force contributions (converted to eV).

        Uses the pre-computed MM function from ``_cached_mm_fn`` (built
        outside JIT by ``_ensure_mm_fn``) so that cell-list pair generation
        never runs inside a JAX trace. When box is provided (NPT), rebuild
        with that box (no cache).
        """
        
        # Ensure positions are finite
        positions = jnp.where(jnp.isfinite(positions), positions, 0.0)

        mm_fn_val = _cached_mm_fn[0]
        if isinstance(mm_fn_val, tuple):
            mm_fn_jaxmd, mm_fn_cell = mm_fn_val
            if mm_pair_idx is not None and mm_pair_mask is not None:
                mm_E, mm_grad = mm_fn_jaxmd(positions, mm_pair_idx, mm_pair_mask, box_override=box)
            else:
                mm_E, mm_grad = mm_fn_cell(positions)
        else:
            mm_E, mm_grad = mm_fn_val(positions)
        
        # Check for NaN/Inf in MM energy and forces
        mm_E = jnp.where(jnp.isfinite(mm_E), mm_E, 0.0)
        mm_grad = jnp.where(jnp.isfinite(mm_grad), mm_grad, 0.0)

        # # MM outputs are in kcal/mol and kcal/mol/Å. Convert to eV and eV/Å.
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
                pbc_cell = None,
                verbose: bool = True,
            ):
                """Initialize calculator with configuration parameters

                Args:
                    pbc_cell: Setup-time cell (3x3) for PBC; used for consistency check
                              and fallback when atoms.cell is not available.
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
                self.pbc_cell = np.asarray(pbc_cell) if pbc_cell is not None else None
                self._pbc_map_cache = None
                self._pbc_map_cache_cell = None
                self.verbose = verbose
                # Expose heterogeneous info
                self.atoms_per_monomer_list = atoms_per_monomer_list
                self.atoms_per_monomer = ATOMS_PER_MONOMER_UNIFORM  # None if heterogeneous
                self.total_atoms = total_atoms
                if self.do_pbc_map and self.pbc_map is None:
                    self.do_pbc_map = False

            # --- Lambda property for runtime adjustment (FEP / TI) ---
            @property
            def lambda_monomer_values(self):
                """Current lambda_monomer array (read-only view)."""
                return lambda_monomer

            def set_lambda_monomer(self, new_lambda):
                """Update the lambda_monomer array at runtime.

                ``new_lambda`` must be array-like of shape ``(n_monomers,)``.
                The update is reflected in subsequent ``calculate()`` calls because
                the closure variables inside the JIT-compiled functions reference
                the same mutable buffer.

                Note: Since JAX traces are keyed on array *identity*, changing
                the lambda values will trigger a re-trace of JIT-compiled
                functions that close over ``lambda_monomer``.  For performance
                in production runs, prefer setting lambda once before the MD loop.
                """
                nonlocal lambda_monomer
                new_arr = jnp.asarray(new_lambda, dtype=jnp.float32)
                if new_arr.shape != (n_monomers,):
                    raise ValueError(
                        f"lambda_monomer must have shape ({n_monomers},), got {new_arr.shape}"
                    )
                lambda_monomer = new_arr

            def calculate(
                self,
                atoms,
                properties,
                system_changes=ase.calculators.calculator.all_changes,
            ):
                """Calculate energy and forces for given atomic configuration"""

                ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
                R = np.asarray(atoms.get_positions(), dtype=np.float64)

                # Do NOT wrap positions during energy/force evaluation. MIC handles unwrapped
                # coordinates; wrapping causes discontinuous energy jumps during BFGS when
                # monomers cross cell boundaries. Wrap only for output (trajectories, structures).

                Z = atoms.get_atomic_numbers()

                # MIC-only PBC: pass R directly. Cell list wraps by molecule for binning.
                expected_atoms = self.total_atoms
                if len(Z) != expected_atoms:
                    raise ValueError(
                        "Atom count mismatch. "
                        f"Got len(Z)={len(Z)}, expected {expected_atoms} "
                        f"(atoms_per_monomer={self.atoms_per_monomer_list}). "
                        "This triggers padding and can yield exact zero forces. "
                        "Fix atoms_per_monomer or trim the input atoms."
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

                # Pre-build MM function: cell list uses wrap-by-molecule for binning (monomer_offsets).
                mm_pair_idx, mm_pair_mask = None, None
                if self.doMM:
                    _ensure_mm_fn(np.asarray(R), self.cutoff_params)
                    update_fn = _cached_update_mm_pairs[0]
                    if update_fn is not None:
                        mm_pair_idx, mm_pair_mask = update_fn(R)

                # Compute ModelOutput to get forces directly (more stable)
                out = spherical_cutoff_calculator(
                    positions=R,
                    atomic_numbers=Z,
                    n_monomers=self.n_monomers,
                    cutoff_params=self.cutoff_params,
                    doML=self.doML,
                    doMM=self.doMM,
                    doML_dimer=self.doML_dimer,
                    debug=self.debug,
                    mm_pair_idx=mm_pair_idx,
                    mm_pair_mask=mm_pair_mask,
                )
                
                # Use forces directly from ModelOutput (MIC-only: no force transform needed)
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



                final_energy = E
                
                self.results["energy"] = final_energy * self.energy_conversion_factor
                # Ensure forces are finite before storing
                forces_final = F * self.force_conversion_factor
                
                # Check for NaN/Inf using JAX operations first (works with JAX arrays)
                forces_final = jnp.where(jnp.isfinite(forces_final), forces_final, 0.0)
                if self.debug:

                    # Hard check: ml_2b outputs must be finite (avoid silent zeroing).
                    if hasattr(out, "ml_2b_E") or hasattr(out, "ml_2b_F"):
                        ml_2b_E_host = np.asarray(jax.device_get(getattr(out, "ml_2b_E", 0.0)))
                        ml_2b_F_host = np.asarray(jax.device_get(getattr(out, "ml_2b_F", np.zeros_like(R))))
                        if not np.all(np.isfinite(ml_2b_E_host)):
                            raise ValueError(f"Non-finite ml_2b_E detected: {ml_2b_E_host}")
                        if not np.all(np.isfinite(ml_2b_F_host)):
                            bad_idx = np.where(~np.isfinite(ml_2b_F_host).all(axis=1))[0]
                            raise ValueError(
                                "Non-finite ml_2b_F detected at atom indices "
                                f"{bad_idx.tolist()}."
                            )

                    if hasattr(out, "internal_F"):
                        internal_F = out.internal_F
                        internal_F = jnp.where(jnp.isfinite(internal_F), internal_F, 0.0)
                        internal_F_host = np.asarray(jax.device_get(internal_F))
                        # Use relaxed threshold (1e-10) - float32 ML models can produce very small forces
                        internal_zero_mask = np.linalg.norm(internal_F_host, axis=1) < 1e-10
                        if np.any(internal_zero_mask):
                            zero_indices = np.where(internal_zero_mask)[0]
                            R_host = np.asarray(jax.device_get(R))
                            # Find which monomer each zero-force atom belongs to (heterogeneous-safe)
                            _offsets_np = np.array([int(monomer_offsets[k]) for k in range(n_monomers + 1)])
                            _mono_ids = np.searchsorted(_offsets_np[1:], zero_indices, side='right')
                            slots = (zero_indices - _offsets_np[_mono_ids]).tolist()
                            monomers = _mono_ids.tolist()
                            pos_sample = R_host[zero_indices[:10]].tolist()
                            # Compute min distance within each monomer for zero-force atoms
                            min_distances = []
                            for idx in zero_indices:
                                mid = int(np.searchsorted(_offsets_np[1:], idx, side='right'))
                                start = int(_offsets_np[mid])
                                end = int(_offsets_np[mid + 1])
                                monomer_positions = R_host[start:end]
                                diffs = monomer_positions - R_host[idx]
                                dists = np.linalg.norm(diffs, axis=1)
                                dists[idx - start] = np.inf  # exclude self-distance
                                min_distances.append(float(np.min(dists)))
                            warnings.warn(
                                "Internal monomer forces near zero. "
                                f"Zero-force atoms at indices {zero_indices.tolist()} "
                                f"(monomer slots {slots}, monomers {monomers}) "
                                f"with Z {Z[zero_indices].tolist()} and positions {pos_sample}. "
                                f"Min in-monomer distances {min_distances}. "
                                "Continuing; if unexpected, check model/checkpoint or PBC mapping.",
                                UserWarning,
                                stacklevel=2,
                            )
                    
                        # Also report zero internal forces in debug mode (should be none due to check above).
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
                            print(
                                f"DEBUG dimer COM distance: {d_com:.4f} "
                                "(ml_2b can be 0 if r > mm_switch_on or the model yields ~0 interaction)"
                            )
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
                    # Compare atoms that changed to zero during conversion
                    if "zero_mask_jax" in locals():
                        became_zero = np.where((~zero_mask_jax) & zero_mask_after)[0]
                        if len(became_zero) > 0:
                            print("WARNING: Forces became zero during numpy conversion!")
                            for idx in became_zero[:10]:
                                print(f"  Atom {idx}: BEFORE={forces_jax_mags_np[idx]:.6e}, AFTER={force_mags_after_conv[idx]:.6e}")
                
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
                        # Report per-monomer slot pattern
                        if self.atoms_per_monomer is not None:
                            _apm = self.atoms_per_monomer
                            zero_mod = zero_indices % _apm
                            mod_counts = np.bincount(zero_mod, minlength=int(_apm))
                            print(f"Calculator storage - zero-force slot counts (mod n_per): {mod_counts}")
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
            do_pbc_map: bool = False,
            pbc_map = None,
            create_ase_calculator: bool = True,
        ) -> Tuple[Any, Callable]:
            """Factory function to create calculator instances.

            doML, doMM, doML_dimer, debug default to the values passed to setup_calculator.
            Pass them explicitly here to override per-call.

            Args:
                verbose: If True, store full ModelOutput breakdown in results.
                         If None, defaults to debug value.
                create_ase_calculator: If False, return a minimal object with pbc_map/do_pbc_map
                    only (for JAX-MD when ASE MD is not run). Saves overhead when nsteps_ase == 0.
            """
            pbc_cell_for_calc = np.asarray(pbc_cell) if pbc_cell is not None else None

            if create_ase_calculator:
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
                    pbc_cell=pbc_cell_for_calc,
                    verbose=verbose,
                )
            else:
                # Minimal object for JAX-MD pbc_map only
                class _PbcMapOnly:
                    def __init__(self, pbc_map_fn, do_pbc_map_val: bool):
                        self.pbc_map = pbc_map_fn
                        self.do_pbc_map = do_pbc_map_val
                calculator = _PbcMapOnly(pbc_map, do_pbc_map)

            # Bind setup-time defaults so direct callers of the returned function
            # inherit doML/doMM/doML_dimer/debug configuration.
            configured_spherical_cutoff = partial(
                spherical_cutoff_calculator,
                doML=doML,
                doMM=doMM,
                doML_dimer=doML_dimer,
                debug=debug,
            )

            def get_update_fn(positions, cutoff_params_arg):
                """Ensure MM fn is built and return update_mm_pairs, or None for cell-list path."""
                _ensure_mm_fn(np.asarray(positions), cutoff_params_arg)
                return _cached_update_mm_pairs[0]

            return calculator, configured_spherical_cutoff, get_update_fn

    else:  # pragma: no cover - exercised when ASE not installed

        class AseDimerCalculator:  # type: ignore[too-few-public-methods]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ModuleNotFoundError("ase is required for AseDimerCalculator")

        def get_spherical_cutoff_calculator(*args: Any, **kwargs: Any):  # type: ignore[override]
            raise ModuleNotFoundError("ase is required for get_spherical_cutoff_calculator")



    def process_monomer_forces(
        ml_monomer_forces: Array,
        monomer_segment_idxs: Array,
        atoms_per_system: int,
        monomer_atomic_numbers_flat: Array | None = None,
        monomer_positions_flat: Array | None = None,
        monomer_atom_mask_flat: Array | None = None,
        debug: bool = False,
    ) -> Array:
        """Process and reshape monomer forces with proper masking.

        Supports heterogeneous monomer sizes.  Forces are extracted from the
        padded batch output and scattered to the correct global atom positions
        via ``segment_sum`` using ``monomer_segment_idxs``.

        Args:
            ml_monomer_forces: Raw forces from ML model, shape ``(n_monomers * atoms_per_system, 3)``
                where ``atoms_per_system`` is the padded (max_atoms) dimension.
            monomer_segment_idxs: Global atom index for every valid monomer atom
                (length ``sum(atoms_per_monomer_list)``).
            atoms_per_system: Padded atom count per batch element (max_atoms).
            debug: Enable debug printing.

        Returns:
            Array of shape ``(total_atoms, 3)`` with monomer forces placed at global positions.
        """
        _n_mono = len(atoms_per_monomer_list)
        # Reshape to (n_monomers, atoms_per_system, 3)
        monomer_forces = ml_monomer_forces.reshape(_n_mono, atoms_per_system, 3)

        # Extract valid atoms per monomer (variable count) and flatten
        valid_forces_parts = []
        for mi in range(_n_mono):
            n_i = atoms_per_monomer_list[mi]
            valid_forces_parts.append(monomer_forces[mi, :n_i, :])
        forces_flat = jnp.concatenate(valid_forces_parts, axis=0)  # (sum(n_i), 3)

        # Scatter to global positions
        processed_forces = jax.ops.segment_sum(
            forces_flat,
            monomer_segment_idxs,
            num_segments=total_atoms
        )

        # Ensure all forces are finite
        processed_forces = jnp.where(jnp.isfinite(processed_forces), processed_forces, 0.0)

        if debug:
            force_mags = jnp.linalg.norm(processed_forces, axis=1)
            zero_mask = force_mags < 1e-12
            zero_count = jnp.sum(zero_mask)
            jax.debug.print("DEBUG monomer forces: zero count {c}", c=zero_count, ordered=False)
            zero_indices = jnp.where(zero_mask, size=10, fill_value=-1)[0]
            jax.debug.print("DEBUG monomer forces: zero indices sample {idx}", idx=zero_indices, ordered=False)
            if monomer_atomic_numbers_flat is not None:
                z_sample = jnp.take(monomer_atomic_numbers_flat, zero_indices, axis=0, mode="clip")
                jax.debug.print("DEBUG monomer forces: Z sample {z}", z=z_sample, ordered=False)
        
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
        """Calculate energy and force contributions from dimers (heterogeneous-safe)."""
        # Compute max_atoms (padded batch dimension) -- heterogeneous
        _dimer_atom_counts = [
            atoms_per_monomer_list[a] + atoms_per_monomer_list[b]
            for a, b in dimer_perms
        ]
        max_monomer_atoms = max(atoms_per_monomer_list)
        max_dimer_atoms = max(_dimer_atom_counts) if _dimer_atom_counts else 2 * max_monomer_atoms
        max_atoms = max(max_monomer_atoms, max_dimer_atoms)

        # Get dimer energies and forces
        ml_dimer_energy = jnp.array(e[n_monomers:]).flatten()
        monomer_batch_atoms = n_monomers * max_atoms
        ml_dimer_forces = f[monomer_batch_atoms:]

        # Calculate force segments for dimers
        force_segments = calculate_dimer_force_segments(n_dimers)

        # Calculate interaction energies (E_dimer - E_mono_a - E_mono_b)
        monomer_contrib = calculate_monomer_contribution_to_dimers(
            monomer_energies, jnp.array(dimer_perms)
        )
        dimer_int_energies = ml_dimer_energy - monomer_contrib

        # Apply lambda scaling: dimer interaction scaled by lambda_i * lambda_j
        dimer_lambda = jnp.array([
            lambda_monomer[a] * lambda_monomer[b] for a, b in dimer_perms
        ])
        dimer_int_energies = dimer_int_energies * dimer_lambda

        debug_print(debug, "Dimer int energies:",
            dimer_int_energies=dimer_int_energies,
            ml_dimer_energy=ml_dimer_energy,
            monomer_contrib=monomer_contrib,
        )

        # Process dimer forces
        dimer_forces = process_dimer_forces(
            ml_dimer_forces, force_segments, n_dimers, max_atoms, debug
        )

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
            "out_F": switched_results["forces"],
            "dH": switched_results["energies"].sum(),
            "ml_2b_E": switched_results["energies"].sum(),
            "ml_2b_F": switched_results["forces"]
        }

    def calculate_dimer_force_segments(n_dimers: int) -> Array:
        """Calculate force segments for dimer force summation (heterogeneous-safe).

        For each dimer (i, j), the segment indices map the dimer's atoms back
        to their global positions using ``monomer_offsets``.
        """
        parts = []
        for di, (mi, mj) in enumerate(dimer_perms):
            n_i = atoms_per_monomer_list[mi]
            n_j = atoms_per_monomer_list[mj]
            off_i = int(monomer_offsets[mi])
            off_j = int(monomer_offsets[mj])
            # First monomer atoms, then second monomer atoms
            seg = np.concatenate([
                np.arange(off_i, off_i + n_i),
                np.arange(off_j, off_j + n_j),
            ])
            parts.append(seg)
        return jnp.array(np.concatenate(parts))

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
        max_atoms: int,
        debug: bool
    ) -> Array:
        """Process and reshape dimer forces (heterogeneous-safe).

        Extracts valid atoms per dimer (variable count) and scatters forces to
        global atom positions via ``segment_sum``.
        """
        # dimer_forces shape: (n_dimers * max_atoms, 3)
        dimer_forces_2d = dimer_forces.reshape(n_dimers, max_atoms, 3)

        # Extract valid atoms per dimer (variable) and flatten
        valid_parts = []
        for di, (mi, mj) in enumerate(dimer_perms):
            n_d = atoms_per_monomer_list[mi] + atoms_per_monomer_list[mj]
            valid_parts.append(dimer_forces_2d[di, :n_d, :])
        forces_flat = jnp.concatenate(valid_parts, axis=0)

        processed_forces = jax.ops.segment_sum(
            forces_flat,
            force_segments,
            num_segments=total_atoms
        )

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
        """Apply switching functions to dimer energies and forces (heterogeneous-safe).

        Forces are computed using the product rule:
        ``F = -d/dR [E * s(R)] = -[dE/dR * s(R) + E * ds/dR]``
        """
        n_dimers = len(all_dimer_idxs)
        force_segments = calculate_dimer_force_segments(n_dimers)

        # Per-dimer atom counts for switch_ML calls
        dimer_n_atoms_a = [atoms_per_monomer_list[a] for a, b in dimer_perms]
        dimer_n_atoms_b = [atoms_per_monomer_list[b] for a, b in dimer_perms]

        # Pad dimer indices to max_atoms for uniform array shape
        _padded_dimer_idxs = []
        for idxs in all_dimer_idxs:
            n_d = len(idxs)
            if n_d < max_atoms:
                padded = np.concatenate([idxs, np.zeros(max_atoms - n_d, dtype=int)])
            else:
                padded = idxs[:max_atoms]
            _padded_dimer_idxs.append(padded)
        padded_dimer_idx_arr = jnp.array(_padded_dimer_idxs)  # (n_dimers, max_atoms)

        # Gather dimer positions: (n_dimers, max_atoms, 3)
        dimer_pos_padded = positions[padded_dimer_idx_arr]

        # --- switched energy, scale, grad per dimer (loop, not vmap, for variable n_atoms_a/b) ---
        switched_energy_list = []
        switching_scale_list = []
        switched_grad_list = []
        for di in range(n_dimers):
            na = dimer_n_atoms_a[di]
            nb = dimer_n_atoms_b[di]
            x = dimer_pos_padded[di]  # (max_atoms, 3)
            e_di = dimer_energies[di]
            se = switch_ML(x, e_di, ml_cutoff=cutoff_params.ml_cutoff,
                           mm_switch_on=cutoff_params.mm_switch_on,
                           n_atoms_a=na, n_atoms_b=nb, pbc_cell=pbc_cell)
            ss = switch_ML(x, 1.0, ml_cutoff=cutoff_params.ml_cutoff,
                           mm_switch_on=cutoff_params.mm_switch_on,
                           n_atoms_a=na, n_atoms_b=nb, pbc_cell=pbc_cell)
            sg = switch_ML_grad(x, e_di, ml_cutoff=cutoff_params.ml_cutoff,
                                mm_switch_on=cutoff_params.mm_switch_on,
                                n_atoms_a=na, n_atoms_b=nb, pbc_cell=pbc_cell)
            switched_energy_list.append(se)
            switching_scale_list.append(ss)
            switched_grad_list.append(sg)

        switched_energy = jnp.array(switched_energy_list)
        switching_scales = jnp.array(switching_scale_list)
        switched_grad = jnp.stack(switched_grad_list)  # (n_dimers, max_atoms, 3)

        # Extract valid atoms per dimer from switched_grad and flatten for segment_sum
        grad_parts = []
        for di, (mi, mj) in enumerate(dimer_perms):
            n_d = atoms_per_monomer_list[mi] + atoms_per_monomer_list[mj]
            grad_parts.append(switched_grad[di, :n_d, :])
        dimer_switching_grads_flat = jnp.concatenate(grad_parts, axis=0)

        energy_weighted_grad = jax.ops.segment_sum(
            dimer_switching_grads_flat,
            force_segments,
            num_segments=total_atoms
        )

        # Per-atom switching scale: expand per-dimer scale to per-atom
        scale_parts = []
        for di, (mi, mj) in enumerate(dimer_perms):
            n_d = atoms_per_monomer_list[mi] + atoms_per_monomer_list[mj]
            scale_parts.append(jnp.full((n_d,), switching_scales[di]))
        switching_scales_per_atom = jnp.concatenate(scale_parts)

        atom_switching_scales_sum = jax.ops.segment_sum(
            switching_scales_per_atom, force_segments, num_segments=total_atoms
        )
        atom_dimer_counts = jax.ops.segment_sum(
            jnp.ones_like(switching_scales_per_atom), force_segments, num_segments=total_atoms
        )

        safe_counts = jnp.maximum(atom_dimer_counts, 1.0)
        safe_counts = jnp.where(jnp.isfinite(safe_counts), safe_counts, 1.0)
        safe_scales_sum = jnp.where(jnp.isfinite(atom_switching_scales_sum), atom_switching_scales_sum, 0.0)
        atom_switching_scales = jnp.where(
            atom_dimer_counts > 0, safe_scales_sum / safe_counts, 1.0
        )
        atom_switching_scales = jnp.where(jnp.isfinite(atom_switching_scales), atom_switching_scales, 1.0)

        dimer_forces_safe = jnp.where(jnp.isfinite(dimer_forces), dimer_forces, 0.0)
        scaled_dimer_forces = dimer_forces_safe * atom_switching_scales[:, None]

        energy_weighted_grad_safe = jnp.where(jnp.isfinite(energy_weighted_grad), energy_weighted_grad, 0.0)
        switched_forces = scaled_dimer_forces - energy_weighted_grad_safe
        switched_forces = jnp.where(jnp.isfinite(switched_forces), switched_forces, 0.0)

        return {
            "energies": switched_energy,
            "forces": switched_forces
        }

    # Expose pbc_map and do_pbc_map so callers (e.g. run_sim) can pass them to the calculator
    get_spherical_cutoff_calculator.pbc_map = pbc_map if do_pbc_map else None
    get_spherical_cutoff_calculator.do_pbc_map = do_pbc_map
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