"""MM (Lennard-Jones + Coulomb) energy and force calculation for hybrid ML/MM potentials."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from mmml.interfaces.pycharmmInterface.calculator_utils import _sharpstep
from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON
from mmml.interfaces.pycharmmInterface.ml_dtypes import resolve_ml_compute_dtype
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
    mic_displacement,
    mic_displacement_smooth,
    mic_displacements_batched,
    mic_displacements_batched_smooth,
)

# Verlet skin for jax-md MM pair reuse: ≤ dr_threshold/2 (dr_threshold=0.5 Å in bundle).
DEFAULT_JAX_MD_CAPACITY_MULTIPLIER = 1.75
DEFAULT_JAX_MD_SKIN_DISTANCE_A = 0.25

try:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
    pycharmm_quiet()
except Exception:
    CGENFF_PRM = CGENFF_RTF = None
    def pycharmm_quiet():
        pass

try:
    from mmml.interfaces.pycharmmInterface.cell_list import (
        PairListTruncationError,
        cell_list_pairs as _cell_list_pairs,
        estimate_max_pairs as _estimate_max_pairs,
    )
except Exception:
    PairListTruncationError = RuntimeError  # type: ignore[misc, assignment]
    _cell_list_pairs = None
    _estimate_max_pairs = None

try:
    from mmml.interfaces.pycharmmInterface.nl_reference import have_vesin
except Exception:
    def have_vesin() -> bool:
        return False

try:
    from mmml.interfaces.pycharmmInterface.nl_backend import (
        build_mm_pairs_with_backend,
        pick_static_rebuild_backend,
        resolve_mm_nl_backend,
    )
except Exception:
    def build_mm_pairs_with_backend(*args, **kwargs):
        raise RuntimeError("nl_backend unavailable")

    def pick_static_rebuild_backend(*args, **kwargs):
        return "cell_list"

    def resolve_mm_nl_backend(name=None):
        return "cell_list"

try:
    from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import (
        have_jax_md,
        create_jax_md_neighbor_list,
    )
except Exception:
    def have_jax_md():
        return False

    def create_jax_md_neighbor_list(*args, **kwargs):
        return None


def _dimer_permutations(n_mol: int) -> List[Tuple[int, int]]:
    from itertools import combinations
    return list(combinations(range(n_mol), 2))


def _filter_pairs_by_com_min(
    positions: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mask: np.ndarray,
    monomer_offsets: np.ndarray,
    monomer_id: np.ndarray,
    mm_r_min: float,
    pbc_cell: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exclude pairs where dimer COM distance < mm_r_min. Returns updated mask."""
    R = np.asarray(positions, dtype=np.float64)
    n_monomers = len(monomer_offsets) - 1
    coms = np.zeros((n_monomers, 3), dtype=np.float64)
    for k in range(n_monomers):
        start, end = int(monomer_offsets[k]), int(monomer_offsets[k + 1])
        coms[k] = R[start:end].mean(axis=0)

    if pbc_cell is not None:
        cell = np.asarray(pbc_cell, dtype=np.float64)
        if cell.ndim == 0:
            cell = np.diag([float(cell)] * 3)
        elif cell.ndim == 1 and cell.shape[0] == 3:
            cell = np.diag(cell)
        inv_cell = np.linalg.inv(cell)
    else:
        cell = inv_cell = None

    out_mask = mask.copy()
    n_pairs = mask.shape[0]
    for k in range(n_pairs):
        if not mask[k]:
            continue
        ai, aj = int(pair_i[k]), int(pair_j[k])
        mi = int(monomer_id[ai])
        mj = int(monomer_id[aj])
        com_i = coms[mi]
        com_j = coms[mj]
        dr = com_j - com_i
        if inv_cell is not None:
            frac_dr = dr @ inv_cell.T
            frac_dr = frac_dr - np.round(frac_dr)
            dr = frac_dr @ cell
        r = float(np.linalg.norm(dr))
        if r < mm_r_min:
            out_mask[k] = False
    return out_mask


def format_mm_pair_update_stats_summary(stats: dict) -> str:
    """One-line neighbor-list cache summary for jaxmd suite logs."""
    calls = int(stats.get("calls", 0))
    reused = int(stats.get("reused", 0))
    updates = int(stats.get("updates", 0))
    reallocs = int(stats.get("reallocs", 0))
    fallbacks = int(stats.get("fallbacks", 0))
    host_syncs = int(stats.get("host_syncs", 0))
    device_skin_checks = int(stats.get("device_skin_checks", 0))
    cpu_rebuilds = int(stats.get("cpu_rebuilds", 0))
    gpu_rebuilds = int(stats.get("gpu_rebuilds", 0))
    capacity_grows = int(stats.get("capacity_grows", 0))
    capacity_changes = int(stats.get("pair_capacity_changes", 0))
    pct = 100.0 * reused / max(1, calls)
    return (
        f"[jaxmd_nbr] pair-list cache: {reused}/{calls} reused ({pct:.1f}%), "
        f"{updates} rebuilds (cpu={cpu_rebuilds}, gpu={gpu_rebuilds}), "
        f"host_syncs={host_syncs}, device_skin_checks={device_skin_checks}, "
        f"capacity_grows={capacity_grows}, capacity_changes={capacity_changes}, "
        f"reallocs={reallocs}, fallbacks={fallbacks}"
    )


def neighbor_pair_cache_should_reuse(
    *,
    calls: int,
    interval: int,
    skin: float,
    R: np.ndarray,
    last_R: np.ndarray | None,
    box: np.ndarray | None,
    last_box: np.ndarray | None,
    have_cache: bool,
    box_delta_tol: float = 1e-8,
) -> bool:
    """Return True when ``update_mm_pairs`` may reuse cached pair_idx/pair_mask."""
    if not have_cache:
        return False

    interval_i = int(max(1, interval))
    skin_f = float(max(0.0, skin))

    box_delta = 0.0
    if (box is None) != (last_box is None):
        return False
    if box is not None and last_box is not None:
        box_delta = float(np.max(np.abs(np.asarray(box) - np.asarray(last_box))))

    if int(calls) % interval_i != 0:
        return box_delta <= box_delta_tol

    if skin_f > 0.0 and last_R is not None:
        max_disp = float(np.max(np.linalg.norm(R - last_R, axis=1)))
        return max_disp <= skin_f and box_delta <= box_delta_tol

    return False


def _fractional_positions_for_jax_md_neighbor_list(
    positions: np.ndarray,
    pbc_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap Cartesian positions to fractional [0,1) and return (R_frac, box_diag)."""
    cell_np = np.asarray(pbc_cell, dtype=np.float64)
    if cell_np.ndim == 0:
        L = float(cell_np)
        cell_3x3 = np.diag([L, L, L])
    elif cell_np.shape == (1,):
        L = float(cell_np[0])
        cell_3x3 = np.diag([L, L, L])
    elif cell_np.shape == (3,):
        cell_3x3 = np.diag(cell_np)
    else:
        cell_3x3 = cell_np

    inv_cell = np.linalg.inv(cell_3x3)
    R_frac = positions @ inv_cell.T
    R_frac = np.asarray(R_frac - np.floor(R_frac), dtype=np.float64)
    box_diag = np.diagonal(cell_3x3).astype(np.float64)
    return R_frac, box_diag


def _validate_dynamic_pair_contract(
    pair_idx: Array,
    pair_mask: Array,
    *,
    total_atoms: int,
    label: str,
) -> None:
    """Debug-only contract checks for dynamic MM neighbor-list providers.

    Dynamic providers return padded atom-pair indices with shape ``(capacity, 2)``
    and a same-length validity mask. Valid entries are inter-atom pairs with
    ``0 <= i,j < total_atoms``; invalid entries may contain padding values.
    """
    pair_idx_np = np.asarray(jax.device_get(pair_idx))
    pair_mask_np = np.asarray(jax.device_get(pair_mask), dtype=bool).reshape(-1)
    if pair_idx_np.ndim != 2 or pair_idx_np.shape[1] != 2:
        raise ValueError(f"{label}: pair_idx must have shape (capacity, 2); got {pair_idx_np.shape}")
    if pair_mask_np.shape[0] != pair_idx_np.shape[0]:
        raise ValueError(
            f"{label}: pair_mask length {pair_mask_np.shape[0]} does not match "
            f"pair_idx capacity {pair_idx_np.shape[0]}"
        )
    valid_idx = pair_idx_np[pair_mask_np]
    if valid_idx.size == 0:
        return
    if valid_idx.min() < 0 or valid_idx.max() >= int(total_atoms):
        raise ValueError(f"{label}: valid pair indices out of atom range [0, {total_atoms})")


def _box_to_cell_3x3(box: Array) -> Array:
    """Convert box (scalar, (1,), (3,), or (3,3)) to 3x3 cell matrix for MIC/frac_coords.
    JIT-safe: uses only JAX ops, no Python float() on traced values."""
    b = jnp.asarray(box)
    # L for isotropic: first element (works for (), (1,), (3,), (3,3))
    L = b.reshape(-1)[0]
    diag_iso = jnp.diag(jnp.broadcast_to(L, (3,)))
    diag_3 = jnp.diag(b) if b.ndim == 1 and b.shape[0] == 3 else diag_iso
    return jnp.where(
        b.ndim == 2,
        jnp.asarray(b, dtype=b.dtype),
        jnp.where(b.ndim == 1, diag_3, diag_iso),
    )


def _resolve_cell_list_max_pairs(
    *,
    total_atoms: int,
    pbc_cell: np.ndarray | None,
    cutoff: float,
    max_pairs: int | None,
    cell_list_safety_factor: float,
    cell_list_density_estimate: float | None,
) -> int:
    if max_pairs is not None:
        return int(max_pairs)
    from mmml.interfaces.pycharmmInterface.cell_list import cubic_box_side_from_cell_matrix

    box_side = cubic_box_side_from_cell_matrix(
        np.asarray(pbc_cell) if pbc_cell is not None else None
    )
    return int(
        _estimate_max_pairs(
            total_atoms,
            cutoff=cutoff,
            safety_factor=cell_list_safety_factor,
            density_estimate=cell_list_density_estimate,
            box_side_A=box_side,
        )
    )


def _build_cell_list_pairs_with_retry(
    *,
    positions: np.ndarray,
    pbc_cell: np.ndarray,
    cutoff: float,
    max_pairs: int | None,
    monomer_offsets: np.ndarray,
    atoms_per_monomer_list: Sequence[int],
    total_atoms: int,
    cell_list_safety_factor: float,
    cell_list_density_estimate: float | None,
    debug: bool,
):
    """Build cell-list pairs; grow ``max_pairs`` if the first estimate is too small."""
    import math

    capacity = _resolve_cell_list_max_pairs(
        total_atoms=total_atoms,
        pbc_cell=pbc_cell,
        cutoff=cutoff,
        max_pairs=max_pairs,
        cell_list_safety_factor=cell_list_safety_factor,
        cell_list_density_estimate=cell_list_density_estimate,
    )
    last_exc: PairListTruncationError | None = None
    for attempt in range(4):
        try:
            cl_i, cl_j, cl_mask, n_valid = _cell_list_pairs(
                np.asarray(positions),
                np.asarray(pbc_cell),
                cutoff=cutoff,
                max_pairs=capacity,
                monomer_offsets=monomer_offsets,
                atoms_per_monomer_list=list(atoms_per_monomer_list),
                exclude_intra_monomer=True,
                suppress_warning=(attempt < 3),
            )
            if debug and attempt > 0:
                print(
                    f"[get_MM] Cell list: max_pairs raised to {capacity} "
                    f"({n_valid} valid pairs)",
                    flush=True,
                )
            return cl_i, cl_j, cl_mask, n_valid, capacity
        except PairListTruncationError as exc:
            last_exc = exc
            capacity = max(
                int(math.ceil(exc.suggested_max_pairs)),
                int(math.ceil(exc.n_found * 1.25)),
            )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("cell-list pair build failed without truncation error")


def _get_actual_psf_charges(total_atoms: int) -> np.ndarray:
    """Fallback mechanism to recover original PSF charges when they have been zeroed.

    Loads charges from `psf.get_charges()`. If any charges are zero (or if we need to recover),
    we parse the CGENFF_RTF file and map the charges back to atoms by residue name and atom type name.
    """
    import pycharmm.psf as psf
    try:
        import pycharmm.atom_info as atom_info
    except ImportError:
        atom_info = None

    charges = np.array(psf.get_charges(), dtype=np.float64)
    if charges.size == 0:
        return charges

    # Read and parse CGENFF_RTF to map residue_name -> {atom_name: charge}
    rtf_charges = {}
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF
    import os

    if CGENFF_RTF and os.path.exists(CGENFF_RTF):
        try:
            current_res = None
            with open(CGENFF_RTF, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("!"):
                        continue
                    if "!" in line:
                        line = line.split("!")[0].strip()
                    tokens = line.split()
                    if not tokens:
                        continue
                    if tokens[0] in ("RESI", "PRES"):
                        current_res = tokens[1].upper()
                        rtf_charges[current_res] = {}
                    elif tokens[0] == "ATOM":
                        if current_res is not None and len(tokens) >= 4:
                            atom_name = tokens[1].upper()
                            try:
                                q_val = float(tokens[3])
                                rtf_charges[current_res][atom_name] = q_val
                            except ValueError:
                                pass
        except Exception as e:
            print(f"WARNING: Failed to parse charges from CGENFF_RTF: {e}", flush=True)

    # Get residue name and atom name for every atom
    n_atoms = charges.size
    atom_res_names = None
    if atom_info is not None:
        try:
            atom_res_names = atom_info.get_res_names(list(range(n_atoms)))
        except Exception as e:
            print(f"WARNING: Failed to get residue names using atom_info: {e}", flush=True)

    try:
        atom_names = psf.get_atype()
    except Exception as e:
        print(f"WARNING: Failed to get atom types using psf.get_atype(): {e}", flush=True)
        atom_names = None

    recovered_charges = charges.copy()
    if atom_res_names and atom_names:
        for i in range(n_atoms):
            # Only override if the active charge is zero (or very close to it)
            if abs(charges[i]) < 1e-9:
                res_name = atom_res_names[i].upper()
                atom_name = atom_names[i].upper()
                rtf_q = rtf_charges.get(res_name, {}).get(atom_name, None)
                if rtf_q is not None:
                    recovered_charges[i] = rtf_q

    return recovered_charges


def _wrap_mm_fn_with_jax_pme_coulomb(
    mm_fn: Callable[..., Tuple[Array, Array]],
    *,
    charges_np: np.ndarray,
    pbc_cell: np.ndarray | None,
    method: str,
    sr_cutoff_A: float,
    dynamic: bool,
    monomer_offsets: np.ndarray,
    monomer_id_np: np.ndarray,
    lambda_monomer: np.ndarray,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
    complementary_handoff: bool,
    mm_r_min: float | None,
    static_pair_i: np.ndarray | None = None,
    static_pair_j: np.ndarray | None = None,
    static_pair_mask: np.ndarray | None = None,
) -> Callable[..., Tuple[Array, Array]]:
    """Add jax-pme Coulomb (minus intra-monomer) to switched-LJ MM (hybrid path).

    Pair loop supplies cross-monomer LJ with COM switching; this wrapper adds
    ``scale * (E_pme - E_intra)`` so intra-monomer electrostatics stay in ML.
    """
    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        hybrid_jax_pme_coulomb_correction,
    )
    from mmml.interfaces.pycharmmInterface.long_range_backend import box_length_from_cell

    charges_host = np.asarray(charges_np, dtype=np.float64)
    offsets_host = np.asarray(monomer_offsets, dtype=np.int64)
    monomer_id_host = np.asarray(monomer_id_np, dtype=np.int64)
    lambda_host = np.asarray(lambda_monomer, dtype=np.float64)
    pbc_host = np.asarray(pbc_cell, dtype=np.float64) if pbc_cell is not None else None
    static_pi = (
        None if static_pair_i is None else np.asarray(static_pair_i, dtype=np.int64)
    )
    static_pj = (
        None if static_pair_j is None else np.asarray(static_pair_j, dtype=np.int64)
    )
    static_mask = (
        None
        if static_pair_mask is None
        else np.asarray(static_pair_mask, dtype=np.float64)
    )

    def _box_length(box_override: Array | np.ndarray | None) -> float:
        if box_override is not None:
            return box_length_from_cell(np.asarray(jax.device_get(box_override)))
        if pbc_cell is None:
            raise ValueError("jax_pme Coulomb requires pbc_cell or box_override")
        return box_length_from_cell(np.asarray(pbc_cell))

    def _pbc_cell(box_override: Array | np.ndarray | None) -> np.ndarray | None:
        if box_override is not None:
            box_np = np.asarray(jax.device_get(box_override), dtype=np.float64)
            return np.diag(box_np) if box_np.ndim == 1 else box_np
        return pbc_host

    def _lr_correction(
        pos_np: np.ndarray,
        box_override: Array | np.ndarray | None,
        pair_i: np.ndarray | None,
        pair_j: np.ndarray | None,
        pair_mask: np.ndarray | None,
    ):
        return hybrid_jax_pme_coulomb_correction(
            pos_np,
            charges_host,
            offsets_host,
            box_length_A=_box_length(box_override),
            method=method,
            sr_cutoff_A=sr_cutoff_A,
            pair_i=pair_i,
            pair_j=pair_j,
            pair_mask=pair_mask,
            monomer_id=monomer_id_host,
            lambda_monomer=lambda_host,
            pbc_cell=_pbc_cell(box_override),
            ml_switch_width=ml_switch_width,
            mm_switch_on=mm_switch_on,
            mm_switch_width=mm_switch_width,
            complementary_handoff=complementary_handoff,
            mm_r_min=mm_r_min,
        )

    if dynamic:

        def wrapped(
            positions: Array,
            pair_idx: Array,
            pair_mask: Array,
            box_override: Optional[Array] = None,
        ) -> Tuple[Array, Array]:
            e_sr, f_sr = mm_fn(positions, pair_idx, pair_mask, box_override=box_override)
            pos_np = np.asarray(jax.device_get(positions), dtype=np.float64)
            pair_i = np.asarray(jax.device_get(pair_idx[:, 0]), dtype=np.int64)
            pair_j = np.asarray(jax.device_get(pair_idx[:, 1]), dtype=np.int64)
            mask = np.asarray(jax.device_get(pair_mask), dtype=np.float64)
            lr = _lr_correction(pos_np, box_override, pair_i, pair_j, mask)
            return (
                e_sr + jnp.asarray(lr.energy_kcalmol, dtype=e_sr.dtype),
                f_sr + jnp.asarray(lr.forces_kcalmol_A, dtype=f_sr.dtype),
            )

        return wrapped

    def wrapped(positions: Array) -> Tuple[Array, Array]:
        e_sr, f_sr = mm_fn(positions)
        pos_np = np.asarray(jax.device_get(positions), dtype=np.float64)
        lr = _lr_correction(pos_np, None, static_pi, static_pj, static_mask)
        return (
            e_sr + jnp.asarray(lr.energy_kcalmol, dtype=e_sr.dtype),
            f_sr + jnp.asarray(lr.forces_kcalmol_A, dtype=f_sr.dtype),
        )

    return wrapped


def build_mm_energy_forces_fn(
    R: np.ndarray,
    *,
    total_atoms: int,
    n_monomers: int,
    monomer_offsets: np.ndarray,
    atoms_per_monomer_list: Sequence[int],
    lambda_monomer: np.ndarray,
    ml_switch_width: float,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_cutoff_distance: float | None = None,
    mm_cutoff: float | None = None,
    complementary_handoff: bool = True,
    ep_scale: Optional[np.ndarray] = None,
    sig_scale: Optional[np.ndarray] = None,
    at_codes_override: Optional[np.ndarray] = None,
    pbc_cell: Optional[np.ndarray] = None,
    max_pairs: Optional[int] = None,
    cell_list_safety_factor: float = 2.5,
    cell_list_density_estimate: Optional[float] = None,
    use_smooth_mic: Optional[bool] = None,
    use_jax_md_neighbor_list: bool = True,
    fractional_coordinates: bool = False,
    mm_r_min: Optional[float] = None,
    jax_md_capacity_multiplier: float = DEFAULT_JAX_MD_CAPACITY_MULTIPLIER,
    jax_md_capacity_growth_factor: float = 1.5,
    jax_md_max_overflow_retries: int = 4,
    jax_md_overflow_fallback_to_cell_list: bool = True,
    jax_md_update_interval: int = 1,
    jax_md_skin_distance: float = DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    mm_nl_backend: str = "auto",
    debug: bool = False,
    ml_compute_dtype: str | None = None,
    defer_xla_gpu_warmup: bool = False,
    lr_solver: str | None = None,
    jax_pme_method: str | None = None,
    jax_pme_sr_cutoff_A: float = 6.0,
) -> Any:
    """Build MM energy/forces function with switching.

    Supports heterogeneous monomer sizes via monomer_offsets and atoms_per_monomer_list.
    Uses cell list for PBC when pbc_cell is provided, otherwise all-pairs.

    Args:
        mm_r_min: Optional inner cutoff (Å). Pairs with dimer COM distance < mm_r_min
            are excluded from the MM neighbor list. Use mm_switch_on to exclude close
            monomers (MM only in switching region). Note: with complementary_handoff,
            mm_scale is nonzero for r in [mm_switch_on - ml_switch_width, mm_switch_on];
            mm_r_min >= mm_switch_on will break the handoff.

    Returns:
        When PBC dynamic neighbor lists are enabled (Vesin rebuild by default, or
        ``mm_nl_backend=jax_md`` for jax-md incremental): ``(mm_fn, update_fn)`` where
        ``mm_fn(positions, pair_idx, pair_mask) -> (energy, forces)`` and
        ``update_fn(positions, box=None) -> (pair_idx, pair_mask)``.
        Otherwise: ``mm_fn(positions) -> (energy, forces)`` (single callable).
    """
    ml_jnp_dtype = resolve_ml_compute_dtype(ml_compute_dtype)
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        pick_lr_solver,
        resolve_jax_pme_method,
    )

    _use_jax_pme_coulomb = pick_lr_solver(lr_solver) == "jax_pme"
    _jax_pme_method = resolve_jax_pme_method(jax_pme_method)
    _jax_pme_sr_cutoff = float(jax_pme_sr_cutoff_A)
    if ml_cutoff_distance is not None:
        ml_switch_width = ml_cutoff_distance
    if mm_cutoff is not None:
        mm_switch_width = mm_cutoff

    if CGENFF_PRM is None or CGENFF_RTF is None:
        raise RuntimeError("CGENFF parameters not available; PyCHARMM may not be initialized")

    import pycharmm.param as param

    def _cgenff_params_loaded() -> bool:
        try:
            atc = param.get_atc()
            return bool(atc) and len(atc) > 0
        except Exception:
            return False

    if not _cgenff_params_loaded():
        from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block
        from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

        reset_block()
        read_cgenff_toppar()
    # Recalibrate XLA delay kernel before hybrid JIT (post-PyCHARMM CGENFF param read).
    # Skip when MPI-linked CHARMM defers GPU until after MLpot SD (CUDA before first
    # gete corrupts OpenMPI registered-memory pools).
    if not defer_xla_gpu_warmup:
        from mmml.utils.jax_gpu_warmup import ensure_xla_gpu_warmed

        ensure_xla_gpu_warmed(force=True)

    cgenff_rtf = open(CGENFF_RTF).readlines()
    atc = param.get_atc()
    cgenff_params_dict_q = {}
    for _ in cgenff_rtf:
        if _.startswith("ATOM"):
            _, atomname, at, q = _.split()[:4]
            try:
                cgenff_params_dict_q[at] = float(q)
            except ValueError:
                cgenff_params_dict_q[at] = float(q.split("!")[0])

    cgenff_params_dict = {}
    for p in open(CGENFF_PRM).readlines():
        if len(p) > 5 and len(p.split()) > 4 and p.split()[1] == "0.0" and p[0] != "!":
            res, _, ep, sig = p.split()[:4]
            cgenff_params_dict[res] = (float(ep), float(sig))

    atc_epsilons = np.array([cgenff_params_dict.get(_, (0.0, 0.0))[0] for _ in atc])
    atc_rmins = np.array([cgenff_params_dict.get(_, (0.0, 0.0))[1] for _ in atc])
    atc_qs = np.array([cgenff_params_dict_q.get(_, 0.0) for _ in atc])

    if ep_scale is None:
        ep_scale = np.ones_like(atc_epsilons)
    if sig_scale is None:
        sig_scale = np.ones_like(atc_epsilons)

    at_ep = -1 * np.abs(atc_epsilons) * ep_scale
    at_rm = atc_rmins * sig_scale
    np.array(atc_qs)
    at_flat_ep = np.array(at_ep)
    at_flat_rm = np.array(at_rm)

    _dp = _dimer_permutations(n_monomers)
    _offsets_np = np.array([int(monomer_offsets[k]) for k in range(len(monomer_offsets))])
    _monomer_id_np = np.empty(total_atoms, dtype=np.int32)
    for mi in range(n_monomers):
        _monomer_id_np[_offsets_np[mi] : _offsets_np[mi + 1]] = mi
    # Smooth MIC avoids discontinuities at cell boundaries during minimization
    _use_smooth_mic = use_smooth_mic if use_smooth_mic is not None else (pbc_cell is not None)
    _resolved_nl_backend = resolve_mm_nl_backend(mm_nl_backend)
    _use_jax_md_nbrs = (
        pbc_cell is not None
        and use_jax_md_neighbor_list
        and have_jax_md()
        and _resolved_nl_backend == "jax_md"
    )
    _use_rebuild_nbrs = (
        pbc_cell is not None
        and not _use_jax_md_nbrs
        and (_cell_list_pairs is not None or have_vesin())
    )
    _use_dynamic_nbrs = _use_jax_md_nbrs or _use_rebuild_nbrs
    _pair_idx_cell = [None]
    _pair_mask_cell = [None]

    def _create_jax_md_bundle(capacity_multiplier: float):
        return create_jax_md_neighbor_list(
            np.asarray(pbc_cell),
            r_cutoff=mm_switch_on + mm_switch_width,
            monomer_offsets=np.asarray(monomer_offsets),
            dr_threshold=0.5,
            capacity_multiplier=capacity_multiplier,
            fractional_coordinates=fractional_coordinates,
        )

    if _use_jax_md_nbrs:
        # jax_md path: pair list is updated externally; mm_fn will accept pair_idx, pair_mask
        jax_md_result = _create_jax_md_bundle(float(jax_md_capacity_multiplier))
        if jax_md_result is not None:
            _neighbor_fn, _filter_fn, _monomer_id_jnp = jax_md_result
            _neighbor_fn_cell = [_neighbor_fn]
            _filter_fn_cell = [_filter_fn]
            _current_capacity_multiplier = [float(jax_md_capacity_multiplier)]
            _nbrs = [None]  # mutable cell for neighbor list state
        else:
            _use_jax_md_nbrs = False
            _use_rebuild_nbrs = pbc_cell is not None and (
                _cell_list_pairs is not None or have_vesin()
            )
            _use_dynamic_nbrs = _use_jax_md_nbrs or _use_rebuild_nbrs

    if _use_rebuild_nbrs:
        _mm_switch_width_dist = mm_switch_on + mm_switch_width
        _static_backend = pick_static_rebuild_backend(
            mm_nl_backend,
            use_jax_md_neighbor_list=False,
        )
        _backend_req = "cell_list" if _static_backend == "cell_list" else "vesin"
        _cl_i, _cl_j, _cl_mask, _n_valid, _max_pairs, _nl_used = build_mm_pairs_with_backend(
            _backend_req,
            positions=np.asarray(R),
            box=np.asarray(pbc_cell),
            cutoff=_mm_switch_width_dist,
            monomer_offsets=_offsets_np,
            atoms_per_monomer_list=atoms_per_monomer_list,
            mm_r_min=mm_r_min,
            max_pairs=max_pairs,
            cell_list_safety_factor=cell_list_safety_factor,
            cell_list_density_estimate=cell_list_density_estimate,
            total_atoms=total_atoms,
            debug=debug,
        )
        if debug:
            print(
                f"[get_MM] {_nl_used} backend: {_n_valid} valid pairs "
                f"out of max_pairs={_max_pairs}"
            )
        pair_idx_atom_atom = jnp.stack([_cl_i, _cl_j], axis=1)
        _cl_mask_jnp = jnp.asarray(_cl_mask, dtype=ml_jnp_dtype)
        _pair_idx_cell[0] = pair_idx_atom_atom
        _pair_mask_cell[0] = _cl_mask_jnp

        _monomer_id_jnp = jnp.array(_monomer_id_np)
        _dimer_lookup_arr = np.full((n_monomers, n_monomers), -1, dtype=np.int32)
        for di, (mi, mj) in enumerate(_dp):
            _dimer_lookup_arr[mi, mj] = _dimer_lookup_arr[mj, mi] = di
        _dimer_lookup_arr = jnp.array(_dimer_lookup_arr)
        pair_dimer_idx = None
        n_pairs_per_dimer_arr = np.zeros(len(_dp), dtype=np.int32)
    elif _use_jax_md_nbrs:
        _mm_switch_width_dist = mm_switch_on + mm_switch_width
        nbrs_init = _neighbor_fn_cell[0].allocate(np.asarray(R))
        _nbrs[0] = nbrs_init
        idx = nbrs_init.idx
        pair_i, pair_j, mask = _filter_fn_cell[0](idx)
        _max_pairs = idx.shape[1]
        if debug:
            n_valid_init = int(np.sum(np.asarray(jax.device_get(mask))))
            print(f"[nbr] allocate: capacity={_max_pairs}, n_valid={n_valid_init}, "
                  f"frac_coords={fractional_coordinates}, r_cutoff={mm_switch_on + mm_switch_width:.2f}")
        pair_idx_atom_atom = jnp.stack([pair_i, pair_j], axis=1)
        _cl_mask_jnp = jnp.asarray(mask, dtype=ml_jnp_dtype)
        _pair_idx_cell[0] = pair_idx_atom_atom
        _pair_mask_cell[0] = _cl_mask_jnp
        _monomer_id_jnp = jnp.array(_monomer_id_np)
        _dimer_lookup_arr = np.full((n_monomers, n_monomers), -1, dtype=np.int32)
        for di, (mi, mj) in enumerate(_dp):
            _dimer_lookup_arr[mi, mj] = _dimer_lookup_arr[mj, mi] = di
        _dimer_lookup_arr = jnp.array(_dimer_lookup_arr)
        pair_dimer_idx = None
        n_pairs_per_dimer_arr = np.zeros(len(_dp), dtype=np.int32)
    else:
        pair_idx_list = []
        pair_lambda_list = []
        n_pairs_per_dimer_list = []
        for mi, mj in _dp:
            off_i = int(monomer_offsets[mi])
            off_j = int(monomer_offsets[mj])
            n_i = atoms_per_monomer_list[mi]
            n_j = atoms_per_monomer_list[mj]
            local_pairs = np.array(
                [(a + off_i, b + off_j) for a in range(n_i) for b in range(n_j)],
                dtype=np.int32,
            )
            pair_idx_list.append(local_pairs)
            n_pairs_per_dimer_list.append(len(local_pairs))
            lam_ij = float(lambda_monomer[mi] * lambda_monomer[mj])
            pair_lambda_list.append(np.full(len(local_pairs), lam_ij))

        pair_idx_atom_atom = jnp.array(np.concatenate(pair_idx_list, axis=0))
        pair_lambda_mm = jnp.array(np.concatenate(pair_lambda_list))
        n_pairs_per_dimer_arr = np.array(n_pairs_per_dimer_list, dtype=np.int32)
        _cl_mask_jnp = None
        pair_dimer_idx = None

    import pycharmm.psf as psf
    charges_full = _get_actual_psf_charges(total_atoms)
    at_codes_full = np.array(psf.get_iac(), dtype=np.int32)
    if charges_full.size == 0:
        raise RuntimeError(
            "PyCHARMM PSF has no atoms. Read or generate a PSF for this system before "
            "building MM energy/forces (e.g. setupRes/setupBox or read.psf)."
        )
    if charges_full.size < total_atoms:
        raise RuntimeError(
            f"PyCHARMM PSF has {charges_full.size} atoms but the calculator expects "
            f"{total_atoms}. Regenerate the PSF for the current system."
        )
    charges = charges_full[:total_atoms]
    at_codes = at_codes_full[:total_atoms]
    if at_codes_override is not None:
        at_codes_override_arr = np.array(at_codes_override)
        if at_codes_override_arr.shape[0] != at_codes.shape[0]:
            raise ValueError(
                f"at_codes_override length {at_codes_override_arr.shape[0]} "
                f"does not match expected {at_codes.shape[0]}"
            )
        at_codes = at_codes_override_arr

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
    q_per_system = jnp.array(charges)

    _n_static_pairs = int(pair_idx_atom_atom.shape[0])
    if not _use_dynamic_nbrs and (_n_static_pairs == 0 or int(q_per_system.shape[0]) == 0):

        @jax.jit
        def calculate_mm_energy_and_forces(positions: Array) -> Tuple[Array, Array]:
            return jnp.array(0.0, dtype=ml_jnp_dtype), jnp.zeros(
                (total_atoms, 3), dtype=ml_jnp_dtype
            )

        return calculate_mm_energy_and_forces

    if not _use_dynamic_nbrs:
        q_a = jnp.take(q_per_system, pair_idx_atom_atom[:, 0])
        q_b = jnp.take(q_per_system, pair_idx_atom_atom[:, 1])
        rm_a = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 0])
        rm_b = jnp.take(rmins_per_system, pair_idx_atom_atom[:, 1])
        ep_a = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 0])
        ep_b = jnp.take(epsilons_per_system, pair_idx_atom_atom[:, 1])

        pair_qq = q_a * q_b * pair_lambda_mm
        pair_rm = rm_a + rm_b
        pair_ep = (ep_a * ep_b) ** 0.5 * pair_lambda_mm

    def lennard_jones(r: Array, sig: Array, ep: Array) -> Array:
        lj_epsilon = 1e-10
        r_safe = jnp.maximum(r, lj_epsilon)
        r6 = (sig / r_safe) ** 6
        return ep * (r6 ** 2 - 2 * r6)

    coulombs_constant = 3.32063711e2
    coulomb_epsilon = 1e-10

    def coulomb(r: Array, qq: Array, constant: float = coulombs_constant, eps: float = coulomb_epsilon) -> Array:
        r_safe = jnp.maximum(r, eps)
        return constant * qq / r_safe

    _dimer_perms_np = jnp.array(_dp)

    _mm_r_min = mm_r_min  # capture for closure

    def get_switching_function(
        ml_switch_width: float = ml_switch_width,
        mm_switch_on: float = mm_switch_on,
        mm_switch_width: float = mm_switch_width,
        complementary_handoff: bool = complementary_handoff,
    ) -> Callable[..., Array]:
        def apply_switching_function(
            positions: Array,
            pair_energies: Array,
            pair_dimer_idx_arg: Optional[Array] = None,
            box_override: Optional[Array] = None,
        ) -> Array:
            coms = jnp.stack([
                positions[monomer_offsets[k]:monomer_offsets[k + 1]].mean(axis=0)
                for k in range(n_monomers)
            ])
            com_i = coms[_dimer_perms_np[:, 0]]
            com_j = coms[_dimer_perms_np[:, 1]]
            cell_for_com = box_override if box_override is not None else pbc_cell
            if cell_for_com is not None:
                mic_fn = mic_displacement_smooth if _use_smooth_mic else mic_displacement
                d_vec = jax.vmap(lambda a, b: mic_fn(a, b, cell_for_com))(com_i, com_j)
                r = jnp.linalg.norm(d_vec, axis=1)
            else:
                r = jnp.linalg.norm(com_j - com_i, axis=1)
            if _mm_r_min is not None:
                r_min_mask = (r >= _mm_r_min)
            else:
                r_min_mask = None
            if complementary_handoff:
                handoff = _sharpstep(r, mm_switch_on - ml_switch_width, mm_switch_on, gamma=GAMMA_ON)
                mm_taper = 1.0 - _sharpstep(r, mm_switch_on, mm_switch_on + mm_switch_width, gamma=GAMMA_OFF)
                mm_scale = handoff * mm_taper
            else:
                mm_on = _sharpstep(r, mm_switch_on, mm_switch_on + mm_switch_width, gamma=GAMMA_ON)
                mm_off = _sharpstep(r, mm_switch_on + mm_switch_width, mm_switch_on + 2.0 * mm_switch_width, gamma=GAMMA_OFF)
                mm_scale = mm_on * (1.0 - mm_off)
            if r_min_mask is not None:
                mm_scale = mm_scale * r_min_mask.astype(mm_scale.dtype)

            use_dimer_idx = (pair_dimer_idx_arg is not None) or (pair_dimer_idx is not None)
            pdi = pair_dimer_idx_arg if pair_dimer_idx_arg is not None else pair_dimer_idx
            if use_dimer_idx and pdi is not None:
                mm_scale_with_dummy = jnp.concatenate([mm_scale, jnp.zeros(1)])
                safe_idx = jnp.where(pdi >= 0, pdi, len(mm_scale))
                mm_scale_expanded = mm_scale_with_dummy[safe_idx]
            else:
                mm_scale_expanded = jnp.concatenate([
                    jnp.full((int(n_pairs_per_dimer_arr[d]),), mm_scale[d])
                    for d in range(len(_dp))
                ])
            return (pair_energies * mm_scale_expanded).sum()
        return apply_switching_function

    apply_switching_function = get_switching_function()

    def calculate_mm_pair_energies(positions: Array) -> Array:
        if pbc_cell is not None:
            pos_dst = positions[pair_idx_atom_atom[:, 1]]
            pos_src = positions[pair_idx_atom_atom[:, 0]]
            mic_batched = mic_displacements_batched_smooth if _use_smooth_mic else mic_displacements_batched
            displacements = mic_batched(pos_dst, pos_src, pbc_cell)
        else:
            displacements = positions[pair_idx_atom_atom[:, 0]] - positions[pair_idx_atom_atom[:, 1]]
        distances = jnp.linalg.norm(displacements, axis=1)

        if _cl_mask_jnp is not None:
            distances = jnp.where(_cl_mask_jnp > 0, distances, 1e6)

        pair_mask = (pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1])
        vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
        if _use_jax_pme_coulomb:
            return vdw_energies
        electrostatic_energies = coulomb(distances, pair_qq) * pair_mask
        return vdw_energies + electrostatic_energies

    def switched_mm_energy(positions: Array) -> Array:
        pair_energies = calculate_mm_pair_energies(positions)
        return apply_switching_function(positions, pair_energies)

    switched_mm_grad = jax.grad(switched_mm_energy)

    @jax.jit
    def calculate_mm_energy_and_forces(positions: Array) -> Tuple[Array, Array]:
        pair_energies = calculate_mm_pair_energies(positions)
        switched_energy = apply_switching_function(positions, pair_energies)
        forces = -1.0 * switched_mm_grad(positions)
        forces = jnp.where(jnp.isfinite(forces), forces, 0.0)
        return switched_energy, forces

    if _use_dynamic_nbrs:
        # Dynamic path: compute pair quantities from pair_idx, pair_mask
        _pbc_cell_jnp = jnp.asarray(pbc_cell)
        _lambda_monomer_jnp = jnp.asarray(lambda_monomer)
        _pair_stats = {
            "calls": 0,
            "updates": 0,
            "reused": 0,
            "reallocs": 0,
            "fallbacks": 0,
            "cache_checks": 0,
            "host_syncs": 0,
            "device_skin_checks": 0,
            "cpu_rebuilds": 0,
            "gpu_rebuilds": 0,
            "capacity_grows": 0,
            "com_filter_calls": 0,
            "capacity_multiplier": float(jax_md_capacity_multiplier),
            "pair_capacity": int(_n_static_pairs),
            "pair_capacity_initial": int(_n_static_pairs),
            "pair_capacity_changes": 0,
            "pair_capacity_history": [int(_n_static_pairs)],
            "update_interval": int(max(1, jax_md_update_interval)),
            "skin_distance": float(max(0.0, jax_md_skin_distance)),
            "cache_reuse_reason": "init",
            "last_reuse_reason": "init",
        }
        _last_positions = [None]
        _last_cartesian_positions = [None]
        _last_cartesian_positions_jax = [None]
        _last_box = [None]
        _fallback_max_pairs_cell = [int(_n_static_pairs)]

        def _record_pair_capacity(capacity: int, reason: str) -> None:
            cap = int(capacity)
            prev = int(_pair_stats.get("pair_capacity", cap))
            if cap == prev:
                return
            _pair_stats["pair_capacity"] = cap
            _pair_stats["pair_capacity_changes"] = int(
                _pair_stats.get("pair_capacity_changes", 0)
            ) + 1
            history = list(_pair_stats.get("pair_capacity_history", []))
            history.append(cap)
            _pair_stats["pair_capacity_history"] = history[-16:]
            _pair_stats["last_capacity_change_reason"] = reason
            if os.environ.get("MMML_MM_NL_STRICT_CAPACITY") == "1":
                raise RuntimeError(
                    f"MM pair-list capacity changed from {prev} to {cap} ({reason}); "
                    "this can trigger JAX recompilation"
                )

        def calculate_mm_pair_energies_dynamic(
            positions: Array,
            pair_idx: Array,
            pair_mask: Array,
            cell_for_mic: Optional[Array] = None,
        ) -> Array:
            pair_i = pair_idx[:, 0]
            pair_j = pair_idx[:, 1]
            lam_a = jnp.take(_lambda_monomer_jnp, _monomer_id_jnp[pair_i])
            lam_b = jnp.take(_lambda_monomer_jnp, _monomer_id_jnp[pair_j])
            pair_lambda_mm_dyn = lam_a * lam_b * pair_mask

            q_a = jnp.take(q_per_system, pair_i)
            q_b = jnp.take(q_per_system, pair_j)
            rm_a = jnp.take(rmins_per_system, pair_i)
            rm_b = jnp.take(rmins_per_system, pair_j)
            ep_a = jnp.take(epsilons_per_system, pair_i)
            ep_b = jnp.take(epsilons_per_system, pair_j)
            pair_qq_dyn = q_a * q_b * pair_lambda_mm_dyn
            pair_rm_dyn = rm_a + rm_b
            pair_ep_dyn = (ep_a * ep_b) ** 0.5 * pair_lambda_mm_dyn

            _cell_raw = cell_for_mic if cell_for_mic is not None else _pbc_cell_jnp
            _cell = _box_to_cell_3x3(_cell_raw)
            pos_dst = positions[pair_j]
            pos_src = positions[pair_i]
            mic_batched = mic_displacements_batched_smooth if _use_smooth_mic else mic_displacements_batched
            displacements = mic_batched(pos_dst, pos_src, _cell)
            distances = jnp.linalg.norm(displacements, axis=1)
            distances = jnp.where(pair_mask > 0, distances, 1e6)

            pair_mask_ij = (pair_i < pair_j)
            vdw = lennard_jones(distances, pair_rm_dyn, pair_ep_dyn) * pair_mask_ij
            if _use_jax_pme_coulomb:
                return vdw
            elec = coulomb(distances, pair_qq_dyn) * pair_mask_ij
            return vdw + elec

        def _mm_dynamic_energy_scalar(
            positions: Array,
            pair_idx: Array,
            pair_mask: Array,
            cell_for_mic: Array,
        ) -> Array:
            pair_i = pair_idx[:, 0]
            pair_j = pair_idx[:, 1]
            mid_i = _monomer_id_jnp[pair_i]
            mid_j = _monomer_id_jnp[pair_j]
            pair_dimer_idx_dyn = _dimer_lookup_arr[mid_i, mid_j]
            pair_energies = calculate_mm_pair_energies_dynamic(
                positions, pair_idx, pair_mask, cell_for_mic=cell_for_mic
            )
            return apply_switching_function(
                positions,
                pair_energies,
                pair_dimer_idx_arg=pair_dimer_idx_dyn,
                box_override=cell_for_mic,
            )

        _mm_dynamic_value_and_grad = jax.jit(
            jax.value_and_grad(_mm_dynamic_energy_scalar, argnums=0),
        )

        @jax.jit
        def calculate_mm_energy_and_forces_dynamic(
            positions: Array,
            pair_idx: Array,
            pair_mask: Array,
            box_override: Optional[Array] = None,
        ) -> Tuple[Array, Array]:
            _cell_raw = box_override if box_override is not None else _pbc_cell_jnp
            _cell_for_mic = _box_to_cell_3x3(_cell_raw)
            switched_energy, grad = _mm_dynamic_value_and_grad(
                positions,
                pair_idx,
                pair_mask,
                _cell_for_mic,
            )
            forces = -1.0 * grad
            forces = jnp.where(jnp.isfinite(forces), forces, 0.0)
            switched_energy = jnp.where(jnp.isfinite(switched_energy), switched_energy, 0.0)
            return switched_energy, forces

        def _fallback_backend_request() -> str:
            resolved = resolve_mm_nl_backend(mm_nl_backend)
            if resolved == "cell_list":
                return "cell_list"
            if resolved == "vesin":
                return "vesin"
            return "vesin"

        def _cell_list_fallback_pairs(
            positions_in: np.ndarray,
            _nbr_debug: bool,
            box_in: Optional[np.ndarray] = None,
        ) -> Tuple[Array, Array]:
            cutoff = mm_switch_on + mm_switch_width
            current_pbc_cell = pbc_cell
            if box_in is not None:
                box_np = np.asarray(box_in, dtype=np.float64)
                if box_np.ndim == 1:
                    current_pbc_cell = np.diag(box_np)
                else:
                    current_pbc_cell = box_np
            try:
                cl_i, cl_j, cl_mask, _, fallback_max_pairs, used = build_mm_pairs_with_backend(
                    _fallback_backend_request(),
                    positions=np.asarray(positions_in, dtype=np.float64),
                    box=np.asarray(current_pbc_cell),
                    cutoff=cutoff,
                    monomer_offsets=_offsets_np,
                    atoms_per_monomer_list=atoms_per_monomer_list,
                    mm_r_min=mm_r_min,
                    max_pairs=_fallback_max_pairs_cell[0],
                    cell_list_safety_factor=max(float(cell_list_safety_factor), 4.0),
                    cell_list_density_estimate=cell_list_density_estimate,
                    total_atoms=total_atoms,
                    debug=_nbr_debug,
                )
            except Exception as exc:
                if (
                    not jax_md_overflow_fallback_to_cell_list
                    or _cell_list_pairs is None
                ):
                    raise RuntimeError(
                        "Neighbor list failed and rebuild fallback is unavailable"
                    ) from exc
                cl_i, cl_j, cl_mask, _, fallback_max_pairs = _build_cell_list_pairs_with_retry(
                    positions=np.asarray(positions_in, dtype=np.float64),
                    pbc_cell=np.asarray(current_pbc_cell),
                    cutoff=cutoff,
                    max_pairs=_fallback_max_pairs_cell[0],
                    monomer_offsets=_offsets_np,
                    atoms_per_monomer_list=atoms_per_monomer_list,
                    total_atoms=total_atoms,
                    cell_list_safety_factor=max(float(cell_list_safety_factor), 4.0),
                    cell_list_density_estimate=cell_list_density_estimate,
                    debug=_nbr_debug,
                )
                used = "cell_list"
                if mm_r_min is not None:
                    cl_mask = _filter_pairs_by_com_min(
                        np.asarray(positions_in, dtype=np.float64),
                        np.asarray(cl_i),
                        np.asarray(cl_j),
                        np.asarray(cl_mask, dtype=bool),
                        _offsets_np,
                        np.asarray(_monomer_id_jnp),
                        mm_r_min,
                        pbc_cell=np.asarray(current_pbc_cell) if current_pbc_cell is not None else None,
                    )
            _fallback_max_pairs_cell[0] = int(fallback_max_pairs)
            _record_pair_capacity(int(fallback_max_pairs), f"fallback_{used}")
            if _nbr_debug:
                print(f"[nbr] fallback via {used} max_pairs={fallback_max_pairs}")
            _pair_stats["fallbacks"] += 1
            return (
                jnp.stack([jnp.asarray(cl_i), jnp.asarray(cl_j)], axis=1),
                jnp.asarray(cl_mask, dtype=ml_jnp_dtype),
            )

        def _cartesian_for_nl_build(
            positions_in: np.ndarray,
            box_in: Optional[np.ndarray],
        ) -> np.ndarray:
            R_np = np.asarray(positions_in, dtype=np.float64)
            if not fractional_coordinates:
                return R_np
            if box_in is not None:
                box_np = np.asarray(box_in, dtype=np.float64)
                cell_3x3 = np.diag(box_np) if box_np.ndim == 1 else box_np
                return np.asarray(R_np @ cell_3x3, dtype=np.float64)
            cell_np = np.asarray(pbc_cell, dtype=np.float64)
            if cell_np.ndim == 0:
                cell_3x3 = np.diag([float(cell_np)] * 3)
            elif cell_np.shape == (3,):
                cell_3x3 = np.diag(cell_np)
            else:
                cell_3x3 = cell_np
            return np.asarray(R_np @ cell_3x3, dtype=np.float64)

        def _pbc_cell_for_nl_build(box_in: Optional[np.ndarray]) -> np.ndarray:
            if box_in is not None:
                box_np = np.asarray(box_in, dtype=np.float64)
                return np.diag(box_np) if box_np.ndim == 1 else box_np
            return np.asarray(pbc_cell, dtype=np.float64)

        def _jax_cartesian_for_nl_build(positions_in: Array, box_in: Optional[np.ndarray]) -> Array:
            pos = jnp.asarray(positions_in)
            if not fractional_coordinates:
                return pos
            if box_in is not None:
                cell = _box_to_cell_3x3(jnp.asarray(box_in, dtype=pos.dtype))
            else:
                cell = _box_to_cell_3x3(jnp.asarray(pbc_cell, dtype=pos.dtype))
            return pos @ cell

        @jax.jit
        def _max_cartesian_displacement(current: Array, previous: Array) -> Array:
            return jnp.max(jnp.linalg.norm(current - previous, axis=1))

        def _box_delta_within_tolerance(box_in: Optional[np.ndarray]) -> bool:
            if box_in is None or _last_box[0] is None:
                return box_in is None and _last_box[0] is None
            box_delta = float(np.max(np.abs(np.asarray(box_in) - np.asarray(_last_box[0]))))
            return box_delta <= 1e-8

        def _gpu_interval_reuse_allowed(box_in: Optional[np.ndarray], interval: int, skin: float) -> bool:
            if not (
                _pair_idx_cell[0] is not None and _pair_mask_cell[0] is not None
            ):
                return False
            if _pair_stats["calls"] % int(max(1, interval)) == 0:
                return False
            return _box_delta_within_tolerance(box_in)

        def _rebuild_pairs_with_static_backend(
            positions_in: np.ndarray,
            box_in: Optional[np.ndarray],
            _nbr_debug: bool,
            *,
            positions_jax=None,
        ) -> Tuple[Array, Array]:
            from mmml.interfaces.pycharmmInterface.nl_gpu import (
                gpu_nl_path_available,
                rebuild_vesin_pairs_gpu,
            )

            if (
                gpu_nl_path_available()
                and positions_jax is not None
                and hasattr(positions_jax, "__dlpack_device__")
            ):
                pbc_for_build = _pbc_cell_for_nl_build(box_in)
                pos_for_gpu = _jax_cartesian_for_nl_build(positions_jax, box_in)
                while True:
                    try:
                        pair_idx, pair_mask, used = rebuild_vesin_pairs_gpu(
                            pos_for_gpu,
                            pbc_for_build,
                            cutoff=mm_switch_on + mm_switch_width,
                            monomer_offsets=_offsets_np,
                            mm_r_min=mm_r_min,
                            max_pairs=_fallback_max_pairs_cell[0],
                            cell_list_safety_factor=cell_list_safety_factor,
                            cell_list_density_estimate=cell_list_density_estimate,
                            total_atoms=total_atoms,
                            debug=_nbr_debug,
                        )
                        break
                    except PairListTruncationError as exc:
                        _fallback_max_pairs_cell[0] = int(exc.suggested_max_pairs)
                        _pair_stats["capacity_grows"] += 1
                        _record_pair_capacity(
                            int(_fallback_max_pairs_cell[0]),
                            "gpu_pair_truncation_growth",
                        )
                if _nbr_debug:
                    print(f"[nbr] rebuild via {used}")
                    _validate_dynamic_pair_contract(
                        pair_idx,
                        pair_mask,
                        total_atoms=total_atoms,
                        label=used,
                    )
                _pair_stats["gpu_rebuilds"] += 1
                return pair_idx, pair_mask

            R_build = _cartesian_for_nl_build(positions_in, box_in)
            pbc_for_build = _pbc_cell_for_nl_build(box_in)
            backend_req = pick_static_rebuild_backend(
                mm_nl_backend,
                use_jax_md_neighbor_list=False,
            )
            if backend_req == "jax_md":
                backend_req = "vesin" if have_vesin() else "cell_list"
            while True:
                try:
                    cl_i, cl_j, cl_mask, n_valid, capacity, used = build_mm_pairs_with_backend(
                        backend_req,
                        positions=R_build,
                        box=pbc_for_build,
                        cutoff=mm_switch_on + mm_switch_width,
                        monomer_offsets=_offsets_np,
                        atoms_per_monomer_list=atoms_per_monomer_list,
                        mm_r_min=mm_r_min,
                        max_pairs=_fallback_max_pairs_cell[0],
                        cell_list_safety_factor=cell_list_safety_factor,
                        cell_list_density_estimate=cell_list_density_estimate,
                        total_atoms=total_atoms,
                        debug=_nbr_debug,
                    )
                    break
                except PairListTruncationError as exc:
                    _fallback_max_pairs_cell[0] = int(exc.suggested_max_pairs)
                    _pair_stats["capacity_grows"] += 1
                    _record_pair_capacity(
                        int(_fallback_max_pairs_cell[0]),
                        "cpu_pair_truncation_growth",
                    )
            if int(capacity) != int(_fallback_max_pairs_cell[0]):
                if int(capacity) > int(_fallback_max_pairs_cell[0]):
                    _pair_stats["capacity_grows"] += 1
                _fallback_max_pairs_cell[0] = int(capacity)
                _record_pair_capacity(int(capacity), f"{used}_returned_capacity")
            if _nbr_debug:
                print(f"[nbr] rebuild via {used}: n_valid={n_valid} capacity={capacity}")
            pair_idx_out = jnp.stack([jnp.asarray(cl_i), jnp.asarray(cl_j)], axis=1)
            pair_mask_out = jnp.asarray(cl_mask, dtype=ml_jnp_dtype)
            if _nbr_debug:
                _validate_dynamic_pair_contract(
                    pair_idx_out,
                    pair_mask_out,
                    total_atoms=total_atoms,
                    label=used,
                )
            _pair_stats["cpu_rebuilds"] += 1
            return (
                pair_idx_out,
                pair_mask_out,
            )

        def update_mm_pairs(positions: np.ndarray, box: Optional[np.ndarray] = None) -> Tuple[Array, Array]:
            """Return padded dynamic MM pairs.

            Contract: callers pass Cartesian positions unless ``fractional_coordinates``
            was configured, in which case positions are fractional and ``box`` supplies
            the current cell. Results have stable shape ``(capacity, 2)`` for
            ``pair_idx`` and ``(capacity,)`` for ``pair_mask`` until an explicit
            capacity grow is required.
            """
            positions_jax = positions if hasattr(positions, "__dlpack_device__") else None
            _nbr_debug = debug
            _pair_stats["calls"] += 1

            interval = int(max(1, jax_md_update_interval))
            skin = float(max(0.0, jax_md_skin_distance))

            have_cache = (
                _pair_idx_cell[0] is not None
                and _pair_mask_cell[0] is not None
            )

            if _use_rebuild_nbrs and positions_jax is not None:
                from mmml.interfaces.pycharmmInterface.nl_gpu import gpu_nl_path_available

                if gpu_nl_path_available():
                    _pair_stats["cache_checks"] += 1
                    if _gpu_interval_reuse_allowed(box, interval, skin):
                        _pair_stats["reused"] += 1
                        _pair_stats["cache_reuse_reason"] = "gpu_interval_box_stable"
                        _pair_stats["last_reuse_reason"] = "gpu_interval_box_stable"
                        return _pair_idx_cell[0], _pair_mask_cell[0]

                    pair_idx, pair_mask = _rebuild_pairs_with_static_backend(
                        np.empty((0, 3), dtype=np.float64),
                        box,
                        _nbr_debug,
                        positions_jax=positions_jax,
                    )
                    _pair_idx_cell[0] = pair_idx
                    _pair_mask_cell[0] = pair_mask
                    _last_cartesian_positions[0] = None
                    _last_cartesian_positions_jax[0] = None
                    _last_box[0] = None if box is None else np.asarray(box, dtype=np.float64).copy()
                    _pair_stats["updates"] += 1
                    _pair_stats["last_reuse_reason"] = (
                        "gpu_rebuild_no_host_skin_cache"
                        if skin > 0.0
                        else "gpu_rebuild"
                    )
                    _pair_stats["cache_reuse_reason"] = _pair_stats["last_reuse_reason"]
                    _record_pair_capacity(int(pair_idx.shape[0]), "gpu_rebuild_shape")
                    return pair_idx, pair_mask

            if (
                _use_rebuild_nbrs
                and positions_jax is not None
                and have_cache
                and skin > 0.0
                and _last_cartesian_positions_jax[0] is not None
                and _box_delta_within_tolerance(box)
            ):
                _pair_stats["cache_checks"] += 1
                if _gpu_interval_reuse_allowed(box, interval, skin):
                    _pair_stats["reused"] += 1
                    _pair_stats["cache_reuse_reason"] = "device_interval"
                    _pair_stats["last_reuse_reason"] = "device_interval"
                    return _pair_idx_cell[0], _pair_mask_cell[0]
                _pair_stats["device_skin_checks"] += 1
                R_cart_jax = _jax_cartesian_for_nl_build(positions_jax, box)
                max_disp = _max_cartesian_displacement(
                    R_cart_jax,
                    _last_cartesian_positions_jax[0],
                )
                if bool(np.asarray(jax.device_get(max_disp <= skin))):
                    _pair_stats["reused"] += 1
                    _pair_stats["cache_reuse_reason"] = "device_skin"
                    _pair_stats["last_reuse_reason"] = "device_skin"
                    return _pair_idx_cell[0], _pair_mask_cell[0]

            if positions_jax is not None:
                _pair_stats["host_syncs"] += 1
                positions_np = np.asarray(jax.device_get(positions), dtype=np.float64)
            else:
                positions_np = np.asarray(positions, dtype=np.float64)
            R_cart = _cartesian_for_nl_build(positions_np, box)

            # Skin/interval reuse on Cartesian coords (skip fractional wrap + inv on hot path).
            _pair_stats["cache_checks"] += 1
            if neighbor_pair_cache_should_reuse(
                calls=_pair_stats["calls"],
                interval=interval,
                skin=skin,
                R=R_cart,
                last_R=_last_cartesian_positions[0],
                box=box,
                last_box=_last_box[0],
                have_cache=have_cache,
            ):
                _pair_stats["reused"] += 1
                if (
                    positions_jax is not None
                    and skin > 0.0
                    and _last_cartesian_positions_jax[0] is None
                    and _last_cartesian_positions[0] is not None
                ):
                    _last_cartesian_positions_jax[0] = jnp.asarray(
                        _last_cartesian_positions[0],
                        dtype=jnp.asarray(positions_jax).dtype,
                    )
                _pair_stats["cache_reuse_reason"] = "cpu_skin_or_interval"
                _pair_stats["last_reuse_reason"] = "cpu_skin_or_interval"
                return _pair_idx_cell[0], _pair_mask_cell[0]

            if _use_rebuild_nbrs:
                pair_idx, pair_mask = _rebuild_pairs_with_static_backend(
                    positions_np,
                    box,
                    _nbr_debug,
                    positions_jax=positions_jax,
                )
                _pair_idx_cell[0] = pair_idx
                _pair_mask_cell[0] = pair_mask
                _last_cartesian_positions[0] = R_cart.copy()
                _last_cartesian_positions_jax[0] = (
                    _jax_cartesian_for_nl_build(positions_jax, box)
                    if positions_jax is not None
                    else None
                )
                _last_box[0] = None if box is None else np.asarray(box, dtype=np.float64).copy()
                _pair_stats["updates"] += 1
                _pair_stats["cache_reuse_reason"] = "cpu_rebuild"
                _pair_stats["last_reuse_reason"] = "cpu_rebuild"
                _record_pair_capacity(int(pair_idx.shape[0]), "cpu_rebuild_shape")
                if _nbr_debug:
                    n_valid = int(np.sum(np.asarray(jax.device_get(pair_mask))))
                    capacity = int(pair_idx.shape[0])
                    print(
                        f"[nbr] vesin rebuild: n_valid={n_valid}, capacity={capacity}, "
                        f"frac_coords={fractional_coordinates}"
                    )
                return pair_idx, pair_mask

            R = positions_np
            if fractional_coordinates and box is None:
                R, box = _fractional_positions_for_jax_md_neighbor_list(positions_np, pbc_cell)

            nbrs = _nbrs[0]
            kwargs = {} if (box is None or not fractional_coordinates) else {"box": jnp.asarray(box)}

            try:
                nbrs = nbrs.update(R, **kwargs)
            except Exception as e:
                if _nbr_debug:
                    print(f"[nbr] update failed before overflow check ({type(e).__name__}): {e}")
                return _cell_list_fallback_pairs(
                    np.asarray(positions, dtype=np.float64),
                    _nbr_debug,
                    box_in=box,
                )

            realloc_count = 0
            for _ in range(int(jax_md_max_overflow_retries)):
                overflow = np.asarray(jax.device_get(nbrs.did_buffer_overflow))
                did_overflow = bool(overflow) if overflow.ndim == 0 else bool(overflow.any())

                if _nbr_debug:
                    print(
                        f"[nbr] update: overflow={did_overflow}, realloc={realloc_count}, "
                        f"box={'None' if box is None else np.asarray(box).tolist()}"
                    )

                if not did_overflow:
                    break

                realloc_count += 1
                _pair_stats["reallocs"] += 1

                next_multiplier = (
                    _current_capacity_multiplier[0]
                    * float(jax_md_capacity_growth_factor)
                )

                rebuilt = _create_jax_md_bundle(next_multiplier)
                if rebuilt is not None:
                    _neighbor_fn_new, _filter_fn_new, _ = rebuilt
                    _neighbor_fn_cell[0] = _neighbor_fn_new
                    _filter_fn_cell[0] = _filter_fn_new
                    _current_capacity_multiplier[0] = next_multiplier
                    _pair_stats["capacity_multiplier"] = float(next_multiplier)

                try:
                    nbrs = _neighbor_fn_cell[0].allocate(R, **kwargs)
                    nbrs = nbrs.update(R, **kwargs)
                except Exception as e:
                    if _nbr_debug:
                        print(
                            f"[nbr] allocate/update failed during retry {realloc_count} "
                            f"({type(e).__name__}): {e}"
                        )
                    return _cell_list_fallback_pairs(
                        np.asarray(positions, dtype=np.float64),
                        _nbr_debug,
                        box_in=box,
                    )
            else:
                if _nbr_debug:
                    print("[nbr] persistent overflow after retries; attempting cell-list fallback")
                return _cell_list_fallback_pairs(
                    np.asarray(positions, dtype=np.float64),
                    _nbr_debug,
                    box_in=box,
                )

            _nbrs[0] = nbrs

            pair_i, pair_j, mask = _filter_fn_cell[0](nbrs.idx)

            if mm_r_min is not None:
                _pair_stats["com_filter_calls"] += 1

                R_for_filter = np.asarray(R, dtype=np.float64)

                if fractional_coordinates and box is not None:
                    cell_np = np.asarray(box, dtype=np.float64)
                    cell_3x3 = np.diag(cell_np) if cell_np.ndim == 1 else cell_np
                    R_for_filter = np.asarray(R_for_filter @ cell_3x3, dtype=np.float64)

                elif fractional_coordinates and box is None:
                    cell_np = np.asarray(pbc_cell, dtype=np.float64)
                    if cell_np.ndim == 0:
                        cell_3x3 = np.diag([float(cell_np)] * 3)
                    elif cell_np.shape == (3,):
                        cell_3x3 = np.diag(cell_np)
                    else:
                        cell_3x3 = cell_np
                    R_for_filter = np.asarray(R_for_filter @ cell_3x3, dtype=np.float64)

                pbc_for_filter = np.asarray(pbc_cell) if pbc_cell is not None else None
                if fractional_coordinates and box is not None:
                    box_np = np.asarray(box)
                    pbc_for_filter = np.diag(box_np) if box_np.ndim == 1 else box_np

                mask = _filter_pairs_by_com_min(
                    R_for_filter,
                    np.asarray(jax.device_get(pair_i)),
                    np.asarray(jax.device_get(pair_j)),
                    np.asarray(jax.device_get(mask), dtype=bool),
                    _offsets_np,
                    np.asarray(_monomer_id_jnp),
                    mm_r_min,
                    pbc_cell=pbc_for_filter,
                )

            pair_idx = jnp.stack([pair_i, pair_j], axis=1)
            pair_mask = jnp.asarray(mask, dtype=ml_jnp_dtype)

            _pair_idx_cell[0] = pair_idx
            _pair_mask_cell[0] = pair_mask
            _last_positions[0] = np.asarray(R, dtype=np.float64)
            _last_cartesian_positions[0] = R_cart.copy()
            _last_box[0] = None if box is None else np.asarray(box, dtype=np.float64).copy()
            _pair_stats["updates"] += 1

            if _nbr_debug:
                n_valid = int(np.sum(np.asarray(jax.device_get(mask))))
                capacity = pair_idx.shape[0] if hasattr(pair_idx, "shape") else len(pair_i)
                print(
                    f"[nbr] pairs: n_valid={n_valid}, capacity={capacity}, "
                    f"frac_coords={fractional_coordinates}"
                )

            return pair_idx, pair_mask
        
        def _get_pair_update_stats() -> dict:
            return dict(_pair_stats)

        update_mm_pairs.get_stats = _get_pair_update_stats

        mm_fn = calculate_mm_energy_and_forces_dynamic
        if _use_jax_pme_coulomb:
            mm_fn = _wrap_mm_fn_with_jax_pme_coulomb(
                mm_fn,
                charges_np=charges,
                pbc_cell=np.asarray(pbc_cell) if pbc_cell is not None else None,
                method=_jax_pme_method,
                sr_cutoff_A=_jax_pme_sr_cutoff,
                dynamic=True,
                monomer_offsets=_offsets_np,
                monomer_id_np=_monomer_id_np,
                lambda_monomer=np.asarray(lambda_monomer),
                ml_switch_width=ml_switch_width,
                mm_switch_on=mm_switch_on,
                mm_switch_width=mm_switch_width,
                complementary_handoff=complementary_handoff,
                mm_r_min=mm_r_min,
            )
        return (mm_fn, update_mm_pairs)

    mm_fn = calculate_mm_energy_and_forces
    if _use_jax_pme_coulomb:
        _static_mask = (
            np.ones(_n_static_pairs, dtype=np.float64)
            if _cl_mask_jnp is None
            else np.asarray(jax.device_get(_cl_mask_jnp), dtype=np.float64)
        )
        mm_fn = _wrap_mm_fn_with_jax_pme_coulomb(
            mm_fn,
            charges_np=charges,
            pbc_cell=np.asarray(pbc_cell) if pbc_cell is not None else None,
            method=_jax_pme_method,
            sr_cutoff_A=_jax_pme_sr_cutoff,
            dynamic=False,
            monomer_offsets=_offsets_np,
            monomer_id_np=_monomer_id_np,
            lambda_monomer=np.asarray(lambda_monomer),
            ml_switch_width=ml_switch_width,
            mm_switch_on=mm_switch_on,
            mm_switch_width=mm_switch_width,
            complementary_handoff=complementary_handoff,
            mm_r_min=mm_r_min,
            static_pair_i=np.asarray(jax.device_get(pair_idx_atom_atom[:, 0]), dtype=np.int64),
            static_pair_j=np.asarray(jax.device_get(pair_idx_atom_atom[:, 1]), dtype=np.int64),
            static_pair_mask=_static_mask,
        )
    return mm_fn
