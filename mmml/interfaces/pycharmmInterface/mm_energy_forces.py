"""MM (Lennard-Jones + Coulomb) energy and force calculation for hybrid ML/MM potentials."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array

from mmml.interfaces.pycharmmInterface.calculator_utils import _sharpstep
from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
    mic_displacement,
    mic_displacement_smooth,
    mic_displacements_batched,
    mic_displacements_batched_smooth,
)

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
        cell_list_pairs as _cell_list_pairs,
        estimate_max_pairs as _estimate_max_pairs,
    )
except Exception:
    _cell_list_pairs = None
    _estimate_max_pairs = None

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


def build_mm_energy_forces_fn(
    R: np.ndarray,
    *,
    total_atoms: int,
    n_monomers: int,
    monomer_offsets: np.ndarray,
    atoms_per_monomer_list: Sequence[int],
    lambda_monomer: np.ndarray,
    ml_cutoff_distance: float,
    mm_switch_on: float,
    mm_cutoff: float,
    complementary_handoff: bool = True,
    ep_scale: Optional[np.ndarray] = None,
    sig_scale: Optional[np.ndarray] = None,
    at_codes_override: Optional[np.ndarray] = None,
    pbc_cell: Optional[np.ndarray] = None,
    max_pairs: Optional[int] = None,
    cell_list_safety_factor: float = 2.5,
    use_smooth_mic: bool = False,
    use_jax_md_neighbor_list: bool = True,
    debug: bool = False,
) -> Any:
    """Build MM energy/forces function with switching.

    Supports heterogeneous monomer sizes via monomer_offsets and atoms_per_monomer_list.
    Uses cell list for PBC when pbc_cell is provided, otherwise all-pairs.

    Returns:
        When use_jax_md_neighbor_list and jax_md available: (mm_fn, update_fn) where
        mm_fn(positions, pair_idx, pair_mask) -> (energy, forces) and
        update_fn(positions, box=None) -> (pair_idx, pair_mask).
        Otherwise: mm_fn(positions) -> (energy, forces) (single callable).
    """
    if CGENFF_PRM is None or CGENFF_RTF is None:
        raise RuntimeError("CGENFF parameters not available; PyCHARMM may not be initialized")

    import pycharmm.param as param
    from mmml.interfaces.pycharmmInterface.import_pycharmm import read, reset_block, settings
    pycharmm_quiet()
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    import pycharmm
    pycharmm.lingo.charmm_script('bomlev 0')

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
    at_flat_q = np.array(atc_qs)
    at_flat_ep = np.array(at_ep)
    at_flat_rm = np.array(at_rm)

    _dp = _dimer_permutations(n_monomers)
    _use_jax_md_nbrs = (
        pbc_cell is not None
        and use_jax_md_neighbor_list
        and have_jax_md()
    )
    _use_cell_list = (
        pbc_cell is not None
        and not _use_jax_md_nbrs
        and _cell_list_pairs is not None
    )

    if _use_jax_md_nbrs:
        # jax_md path: pair list is updated externally; mm_fn will accept pair_idx, pair_mask
        jax_md_result = create_jax_md_neighbor_list(
            np.asarray(pbc_cell),
            r_cutoff=mm_switch_on + mm_cutoff,
            monomer_offsets=np.asarray(monomer_offsets),
            dr_threshold=0.5,
            capacity_multiplier=1.25,
        )
        if jax_md_result is not None:
            _neighbor_fn, _filter_fn, _monomer_id_jnp = jax_md_result
            _nbrs = [None]  # mutable cell for neighbor list state
            _pair_idx_cell = [None]
            _pair_mask_cell = [None]
        else:
            _use_jax_md_nbrs = False
            _use_cell_list = pbc_cell is not None and _cell_list_pairs is not None

    if _use_cell_list:
        _mm_cutoff_dist = mm_switch_on + mm_cutoff
        if max_pairs is not None:
            _max_pairs = int(max_pairs)
        else:
            _max_pairs = _estimate_max_pairs(
                total_atoms, cutoff=_mm_cutoff_dist, safety_factor=cell_list_safety_factor
            )
        _offsets_np = np.array([int(monomer_offsets[k]) for k in range(len(monomer_offsets))])
        _cl_i, _cl_j, _cl_mask, _n_valid = _cell_list_pairs(
            np.asarray(R),
            np.asarray(pbc_cell),
            cutoff=_mm_cutoff_dist,
            max_pairs=_max_pairs,
            monomer_offsets=_offsets_np,
            atoms_per_monomer_list=atoms_per_monomer_list,
            exclude_intra_monomer=True,
        )
        pair_idx_atom_atom = jnp.stack([_cl_i, _cl_j], axis=1)
        _cl_mask_jnp = jnp.asarray(_cl_mask, dtype=jnp.float32)

        if debug:
            print(f"[get_MM] Cell list: {_n_valid} valid pairs out of max_pairs={_max_pairs}")

        _monomer_id_np = np.empty(total_atoms, dtype=np.int32)
        for mi in range(n_monomers):
            _monomer_id_np[_offsets_np[mi]:_offsets_np[mi + 1]] = mi
        _monomer_id_jnp = jnp.array(_monomer_id_np)
        _lam_a = jnp.take(lambda_monomer, _monomer_id_jnp[_cl_i])
        _lam_b = jnp.take(lambda_monomer, _monomer_id_jnp[_cl_j])
        pair_lambda_mm = _lam_a * _lam_b * _cl_mask_jnp

        _dimer_lookup = {}
        for di, (mi, mj) in enumerate(_dp):
            _dimer_lookup[(mi, mj)] = di
            _dimer_lookup[(mj, mi)] = di
        _pair_dimer_idx = np.full(_max_pairs, -1, dtype=np.int32)
        _n_pairs_per_dimer_count = np.zeros(len(_dp), dtype=np.int32)
        for k in range(_n_valid):
            ai, aj = int(_cl_i[k]), int(_cl_j[k])
            di_key = (int(_monomer_id_np[ai]), int(_monomer_id_np[aj]))
            di_idx = _dimer_lookup.get(di_key, -1)
            _pair_dimer_idx[k] = di_idx
            if di_idx >= 0:
                _n_pairs_per_dimer_count[di_idx] += 1
        n_pairs_per_dimer_arr = _n_pairs_per_dimer_count
        pair_dimer_idx = jnp.array(_pair_dimer_idx)
    elif _use_jax_md_nbrs:
        _mm_cutoff_dist = mm_switch_on + mm_cutoff
        _offsets_np = np.array([int(monomer_offsets[k]) for k in range(len(monomer_offsets))])
        nbrs_init = _neighbor_fn.allocate(np.asarray(R))
        _nbrs[0] = nbrs_init
        idx = nbrs_init.idx
        pair_i, pair_j, mask = _filter_fn(idx)
        _max_pairs = idx.shape[1]
        pair_idx_atom_atom = jnp.stack([pair_i, pair_j], axis=1)
        _cl_mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
        _pair_idx_cell[0] = pair_idx_atom_atom
        _pair_mask_cell[0] = _cl_mask_jnp
        _monomer_id_np = np.empty(total_atoms, dtype=np.int32)
        for mi in range(n_monomers):
            _monomer_id_np[_offsets_np[mi]:_offsets_np[mi + 1]] = mi
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
    charges = np.array(psf.get_charges())[:total_atoms]
    at_codes = np.array(psf.get_iac())[:total_atoms]
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

    if not _use_jax_md_nbrs:
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

    def get_switching_function(
        ml_cutoff_distance: float = ml_cutoff_distance,
        mm_switch_on: float = mm_switch_on,
        mm_cutoff: float = mm_cutoff,
        complementary_handoff: bool = complementary_handoff,
    ) -> Callable[..., Array]:
        @jax.jit
        def apply_switching_function(
            positions: Array,
            pair_energies: Array,
            pair_dimer_idx_arg: Optional[Array] = None,
        ) -> Array:
            coms = jnp.stack([
                positions[monomer_offsets[k]:monomer_offsets[k + 1]].mean(axis=0)
                for k in range(n_monomers)
            ])
            com_i = coms[_dimer_perms_np[:, 0]]
            com_j = coms[_dimer_perms_np[:, 1]]
            if pbc_cell is not None:
                mic_fn = mic_displacement_smooth if use_smooth_mic else mic_displacement
                d_vec = jax.vmap(lambda a, b: mic_fn(a, b, pbc_cell))(com_i, com_j)
                r = jnp.linalg.norm(d_vec, axis=1)
            else:
                r = jnp.linalg.norm(com_j - com_i, axis=1)
            if complementary_handoff:
                handoff = _sharpstep(r, mm_switch_on - ml_cutoff_distance, mm_switch_on, gamma=GAMMA_ON)
                mm_taper = 1.0 - _sharpstep(r, mm_switch_on, mm_switch_on + mm_cutoff, gamma=GAMMA_OFF)
                mm_scale = handoff * mm_taper
            else:
                mm_on = _sharpstep(r, mm_switch_on, mm_switch_on + mm_cutoff, gamma=GAMMA_ON)
                mm_off = _sharpstep(r, mm_switch_on + mm_cutoff, mm_switch_on + 2.0 * mm_cutoff, gamma=GAMMA_OFF)
                mm_scale = mm_on * (1.0 - mm_off)

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

    @jax.jit
    def calculate_mm_pair_energies(positions: Array) -> Array:
        if pbc_cell is not None:
            pos_dst = positions[pair_idx_atom_atom[:, 1]]
            pos_src = positions[pair_idx_atom_atom[:, 0]]
            mic_batched = mic_displacements_batched_smooth if use_smooth_mic else mic_displacements_batched
            displacements = mic_batched(pos_dst, pos_src, pbc_cell)
        else:
            displacements = positions[pair_idx_atom_atom[:, 0]] - positions[pair_idx_atom_atom[:, 1]]
        distances = jnp.linalg.norm(displacements, axis=1)

        if _cl_mask_jnp is not None:
            distances = jnp.where(_cl_mask_jnp > 0, distances, 1e6)

        pair_mask = (pair_idx_atom_atom[:, 0] < pair_idx_atom_atom[:, 1])
        vdw_energies = lennard_jones(distances, pair_rm, pair_ep) * pair_mask
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

    if _use_jax_md_nbrs:
        # Dynamic path: compute pair quantities from pair_idx, pair_mask
        _pbc_cell_jnp = jnp.asarray(pbc_cell)
        _lambda_monomer_jnp = jnp.asarray(lambda_monomer)

        @jax.jit
        def calculate_mm_pair_energies_dynamic(
            positions: Array,
            pair_idx: Array,
            pair_mask: Array,
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

            pos_dst = positions[pair_j]
            pos_src = positions[pair_i]
            mic_batched = mic_displacements_batched_smooth if use_smooth_mic else mic_displacements_batched
            displacements = mic_batched(pos_dst, pos_src, _pbc_cell_jnp)
            distances = jnp.linalg.norm(displacements, axis=1)
            distances = jnp.where(pair_mask > 0, distances, 1e6)

            pair_mask_ij = (pair_i < pair_j)
            vdw = lennard_jones(distances, pair_rm_dyn, pair_ep_dyn) * pair_mask_ij
            elec = coulomb(distances, pair_qq_dyn) * pair_mask_ij
            return vdw + elec

        @jax.jit
        def calculate_mm_energy_and_forces_dynamic(
            positions: Array,
            pair_idx: Array,
            pair_mask: Array,
        ) -> Tuple[Array, Array]:
            pair_i = pair_idx[:, 0]
            pair_j = pair_idx[:, 1]
            mid_i = _monomer_id_jnp[pair_i]
            mid_j = _monomer_id_jnp[pair_j]
            pair_dimer_idx_dyn = _dimer_lookup_arr[mid_i, mid_j]

            pair_energies = calculate_mm_pair_energies_dynamic(positions, pair_idx, pair_mask)
            switched_energy = apply_switching_function(
                positions, pair_energies, pair_dimer_idx_arg=pair_dimer_idx_dyn
            )
            switched_mm_energy_dyn = lambda pos: apply_switching_function(
                pos,
                calculate_mm_pair_energies_dynamic(pos, pair_idx, pair_mask),
                pair_dimer_idx_arg=pair_dimer_idx_dyn,
            )
            forces = -1.0 * jax.grad(switched_mm_energy_dyn)(positions)
            forces = jnp.where(jnp.isfinite(forces), forces, 0.0)
            return switched_energy, forces

        def update_mm_pairs(positions: np.ndarray, box: Optional[np.ndarray] = None) -> Tuple[Array, Array]:
            R = np.asarray(positions, dtype=np.float64)
            nbrs = _nbrs[0]
            kwargs = {} if box is None else {"box": jnp.asarray(box)}
            nbrs = nbrs.update(R, **kwargs)
            for _ in range(3):
                if not bool(np.asarray(nbrs.did_buffer_overflow)):
                    break
                nbrs = _neighbor_fn.allocate(R, **kwargs)
                nbrs = nbrs.update(R, **kwargs)
            else:
                raise RuntimeError("Neighbor list buffer overflow persisted after reallocation")
            _nbrs[0] = nbrs
            pair_i, pair_j, mask = _filter_fn(nbrs.idx)
            pair_idx = jnp.stack([pair_i, pair_j], axis=1)
            pair_mask = jnp.asarray(mask, dtype=jnp.float32)
            return pair_idx, pair_mask

        return (calculate_mm_energy_and_forces_dynamic, update_mm_pairs)

    return calculate_mm_energy_and_forces
