#!/usr/bin/env python
# coding: utf-8

"""Example script demonstrating a hybrid ML/MM simulation workflow using JAX-MD.

By bypassing ASE and running the optimization and molecular dynamics entirely
within JAX-MD, we achieve significantly faster execution times through full-program
compilation (JIT) and hardware acceleration (GPU/TPU).

GPU Performance Optimizations vs. original:
  1. Zero-retrace pair-list pattern: pair arrays are padded to a fixed MAX_PAIRS
     shape and stored in mutable Python list slots (_pi_ref, _pj_ref). The energy
     function closes over these references; since the shape never changes JAX only
     traces once, eliminating thousands of recompilations.
  2. e14/vdw14 scale arrays precomputed on CPU once per NL update (not inside JIT).
  3. Runner step functions compiled once and reused across all blocks.
  4. value_and_grad: energy + forces computed in a single kernel launch.
  5. Vectorized helpers: unfold_coordinates, get_max_h_x_bond, scale_broken_h_bonds
     now use NumPy broadcasting instead of Python for-loops.
  6. Larger block sizes (FIRE: 200→1000, NVT/NVE: 100→500) to amortize Python
     overhead across more GPU work per outer-loop iteration.
"""

from pathlib import Path
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad

# ASE trajectory, calculator, and Atoms imports
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

# JAX-MD imports
from jax_md import space, minimize, simulate

# PyCHARMM lingo imports
import pycharmm.lingo as lingo

import mmml
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    ensure_pycharmm_loaded,
    CGENFF_PRM,
    pycharmm_loud,
    coor,
)
from mmml.interfaces.pycharmmInterface.trialanine_water_box import build_trialanine_water_box_in_charmm
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    set_charmm_positions,
    setup_nonbonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    load_nonbonded_system_from_charmm,
    nonbonded_energy_and_forces,
    _build_pair_indices,
    resolve_nonbonded_excluded_pairs,
    charmm_vfswitch_coeffs,
    charmm_fswitch_coeffs,
    _pair_lj_epsilon,
    _pair_vdw_energy,
    _pair_elec_energy,
)
from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import _nbond_settings_from_cutoffs
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_loud
from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.data.units import KCAL_MOL_TO_EV
from mmml.interfaces.jaxmdInterface import (
    make_monomer_energy_fn,
    make_peptide_water_ml_energy_fn,
    get_intermolecular_pairs,
)
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

# 1. Initialize JAX and PyCHARMM configuration
jax.config.update("jax_enable_x64", True)
ensure_pycharmm_loaded()
pycharmm_loud()

# Helper function to parse peptide H-X bonds from the CHARMM PSF file
def parse_peptide_h_x_bonds(psf_path, z_array):
    bonds = []
    in_bonds = False
    with open(psf_path, 'r') as f:
        for line in f:
            if '!NBOND' in line:
                in_bonds = True
                continue
            if in_bonds:
                if not line.strip() or '!' in line:
                    if '!NBOND' not in line:
                        in_bonds = False
                        continue
                parts = line.split()
                for i in range(0, len(parts), 2):
                    idx1 = int(parts[i]) - 1
                    idx2 = int(parts[i+1]) - 1
                    # Peptide atoms are the first 42 atoms
                    if idx1 < 42 and idx2 < 42:
                        z1 = z_array[idx1]
                        z2 = z_array[idx2]
                        # Identify hydrogen (Z=1) to heavy atom (Z>1) bonds
                        if (z1 == 1 and z2 > 1) or (z2 == 1 and z1 > 1):
                            h_idx = idx1 if z1 == 1 else idx2
                            x_idx = idx2 if z1 == 1 else idx1
                            bonds.append((h_idx, x_idx))
    return bonds


# Path to the pretrained neural network checkpoint parameters
CKPT_PATH = "params_aaa_long_2026-07-04_22-30-27.json"

FIRE_STEPS = 500
FIRE_PRINT_FREQ = 100
# FIRE_BLOCK_STEPS kept small (100) because FIRE adaptively grows its step size:
# running 1000 steps without checking allows catastrophic divergence before repair.
FIRE_BLOCK_STEPS = 100
NVT_TOTAL_STEPS = 50000
NVT_BLOCK_STEPS = 500

NVE_TOTAL_STEPS = 20000
NVE_BLOCK_STEPS = 500
FIRE_CYCLES = 10
NWATER = 1500
BOX_SIDE_A = 30.0
NL_BUFFER = 2.0
# Extra headroom fraction for padded pair array (5%)
MAX_PAIRS_HEADROOM = 1.05
MAX_HX_BOND_LIMIT = 1.2
SEED = 42
# Define simulation conditions
temperature = 100.0  # Kelvin
kb = 8.617333262145e-5  # eV/K (Boltzmann constant in eV/K)
target_temp_ev = temperature * kb
dt_fs = 0.25  # time step in femtoseconds
dt = dt_fs * 0.001  # convert to picoseconds (JAX-MD metal units)


# 2. Build the initial system in PyCHARMM and minimize
print("--- Building Trialanine Water Box in CHARMM ---")
workdir = Path('/tmp/tria_box')
box = build_trialanine_water_box_in_charmm(n_waters=NWATER,
    box_side_A=BOX_SIDE_A, seed=SEED, workdir=workdir
    )

pos = np.asarray(box.positions, dtype=np.float64)
pos = np.random.uniform(-0.1, 0.1, pos.shape) + pos

# Translate the entire system so that the peptide is centered in the box
n_trialanine = 42
peptide_center = pos[:n_trialanine].mean(axis=0)
box_center = np.array([box.box_side_A / 2, box.box_side_A / 2, box.box_side_A / 2])
translation = box_center - peptide_center
pos += translation / 2.0

set_charmm_positions(pos)

lingo.charmm_script("MINI SD 10000")
lingo.charmm_script("MINI ABNR 10000")

# Retrieve positions and atomic numbers
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
z = get_Z_from_psf()
pep_h_x_bonds = parse_peptide_h_x_bonds(box.psf_path, z)
print(f"Parsed {len(pep_h_x_bonds)} peptide H-X bonds from PSF.")

# Precompute H-X bond index arrays for vectorized operations
h_idx_arr = np.array([b[0] for b in pep_h_x_bonds], dtype=np.int32)
x_idx_arr = np.array([b[1] for b in pep_h_x_bonds], dtype=np.int32)

# Construct ASE Atoms object to act as a template for trajectory writing
atoms = Atoms(z, pos)
atoms.set_cell(box.cell)
atoms.set_pbc(True)

# Define box size and cell
box_size = float(box.box_side_A)
cell = np.asarray(box.cell, dtype=np.float64)

# 3. Load JAX nonbonded parameters and setup cutoffs
psf_path = box.psf_path
prm_path = CGENFF_PRM
nb_settings = _nbond_settings_from_cutoffs(box.nbond_cutoffs)
nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)

calc = create_calculator_from_checkpoint(CKPT_PATH)
model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))

if model is None or params is None:
    raise ValueError("Could not extract model or params from the loaded calculator.")

# Setup monomer groupings (first 42 atoms: peptide; then groups of 3 for water)
monomer_indices = [np.arange(n_trialanine)]
for i in range(n_trialanine, len(pos), 3):
    monomer_indices.append(np.arange(i, i + 3))

# Pre-convert to JAX arrays for performance
jax_monomer_indices = [jnp.array(idx, dtype=jnp.int32) for idx in monomer_indices]
jax_z = jnp.array(z, dtype=jnp.int32)

# Set up molecule IDs for nonbonded interactions
molecule_id = np.empty(len(pos), dtype=np.int32)
molecule_id[:n_trialanine] = 0
for mol_id, start in enumerate(range(n_trialanine, len(pos), 3), start=1):
    molecule_id[start:start + 3] = mol_id
jax_molecule_id = jnp.array(molecule_id, dtype=jnp.int32)

# Precompute stacked monomer index arrays grouped by size for vectorized unfolding
from collections import defaultdict
_mon_by_size = defaultdict(list)
for idx in monomer_indices:
    _mon_by_size[len(idx)].append(idx)
# List of (stacked_indices, size) tuples — one entry per unique monomer size
_mon_stacked_groups = [(np.stack(lst), sz) for sz, lst in _mon_by_size.items()]

# 5. Define displacement and shift functions for Periodic Boundary Conditions
displacement_fn, shift_fn = space.periodic(box_size)

# Option: Treat peptide-water intermolecular interactions with ML instead of MM
PEPTIDE_WATER_ML = False

# Configure compute_monomer_energy function based on selection
if PEPTIDE_WATER_ML:
    print("--- Configuring PEPTIDE-WATER interactions with ML (dimer approach) ---")
    peptide_idx = monomer_indices[0]
    water_indices = monomer_indices[1:]
    compute_monomer_energy = make_peptide_water_ml_energy_fn(
        model, params, jax_z, peptide_idx, water_indices, displacement_fn
    )
else:
    print("--- Configuring PEPTIDE-WATER interactions with MM ---")
    compute_monomer_energy = make_monomer_energy_fn(
        model, params, jax_z, jax_monomer_indices, displacement_fn
    )

# Precompute initial pair list
print("--- Precomputing nonbonded pair list indices ---")
excluded_pairs = nbond_data.excluded_pairs
if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
    excluded_pairs = resolve_nonbonded_excluded_pairs(
        nbond_data.psf_path,
        nbond_data.psf_bonds,
        natom=int(np.asarray(nbond_data.charges).shape[0]),
    )

# ─────────────────────────────────────────────────────────────────────────────
# GPU OPTIMIZATION: Padded pair-list infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_e14_vdw14(pair_i_np, pair_j_np):
    """Precompute per-pair 1-4 scaling arrays on the CPU.

    Called on every neighbor-list update (cheap — runs on CPU once per NL rebuild).
    Returns JAX arrays of fixed shape equal to MAX_PAIRS with padding zeros.
    """
    n = len(pair_i_np)
    e14 = np.ones(n, dtype=np.float64)
    vdw14 = np.ones(n, dtype=np.float64)
    for k, (i, j) in enumerate(zip(pair_i_np, pair_j_np)):
        if (int(i), int(j)) in nbond_data.e14_pairs:
            e14[k] = nb_settings.e14fac
            vdw14[k] = nb_settings.vdw14fac
    return e14, vdw14


def _get_inter_pairs_np(pos_np):
    """Compute intermolecular pair list on host, return numpy arrays."""
    pair_i_raw, pair_j_raw = _build_pair_indices(
        pos_np, cell, excluded_pairs, nb_settings.cutnb + NL_BUFFER
    )
    inter = molecule_id[pair_i_raw] != molecule_id[pair_j_raw]
    return pair_i_raw[inter], pair_j_raw[inter]


# Sample the pair count from the initial structure to size the padded arrays
_pi0, _pj0 = _get_inter_pairs_np(pos)
MAX_PAIRS = int(len(_pi0) * MAX_PAIRS_HEADROOM) + 16  # add 16 as safety margin
print(f"--- Initial pair count: {len(_pi0)}, MAX_PAIRS buffer: {MAX_PAIRS} ---")


def _pad_pairs(pi_np, pj_np):
    """Pad pair index arrays to MAX_PAIRS with (0,0) sentinel entries.

    Sentinel pairs (both index 0) contribute zero energy because same-atom pairs
    have zero charge product and the VDW self-energy is zero (r→0 handled by cutoff
    mask which evaluates False for r=0 < c2of, but since atom 0 contributes to
    real pairs we use a zero-mask column instead).
    """
    n = len(pi_np)
    assert n <= MAX_PAIRS, (
        f"Pair count {n} exceeds MAX_PAIRS={MAX_PAIRS}. "
        "Increase MAX_PAIRS_HEADROOM or decrease NL_BUFFER."
    )
    pi_pad = np.zeros(MAX_PAIRS, dtype=np.int32)
    pj_pad = np.zeros(MAX_PAIRS, dtype=np.int32)
    mask = np.zeros(MAX_PAIRS, dtype=np.float64)   # 1.0 for real pairs, 0.0 for padding
    pi_pad[:n] = pi_np
    pj_pad[:n] = pj_np
    mask[:n] = 1.0
    return pi_pad, pj_pad, mask


def _pad_e14_vdw14(e14_np, vdw14_np):
    """Pad 1-4 scaling arrays to MAX_PAIRS with 1.0 (neutral scaling)."""
    e14_pad = np.ones(MAX_PAIRS, dtype=np.float64)
    vdw14_pad = np.ones(MAX_PAIRS, dtype=np.float64)
    n = len(e14_np)
    e14_pad[:n] = e14_np
    vdw14_pad[:n] = vdw14_np
    return e14_pad, vdw14_pad


# ─────────────────────────────────────────────────────────────────────────────
# Mutable reference slots for pair data — JAX sees fixed shapes → traces once
# ─────────────────────────────────────────────────────────────────────────────
# We close over these Python list slots in the JIT-compiled energy function.
# Updating _pi_ref[0] etc. before each block is a Python-level mutation that
# does NOT trigger JAX re-tracing because the array shape is unchanged.

_pi_ref = [jnp.zeros(MAX_PAIRS, dtype=jnp.int32)]
_pj_ref = [jnp.zeros(MAX_PAIRS, dtype=jnp.int32)]
_mask_ref = [jnp.zeros(MAX_PAIRS, dtype=jnp.float64)]
_e14_ref = [jnp.ones(MAX_PAIRS, dtype=jnp.float64)]
_vdw14_ref = [jnp.ones(MAX_PAIRS, dtype=jnp.float64)]

# Precompute static per-atom arrays used inside the energy function
_q_jax = jnp.asarray(nbond_data.charges, dtype=jnp.float64)
_eps_jax = jnp.asarray(nbond_data.epsilon, dtype=jnp.float64)
_rmin_jax = jnp.asarray(nbond_data.rmin, dtype=jnp.float64)
_cell_jax = jnp.asarray(cell if cell.ndim == 2 else np.diag(cell), dtype=jnp.float64)
_vfswitch = charmm_vfswitch_coeffs(nb_settings)
_fswitch = charmm_fswitch_coeffs(nb_settings)
_c2of = nb_settings.c2ofnb


def update_pair_refs(pos_np):
    """Rebuild pair list on CPU and update mutable reference slots.

    This is the ONLY function that touches the pair data between blocks.
    Because array shapes are constant, the compiled energy_fn is never re-traced.
    """
    pi_np, pj_np = _get_inter_pairs_np(pos_np)
    e14_np, vdw14_np = _precompute_e14_vdw14(pi_np, pj_np)

    pi_pad, pj_pad, mask = _pad_pairs(pi_np, pj_np)
    e14_pad, vdw14_pad = _pad_e14_vdw14(e14_np, vdw14_np)

    _pi_ref[0] = jnp.array(pi_pad, dtype=jnp.int32)
    _pj_ref[0] = jnp.array(pj_pad, dtype=jnp.int32)
    _mask_ref[0] = jnp.array(mask, dtype=jnp.float64)
    _e14_ref[0] = jnp.array(e14_pad, dtype=jnp.float64)
    _vdw14_ref[0] = jnp.array(vdw14_pad, dtype=jnp.float64)


# Populate refs with the initial pair list
update_pair_refs(pos)


def hybrid_energy_fn(r) -> jnp.ndarray:
    """Single JIT-compiled hybrid ML/MM energy function.

    Closes over mutable _pi_ref/_pj_ref/_mask_ref/_e14_ref/_vdw14_ref slots.
    JAX traces this once; pair data is updated by mutating the slots.
    """
    # (A) Intramolecular terms from ML potential
    e_intra = compute_monomer_energy(r)

    # (B) Intermolecular MM nonbonded terms with padded pair list
    pi = _pi_ref[0]
    pj = _pj_ref[0]
    mask = _mask_ref[0]      # 1.0 for real pairs, 0.0 for padding
    e14 = _e14_ref[0]
    vdw14 = _vdw14_ref[0]

    ri = r[pi]
    rj = r[pj]
    disp = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri, rj)
    dist = jnp.linalg.norm(disp, axis=-1)
    within_ctof = (dist * dist < _c2of) * mask   # 0.0 for padding and out-of-cutoff

    # CRITICAL: Clamp distances for padding pairs (mask=0) to a safe non-zero value
    # before passing to VDW/elec kernels.  Padding pairs have pi=pj=0 → dist=0 → inf.
    # JAX's jnp.where evaluates BOTH branches, so inf survives and 0.0 * inf = nan.
    # Using a safe sentinel distance (1.0 Å, well within cutoff but harmless since
    # within_ctof already zeros the result for masked pairs via the mask column).
    safe_dist = jnp.where(mask > 0.5, dist, 1.0)

    ep = _pair_lj_epsilon(_eps_jax[pi], _eps_jax[pj])
    sig = _rmin_jax[pi] + _rmin_jax[pj]
    qq = _q_jax[pi] * _q_jax[pj] * e14 / nb_settings.eps

    vdw = _pair_vdw_energy(safe_dist, ep, sig, nb_settings, _vfswitch, use_jax_pme_dispersion=False)
    vdw = vdw * vdw14
    elec = _pair_elec_energy(safe_dist, qq, nb_settings, _fswitch)

    vdw = jnp.where(within_ctof, vdw, 0.0)
    elec = jnp.where(within_ctof, elec, 0.0)
    e_inter = (jnp.sum(vdw) + jnp.sum(elec)) * KCAL_MOL_TO_EV

    return e_intra + e_inter


@jit
def _debug_ml_energy(r):
    """JIT-compiled ML intramolecular energy only (for diagnostics)."""
    return compute_monomer_energy(r)


@jit
def _debug_mm_energy(r):
    """JIT-compiled MM intermolecular energy only (for diagnostics)."""
    pi = _pi_ref[0]
    pj = _pj_ref[0]
    mask = _mask_ref[0]
    e14 = _e14_ref[0]
    vdw14 = _vdw14_ref[0]
    ri = r[pi]; rj = r[pj]
    disp = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri, rj)
    dist = jnp.linalg.norm(disp, axis=-1)
    within_ctof = (dist * dist < _c2of) * mask
    safe_dist = jnp.where(mask > 0.5, dist, 1.0)
    ep = _pair_lj_epsilon(_eps_jax[pi], _eps_jax[pj])
    sig = _rmin_jax[pi] + _rmin_jax[pj]
    qq = _q_jax[pi] * _q_jax[pj] * e14 / nb_settings.eps
    vdw = _pair_vdw_energy(safe_dist, ep, sig, nb_settings, _vfswitch, use_jax_pme_dispersion=False) * vdw14
    elec = _pair_elec_energy(safe_dist, qq, nb_settings, _fswitch)
    vdw = jnp.where(within_ctof, vdw, 0.0)
    elec = jnp.where(within_ctof, elec, 0.0)
    return (jnp.sum(vdw) + jnp.sum(elec)) * KCAL_MOL_TO_EV


def diagnose_energy(r, label=""):
    """Print energy components to identify NaN source."""
    r_np = np.asarray(r)
    if not np.isfinite(r_np).all():
        n_bad = int((~np.isfinite(r_np)).any(axis=-1).sum())
        print(f"[DIAG{label}] ⚠ Positions: {n_bad} atoms with NaN/Inf — energy will be NaN")
        return
    e_ml = float(_debug_ml_energy(r))
    e_mm = float(_debug_mm_energy(r))
    e_tot = e_ml + e_mm
    ml_ok = "✓" if np.isfinite(e_ml) else "✗ NaN/Inf"
    mm_ok = "✓" if np.isfinite(e_mm) else "✗ NaN/Inf"
    print(f"[DIAG{label}] E_ML={e_ml:.4f} eV {ml_ok} | E_MM={e_mm:.4f} eV {mm_ok} | "
          f"E_tot={e_tot:.4f} eV | Pairs_real={int((_mask_ref[0]>0.5).sum())}")


# Compile energy + forces in a single kernel launch
@jit
def energy_and_forces_fn(r):
    """Returns (energy_scalar, forces_array) in one JIT-compiled call."""
    e, neg_f = jax.value_and_grad(hybrid_energy_fn)(r)
    return e, -neg_f


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized helper functions (no Python loops)
# ─────────────────────────────────────────────────────────────────────────────

def unfold_coordinates(positions, L, mon_stacked_groups):
    """Vectorized coordinate unfolding under PBC.

    Uses precomputed _mon_stacked_groups (list of (stacked_idx, size)) to avoid
    per-monomer Python iteration. Each group is handled with NumPy broadcasting.
    """
    unfolded = np.array(positions)
    for stacked, _sz in mon_stacked_groups:
        # stacked: (N_monomers, size) index array
        coords = unfolded[stacked]           # (N, size, 3)
        ref = coords[:, 0:1, :]             # (N, 1, 3)
        diff = coords - ref
        diff -= L * np.round(diff / L)
        unfolded[stacked] = ref + diff
    return unfolded


def get_max_h_x_bond(positions, box_sz, h_idx, x_idx):
    """Vectorized max H-X bond length computation under PBC."""
    if len(h_idx) == 0:
        return 0.0
    pos_np = np.asarray(positions)
    diff = pos_np[h_idx] - pos_np[x_idx]      # (N_bonds, 3)
    diff -= box_sz * np.round(diff / box_sz)
    return float(np.linalg.norm(diff, axis=-1).max())


def scale_broken_h_bonds(positions, box_sz, h_idx, x_idx, threshold=1.3, target=1.02):
    """Vectorized H-X bond repair: scales back bonds longer than threshold."""
    new_pos = np.array(np.asarray(positions))
    if len(h_idx) == 0:
        return new_pos, False
    diff = new_pos[h_idx] - new_pos[x_idx]    # (N_bonds, 3)
    diff -= box_sz * np.round(diff / box_sz)
    dists = np.linalg.norm(diff, axis=-1)      # (N_bonds,)
    broken = dists > threshold
    if broken.any():
        dirs = diff[broken] / dists[broken, np.newaxis]
        new_pos[h_idx[broken]] = new_pos[x_idx[broken]] + dirs * target
    return new_pos, bool(broken.any())


# Helper function to repair structures using PyCHARMM minimization
def repair_structure_in_charmm(positions):
    print("\n[REPAIR] Temperature spike or NaN detected! Unfolding and repairing structure in CHARMM...")
    pos_np = np.asarray(positions)
    if not np.isfinite(pos_np).all():
        n_bad = int((~np.isfinite(pos_np)).any(axis=-1).sum())
        raise RuntimeError(
            f"[REPAIR] Positions contain NaN/Inf on {n_bad} atoms — "
            "cannot pass to CHARMM (would segfault). "
            "The simulation exploded before the repair point. "
            "Consider reducing dt, FIRE_BLOCK_STEPS, or dt_start/dt_max."
        )
    unfolded_pos = unfold_coordinates(pos_np, box_size, _mon_stacked_groups)
    set_charmm_positions(unfolded_pos)
    lingo.charmm_script("MINI SD 1000")
    lingo.charmm_script("MINI ABNR 1000")
    repaired_pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    print("[REPAIR] Structure repaired successfully. Re-initializing state.\n")
    return repaired_pos


# ─────────────────────────────────────────────────────────────────────────────
# Build simulation runner functions ONCE — never rebuilt inside loops
# ─────────────────────────────────────────────────────────────────────────────

# FIRE minimization
print("--- Building FIRE minimizer (compiled once) ---")
_init_fn_fire, _step_fn_fire = minimize.fire_descent(
    hybrid_energy_fn, shift_fn, dt_start=0.0001, dt_max=0.001
)
_step_fn_fire = jit(_step_fn_fire)

@jit
def run_fire_block(state):
    """Run FIRE_BLOCK_STEPS steps of FIRE (compiled once)."""
    def body_fn(i, s):
        return _step_fn_fire(s)
    return jax.lax.fori_loop(0, FIRE_BLOCK_STEPS, body_fn, state)


@jit
def run_fire_block_repair(state):
    """Run 200 steps of FIRE for post-repair minimization (compiled once)."""
    def body_fn(i, s):
        return _step_fn_fire(s)
    return jax.lax.fori_loop(0, 200, body_fn, state)


# NVT Nose-Hoover
print("--- Building NVT NHC simulator (compiled once) ---")
from jax_md import quantity

mass = np.zeros(len(pos))
mass[:n_trialanine] = 12.0   # average mass approximation for peptide
mass[n_trialanine::3] = 16.0  # Oxygen
mass[n_trialanine+1::3] = 1.0  # Hydrogen
mass[n_trialanine+2::3] = 1.0  # Hydrogen
jax_mass = jnp.array(mass, dtype=jnp.float64)

_init_fn_nvt, _step_fn_nvt = simulate.nvt_nose_hoover(hybrid_energy_fn, shift_fn, dt, target_temp_ev)
_step_fn_nvt = jit(_step_fn_nvt)

@jit
def run_nvt_block(state):
    def body_fn(i, s):
        return _step_fn_nvt(s)
    return jax.lax.fori_loop(0, NVT_BLOCK_STEPS, body_fn, state)


# NVE
print("--- Building NVE simulator (compiled once) ---")
_init_fn_nve, _step_fn_nve = simulate.nve(hybrid_energy_fn, shift_fn, dt)
_step_fn_nve = jit(_step_fn_nve)

@jit
def run_nve_block(state):
    def body_fn(i, s):
        return _step_fn_nve(s)
    return jax.lax.fori_loop(0, NVE_BLOCK_STEPS, body_fn, state)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Structure Minimization with JAX-MD FIRE and PyCHARMM Repair Loops
# ─────────────────────────────────────────────────────────────────────────────
print("--- Minimizing System with JAX-MD FIRE and PyCHARMM Repair Loops ---")
init_r = jnp.array(pos, dtype=jnp.float64)
pos_current = init_r

traj_path_fire = "cg_fire.traj"
print(f"--- Saving minimization trajectory to {traj_path_fire} ---")
traj_fire = Trajectory(traj_path_fire, "w", atoms)

for cycle in range(FIRE_CYCLES):
    print(f"\n--- Minimization Cycle {cycle+1}/{FIRE_CYCLES} ---")

    # Update pair list from current coordinates (CPU, once per cycle start)
    update_pair_refs(np.asarray(pos_current))

    # --- Pre-FIRE diagnostic: print energy components before any step ---
    diagnose_energy(pos_current, label=f" Cycle{cycle+1} init")

    fire_state = _init_fn_fire(pos_current)

    # Write starting configuration of this cycle to trajectory
    curr_e, curr_f = energy_and_forces_fn(pos_current)
    curr_e = float(curr_e)
    curr_f = np.asarray(curr_f)
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(pos_current), box_size, _mon_stacked_groups))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_fire.write(frame)

    # Run FIRE blocks — pair list rebuilt every FIRE_BLOCK_STEPS steps
    nan_detected = False
    last_good_pos = np.asarray(pos_current)   # checkpoint: last positions with finite energy
    for step in range(0, FIRE_STEPS, FIRE_BLOCK_STEPS):
        # Save current positions before the block so we can recover if FIRE diverges
        pos_before_block = np.asarray(fire_state.position)

        # Update pair list (CPU side only; no JAX re-trace)
        update_pair_refs(pos_before_block)

        fire_state = run_fire_block(fire_state)

        curr_e, curr_f = energy_and_forces_fn(fire_state.position)
        curr_e = float(curr_e)
        curr_f = np.asarray(curr_f)
        max_bond = get_max_h_x_bond(fire_state.position, box_size, h_idx_arr, x_idx_arr)
        print(f"Cycle {cycle+1} | FIRE Step {step+FIRE_BLOCK_STEPS:4d} | "
              f"Energy: {curr_e:.4f} eV | Max H-X Bond: {max_bond:.2f} Å")

        if not np.isfinite(curr_e) or not np.isfinite(max_bond):
            print(f"[FIRE] NaN/Inf energy detected at step {step+FIRE_BLOCK_STEPS} — "
                  f"reverting to last-good positions (before this block) and repairing")
            # Diagnose using pre-block positions (which are still finite)
            good_pos_jax = jnp.array(pos_before_block, dtype=jnp.float64)
            diagnose_energy(good_pos_jax,
                            label=f" Cycle{cycle+1} pre-step{step+FIRE_BLOCK_STEPS}")
            # Re-initialize FIRE from pre-block (finite) positions for CHARMM repair.
            # FireDescentState is a plain dataclass so we use _init_fn_fire, not .replace().
            update_pair_refs(pos_before_block)
            fire_state = _init_fn_fire(good_pos_jax)
            nan_detected = True
            break

        last_good_pos = pos_before_block
        frame = atoms.copy()
        frame.set_positions(unfold_coordinates(np.asarray(fire_state.position), box_size, _mon_stacked_groups))
        frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
        traj_fire.write(frame)

    # Repair the minimized structures in CHARMM at the end of every cycle
    # Use fire_state.position (reverted to pre-NaN if divergence occurred)
    pos_current = jnp.array(repair_structure_in_charmm(fire_state.position), dtype=jnp.float64)

traj_fire.close()
min_r = pos_current
print(f"\nMinimization completed over {FIRE_CYCLES} cycles.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Molecular Dynamics (NVT Nose-Hoover)
# ─────────────────────────────────────────────────────────────────────────────
print("--- Running NVT Nose-Hoover Dynamics with JAX-MD ---")

key = jax.random.PRNGKey(42)

# Initialize starting NVT state
update_pair_refs(np.asarray(min_r))
state = _init_fn_nvt(key, min_r, mass=jax_mass)

traj_path_nvt = "cg_nvt.traj"
print(f"--- Running NVT dynamics and saving trajectory to {traj_path_nvt} ---")
traj_nvt = Trajectory(traj_path_nvt, "w", atoms)

for step in range(0, NVT_TOTAL_STEPS, NVT_BLOCK_STEPS):
    # Update pair list on CPU (no JAX re-trace — shapes are fixed)
    update_pair_refs(np.asarray(state.position))

    # Run the compiled block of steps
    state = run_nvt_block(state)

    # Compute diagnostics (energy + forces in one kernel launch)
    curr_e, curr_f = energy_and_forces_fn(state.position)
    curr_e = float(curr_e)
    curr_f = np.asarray(curr_f)
    ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
    temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
    max_bond = get_max_h_x_bond(state.position, box_size, h_idx_arr, x_idx_arr)

    # Check for instability and repair
    if temp > 400.0 or np.isnan(curr_e) or max_bond > MAX_HX_BOND_LIMIT:
        if max_bond > MAX_HX_BOND_LIMIT:
            print(f"[REPAIR] Peptide H-X bond broke! Max bond length: {max_bond:.2f} Å "
                  f"(max limit {MAX_HX_BOND_LIMIT} Å)")
        # 1. Scale back stretched H-X bonds
        scaled_pos, _ = scale_broken_h_bonds(
            np.asarray(state.position), box_size, h_idx_arr, x_idx_arr
        )
        # 2. Minimize in PyCHARMM
        repaired_pos = repair_structure_in_charmm(scaled_pos)
        # 3. Post-repair JAX FIRE minimization
        print("[REPAIR] Running post-repair JAX FIRE minimization (200 steps)...")
        repaired_jax = jnp.array(repaired_pos, dtype=jnp.float64)
        update_pair_refs(repaired_pos)
        fire_state_rep = _init_fn_fire(repaired_jax)
        fire_state_rep = run_fire_block_repair(fire_state_rep)
        final_min_pos = jnp.array(fire_state_rep.position, dtype=jnp.float64)

        # Re-initialize NVT state
        update_pair_refs(np.asarray(final_min_pos))
        state = _init_fn_nvt(key, final_min_pos, mass=jax_mass)
        # Re-evaluate diagnostics
        curr_e, curr_f = energy_and_forces_fn(state.position)
        curr_e = float(curr_e)
        curr_f = np.asarray(curr_f)
        ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
        max_bond = get_max_h_x_bond(state.position, box_size, h_idx_arr, x_idx_arr)

    print(f"NVT Step {step+NVT_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | "
          f"Temp: {temp:.1f} K | Max H-X Bond: {max_bond:.2f} Å")

    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(state.position), box_size, _mon_stacked_groups))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nvt.write(frame)

traj_nvt.close()
print("NVT dynamics complete!")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Molecular Dynamics (NVE) to check stability
# ─────────────────────────────────────────────────────────────────────────────
print("--- Running NVE Dynamics with JAX-MD to check stability ---")

# Initialize NVE state from final NVT positions
update_pair_refs(np.asarray(state.position))
state_nve = _init_fn_nve(key, state.position, target_temp_ev, mass=jax_mass)

traj_path_nve = "cg_nve.traj"
print(f"--- Running NVE dynamics and saving trajectory to {traj_path_nve} ---")
traj_nve = Trajectory(traj_path_nve, "w", atoms)

for step in range(0, NVE_TOTAL_STEPS, NVE_BLOCK_STEPS):
    # Update pair list on CPU (no JAX re-trace)
    update_pair_refs(np.asarray(state_nve.position))

    # Run the compiled block of steps
    state_nve = run_nve_block(state_nve)

    # Compute diagnostics
    curr_e, curr_f = energy_and_forces_fn(state_nve.position)
    curr_e = float(curr_e)
    curr_f = np.asarray(curr_f)
    ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
    temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
    max_bond = get_max_h_x_bond(state_nve.position, box_size, h_idx_arr, x_idx_arr)

    # Check for instability and repair
    if temp > 400.0 or np.isnan(curr_e) or max_bond > MAX_HX_BOND_LIMIT:
        if max_bond > MAX_HX_BOND_LIMIT:
            print(f"[REPAIR] Peptide H-X bond broke! Max bond length: {max_bond:.2f} Å "
                  f"(max limit {MAX_HX_BOND_LIMIT} Å)")
        # 1. Scale back stretched H-X bonds
        scaled_pos, _ = scale_broken_h_bonds(
            np.asarray(state_nve.position), box_size, h_idx_arr, x_idx_arr
        )
        # 2. Minimize in PyCHARMM
        repaired_pos = repair_structure_in_charmm(scaled_pos)
        # 3. Post-repair JAX FIRE minimization
        print("[REPAIR] Running post-repair JAX FIRE minimization (200 steps)...")
        repaired_jax = jnp.array(repaired_pos, dtype=jnp.float64)
        update_pair_refs(repaired_pos)
        fire_state_rep = _init_fn_fire(repaired_jax)
        fire_state_rep = run_fire_block_repair(fire_state_rep)
        final_min_pos = jnp.array(fire_state_rep.position, dtype=jnp.float64)

        # Re-initialize NVE state
        update_pair_refs(np.asarray(final_min_pos))
        state_nve = _init_fn_nve(key, final_min_pos, target_temp_ev, mass=jax_mass)
        # Re-evaluate diagnostics
        curr_e, curr_f = energy_and_forces_fn(state_nve.position)
        curr_e = float(curr_e)
        curr_f = np.asarray(curr_f)
        ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
        temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
        max_bond = get_max_h_x_bond(state_nve.position, box_size, h_idx_arr, x_idx_arr)

    print(f"NVE Step {step+NVE_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | "
          f"Temp: {temp:.1f} K | Max H-X Bond: {max_bond:.2f} Å")

    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(state_nve.position), box_size, _mon_stacked_groups))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nve.write(frame)

traj_nve.close()
print("NVE dynamics complete!")
