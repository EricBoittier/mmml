"""JAX CGENFF bonded minimization for recovery without MLpot detach / BLOCK toggle."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array, jit

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import BondedMmMiniConfig
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

PathLike = str | Path


@dataclass(frozen=True)
class RecoveryPsfSource:
    path: Path
    temporary: bool = False

    def cleanup(self) -> None:
        if self.temporary:
            self.path.unlink(missing_ok=True)


def _ml_atom_indices(ctx: Any) -> tuple[int, ...]:
    ml_selection = getattr(ctx, "ml_selection", None)
    if ml_selection is None:
        return ()
    return tuple(int(i) for i in ml_selection.get_atom_indexes())


def resolve_recovery_psf_source(
    ctx: Any,
    topology_psf: PathLike | None = None,
) -> RecoveryPsfSource:
    """Return a PSF path suitable for ``load_cgenff_bonded_from_psf``."""
    if topology_psf is not None:
        path = Path(topology_psf).expanduser()
        if path.is_file():
            return RecoveryPsfSource(path=path.resolve(), temporary=False)
    topo = getattr(ctx, "topology_psf_path", None)
    if topo is not None and Path(topo).is_file():
        return RecoveryPsfSource(path=Path(topo).resolve(), temporary=False)

    import pycharmm.write as write

    fd, name = tempfile.mkstemp(suffix=".psf")
    import os

    os.close(fd)
    path = Path(name)
    write.psf_card(str(path))
    return RecoveryPsfSource(path=path, temporary=True)


def load_bonded_system_for_recovery(
    ctx: Any,
    positions: np.ndarray,
    *,
    topology_psf: PathLike | None = None,
    ml_atom_indices: Sequence[int] | None = None,
) -> tuple[Any, RecoveryPsfSource]:
    """Load MM-filtered CGENFF bonded topology for hybrid recovery."""
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        CgenffBondedSystem,
        filter_bonded_topology_for_mm,
        load_cgenff_bonded_from_psf,
        mm_atom_mask_complement,
    )

    psf_source = resolve_recovery_psf_source(ctx, topology_psf)
    system = load_cgenff_bonded_from_psf(psf_source.path, positions)
    if ml_atom_indices is None:
        ml_atom_indices = _ml_atom_indices(ctx)
    if ml_atom_indices:
        mm_mask = mm_atom_mask_complement(ml_atom_indices, system.n_atoms)
        topology, bonded = filter_bonded_topology_for_mm(
            system.topology,
            system.bonded,
            mm_mask,
        )
        system = CgenffBondedSystem(
            positions=system.positions,
            topology=topology,
            bonded=bonded,
            atom_types=system.atom_types,
            charges=system.charges,
        )
    return system, psf_source


def bonded_forces_grms_kcalmol_A(forces: np.ndarray | Array) -> float:
    f = np.asarray(forces, dtype=np.float64)
    if f.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(f * f, axis=-1))))


def _freeze_atom_indices(ctx: Any, ml_atom_indices: Sequence[int]) -> np.ndarray:
    """Atoms whose coordinates must stay fixed during JAX bonded mini."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    n_atoms = int(get_charmm_positions_array().shape[0])
    frozen = set(int(i) for i in ml_atom_indices)
    return np.fromiter(
        (i for i in range(n_atoms) if i in frozen),
        dtype=np.int32,
        count=len(frozen),
    )


def _apply_frozen_positions_to_fire_state(
    state: Any,
    *,
    pos0_frozen: jnp.ndarray | None,
    freeze_idx: jnp.ndarray,
) -> Any:
    """Restore fixed atom coordinates on a jax-md ``FireDescentState`` (dataclass)."""
    if pos0_frozen is None or int(freeze_idx.size) == 0:
        return state
    return replace(
        state,
        position=state.position.at[freeze_idx].set(pos0_frozen),
    )


def _run_jax_bonded_fire(
    positions: np.ndarray,
    system: Any,
    *,
    nstep: int,
    nprint: int,
    tolgrd: float,
    verbose: bool,
    freeze_indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    from jax_md import minimize as jax_minimize
    from jax_md import space

    from mmml.interfaces.pycharmmInterface.cgenff_bonded import build_bonded_energy_fn

    bonded_eval = build_bonded_energy_fn(
        system.topology,
        system.bonded,
        energy_unit="kcal/mol",
    )
    pos0 = jnp.asarray(positions, dtype=jnp.float64)
    freeze_idx = jnp.asarray(freeze_indices, dtype=jnp.int32)
    pos0_frozen = pos0[freeze_idx] if int(freeze_idx.size) > 0 else None

    def force_fn(pos: Array) -> Array:
        _, forces = bonded_eval(pos)
        if int(freeze_idx.size) > 0:
            forces = forces.at[freeze_idx].set(0.0)
        return forces

    _, shift_fn = space.free()
    init_fn, step_fn = jax_minimize.fire_descent(
        force_fn,
        shift_fn,
        dt_start=0.05,
        dt_max=0.05,
    )
    step_fn = jit(step_fn)
    state = init_fn(pos0)
    grms = bonded_forces_grms_kcalmol_A(np.asarray(force_fn(state.position)))

    if nstep <= 0:
        return np.asarray(state.position, dtype=np.float64), grms

    if verbose:
        print(
            f"Bonded JAX mini start: GRMS={grms:.4f} kcal/mol/Å "
            f"(MM bonded only, MLpot stays attached)",
            flush=True,
        )

    for step in range(1, max(1, int(nstep)) + 1):
        state = step_fn(state)
        state = _apply_frozen_positions_to_fire_state(
            state,
            pos0_frozen=pos0_frozen,
            freeze_idx=freeze_idx,
        )
        grms = bonded_forces_grms_kcalmol_A(np.asarray(force_fn(state.position)))
        if step % max(1, int(nprint)) == 0 and verbose:
            print(
                f"Bonded JAX mini step {step}/{nstep}: GRMS={grms:.4f} kcal/mol/Å",
                flush=True,
            )
        if grms <= float(tolgrd):
            if verbose:
                print(
                    f"Bonded JAX mini converged at step {step}: GRMS={grms:.4f} kcal/mol/Å",
                    flush=True,
                )
            break

    if verbose:
        print(
            f"Bonded JAX mini end: GRMS={grms:.4f} kcal/mol/Å",
            flush=True,
        )
    return np.asarray(state.position, dtype=np.float64), grms


def minimize_bonded_jax_recovery(
    ctx: Any,
    config: "BondedMmMiniConfig",
    *,
    topology_psf: PathLike | None = None,
) -> float | None:
    """Relax MM bonded strain in JAX; sync coords to CHARMM without detaching MLpot."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        invalidate_mlpot_calculator_caches,
        sync_charmm_lists_after_mini,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    positions = np.asarray(get_charmm_positions_array(), dtype=np.float64)
    ml_indices = _ml_atom_indices(ctx)
    system, psf_source = load_bonded_system_for_recovery(
        ctx,
        positions,
        topology_psf=topology_psf,
        ml_atom_indices=ml_indices,
    )
    try:
        n_bonds = int(np.asarray(system.topology.bonds).shape[0])
        if n_bonds <= 0:
            if config.verbose:
                print(
                    "Bonded JAX mini skipped: no MM bonded terms (all-ML cluster)",
                    flush=True,
                )
            return None
        nstep = int(config.nstep_jax if config.nstep_jax is not None else config.nstep_sd)
        freeze_indices = _freeze_atom_indices(ctx, ml_indices)
        new_positions, grms = _run_jax_bonded_fire(
            positions,
            system,
            nstep=nstep,
            nprint=max(1, int(config.nprint)),
            tolgrd=float(config.tolgrd),
            verbose=bool(config.verbose),
            freeze_indices=freeze_indices,
        )
        sync_charmm_positions(new_positions)
        sync_charmm_lists_after_mini(quiet=True)
        invalidate_mlpot_calculator_caches(ctx)
        _ = get_charmm_positions_array()
        return grms
    finally:
        psf_source.cleanup()
