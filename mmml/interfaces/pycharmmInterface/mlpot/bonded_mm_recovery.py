"""Bonded-MM recovery mini after bad conformations (e.g. post-heat)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    BondedMmMiniConfig,
    charmm_bonded_term_kcalmol,
    charmm_internal_energy_kcalmol,
    measure_mm_grms_with_full_block,
    minimize_bonded_mm_recovery,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

PathLike = str | Path


@dataclass(frozen=True)
class MmStrainBaseline:
    """Reference MM strain after the first CHARMM MM-only pre-minimize."""

    grms_kcalmol_A: float
    internal_kcalmol: float | None = None
    angl_kcalmol: float | None = None


@dataclass(frozen=True)
class BondedMmRecoveryResult:
    """Outcome of a bonded-MM recovery check."""

    ran_recovery: bool
    current: MmStrainBaseline | None = None
    reasons: tuple[str, ...] = ()
    heavy_reload: bool = False


def rewrite_dynamics_restart_from_current_state(
    restart_path: PathLike | None,
    *,
    write_unit: int = 92,
) -> None:
    """Write a ``WRITe restart`` snapshot from the current in-memory CHARMM state.

    Used before NPT stages after ``memory_handoff`` (e.g. post-heat bonded-MM mini)
    so overlap-chunk ``READYN`` has a valid restart (coords, velocities, crystal).
    """
    if restart_path is None:
        return
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    path = Path(restart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script(
            f"open write form unit {int(write_unit)} name {path}\n"
            f"write restart unit {int(write_unit)}\n"
            f"close unit {int(write_unit)}\n"
        )


def rewrite_dynamics_restart_validated(
    restart_path: PathLike | None,
    *,
    write_unit: int = 92,
) -> bool:
    """Write restart from memory; return False when Cartesian coords are non-finite."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        restart_has_nonfinite_coordinates,
    )

    if restart_path is None:
        return True
    rewrite_dynamics_restart_from_current_state(restart_path, write_unit=write_unit)
    return not restart_has_nonfinite_coordinates(Path(restart_path))


def restore_post_rescue_coordinates(
    *,
    rescued_positions: np.ndarray | None = None,
    rescue_crd: PathLike | None = None,
    prior_restart: PathLike | None = None,
) -> str:
    """Load finite coordinates into CHARMM after a bad post-rescue restart write."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import load_minimized_coordinates
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        restart_has_nonfinite_coordinates,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    if rescued_positions is not None:
        pos = np.asarray(rescued_positions, dtype=float)
        if pos.size and np.all(np.isfinite(pos)):
            sync_charmm_positions(pos)
            return "in-memory rescue positions"

    if rescue_crd is not None:
        crd = Path(rescue_crd).expanduser()
        if crd.is_file():
            load_minimized_coordinates(crd)
            pos = get_charmm_positions_array()
            if np.all(np.isfinite(pos)):
                return f"rescue CRD {crd.name}"

    if prior_restart is not None:
        prior = Path(prior_restart).expanduser()
        if prior.is_file() and not restart_has_nonfinite_coordinates(prior):
            restore_charmm_state_from_restart(prior)
            return f"prior restart {prior.name}"

    raise RuntimeError(
        "post-rescue restart fallback failed: no finite coordinates in rescued "
        "positions, rescue CRD, or prior segment restart"
    )


def restore_charmm_state_from_restart(
    restart_path: PathLike,
    *,
    read_unit: int = 93,
) -> None:
    """Load coordinates (and crystal state) from a CHARMM restart into memory."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinates,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    path = Path(restart_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"restart not found: {path}")
    pos = read_restart_coordinates(path)
    if pos is None:
        raise RuntimeError(
            f"restart {path.name} has no finite Cartesian coordinates in !X, Y, Z"
        )

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script(
            f"open read unit {int(read_unit)} name {path}\n"
            f"read restart unit {int(read_unit)}\n"
            f"close unit {int(read_unit)}\n"
        )
    sync_charmm_positions(pos)


def _run_all_ml_extent_recovery(
    ctx: MlpotContext,
    config: Any,
    bonded_cfg: BondedMmMiniConfig,
    *,
    positions: np.ndarray,
) -> None:
    """Fly-off recovery for all-ML: reload PSF using restart-file coords (no noise)."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        MinimizeWithMlpotConfig,
        minimize_with_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import clear_mmfp_restraints

    topo = getattr(config, "topology_psf", None)
    if topo is None or not Path(topo).is_file():
        raise RuntimeError(
            "all-ML fly-off recovery requires pre-MLpot topology PSF "
            "(cluster_for_vmd_*.psf)"
        )
    if bonded_cfg.verbose:
        print(
            f"Fly-off recovery: reloading pre-MLpot topology {Path(topo).name} "
            f"with coords from prior restart",
            flush=True,
        )
    _reload_pre_mlpot_topology(ctx, topology_psf=topo, positions=positions)
    try:
        clear_mmfp_restraints()
        assert_bonded_mm_energy_active(context="Fly-off recovery (reloaded PSF)")
        _run_bonded_sd_without_mlpot(ctx, bonded_cfg)
    finally:
        _reregister_mlpot_after_topology_reload(ctx)

    n_mlpot = int(getattr(config, "mlpot_rescue_mini_nstep", 0) or 0)
    pyCModel = getattr(config, "pyCModel", None)
    if n_mlpot > 0 and pyCModel is not None:
        if bonded_cfg.verbose:
            print(
                f"Fly-off recovery: MLpot SD mini ({n_mlpot} steps)",
                flush=True,
            )
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=n_mlpot,
                nprint=bonded_cfg.nprint,
                verbose=bonded_cfg.verbose,
                pyCModel=pyCModel,
                mlpot_ctx=ctx,
                save=False,
                skip_if_crd_exists=False,
            )
        )


def run_extent_recovery_from_prior_restart(
    ctx: MlpotContext,
    config: Any,
    *,
    prior_restart: PathLike,
) -> None:
    """Restore the prior segment restart, then run bonded-MM SD on those coords."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_bonded_mm_recovery
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_coordinates,
    )

    path = Path(prior_restart).expanduser().resolve()
    restore_charmm_state_from_restart(path)
    bonded_cfg = _bonded_cfg_from_overlap_config(config)
    pos = read_restart_coordinates(path)
    if pos is None:
        raise RuntimeError(
            f"fly-off recovery: parsed no finite coordinates from {path.name}"
        )
    if _mlpot_covers_all_atoms(ctx):
        _run_all_ml_extent_recovery(ctx, config, bonded_cfg, positions=pos)
        return
    minimize_bonded_mm_recovery(ctx, bonded_cfg)


def record_mm_baseline_strain(*, verbose: bool = False) -> MmStrainBaseline | None:
    """Record MM GRMS (+ internal energy when readable) after MM-only pre-minimize."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    apply_charmm_mm_block()
    run_charmm_script_quiet("ENER")
    baseline = MmStrainBaseline(
        grms_kcalmol_A=float(charmm_grms()),
        internal_kcalmol=charmm_internal_energy_kcalmol(),
        angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
    )
    if verbose:
        msg = f"MM baseline GRMS: {baseline.grms_kcalmol_A:.4f} kcal/mol/Å"
        if baseline.angl_kcalmol is not None:
            msg += f", ANGL: {baseline.angl_kcalmol:.4f} kcal/mol"
        if baseline.internal_kcalmol is not None:
            msg += f", internal: {baseline.internal_kcalmol:.4f} kcal/mol"
        print(msg, flush=True)
    return baseline


def measure_mm_bonded_strain_with_full_block(ctx: MlpotContext) -> MmStrainBaseline:
    """GRMS + bonded internal + ANGL after ``ENER`` with full MM block (MLpot detached)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _with_mlpot_detached

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    def _measure() -> MmStrainBaseline:
        apply_charmm_mm_block()
        run_charmm_script_quiet("ENER")
        return MmStrainBaseline(
            grms_kcalmol_A=float(charmm_grms()),
            internal_kcalmol=charmm_internal_energy_kcalmol(),
            angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
        )

    return _with_mlpot_detached(ctx, _measure)


def _bonded_strain_margins(args: argparse.Namespace) -> tuple[float, float, float]:
    grms_margin = float(
        getattr(args, "bonded_mm_grms_margin", None)
        if getattr(args, "bonded_mm_grms_margin", None) is not None
        else getattr(args, "bonded_mm_internal_margin", 0.0)
    )
    internal_margin = float(getattr(args, "bonded_mm_internal_energy_margin", 0.0) or 0.0)
    angl_margin = float(getattr(args, "bonded_mm_angl_margin", 0.0) or 0.0)
    return grms_margin, internal_margin, angl_margin


def _mlpot_covers_all_atoms(ctx: MlpotContext) -> bool:
    """True when CHARMM MM terms cannot be used as a recovery potential."""
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm

        n_atoms = int(pycharmm.coor.get_natom())
        n_ml = len(ctx.ml_selection.get_atom_indexes()) if ctx.ml_selection is not None else 0
        return n_atoms > 0 and n_ml >= n_atoms
    except Exception:
        return False


def apply_charmm_position_noise(
    *,
    amplitude_A: float,
    seed: int | None = None,
) -> None:
    """Perturb in-memory coordinates (Å) to escape bad local minima before MM rescue."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    amp = float(amplitude_A)
    if amp <= 0.0:
        return
    pos = get_charmm_positions_array()
    rng = np.random.default_rng(None if seed is None else int(seed))
    sync_charmm_positions(pos + rng.normal(0.0, amp, pos.shape))


def assert_bonded_mm_energy_active(*, context: str = "bonded-MM rescue") -> None:
    """Raise when CHARMM bonded internals read zero after MM BLOCK setup."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    run_charmm_script_quiet("ENER")
    bond = charmm_bonded_term_kcalmol("BOND")
    angl = charmm_bonded_term_kcalmol("ANGL")
    grms = float(charmm_grms())
    bonded_zero = (
        (bond is None or abs(float(bond)) <= 1.0e-8)
        and (angl is None or abs(float(angl)) <= 1.0e-8)
        and grms <= 1.0e-8
    )
    if bonded_zero:
        raise RuntimeError(
            f"{context}: CHARMM bonded MM terms read zero after BLOCK setup "
            "(BOND/ANGL/GRMS). All-ML workflows must reload the pre-MLpot topology "
            "PSF before bonded rescue."
        )


def _bonded_cfg_from_overlap_config(config: Any) -> BondedMmMiniConfig:
    sd_steps = getattr(config, "intra_rescue_sd_steps", None)
    if sd_steps is None:
        sd_steps = config.rescue.nstep_sd
    return BondedMmMiniConfig(
        nstep_sd=int(sd_steps),
        nprint=max(1, int(config.rescue.nprint)),
        tolenr=float(config.rescue.tolenr),
        tolgrd=float(config.rescue.tolgrd),
        verbose=config.rescue.verbose,
    )


def _run_all_ml_intra_overlap_rescue(
    ctx: MlpotContext,
    config: Any,
    bonded_cfg: BondedMmMiniConfig,
) -> None:
    """All-ML intra rescue: noise + MLpot SD only (no PSF reload).

    Reloading ``cluster_for_vmd_*.psf`` via ``DELETE ATOM ALL`` after dynamics
    segfaults on MPI-linked CHARMM.  Bonded MM is inactive under all-ML BLOCK
    anyway, so MLpot SD is the appropriate fix for intra-monomer close contacts.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        MinimizeWithMlpotConfig,
        minimize_with_mlpot,
    )

    noise = float(getattr(config, "position_noise_A", 0.05) or 0.0)
    seed = getattr(config, "recovery_seed", None)
    if noise > 0.0:
        if bonded_cfg.verbose:
            print(
                f"Intra overlap rescue: applying {noise:.3f} Å coordinate noise "
                f"(seed={seed})",
                flush=True,
            )
        apply_charmm_position_noise(amplitude_A=noise, seed=seed)

    pyCModel = getattr(config, "pyCModel", None)
    if pyCModel is None:
        raise RuntimeError(
            "all-ML intra overlap rescue requires pyCModel on overlap config"
        )

    sd_steps = int(
        getattr(config, "intra_rescue_sd_steps", None) or bonded_cfg.nstep_sd
    )
    if bonded_cfg.verbose:
        print(
            f"Intra overlap rescue (all-ML): MLpot SD {sd_steps} steps "
            "(no pre-MLpot PSF reload)",
            flush=True,
        )
    minimize_with_mlpot(
        MinimizeWithMlpotConfig(
            nstep=sd_steps,
            nprint=bonded_cfg.nprint,
            tolenr=bonded_cfg.tolenr,
            tolgrd=bonded_cfg.tolgrd,
            verbose=bonded_cfg.verbose,
            pyCModel=pyCModel,
            mlpot_ctx=ctx,
            save=False,
            skip_if_crd_exists=False,
        )
    )


def _run_bonded_vdw_sd_without_mlpot(
    ctx: MlpotContext,
    rescue: Any,
) -> float | None:
    """Bonded+VDW SD/ABNR on a reloaded pre-MLpot PSF (MLpot not registered)."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_vdw_recovery_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _bonded_recovery_sd_kwargs,
        _import_pycharmm_modules,
        _prepare_overlap_rescue_lists,
    )

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    bonded_cfg = BondedMmMiniConfig(
        nstep_sd=int(rescue.nstep_sd),
        nprint=max(1, int(rescue.nprint)),
        tolenr=float(rescue.tolenr),
        tolgrd=float(rescue.tolgrd),
        verbose=bool(rescue.verbose),
        show_energy=False,
    )
    apply_bonded_vdw_recovery_block()
    _prepare_overlap_rescue_lists(ctx)
    pycharmm, cons_fix, *_ = _import_pycharmm_modules()
    minimize = _import_pycharmm_modules()[3]
    run_charmm_script_quiet("ENER")
    assert_bonded_mm_energy_active(context="Inter overlap rescue (reloaded PSF)")
    grms_before = float(charmm_grms())
    if rescue.verbose:
        print(
            f"Inter overlap rescue start: GRMS={grms_before:.4f} kcal/mol/Å "
            "(pre-MLpot topology, bonded+VDW, MLpot detached)",
            flush=True,
        )
    try:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output

        if int(rescue.nstep_sd) > 0:
            with charmm_quiet_output():
                minimize.run_sd(**_bonded_recovery_sd_kwargs(ctx, bonded_cfg))
        if int(rescue.nstep_abnr) > 0:
            with charmm_quiet_output():
                minimize.run_abnr(
                    nstep=int(rescue.nstep_abnr),
                    tolenr=float(rescue.tolenr),
                    tolgrd=float(rescue.tolgrd),
                )
        grms = float(charmm_grms())
        if rescue.verbose:
            print(
                f"Inter overlap rescue end: GRMS={grms:.4f} kcal/mol/Å",
                flush=True,
            )
        return grms
    finally:
        cons_fix.turn_off()


def _run_all_ml_inter_overlap_rescue(
    ctx: MlpotContext,
    config: Any,
) -> None:
    """Reload clean PSF, bonded+VDW SD/ABNR, reattach MLpot, optional ML SD mini."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        MinimizeWithMlpotConfig,
        minimize_with_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import clear_mmfp_restraints

    topo = getattr(config, "topology_psf", None)
    rescue = config.rescue
    if topo is None or not Path(topo).is_file():
        raise RuntimeError(
            "all-ML inter-monomer rescue requires pre-MLpot topology PSF "
            "(cluster_for_vmd_*.psf); reload restores CHARMM VDW between monomers "
            "while MLpot is detached"
        )

    noise = float(getattr(config, "position_noise_A", 0.05) or 0.0)
    seed = getattr(config, "recovery_seed", None)
    if noise > 0.0:
        if rescue.verbose:
            print(
                f"Inter overlap rescue: applying {noise:.3f} Å coordinate noise "
                f"(seed={seed})",
                flush=True,
            )
        apply_charmm_position_noise(amplitude_A=noise, seed=seed)

    if rescue.verbose:
        print(
            f"Inter overlap rescue: reloading pre-MLpot topology {Path(topo).name}",
            flush=True,
        )
    _reload_pre_mlpot_topology(ctx, topology_psf=topo)
    try:
        clear_mmfp_restraints()
        _run_bonded_vdw_sd_without_mlpot(ctx, rescue)
    finally:
        _reregister_mlpot_after_topology_reload(ctx)

    n_mlpot = int(getattr(config, "mlpot_rescue_mini_nstep", 0) or 0)
    pyCModel = getattr(config, "pyCModel", None)
    if n_mlpot > 0 and pyCModel is not None:
        if rescue.verbose:
            print(
                f"Inter overlap rescue: MLpot SD mini ({n_mlpot} steps)",
                flush=True,
            )
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=n_mlpot,
                nprint=max(1, int(rescue.nprint)),
                verbose=rescue.verbose,
                pyCModel=pyCModel,
                mlpot_ctx=ctx,
                save=False,
                skip_if_crd_exists=False,
            )
        )


def run_inter_monomer_overlap_rescue(
    ctx: MlpotContext,
    config: Any,
) -> None:
    """Recover from inter-monomer close contacts during dynamics overlap checks."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_overlap_rescue

    if _mlpot_covers_all_atoms(ctx):
        _run_all_ml_inter_overlap_rescue(ctx, config)
        return
    minimize_overlap_rescue(ctx, config.rescue)


def run_intra_monomer_overlap_rescue(
    ctx: MlpotContext,
    config: Any,
) -> None:
    """Recover from intra-monomer close contacts during dynamics overlap checks."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_bonded_mm_recovery

    bonded_cfg = _bonded_cfg_from_overlap_config(config)
    if _mlpot_covers_all_atoms(ctx):
        _run_all_ml_intra_overlap_rescue(ctx, config, bonded_cfg)
        return
    minimize_bonded_mm_recovery(ctx, bonded_cfg)


def _copy_mlpot_context_state(dst: MlpotContext, src: MlpotContext) -> None:
    """Keep the caller's context object valid after rebuilding CHARMM topology."""
    for name in (
        "mlpot",
        "pyCModel",
        "params",
        "model",
        "ml_selection",
        "block_tag",
        "ml_Z",
        "use_pbc",
        "cubic_box_side_A",
        "charmm_cubic_box_side_A",
        "ml_charge",
        "ml_fq",
        "mm_internal_scale",
    ):
        setattr(dst, name, getattr(src, name))


def _measure_current_mm_strain() -> MmStrainBaseline:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    apply_charmm_mm_block()
    run_charmm_script_quiet("ENER")
    return MmStrainBaseline(
        grms_kcalmol_A=float(charmm_grms()),
        internal_kcalmol=charmm_internal_energy_kcalmol(),
        angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
    )


def _reload_pre_mlpot_topology(
    ctx: MlpotContext,
    *,
    topology_psf: PathLike,
    positions: np.ndarray | None = None,
) -> None:
    """Replace MLpot-mutated PSF with the saved pre-MLpot topology."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    if positions is None:
        current_positions = get_charmm_positions_array()
    else:
        current_positions = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(current_positions)):
        raise RuntimeError(
            "pre-MLpot topology reload requires finite coordinates "
            f"(got {int(np.sum(~np.isfinite(current_positions)))} non-finite values)"
        )

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

    ctx.unset()
    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script(
            "DELETE ATOM SELE ALL END\nDELETE PSF SELE ALL END"
        )
        read.psf_card(str(Path(topology_psf).expanduser().resolve()))
    sync_charmm_positions(current_positions)
    charmm_side = getattr(ctx, "charmm_cubic_box_side_A", None) or (
        ctx.cubic_box_side_A if ctx.use_pbc else None
    )
    if charmm_side is not None:
        setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(charmm_side))
    else:
        setup_default_nbonds()


def _reregister_mlpot_after_topology_reload(ctx: MlpotContext) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    positions = get_charmm_positions_array()
    # ``_reload_pre_mlpot_topology`` already rebuilt CHARMM PBC/vacuum nbonds with
    # MLpot detached. A fresh ``register_mlpot`` or ``refresh_nbonds_after_mlpot_pbc``
    # (``update_bnbnd`` / ``upinb``) segfaults for all-ML PBC clusters.
    ctx.reregister_mlpot()
    sync_charmm_positions(positions)


def _run_bonded_sd_without_mlpot(
    ctx: MlpotContext,
    config: BondedMmMiniConfig,
) -> float | None:
    """Run bonded-only SD after the pre-MLpot PSF has already been reloaded."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_mm_only_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _bonded_recovery_sd_kwargs,
        _import_pycharmm_modules,
    )

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    apply_bonded_mm_only_block()
    pycharmm, cons_fix, *_ = _import_pycharmm_modules()
    minimize = _import_pycharmm_modules()[3]
    run_charmm_script_quiet("ENER")
    assert_bonded_mm_energy_active(context="Bonded-MM heavy mini")
    grms_before = float(charmm_grms())
    angl_before = charmm_bonded_term_kcalmol("ANGL")
    if config.verbose:
        msg = (
            f"Bonded-MM heavy mini start: GRMS={grms_before:.4f} kcal/mol/Å "
            "(pre-MLpot topology, MLpot detached)"
        )
        if angl_before is not None:
            msg += f", ANGL={angl_before:.4f} kcal/mol"
        print(msg, flush=True)
    if config.nstep_sd > 0:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output

        with charmm_quiet_output():
            minimize.run_sd(**_bonded_recovery_sd_kwargs(ctx, config))
    run_charmm_script_quiet("ENER")
    grms_after = float(charmm_grms())
    if config.verbose:
        msg = f"Bonded-MM heavy mini end: GRMS={grms_after:.4f} kcal/mol/Å"
        angl_after = charmm_bonded_term_kcalmol("ANGL")
        if angl_after is not None:
            msg += f", ANGL={angl_after:.4f} kcal/mol"
        print(msg, flush=True)
    cons_fix.turn_off()
    return grms_after


def _run_heavy_bonded_recovery_check(
    ctx: MlpotContext,
    args: argparse.Namespace,
    *,
    stage: str,
    baseline: MmStrainBaseline,
    topology_psf: PathLike,
) -> BondedMmRecoveryResult:
    """Reload the clean topology, check real MM strain, optionally recover, and reattach MLpot."""
    path = Path(topology_psf).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pre-MLpot topology PSF not found: {path}")

    if not args.quiet:
        print(
            f"bonded-MM-mini: reloading pre-MLpot topology for all-ML recovery: {path.name}",
            flush=True,
        )
    _reload_pre_mlpot_topology(ctx, topology_psf=path)
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.restraints import clear_mmfp_restraints

        clear_mmfp_restraints()
        current = _measure_current_mm_strain()
        grms_margin, internal_margin, angl_margin = _bonded_strain_margins(args)
        reasons = tuple(
            _recovery_reasons(
                current,
                baseline,
                grms_margin=grms_margin,
                internal_margin=internal_margin,
                angl_margin=angl_margin,
            )
        )
        if not reasons:
            if not args.quiet:
                print(
                    f"bonded-MM-mini: strain OK after {stage} with reloaded topology "
                    f"(GRMS {current.grms_kcalmol_A:.4f} kcal/mol/Å)",
                    flush=True,
                )
            return BondedMmRecoveryResult(
                ran_recovery=False,
                current=current,
                reasons=reasons,
                heavy_reload=True,
            )

        if not args.quiet:
            print(
                f"bonded-MM-mini: after {stage}: {'; '.join(reasons)}; "
                "running heavy bonded SD (pre-MLpot topology)",
                flush=True,
            )
        nstep = int(getattr(args, "bonded_mm_mini_steps", 50))
        _run_bonded_sd_without_mlpot(
            ctx,
            BondedMmMiniConfig(
                nstep_sd=nstep,
                nprint=max(1, int(getattr(args, "dyn_nprint", 500))),
                verbose=not args.quiet,
                show_energy=bool(getattr(args, "show_energy", False)),
            ),
        )
        return BondedMmRecoveryResult(
            ran_recovery=True,
            current=current,
            reasons=reasons,
            heavy_reload=True,
        )
    finally:
        _reregister_mlpot_after_topology_reload(ctx)
        _restore_flat_bottom_after_heavy_recovery(args)


def _restore_flat_bottom_after_heavy_recovery(args: argparse.Namespace) -> None:
    """Reinstall the workflow MMFP wall after temporarily clearing it for MM recovery."""
    fb_rad = getattr(args, "fb_rad", None)
    if fb_rad is None or float(fb_rad) <= 0:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_flat_bottom_selection,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
        FlatBottomSphereConfig,
        setup_flat_bottom_sphere_mmfp,
    )

    setup_flat_bottom_sphere_mmfp(
        FlatBottomSphereConfig(
            radius=float(fb_rad),
            force=float(getattr(args, "fb_forc", 1.0)),
            xref=float(getattr(args, "fb_xref", 0.0)),
            yref=float(getattr(args, "fb_yref", 0.0)),
            zref=float(getattr(args, "fb_zref", 0.0)),
            selection=resolve_flat_bottom_selection(args),
        )
    )


def _recovery_reasons(
    current: MmStrainBaseline,
    baseline: MmStrainBaseline,
    *,
    grms_margin: float,
    internal_margin: float,
    angl_margin: float,
) -> list[str]:
    reasons: list[str] = []
    grms_thr = float(baseline.grms_kcalmol_A) + grms_margin
    if current.grms_kcalmol_A > grms_thr:
        reasons.append(
            f"GRMS {current.grms_kcalmol_A:.4f} > {grms_thr:.4f} kcal/mol/Å"
        )
    if (
        baseline.internal_kcalmol is not None
        and current.internal_kcalmol is not None
        and internal_margin > 0.0
    ):
        int_thr = float(baseline.internal_kcalmol) + internal_margin
        if current.internal_kcalmol > int_thr:
            reasons.append(
                f"internal {current.internal_kcalmol:.4f} > {int_thr:.4f} kcal/mol"
            )
    if (
        baseline.angl_kcalmol is not None
        and current.angl_kcalmol is not None
        and angl_margin > 0.0
    ):
        angl_thr = float(baseline.angl_kcalmol) + angl_margin
        if current.angl_kcalmol > angl_thr:
            reasons.append(
                f"ANGL {current.angl_kcalmol:.4f} > {angl_thr:.4f} kcal/mol"
            )
    return reasons


def assert_pre_min_bonded_geometry(
    args: argparse.Namespace,
    *,
    baseline: MmStrainBaseline | None = None,
) -> None:
    """Abort (unless allowed) when Packmol/MM pre-min leaves high ANGL or internal strain."""
    import os
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block

    if not getattr(args, "bonded_mm_mini", False):
        return

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    apply_charmm_mm_block()
    run_charmm_script_quiet("ENER")
    angl = charmm_bonded_term_kcalmol("ANGL")
    internal = charmm_internal_energy_kcalmol()
    max_angl = getattr(args, "bonded_mm_max_angl_kcal", None)
    max_internal = getattr(args, "bonded_mm_max_internal_kcal", None)
    problems: list[str] = []
    if max_angl is not None and angl is not None and float(angl) > float(max_angl):
        problems.append(f"ANGL {float(angl):.4f} > limit {float(max_angl):.4f} kcal/mol")
    if (
        max_internal is not None
        and internal is not None
        and float(internal) > float(max_internal)
    ):
        problems.append(
            f"internal {float(internal):.4f} > limit {float(max_internal):.4f} kcal/mol"
        )
    if not problems:
        if not args.quiet and angl is not None:
            msg = f"Pre-min bonded geometry: ANGL={float(angl):.4f} kcal/mol"
            if internal is not None:
                msg += f", internal={float(internal):.4f} kcal/mol"
            if baseline is not None:
                msg += f" (baseline GRMS {baseline.grms_kcalmol_A:.4f} kcal/mol/Å)"
            print(msg, flush=True)
        return

    allow = bool(getattr(args, "allow_high_bonded_strain", False)) or (
        (os.environ.get("MMML_MLPOT_ALLOW_HIGH_BONDED_STRAIN") or "")
        .strip()
        .lower()
        in ("1", "yes", "true")
    )
    text = (
        "Pre-min bonded strain limits exceeded after CHARMM MM minimize:\n  "
        + "\n  ".join(problems)
        + "\nIncrease --charmm-sd-steps / --charmm-abnr-steps, relax Packmol, "
        "or pass --allow-high-bonded-strain (not recommended)."
    )
    if allow:
        print(f"WARN: {text}", file=sys.stderr, flush=True)
        return
    print(text, file=sys.stderr, flush=True)
    raise SystemExit(1)


def record_mm_baseline_internal_energy(
    *,
    verbose: bool = False,
) -> float | None:
    """Legacy helper: internal energy only (prefer :func:`record_mm_baseline_strain`)."""
    baseline = record_mm_baseline_strain(verbose=verbose)
    return baseline.internal_kcalmol if baseline is not None else None


def _bonded_mm_skip_reason_after_heat_overlap(
    ctx: MlpotContext,
    args: argparse.Namespace,
    *,
    stage: str,
) -> str | None:
    """Block heavy bonded-MM reload when heat left inter-monomer overlaps."""
    if stage.lower() != "heat":
        return None
    action = str(getattr(args, "dynamics_overlap_action", "rescue") or "rescue").lower()
    if action == "off":
        return None
    n_monomers = int(getattr(ctx, "n_monomers", 0) or 0)
    if n_monomers < 2:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        measure_worst_intermonomer_distance,
        resolve_dynamics_overlap_config,
    )

    cfg = resolve_dynamics_overlap_config(
        args,
        n_monomers=n_monomers,
        use_pbc=bool(getattr(ctx, "use_pbc", False)),
    )
    if not cfg.enabled:
        return None
    worst = measure_worst_intermonomer_distance(cfg)
    if worst >= cfg.min_distance_A:
        return None
    return (
        f"worst inter-monomer distance {worst:.3f} Å < "
        f"--dynamics-overlap-min-distance {cfg.min_distance_A:.3f} Å "
        "(fix heat/overlap before bonded-MM on pre-MLpot topology)"
    )


def maybe_run_bonded_mm_mini_after_stage(
    ctx: MlpotContext,
    args: argparse.Namespace,
    *,
    stage: str,
    baseline: MmStrainBaseline | None,
    restart_path: PathLike | None = None,
    topology_psf: PathLike | None = None,
    mini_registry: Any = None,
    snapshot_spec: Any = None,
    snapshot_paths: dict[str, Path] | None = None,
) -> bool:
    """If MM bonded strain rose above baseline, run bonded-only SD (BLOCK toggle only).

    Triggers when GRMS exceeds baseline + ``--bonded-mm-grms-margin``, and/or (if margins
    > 0) internal or ANGL exceed baseline + their margins.

    Returns True when recovery SD ran (caller should hand off next stage from memory).
    """
    if not getattr(args, "bonded_mm_mini", False):
        return False
    raw = str(getattr(args, "bonded_mm_mini_after", "mini") or "mini")
    watch = {s.strip().lower() for s in raw.split(",") if s.strip()}
    if stage.lower() not in watch:
        return False
    if baseline is None:
        if not args.quiet:
            print(
                "bonded-MM-mini: no baseline strain recorded; skipping check",
                flush=True,
            )
        return False

    def _record_bonded_snapshot() -> None:
        if (
            mini_registry is None
            or snapshot_spec is None
            or snapshot_paths is None
            or not bool(getattr(args, "save", True))
        ):
            return
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import save_minimization_results

        written = save_minimization_results(
            pdb_path=snapshot_paths.get("pdb"),
            crd_path=snapshot_paths.get("crd"),
            title=snapshot_spec.label,
        )
        mini_registry.record(
            snapshot_spec,
            written,
            grms_kcalmol_A=float(charmm_grms()),
        )

    if _mlpot_covers_all_atoms(ctx):
        if topology_psf is not None:
            skip_reason = _bonded_mm_skip_reason_after_heat_overlap(ctx, args, stage=stage)
            if skip_reason is not None:
                if not args.quiet:
                    print(
                        f"bonded-MM-mini: skipping heavy recovery after {stage}: {skip_reason}",
                        flush=True,
                    )
                return False
            result = _run_heavy_bonded_recovery_check(
                ctx,
                args,
                stage=stage,
                baseline=baseline,
                topology_psf=topology_psf,
            )
            if result.ran_recovery:
                _record_bonded_snapshot()
            return result.ran_recovery
        if not args.quiet:
            print(
                f"bonded-MM-mini: skipping after {stage} for all-ML system; "
                "CHARMM bonded terms are unavailable after MLpot registration "
                "and no pre-MLpot topology PSF was provided",
                flush=True,
            )
        return False

    grms_margin, internal_margin, angl_margin = _bonded_strain_margins(args)
    current = measure_mm_bonded_strain_with_full_block(ctx)
    reasons = _recovery_reasons(
        current,
        baseline,
        grms_margin=grms_margin,
        internal_margin=internal_margin,
        angl_margin=angl_margin,
    )
    if not reasons:
        if not args.quiet:
            msg = f"bonded-MM-mini: strain OK after {stage} (GRMS {current.grms_kcalmol_A:.4f}"
            if current.angl_kcalmol is not None:
                msg += f", ANGL {current.angl_kcalmol:.4f}"
            if current.internal_kcalmol is not None:
                msg += f", internal {current.internal_kcalmol:.4f}"
            msg += " kcal/mol)"
            print(msg, flush=True)
        return False

    if not args.quiet:
        print(
            f"bonded-MM-mini: after {stage}: {'; '.join(reasons)}; "
            f"running bonded SD (MLpot detached)",
            flush=True,
        )
    nstep = int(getattr(args, "bonded_mm_mini_steps", 50))
    minimize_bonded_mm_recovery(
        ctx,
        BondedMmMiniConfig(
            nstep_sd=nstep,
            nprint=max(1, int(getattr(args, "dyn_nprint", 500))),
            verbose=not args.quiet,
            show_energy=bool(getattr(args, "show_energy", False)),
        ),
    )
    _record_bonded_snapshot()
    if not args.quiet:
        print(
            f"bonded-MM-mini: next stage will continue from in-memory coordinates "
            f"(restart file unchanged)",
            flush=True,
        )
    return True
