"""Inter-monomer overlap checks during PyCHARMM MLpot dynamics."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

DynamicsOverlapAction = Literal["error", "warn", "rescue", "off"]


@dataclass(frozen=True)
class OverlapRescueConfig:
    """CHARMM bonded+VDW SD/ABNR while MLpot stays registered."""

    nstep_sd: int = 200
    nstep_abnr: int = 400
    nprint: int = 50
    tolenr: float = 1e-3
    tolgrd: float = 1e-3
    verbose: bool = False


@dataclass(frozen=True)
class DynamicsOverlapConfig:
    """Chunked dynamics overlap guard (see :func:`run_dynamics_with_io`)."""

    action: DynamicsOverlapAction = "rescue"
    min_distance_A: float = 1.5
    intra_min_distance_A: float = 1.0
    intra_exclude_1_3: bool = True
    intra_rescue_sd_steps: int | None = None
    check_interval: int = 500
    n_monomers: int = 1
    use_pbc: bool = False
    fallback_box_side_A: float | None = None
    rescue: OverlapRescueConfig = field(default_factory=OverlapRescueConfig)
    separate_on_rescue_fail: bool = True
    separate_margin_A: float = 0.2

    @property
    def enabled(self) -> bool:
        return (
            self.action != "off"
            and float(self.min_distance_A) > 0.0
            and int(self.n_monomers) > 1
        )

    @property
    def intra_enabled(self) -> bool:
        return self.action != "off" and float(self.intra_min_distance_A) > 0.0


def add_dynamics_overlap_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Dynamics overlap guard (PyCHARMM MLpot)")
    group.add_argument(
        "--dynamics-overlap-action",
        choices=("error", "warn", "rescue", "off"),
        default="rescue",
        help=(
            "On inter-monomer overlap during MD: rescue=CHARMM bonded+VDW mini, then "
            "rigid monomer push if still overlapped (default); error=abort, "
            "warn=log only, off=disable. Also controls intra-monomer close-contact checks."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-charmm-sd-steps",
        type=int,
        default=200,
        help="CHARMM SD steps for overlap rescue (default: 200).",
    )
    group.add_argument(
        "--dynamics-overlap-charmm-abnr-steps",
        type=int,
        default=400,
        help="CHARMM ABNR steps for overlap rescue (default: 400).",
    )
    group.add_argument(
        "--dynamics-overlap-min-distance",
        type=float,
        default=1.5,
        metavar="ANG",
        help=(
            "Minimum allowed inter-monomer atom distance in Å during dynamics "
            "(default: 1.5; CHARMM close-contact warnings often appear near this)."
        ),
    )
    group.add_argument(
        "--dynamics-intra-min-distance",
        type=float,
        default=1.0,
        metavar="ANG",
        help=(
            "Minimum allowed nonbonded atom distance within each monomer (1–2 and 1–3 "
            "pairs excluded from PSF bonds). Set 0 to disable (default: 1.0 Å)."
        ),
    )
    group.add_argument(
        "--no-dynamics-intra-exclude-1-3",
        action="store_true",
        help="Intra-monomer checks: only exclude PSF 1–2 bonds, not 1–3 pairs.",
    )
    group.add_argument(
        "--dynamics-intra-rescue-sd-steps",
        type=int,
        default=None,
        help=(
            "Bonded-only SD steps for intra-monomer close-contact rescue "
            "(default: --dynamics-overlap-charmm-sd-steps)."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-check-interval",
        type=int,
        default=500,
        help=(
            "Integration steps between overlap checks (default: 500). "
            "Per stage, the effective interval is the largest divisor of the stage "
            "step count not exceeding this value (and at least dcd-nsavc + 1 when set)."
        ),
    )
    group.add_argument(
        "--no-dynamics-overlap-separate",
        action="store_true",
        help=(
            "Do not rigidly push overlapped monomers apart when bonded+VDW rescue "
            "minimization fails to restore min inter-monomer distance."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-separate-margin",
        type=float,
        default=0.2,
        metavar="ANG",
        help=(
            "Extra Å added beyond --dynamics-overlap-min-distance when last-resort "
            "monomer separation is applied (default: 0.2)."
        ),
    )


def resolve_dynamics_overlap_config(
    args: argparse.Namespace,
    *,
    n_monomers: int,
    use_pbc: bool,
    fallback_box_side_A: float | None = None,
) -> DynamicsOverlapConfig:
    action = str(
        getattr(args, "dynamics_overlap_action", "rescue")
    ).lower()
    if action not in ("error", "warn", "rescue", "off"):
        raise ValueError(f"unknown dynamics_overlap_action: {action!r}")

    min_dist = getattr(args, "dynamics_overlap_min_distance", None)
    if min_dist is None:
        min_dist = getattr(args, "min_intermonomer_atom_distance", 1.5)

    interval = int(getattr(args, "dynamics_overlap_check_interval", 500))
    if use_pbc and fallback_box_side_A is None:
        box_size = getattr(args, "box_size", None)
        if box_size is not None:
            fallback_box_side_A = float(box_size)
    rescue = OverlapRescueConfig(
        nstep_sd=int(getattr(args, "dynamics_overlap_charmm_sd_steps", 200)),
        nstep_abnr=int(getattr(args, "dynamics_overlap_charmm_abnr_steps", 400)),
        nprint=max(1, int(getattr(args, "dyn_nprint", 50))),
        tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
        tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    return DynamicsOverlapConfig(
        action=action,  # type: ignore[arg-type]
        min_distance_A=float(min_dist),
        intra_min_distance_A=float(
            getattr(args, "dynamics_intra_min_distance", 1.0) or 0.0
        ),
        intra_exclude_1_3=not bool(
            getattr(args, "no_dynamics_intra_exclude_1_3", False)
        ),
        intra_rescue_sd_steps=getattr(args, "dynamics_intra_rescue_sd_steps", None),
        check_interval=max(1, interval),
        n_monomers=int(n_monomers),
        use_pbc=bool(use_pbc),
        fallback_box_side_A=(
            float(fallback_box_side_A)
            if use_pbc and fallback_box_side_A is not None and float(fallback_box_side_A) > 0.0
            else None
        ),
        rescue=rescue,
        separate_on_rescue_fail=not bool(
            getattr(args, "no_dynamics_overlap_separate", False)
        ),
        separate_margin_A=float(getattr(args, "dynamics_overlap_separate_margin", 0.2)),
    )


def monomer_offsets(n_atoms: int, n_monomers: int) -> np.ndarray:
    """Uniform monomer atom offsets (length ``n_monomers + 1``)."""
    n_atoms = int(n_atoms)
    n_monomers = int(n_monomers)
    if n_monomers <= 0:
        raise ValueError(f"n_monomers must be > 0, got {n_monomers}")
    if n_atoms % n_monomers != 0:
        raise ValueError(
            f"atom count {n_atoms} not divisible by n_monomers={n_monomers}"
        )
    per = n_atoms // n_monomers
    return np.arange(0, n_atoms + 1, per, dtype=int)


@lru_cache(maxsize=1)
def _geometry_checks_mod():
    """Load geometry_checks without importing ``mmml.utils`` (pulls JAX)."""
    path = Path(__file__).resolve().parents[3] / "utils" / "geometry_checks.py"
    name = "_mmml_geometry_checks"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load geometry checks from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _assert_no_intermonomer_atom_overlap_fn():
    return _geometry_checks_mod().assert_no_intermonomer_atom_overlap


@lru_cache(maxsize=1)
def _find_worst_intermonomer_overlap_fn():
    return _geometry_checks_mod().find_worst_intermonomer_overlap


@lru_cache(maxsize=1)
def _separate_intermonomer_overlaps_fn():
    return _geometry_checks_mod().separate_intermonomer_overlaps


@lru_cache(maxsize=1)
def _assert_no_intramonomer_close_contact_fn():
    return _geometry_checks_mod().assert_no_intramonomer_close_contact


@lru_cache(maxsize=1)
def _build_bond_exclusion_pairs_fn():
    return _geometry_checks_mod().build_bond_exclusion_pairs


_bond_exclusion_cache: tuple[int, bool, frozenset[tuple[int, int]]] | None = None


def _bond_exclusion_pairs(*, exclude_1_3: bool) -> frozenset[tuple[int, int]]:
    """PSF 1–2 / 1–3 pairs to skip during intra-monomer scans."""
    global _bond_exclusion_cache
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.psf as psf

    nbond = int(psf.get_nbond())
    if (
        _bond_exclusion_cache is not None
        and _bond_exclusion_cache[0] == nbond
        and _bond_exclusion_cache[1] == exclude_1_3
    ):
        return _bond_exclusion_cache[2]

    ib, jb = psf.get_ib_jb()
    pairs = _build_bond_exclusion_pairs_fn()(ib, jb, exclude_1_3=exclude_1_3)
    _bond_exclusion_cache = (nbond, exclude_1_3, pairs)
    return pairs


def _overlap_cell(
    *,
    use_pbc: bool,
    fallback_box_side_A: float | None = None,
) -> float | np.ndarray | None:
    if not use_pbc:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        resolve_charmm_cubic_box_side_A,
    )

    side, _ = resolve_charmm_cubic_box_side_A(
        fallback_side_A=fallback_box_side_A,
    )
    return float(side)


def _overlap_check(
    config: DynamicsOverlapConfig,
    *,
    context: str,
) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    assert_no_intermonomer_atom_overlap = _assert_no_intermonomer_atom_overlap_fn()
    return assert_no_intermonomer_atom_overlap(
        pos,
        offsets,
        min_distance=config.min_distance_A,
        cell=cell,
        context=context,
    )


def _intramonomer_check(
    config: DynamicsOverlapConfig,
    *,
    context: str,
) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    excluded = _bond_exclusion_pairs(exclude_1_3=config.intra_exclude_1_3)
    assert_no_intramonomer_close_contact = _assert_no_intramonomer_close_contact_fn()
    return assert_no_intramonomer_close_contact(
        pos,
        offsets,
        excluded,
        min_distance=config.intra_min_distance_A,
        cell=cell,
        context=context,
    )


def _run_intramonomer_bonded_rescue(
    mlpot_ctx: "MlpotContext",
    config: DynamicsOverlapConfig,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        BondedMmMiniConfig,
        minimize_bonded_mm_recovery,
    )

    sd_steps = config.intra_rescue_sd_steps
    if sd_steps is None:
        sd_steps = config.rescue.nstep_sd
    minimize_bonded_mm_recovery(
        mlpot_ctx,
        BondedMmMiniConfig(
            nstep_sd=int(sd_steps),
            nprint=max(1, int(config.rescue.nprint)),
            tolenr=float(config.rescue.tolenr),
            tolgrd=float(config.rescue.tolgrd),
            verbose=config.rescue.verbose,
        ),
    )


def apply_overlap_separation_last_resort(config: DynamicsOverlapConfig) -> float:
    """Rigidly push overlapped monomer pairs apart (symmetric COM translation)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    new_pos = _separate_intermonomer_overlaps_fn()(
        pos,
        offsets,
        min_distance=config.min_distance_A,
        margin=config.separate_margin_A,
        cell=cell,
    )
    sync_charmm_positions(new_pos)
    best_dist, _ = _find_worst_intermonomer_overlap_fn()(new_pos, offsets, cell=cell)
    return float(best_dist)


def _mlpot_covers_all_atoms(mlpot_ctx: "MlpotContext | None") -> bool:
    if mlpot_ctx is None or getattr(mlpot_ctx, "ml_selection", None) is None:
        return False
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm

        n_atoms = int(pycharmm.coor.get_natom())
        n_ml = len(mlpot_ctx.ml_selection.get_atom_indexes())
        return n_atoms > 0 and n_ml >= n_atoms
    except Exception:
        return False


def _apply_separation_or_raise(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    cause: RuntimeError,
) -> float:
    if not config.separate_on_rescue_fail:
        raise cause
    print(
        f"{cause}\nApplying last-resort monomer separation "
        f"(target {config.min_distance_A + config.separate_margin_A:.2f} Å)...",
        flush=True,
    )
    try:
        d_sep = apply_overlap_separation_last_resort(config)
        print(
            f"Overlap separation: min inter-monomer distance now {d_sep:.4f} Å",
            flush=True,
        )
        return _overlap_check(
            config,
            context=f"{label} after overlap separation",
        )
    except RuntimeError as sep_still_bad:
        raise RuntimeError(
            f"{sep_still_bad}; rigid monomer separation did not restore "
            f"{config.min_distance_A:.2f} Å — increase Packmol spacing, "
            f"relax --dynamics-overlap-min-distance, or increase "
            f"--dynamics-overlap-separate-margin"
        ) from sep_still_bad


def _handle_inter_monomer_rescue(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    exc: RuntimeError,
    mlpot_ctx: "MlpotContext",
) -> float:
    print(
        f"{exc}\nAttempting MLpot overlap rescue "
        f"(bonded+VDW SD={config.rescue.nstep_sd}, "
        f"ABNR={config.rescue.nstep_abnr})...",
        flush=True,
    )
    if _mlpot_covers_all_atoms(mlpot_ctx):
        print(
            "Skipping CHARMM bonded+VDW overlap rescue for all-ML system; "
            "ML-ML pairs are excluded from CHARMM nonbond lists.",
            flush=True,
        )
        return _apply_separation_or_raise(config, label=label, cause=exc)

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import minimize_overlap_rescue

    try:
        minimize_overlap_rescue(mlpot_ctx, config.rescue)
    except Exception as rescue_exc:
        if config.separate_on_rescue_fail:
            print(f"MLpot overlap rescue failed: {rescue_exc}", flush=True)
            return _apply_separation_or_raise(config, label=label, cause=exc)
        raise RuntimeError(
            f"{exc}; MLpot overlap rescue failed: {rescue_exc}"
        ) from rescue_exc
    try:
        return _overlap_check(config, context=f"{label} after overlap rescue")
    except RuntimeError as still_bad:
        if config.separate_on_rescue_fail:
            return _apply_separation_or_raise(config, label=label, cause=still_bad)
        raise RuntimeError(
            f"{still_bad}; overlap rescue "
            f"(SD={config.rescue.nstep_sd}, ABNR={config.rescue.nstep_abnr}) "
            f"did not separate monomers — try larger "
            f"--dynamics-overlap-charmm-sd-steps / "
            f"--dynamics-overlap-charmm-abnr-steps, "
            f"increase Packmol spacing, or relax "
            f"--dynamics-overlap-min-distance"
        ) from still_bad


def _handle_intramonomer_rescue(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    exc: RuntimeError,
    mlpot_ctx: "MlpotContext",
) -> float:
    sd_steps = config.intra_rescue_sd_steps
    if sd_steps is None:
        sd_steps = config.rescue.nstep_sd
    print(
        f"{exc}\nAttempting intra-monomer bonded-MM rescue "
        f"(SD={sd_steps})...",
        flush=True,
    )
    try:
        _run_intramonomer_bonded_rescue(mlpot_ctx, config)
    except Exception as rescue_exc:
        raise RuntimeError(
            f"{exc}; intra-monomer bonded-MM rescue failed: {rescue_exc}"
        ) from rescue_exc
    try:
        return _intramonomer_check(config, context=f"{label} after intra-monomer rescue")
    except RuntimeError as still_bad:
        raise RuntimeError(
            f"{still_bad}; intra-monomer bonded rescue (SD={sd_steps}) "
            f"did not restore {config.intra_min_distance_A:.2f} Å — "
            f"try larger --dynamics-intra-rescue-sd-steps / "
            f"--dynamics-overlap-charmm-sd-steps, longer minimization, "
            f"or relax --dynamics-intra-min-distance"
        ) from still_bad


def _run_geometry_guard(
    check_fn,
    *,
    config: DynamicsOverlapConfig,
    label: str,
    mlpot_ctx: "MlpotContext | None",
    inter_monomer: bool,
) -> float:
    if config.action == "error":
        return check_fn(label)

    try:
        return check_fn(label)
    except RuntimeError as exc:
        if config.action == "warn":
            print(f"WARNING: {exc}", flush=True)
            return float("nan")
        if config.action != "rescue":
            raise
        if mlpot_ctx is None:
            raise RuntimeError(
                f"{exc}; geometry rescue requires MlpotContext"
            ) from exc
        if inter_monomer:
            return _handle_inter_monomer_rescue(
                config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
            )
        return _handle_intramonomer_rescue(
            config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
        )


def check_dynamics_overlap(
    config: DynamicsOverlapConfig,
    *,
    context: str,
    step: int | None = None,
    mlpot_ctx: "MlpotContext | None" = None,
) -> float:
    """Check inter- and intra-monomer geometry; raise, warn, or rescue per action."""
    if not config.enabled and not config.intra_enabled:
        return float("inf")

    label = context if step is None else f"{context} at step {step}"
    best = float("inf")

    if config.enabled:
        dist = _run_geometry_guard(
            lambda ctx: _overlap_check(config, context=ctx),
            config=config,
            label=label,
            mlpot_ctx=mlpot_ctx,
            inter_monomer=True,
        )
        if np.isfinite(dist):
            best = min(best, dist)

    if config.intra_enabled:
        dist = _run_geometry_guard(
            lambda ctx: _intramonomer_check(config, context=ctx),
            config=config,
            label=label,
            mlpot_ctx=mlpot_ctx,
            inter_monomer=False,
        )
        if np.isfinite(dist):
            best = min(best, dist)

    return best
