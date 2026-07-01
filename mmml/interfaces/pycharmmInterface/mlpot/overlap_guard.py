"""Inter-monomer overlap checks during PyCHARMM MLpot dynamics."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
    intra_min_distance_A: float = 0.5
    intra_exclude_1_3: bool = True
    intra_rescue_sd_steps: int | None = None
    check_interval: int = 100
    n_monomers: int = 1
    use_pbc: bool = False
    fallback_box_side_A: float | None = None
    rescue: OverlapRescueConfig = field(default_factory=OverlapRescueConfig)
    separate_on_rescue_fail: bool = True
    separate_margin_A: float = 0.2
    repack_spacing_A: float | None = None
    max_monomer_extent_A: float = 12.0
    prior_segment_restart: Path | None = None
    geometry_baseline_restart: Path | None = None
    geometry_fallback_restarts: tuple[Path, ...] = ()
    segment_index: int | None = None
    segment_out_dir: Path | None = None
    segment_restart_prefix: str | None = None
    topology_psf: Path | None = None
    recovery_seed: int | None = None
    position_noise_A: float = 0.05
    mlpot_rescue_mini_nstep: int = 25
    pyCModel: Any = field(default=None, compare=False, hash=False)
    artifact_registry: Any = field(default=None, compare=False, hash=False)
    bonded_recovery_backend: str = "auto"
    # When False (default), overlap chunks hand off via alternating scratch ``.res``
    # files and ``dyna restart``.  True keeps coords/vel in RAM (legacy MLpot path).
    memory_handoff: bool = False
    # When True, heat uses one overlap chunk per segment (checks only at segment end).
    # Default False: mid-segment checks every ``check_interval`` (extent/overlap early).
    heat_segment_boundary_only: bool = False
    density_prep_ladder_fallback: bool = False
    cleanup_mode: bool = False

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

    @property
    def extent_enabled(self) -> bool:
        return (
            self.action != "off"
            and float(self.max_monomer_extent_A) > 0.0
            and int(self.n_monomers) >= 1
        )


def add_dynamics_overlap_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Dynamics overlap guard (PyCHARMM MLpot)")
    group.add_argument(
        "--dynamics-overlap-action",
        choices=("error", "warn", "rescue", "off"),
        default="rescue",
        help=(
            "On inter-monomer overlap during MD: rescue=CHARMM bonded+VDW mini, then "
            "monomer repack (re-place COMs) if still overlapped (default); error=abort, "
            "warn=log only, off=disable. Also controls intra-monomer close-contact checks "
            "and max monomer extent (fly-off) recovery."
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
        default=0.5,
        metavar="ANG",
        help=(
            "Minimum allowed nonbonded atom distance within each monomer (1–2 and 1–3 "
            "pairs excluded from PSF bonds). Set 0 to disable (default: 0.5 Å)."
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
            "Integration steps between overlap/extent checks during dynamics "
            "(default: 500). Effective interval is the largest divisor of the stage "
            "step count not exceeding this value (and at least dcd-nsavc + 1 when set). "
            "Heat uses this mid-segment interval by default; see "
            "--heat-overlap-segment-boundary-only for legacy end-only checks."
        ),
    )
    group.add_argument(
        "--heat-overlap-segment-boundary-only",
        action="store_true",
        help=(
            "Heat only: run one overlap chunk per heat segment (geometry check at "
            "segment end only). Default runs checks every --dynamics-overlap-check-interval "
            "inside each segment so extent/T blow-ups fail faster."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-memory-handoff",
        action="store_true",
        help=(
            "Continue overlap chunks in-process without READYN on scratch restarts. "
            "Default on MPI-linked CHARMM under mpirun (set MMML_NO_OVERLAP_MEMORY_HANDOFF=1 "
            "to force scratch .overlap_a/.b.res handoffs)."
        ),
    )
    group.add_argument(
        "--no-dynamics-overlap-separate",
        action="store_true",
        help=(
            "Do not repack overlapped monomers (re-place COMs with preserved internal "
            "geometry) when bonded+VDW rescue minimization fails to restore min "
            "inter-monomer distance."
        ),
    )
    group.add_argument(
        "--dynamics-overlap-separate-margin",
        type=float,
        default=0.2,
        metavar="ANG",
        help=(
            "Extra Å added beyond --dynamics-overlap-min-distance when last-resort "
            "monomer repack spacing is derived automatically (default: 0.2)."
        ),
    )
    group.add_argument(
        "--dynamics-max-monomer-extent",
        type=float,
        default=12.0,
        metavar="ANG",
        help=(
            "Maximum allowed axis-aligned monomer extent in Å during dynamics "
            "(default: 12.0, aligned with CHARMM NBONDA group limit). "
            "On violation, restore the prior segment restart and run bonded-MM SD."
        ),
    )
    group.add_argument(
        "--no-dynamics-max-monomer-extent",
        action="store_true",
        help="Disable max monomer extent / fly-off guard.",
    )


def _truthy_env(name: str) -> bool:
    import os

    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def resolve_overlap_memory_handoff(args: argparse.Namespace) -> bool:
    """Whether overlap chunks continue in-process without ``READYN`` on scratch restarts.

    Explicit ``--dynamics-overlap-memory-handoff`` always wins.  Otherwise default on
    MPI-linked CHARMM under ``mpirun`` (scratch ``.overlap_a/.b.res`` often get REST -1).
    """
    if bool(getattr(args, "dynamics_overlap_memory_handoff", False)):
        return True
    if _truthy_env("MMML_NO_OVERLAP_MEMORY_HANDOFF"):
        return False
    if _truthy_env("MMML_OVERLAP_MEMORY_HANDOFF"):
        return True
    try:
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            _under_mpirun,
            charmm_lib_links_mpi,
        )

        return charmm_lib_links_mpi() and _under_mpirun()
    except Exception:
        return False


def overlap_config_for_stage(
    overlap: DynamicsOverlapConfig | None,
    *,
    stage: str,
    nstep: int,
    n_segments: int = 1,
) -> DynamicsOverlapConfig | None:
    """Per-stage overlap settings.

    Heat defaults to the configured ``check_interval`` inside each segment so
    extent/overlap guards fire before segment end (CHARMM list-update storms /
    T spikes).  Set ``heat_segment_boundary_only`` (CLI:
    ``--heat-overlap-segment-boundary-only``) to restore one check per segment
    (legacy Hoover / IHTFRQ handoff).
    """
    if overlap is None or not (
        overlap.enabled or overlap.intra_enabled or overlap.extent_enabled
    ):
        return overlap
    if stage.lower() != "heat":
        return overlap
    if not overlap.heat_segment_boundary_only:
        return overlap
    from dataclasses import replace

    n = max(1, int(nstep))
    if int(overlap.check_interval) >= n:
        return overlap
    return replace(overlap, check_interval=n)


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

    interval = int(getattr(args, "dynamics_overlap_check_interval", 100))
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
            getattr(args, "dynamics_intra_min_distance", 0.5) or 0.0
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
        bonded_recovery_backend=str(
            getattr(args, "bonded_recovery_backend", "auto") or "auto"
        ),
        separate_on_rescue_fail=not bool(
            getattr(args, "no_dynamics_overlap_separate", False)
        ),
        separate_margin_A=float(getattr(args, "dynamics_overlap_separate_margin", 0.2)),
        repack_spacing_A=(
            float(spacing)
            if (spacing := getattr(args, "spacing", None)) is not None
            and float(spacing) > 0.0
            else None
        ),
        max_monomer_extent_A=(
            0.0
            if bool(getattr(args, "no_dynamics_max_monomer_extent", False))
            else float(getattr(args, "dynamics_max_monomer_extent", 12.0))
        ),
        memory_handoff=resolve_overlap_memory_handoff(args),
        heat_segment_boundary_only=bool(
            getattr(args, "heat_overlap_segment_boundary_only", False)
        ),
        density_prep_ladder_fallback=_cleanup_overlap_fallback_from_args(args),
        cleanup_mode=_cleanup_mode_from_args(args),
    )


def _cleanup_mode_from_args(args: argparse.Namespace) -> bool:
    from mmml.interfaces.pycharmmInterface.mlpot.cleanup_mode import cleanup_enabled

    return cleanup_enabled(args)


def _cleanup_overlap_fallback_from_args(args: argparse.Namespace) -> bool:
    from mmml.interfaces.pycharmmInterface.mlpot.cleanup_mode import (
        cleanup_overlap_fallback_enabled,
    )

    return cleanup_overlap_fallback_enabled(args)


def augment_overlap_config_for_rescue(
    config: DynamicsOverlapConfig,
    *,
    ctx: "MlpotContext",
    args: argparse.Namespace,
    topology_psf: Path | None,
    artifact_registry: Any = None,
) -> DynamicsOverlapConfig:
    """Attach topology / MLpot handles needed for all-ML overlap rescue."""
    from dataclasses import replace

    topo_path: Path | None = None
    if topology_psf is not None:
        candidate = Path(topology_psf).expanduser()
        if candidate.is_file():
            topo_path = candidate.resolve()
    mini_nstep = int(getattr(args, "charmm_sd_steps", 25) or 25)
    return replace(
        config,
        topology_psf=topo_path,
        recovery_seed=getattr(args, "seed", None),
        mlpot_rescue_mini_nstep=max(0, mini_nstep),
        pyCModel=getattr(ctx, "pyCModel", None),
        artifact_registry=artifact_registry,
    )


def infer_prior_restart_from_write_path(restart_write: Path | str | None) -> Path | None:
    """Infer ``{prefix}.{n-1}.res`` from a chained segment restart write path."""
    if restart_write is None:
        return None
    path = Path(restart_write)
    stem = path.stem
    if "." not in stem:
        return None
    prefix, seg_raw = stem.rsplit(".", 1)
    if not seg_raw.isdigit():
        return None
    seg_idx = int(seg_raw)
    if seg_idx <= 0:
        return None
    prior = path.parent / f"{prefix}.{seg_idx - 1}.res"
    return prior.resolve() if prior.is_file() else None


def resolve_prior_segment_restart_path(
    *,
    segment_index: int,
    prev_restart: Path | str | None = None,
    out_dir: Path | str | None = None,
    restart_prefix: str | None = None,
    geometry_fallback_restarts: tuple[Path, ...] | list[Path] = (),
    geometry_baseline_restart: Path | str | None = None,
) -> Path | None:
    """Return the best on-disk checkpoint before the current segment."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file

    candidates: list[Path] = []
    seg_i = int(segment_index)
    if geometry_baseline_restart is not None:
        candidates.append(Path(geometry_baseline_restart))
    if seg_i > 0 and out_dir is not None and restart_prefix:
        candidates.append(Path(out_dir) / f"{restart_prefix}.{seg_i - 1}.res")
    if prev_restart is not None:
        candidates.append(Path(prev_restart))
    for cand in geometry_fallback_restarts:
        p = Path(cand)
        if geometry_baseline_restart is not None and p == Path(geometry_baseline_restart):
            continue
        from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
            is_pretreat_mm_restart_path,
        )

        if is_pretreat_mm_restart_path(p):
            continue
        candidates.append(p)
    seen: set[str] = set()
    for cand in candidates:
        p = cand.expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        valid = _valid_restart_file(p)
        if valid is not None:
            return valid
    return None


def refresh_overlap_prior_segment_restart(
    overlap: DynamicsOverlapConfig | None,
    *,
    restart_path: Path | str | None,
) -> DynamicsOverlapConfig | None:
    """Persist the last good geometry checkpoint for extent fly-off recovery."""
    if overlap is None or not overlap.extent_enabled or restart_path is None:
        return overlap
    from dataclasses import replace

    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        ensure_segment_restart_checkpoint,
    )

    prior = ensure_segment_restart_checkpoint(restart_path)
    if prior is None:
        return overlap
    return replace(overlap, prior_segment_restart=prior)


def attach_prior_segment_restart(
    overlap: DynamicsOverlapConfig | None,
    *,
    segment_index: int | None = None,
    prev_restart: Path | str | None = None,
    out_dir: Path | str | None = None,
    restart_prefix: str | None = None,
    restart_write: Path | str | None = None,
) -> DynamicsOverlapConfig | None:
    """Attach a fly-off recovery checkpoint when extent guard is enabled."""
    if overlap is None or not overlap.extent_enabled:
        return overlap
    existing = overlap.prior_segment_restart
    if existing is not None and Path(existing).is_file():
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _valid_restart_file

        if _valid_restart_file(existing) is not None:
            return overlap
    from dataclasses import replace

    seg_i = segment_index if segment_index is not None else overlap.segment_index
    seg_out = out_dir if out_dir is not None else overlap.segment_out_dir
    seg_prefix = restart_prefix if restart_prefix is not None else overlap.segment_restart_prefix
    tagged = overlap
    if seg_i is not None or seg_out is not None or seg_prefix is not None:
        tagged = replace(
            overlap,
            segment_index=seg_i if seg_i is not None else overlap.segment_index,
            segment_out_dir=Path(seg_out) if seg_out is not None else overlap.segment_out_dir,
            segment_restart_prefix=(
                str(seg_prefix) if seg_prefix is not None else overlap.segment_restart_prefix
            ),
        )

    prior = resolve_prior_segment_restart_path(
        segment_index=int(seg_i or 0),
        prev_restart=prev_restart,
        out_dir=seg_out,
        restart_prefix=seg_prefix,
        geometry_fallback_restarts=tagged.geometry_fallback_restarts,
        geometry_baseline_restart=tagged.geometry_baseline_restart,
    )
    if prior is None:
        from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
            build_geometry_recovery_candidates,
            first_valid_restart_path,
        )

        prior = first_valid_restart_path(build_geometry_recovery_candidates(tagged))
    if prior is None:
        prior = infer_prior_restart_from_write_path(restart_write)
    if prior is None:
        return tagged
    return replace(tagged, prior_segment_restart=prior)


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
def _repack_monomers_clear_overlap_fn():
    return _geometry_checks_mod().repack_monomers_clear_overlap


@lru_cache(maxsize=1)
def _assert_no_intramonomer_close_contact_fn():
    return _geometry_checks_mod().assert_no_intramonomer_close_contact


@lru_cache(maxsize=1)
def _build_bond_exclusion_pairs_fn():
    return _geometry_checks_mod().build_bond_exclusion_pairs


@lru_cache(maxsize=1)
def _assert_monomer_extent_within_limit_fn():
    return _geometry_checks_mod().assert_monomer_extent_within_limit


@lru_cache(maxsize=1)
def _find_worst_monomer_extent_fn():
    return _geometry_checks_mod().find_worst_monomer_extent


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

    raw_ib_jb = psf.get_ib_jb()
    if isinstance(raw_ib_jb, tuple) and len(raw_ib_jb) == 2:
        ib, jb = raw_ib_jb
    elif isinstance(raw_ib_jb, (list, tuple)) and len(raw_ib_jb) == 0:
        # PyCHARMM <= vendored bug: ``get_ib_jb()`` returned ``[]`` when ``nbond==0``.
        ib, jb = [], []
    else:
        raise ValueError(f"unexpected psf.get_ib_jb() return: {raw_ib_jb!r}")
    if nbond <= 0:
        if ib or jb:
            ib, jb = [], []
        pairs = frozenset()
    else:
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


def measure_worst_intermonomer_distance(
    config: DynamicsOverlapConfig,
) -> float:
    """Return closest inter-monomer atom–atom distance (Å) without raising."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    best_dist, _ = _find_worst_intermonomer_overlap_fn()(pos, offsets, cell=cell)
    return float(best_dist)


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


def relieve_intramonomer_clashes(
    config: DynamicsOverlapConfig,
    *,
    context: str = "intra-monomer separation",
    verbose: bool = False,
    margin_A: float | None = None,
) -> float:
    """Push apart intra-monomer atom pairs below ``intra_min_distance_A``."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )
    from mmml.utils.geometry_checks import (
        find_worst_intramonomer_close_contact,
        separate_intramonomer_contacts,
    )

    if not config.intra_enabled and config.intra_min_distance_A <= 0.0:
        return float("inf")

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    excluded = _bond_exclusion_pairs(exclude_1_3=config.intra_exclude_1_3)
    threshold = float(config.intra_min_distance_A)
    margin = float(margin_A if margin_A is not None else config.separate_margin_A)
    before, violation = find_worst_intramonomer_close_contact(
        pos,
        offsets,
        excluded,
        cell=cell,
        min_distance=threshold,
    )
    if violation is None or before >= threshold:
        return float(before)

    new_pos = separate_intramonomer_contacts(
        pos,
        offsets,
        excluded,
        min_distance=threshold,
        margin=margin,
        cell=cell,
    )
    sync_charmm_positions(new_pos)
    after, _ = find_worst_intramonomer_close_contact(
        new_pos,
        offsets,
        excluded,
        cell=cell,
        min_distance=threshold,
    )
    if verbose:
        after_txt = (
            "cleared (no pairs below threshold)"
            if not np.isfinite(after) or after >= threshold
            else f"{after:.4f} Å"
        )
        print(
            f"{context}: intra-monomer distance {before:.4f} -> {after_txt} "
            f"(target {threshold:.4f} Å)",
            flush=True,
        )
    return float(after)


def _extent_check(
    config: DynamicsOverlapConfig,
    *,
    context: str,
) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    assert_monomer_extent_within_limit = _assert_monomer_extent_within_limit_fn()
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    return assert_monomer_extent_within_limit(
        pos,
        offsets,
        max_extent_A=config.max_monomer_extent_A,
        cell=cell,
        context=context,
    )


def save_stabilized_overlap_rescue_snapshot(
    mlpot_ctx: "MlpotContext",
    config: DynamicsOverlapConfig,
    *,
    label: str,
) -> None:
    """Write PDB/CRD after MLpot re-register + rescue mini (not mid-bonded-MM)."""
    registry = getattr(config, "artifact_registry", None)
    if registry is None:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_mlpot_grms_kcalmol_A,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
        save_snapshot_from_charmm,
    )

    grms = resolve_mlpot_grms_kcalmol_A(
        mlpot_ctx,
        context=f"{label} rescue snapshot",
    )
    spec = registry.allocate_rescue_spec(label)
    save_snapshot_from_charmm(
        registry,
        spec,
        out_dir=registry.out_dir,
        tag=registry.tag,
        title=spec.label,
        grms_kcalmol_A=float(grms),
        include_psf=False,
    )


def _run_intramonomer_bonded_rescue(
    mlpot_ctx: "MlpotContext",
    config: DynamicsOverlapConfig,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_intra_monomer_overlap_rescue,
    )

    run_intra_monomer_overlap_rescue(mlpot_ctx, config)


def apply_overlap_repack_last_resort(
    config: DynamicsOverlapConfig,
    *,
    mlpot_ctx: "MlpotContext | None" = None,
) -> float:
    """Re-place monomer COMs with preserved internal geometry (Packmol-style repack)."""
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
    find_overlap = _find_worst_intermonomer_overlap_fn()
    best_dist, violation = find_overlap(pos, offsets, cell=cell)
    overlap_hint: tuple[int, ...] = ()
    if violation is not None:
        overlap_hint = (int(violation.monomer_i), int(violation.monomer_j))

    repack_fn = _repack_monomers_clear_overlap_fn()
    new_pos = None
    if mlpot_ctx is not None:
        from mmml.utils.geometry_checks import repack_selected_monomers_clear_overlap
        from mmml.utils.monomer_force_diag import resolve_selective_repack_monomers

        diag = resolve_selective_repack_monomers(
            mlpot_ctx,
            offsets,
            overlap_monomers=overlap_hint,
            positions=pos,
        )
        if diag is not None:
            print(
                f"Overlap repack: selective patch monomers "
                f"{list(diag.flagged)} "
                f"(per-mono GRMS "
                f"{', '.join(f'{i}:{diag.grms_per_monomer[i]:.1f}' for i in diag.flagged)} "
                f"kcal/mol/Å; cluster {diag.cluster_grms:.1f})",
                flush=True,
            )
            new_pos = repack_selected_monomers_clear_overlap(
                pos,
                offsets,
                list(diag.flagged),
                min_distance=config.min_distance_A,
                spacing=config.repack_spacing_A,
                margin=config.separate_margin_A,
                seed=config.recovery_seed,
                cell=cell,
            )

    if new_pos is None:
        new_pos = repack_fn(
            pos,
            offsets,
            min_distance=config.min_distance_A,
            spacing=config.repack_spacing_A,
            margin=config.separate_margin_A,
            seed=config.recovery_seed,
            cell=cell,
        )
    sync_charmm_positions(new_pos)
    best_dist, _ = find_overlap(new_pos, offsets, cell=cell)
    return float(best_dist)


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


def _apply_repack_or_raise(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    cause: RuntimeError,
    mlpot_ctx: "MlpotContext | None" = None,
) -> float:
    if not config.separate_on_rescue_fail:
        raise cause
    spacing_msg = (
        f"{config.repack_spacing_A:.2f} Å spacing"
        if config.repack_spacing_A is not None
        else "derived COM spacing"
    )
    print(
        f"{cause}\nApplying last-resort monomer repack ({spacing_msg})...",
        flush=True,
    )
    try:
        d_repack = apply_overlap_repack_last_resort(config, mlpot_ctx=mlpot_ctx)
        print(
            f"Overlap repack: min inter-monomer distance now {d_repack:.4f} Å",
            flush=True,
        )
        relieve_intramonomer_clashes(
            config,
            context=f"{label} after overlap repack (intra preflight)",
            verbose=True,
        )
        return _overlap_check(
            config,
            context=f"{label} after overlap repack",
        )
    except RuntimeError as repack_still_bad:
        print(
            f"{repack_still_bad}\nApplying rigid monomer separation fallback "
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
                f"{sep_still_bad}; monomer repack and rigid separation did not restore "
                f"{config.min_distance_A:.2f} Å — increase Packmol spacing, "
                f"relax --dynamics-overlap-min-distance, or increase "
                f"--dynamics-overlap-separate-margin"
            ) from sep_still_bad


def _apply_separation_or_raise(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    cause: RuntimeError,
    mlpot_ctx: "MlpotContext | None" = None,
) -> float:
    return _apply_repack_or_raise(
        config, label=label, cause=cause, mlpot_ctx=mlpot_ctx
    )


def _handle_inter_monomer_rescue(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    exc: RuntimeError,
    mlpot_ctx: "MlpotContext",
) -> float:
    print(
        f"{exc}\nAttempting MLpot overlap rescue "
        f"(SD={config.rescue.nstep_sd}, ABNR={config.rescue.nstep_abnr})...",
        flush=True,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_inter_monomer_overlap_rescue,
    )

    try:
        run_inter_monomer_overlap_rescue(mlpot_ctx, config)
    except Exception as rescue_exc:
        if config.separate_on_rescue_fail:
            print(f"MLpot overlap rescue failed: {rescue_exc}", flush=True)
            return _apply_separation_or_raise(
                config, label=label, cause=exc, mlpot_ctx=mlpot_ctx
            )
        raise RuntimeError(
            f"{exc}; MLpot overlap rescue failed: {rescue_exc}"
        ) from rescue_exc
    try:
        return _overlap_check(config, context=f"{label} after overlap rescue")
    except RuntimeError as still_bad:
        if config.separate_on_rescue_fail:
            return _apply_separation_or_raise(
                config, label=label, cause=still_bad, mlpot_ctx=mlpot_ctx
            )
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


def _load_extent_reference_positions(
    candidates: list[Path],
) -> tuple[np.ndarray, Path]:
    """Load reference coordinates for selective monomer rebuild (CRD preferred)."""
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        first_valid_geometry_crd_path,
        first_valid_restart_path,
        is_geometry_recovery_crd_path,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_crd_coordinates,
        read_restart_coordinates,
    )

    crd = first_valid_geometry_crd_path(candidates)
    if crd is not None:
        ref = read_crd_coordinates(crd)
        if ref is not None:
            return ref, crd.resolve()

    restart = first_valid_restart_path(candidates)
    if restart is not None:
        ref = read_restart_coordinates(restart)
        if ref is not None:
            return ref, restart.resolve()

    for cand in candidates:
        p = Path(cand).expanduser()
        if not p.is_file():
            continue
        if is_geometry_recovery_crd_path(p):
            ref = read_crd_coordinates(p)
        else:
            ref = read_restart_coordinates(p)
        if ref is not None:
            return ref, p.resolve()

    names = ", ".join(p.name for p in candidates) or "(none)"
    raise RuntimeError(
        f"extent cleanup: no readable reference coordinates in ladder ({names})"
    )


def _handle_extent_cleanup_rescue(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    exc: RuntimeError,
    mlpot_ctx: "MlpotContext",
) -> float:
    """Rebuild fly-off monomer(s) from mini checkpoint, then minimize + cold restart."""
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        build_extent_recovery_candidates,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )
    from mmml.utils.geometry_checks import (
        coords_pathological_for_repack,
        find_worst_monomer_extent,
        rebuild_monomers_from_reference,
        repack_monomers_clear_overlap,
        repack_selected_monomers_clear_overlap,
    )

    pos = get_charmm_positions_array()
    max_abs = float(np.max(np.abs(pos))) if pos.size else 0.0
    if max_abs > 1000.0:
        raise RuntimeError(
            f"{exc}; coordinates span ±{max_abs:.0f} Å — refuse cleanup extent rescue "
            "(integration blow-up; fix heat ihtfrq < chunk nstep or use Hoover heat)"
        ) from exc
    offsets = monomer_offsets(int(pos.shape[0]), config.n_monomers)
    cell = _overlap_cell(
        use_pbc=config.use_pbc,
        fallback_box_side_A=config.fallback_box_side_A,
    )
    _, violation = find_worst_monomer_extent(pos, offsets, cell=cell)
    if violation is None:
        raise exc

    candidates = build_extent_recovery_candidates(config)
    from mmml.interfaces.pycharmmInterface.mlpot.extent_repack_recovery import (
        polish_after_extent_repack,
        resolve_extent_reference_positions,
    )

    try:
        ref_pos, ref_path = resolve_extent_reference_positions(candidates, mlpot_ctx)
    except RuntimeError as ref_exc:
        raise RuntimeError(
            f"{exc}; cleanup extent rescue requires a geometry baseline / checkpoint "
            f"ladder (prior_segment_restart={config.prior_segment_restart!r}) or "
            f"in-memory mini/baseline coordinates on MlpotContext"
        ) from ref_exc
    if ref_pos.shape != pos.shape:
        raise RuntimeError(
            f"{exc}; cleanup extent rescue: reference {ref_path.name} has "
            f"{ref_pos.shape[0]} atoms but system has {pos.shape[0]}"
        ) from exc

    flagged = [int(violation.monomer)]
    extent_cap = float(config.max_monomer_extent_A)
    repack_common = dict(
        min_distance=config.min_distance_A,
        spacing=config.repack_spacing_A,
        margin=config.separate_margin_A,
        seed=config.recovery_seed,
        cell=cell,
        template_positions=ref_pos,
        max_monomer_radius_A=extent_cap if extent_cap > 0.0 else None,
    )
    print(
        f"{exc}\nCleanup extent rescue: rebuild monomer {violation.monomer + 1} "
        f"from {ref_path.name}, minimize, cold restart...",
        flush=True,
    )
    new_pos = rebuild_monomers_from_reference(pos, ref_pos, offsets, flagged)
    if coords_pathological_for_repack(new_pos, offsets, max_monomer_extent_A=extent_cap):
        print(
            "Cleanup extent rescue: pathological coordinates detected — "
            "full repack from reference geometry",
            flush=True,
        )
        new_pos = repack_monomers_clear_overlap(ref_pos.copy(), offsets, **repack_common)
    else:
        new_pos = repack_selected_monomers_clear_overlap(
            new_pos,
            offsets,
            flagged,
            **repack_common,
        )
    sync_charmm_positions(new_pos)
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        invalidate_mlpot_calculator_caches,
        sync_charmm_lists_after_mini,
    )

    sync_charmm_lists_after_mini(quiet=True)
    invalidate_mlpot_calculator_caches(mlpot_ctx)
    mlpot_ctx.reregister_mlpot(verbose=False, reregister_params=False)
    polish_after_extent_repack(
        mlpot_ctx,
        config,
        label=f"{label} after monomer repack",
    )
    setattr(mlpot_ctx, "_overlap_post_rescue_cold_start", True)
    return _extent_check(config, context=f"{label} after cleanup extent rescue")


def _escalate_extent_rescue(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    still_bad: RuntimeError,
    mlpot_ctx: "MlpotContext",
    recovery_path: Path | None = None,
    sd_steps: int | None = None,
) -> float:
    """Try density prep ladder, then monomer repack, before aborting extent rescue."""
    exc: RuntimeError = still_bad
    if config.density_prep_ladder_fallback:
        try:
            return _try_density_prep_ladder_after_extent_failure(
                config,
                label=label,
                exc=exc,
                mlpot_ctx=mlpot_ctx,
            )
        except Exception as ladder_exc:
            if not isinstance(ladder_exc, RuntimeError):
                raise
            exc = ladder_exc

    try:
        return _handle_extent_cleanup_rescue(
            config,
            label=label,
            exc=exc,
            mlpot_ctx=mlpot_ctx,
        )
    except RuntimeError as repack_exc:
        exc = repack_exc

    if recovery_path is not None:
        sd_txt = f" (SD={sd_steps})" if sd_steps is not None else ""
        raise RuntimeError(
            f"{exc}; fly-off recovery from {recovery_path.name}{sd_txt}, density "
            f"prep ladder, and monomer repack did not restore monomer extent "
            f"<= {config.max_monomer_extent_A:.2f} A — inspect the prior restart "
            f"or reduce the timestep / heating rate"
        ) from exc
    raise RuntimeError(
        f"{exc}; extent rescue (density prep ladder and monomer repack) did not "
        f"restore monomer extent <= {config.max_monomer_extent_A:.2f} A"
    ) from exc


def _handle_extent_rescue(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    exc: RuntimeError,
    mlpot_ctx: "MlpotContext",
) -> float:
    if config.cleanup_mode:
        return _handle_extent_cleanup_rescue(
            config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
        )

    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        build_extent_recovery_candidates,
        resolve_extent_recovery_source,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.extent_repack_recovery import (
        in_memory_extent_reference_available,
    )

    candidates = build_extent_recovery_candidates(config)
    if not candidates and not in_memory_extent_reference_available(mlpot_ctx):
        raise RuntimeError(
            f"{exc}; extent recovery requires a geometry baseline / checkpoint ladder "
            f"(prior_segment_restart={config.prior_segment_restart!r}) or in-memory "
            f"mini/baseline coordinates on MlpotContext"
        ) from exc
    if not candidates:
        print(
            f"{exc}\nNo disk geometry checkpoint; repacking from in-memory "
            "mini/baseline snapshot...",
            flush=True,
        )
        return _handle_extent_cleanup_rescue(
            config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
        )
    recovery_path = resolve_extent_recovery_source(candidates) or candidates[0]
    sd_steps = config.intra_rescue_sd_steps
    if sd_steps is None:
        sd_steps = config.rescue.nstep_sd
    backend = str(getattr(config, "bonded_recovery_backend", "auto") or "auto").lower()
    recovery_method = "JAX bonded FIRE" if backend in ("auto", "jax") else "CHARMM bonded SD"
    print(
        f"{exc}\nAttempting fly-off recovery from "
        f"{recovery_path.name} ({recovery_method}, steps={sd_steps})...",
        flush=True,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        run_extent_recovery_from_prior_restart,
    )

    try:
        run_extent_recovery_from_prior_restart(
            mlpot_ctx,
            config,
            candidates=candidates,
        )
    except Exception as rescue_exc:
        return _escalate_extent_rescue(
            config,
            label=label,
            still_bad=RuntimeError(f"{exc}; fly-off recovery failed: {rescue_exc}"),
            mlpot_ctx=mlpot_ctx,
            recovery_path=recovery_path,
            sd_steps=sd_steps,
        )
    try:
        return _extent_check(config, context=f"{label} after fly-off recovery")
    except RuntimeError as still_bad:
        return _escalate_extent_rescue(
            config,
            label=label,
            still_bad=still_bad,
            mlpot_ctx=mlpot_ctx,
            recovery_path=recovery_path,
            sd_steps=sd_steps,
        )


def _try_density_prep_ladder_after_extent_failure(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    exc: RuntimeError,
    mlpot_ctx: "MlpotContext",
) -> float:
    """Fall back to the liquid-prep density ladder when fly-off recovery fails."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        refresh_mlpot_energy_and_grms,
        resolve_max_grms_before_dyn,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        density_prep_ladder_enabled,
        run_density_prep_ladder,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    args = getattr(mlpot_ctx, "workflow_args", None)
    if args is None or not density_prep_ladder_enabled(args):
        raise exc

    atoms_per_list = getattr(mlpot_ctx, "atoms_per_monomer", None)
    if atoms_per_list is None:
        atoms_per_list = getattr(args, "_cluster_atoms_per_list", None)
    if atoms_per_list is None:
        raise exc

    n_mol = int(config.n_monomers)
    n_atoms = int(sum(atoms_per_list))
    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if pyCModel is None:
        raise exc

    box_side = config.fallback_box_side_A
    if box_side is None:
        box_side = getattr(mlpot_ctx, "cubic_box_side_A", None)
    current_grms = refresh_mlpot_energy_and_grms(
        mlpot_ctx,
        context=f"{label} pre-density-ladder",
    )
    max_grms = resolve_max_grms_before_dyn(
        args,
        n_mol,
        n_atoms,
        pbc=bool(config.use_pbc),
        mlpot_ctx=mlpot_ctx,
    )
    print(
        f"{exc}\nFly-off recovery insufficient; running density prep ladder "
        f"(GRMS {current_grms:.4f}, limit {max_grms:.4f})...",
        flush=True,
    )
    composition = getattr(args, "_cluster_composition_summary", None)
    ladder_grms, _new_side, summary = run_density_prep_ladder(
        args,
        mlpot_ctx=mlpot_ctx,
        pyCModel=pyCModel,
        max_grms=max_grms,
        current_grms=current_grms,
        n_mol=n_mol,
        n_atoms=n_atoms,
        box_side=float(box_side) if box_side is not None else None,
        charmm_pbc=bool(config.use_pbc),
        atoms_per_list=list(atoms_per_list),
        composition=composition,
        mini_nstep=int(getattr(args, "mini_nstep", 500) or 500),
        mini_nprint=max(1, int(getattr(args, "nprint", 100) or 100)),
        fix_resids=[],
        show_energy=False,
        z=getattr(mlpot_ctx, "ml_Z", None),
    )
    setattr(args, "_density_prep_ladder_extent_fallback_summary", summary.to_dict())
    sync_positions = get_charmm_positions_array()
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(sync_positions)
    return _extent_check(config, context=f"{label} after density prep ladder")


def _run_geometry_guard(
    check_fn,
    *,
    config: DynamicsOverlapConfig,
    label: str,
    mlpot_ctx: "MlpotContext | None",
    inter_monomer: bool,
) -> tuple[float, bool]:
    if config.action == "error":
        return check_fn(label), False

    try:
        return check_fn(label), False
    except RuntimeError as exc:
        if config.action == "warn":
            print(f"WARNING: {exc}", flush=True)
            return float("nan"), False
        if config.action != "rescue":
            raise
        if mlpot_ctx is None:
            raise RuntimeError(
                f"{exc}; geometry rescue requires MlpotContext"
            ) from exc
        if inter_monomer:
            return (
                _handle_inter_monomer_rescue(
                    config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
                ),
                True,
            )
        return (
            _handle_intramonomer_rescue(
                config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
            ),
            True,
        )


def _run_extent_guard(
    config: DynamicsOverlapConfig,
    *,
    label: str,
    mlpot_ctx: "MlpotContext | None",
) -> tuple[float, bool]:
    if config.action == "error":
        return _extent_check(config, context=label), False

    try:
        return _extent_check(config, context=label), False
    except RuntimeError as exc:
        if config.action == "warn":
            print(f"WARNING: {exc}", flush=True)
            return float("nan"), False
        if config.action != "rescue":
            raise
        if mlpot_ctx is None:
            raise RuntimeError(
                f"{exc}; fly-off recovery requires MlpotContext"
            ) from exc
        return (
            _handle_extent_rescue(
                config, label=label, exc=exc, mlpot_ctx=mlpot_ctx
            ),
            True,
        )


def probe_dynamics_geometry_violation(
    config: DynamicsOverlapConfig,
    *,
    context: str,
    step: int | None = None,
) -> bool:
    """Return True when extent or intra/inter-monomer geometry checks would fail.

    Read-only probe (no bonded-MM or MLpot rescue). Used after echeck/CHARMM
  aborts to decide whether full geometry cleanup is required before retry.
    """
    if not config.enabled and not config.intra_enabled and not config.extent_enabled:
        return False

    label = context if step is None else f"{context} at step {step}"
    if config.extent_enabled:
        try:
            _extent_check(config, context=label)
        except RuntimeError:
            return True
    if config.enabled:
        try:
            _overlap_check(config, context=label)
        except RuntimeError:
            return True
    if config.intra_enabled:
        try:
            _intramonomer_check(config, context=label)
        except RuntimeError:
            return True
    return False


def check_dynamics_overlap(
    config: DynamicsOverlapConfig,
    *,
    context: str,
    step: int | None = None,
    mlpot_ctx: "MlpotContext | None" = None,
) -> tuple[float, bool]:
    """Check inter- and intra-monomer geometry; raise, warn, or rescue per action.

    Returns ``(min_distance, rescued)`` where ``rescued`` is true when bonded-MM
    recovery rewrote coordinates (PSF reload invalidates in-memory dyna state).
    """
    if not config.enabled and not config.intra_enabled and not config.extent_enabled:
        return float("inf"), False

    label = context if step is None else f"{context} at step {step}"
    best = float("inf")
    rescued = False

    if config.extent_enabled:
        _, did_rescue = _run_extent_guard(
            config,
            label=label,
            mlpot_ctx=mlpot_ctx,
        )
        rescued = rescued or did_rescue

    if config.enabled:
        dist, did_rescue = _run_geometry_guard(
            lambda ctx: _overlap_check(config, context=ctx),
            config=config,
            label=label,
            mlpot_ctx=mlpot_ctx,
            inter_monomer=True,
        )
        rescued = rescued or did_rescue
        if np.isfinite(dist):
            best = min(best, dist)

    if config.intra_enabled:
        dist, did_rescue = _run_geometry_guard(
            lambda ctx: _intramonomer_check(config, context=ctx),
            config=config,
            label=label,
            mlpot_ctx=mlpot_ctx,
            inter_monomer=False,
        )
        rescued = rescued or did_rescue
        if np.isfinite(dist):
            best = min(best, dist)

    if best == float("inf") and config.extent_enabled:
        try:
            best = _extent_check(config, context=f"{label} summary")
        except RuntimeError:
            pass

    return best, rescued
