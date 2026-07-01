"""Minimization and MD workflows with MLpot active (PyCHARMM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

NptThermostat = Literal["hoover", "berendsen"]
HeatThermostat = Literal["scale", "hoover"]

# NPT/CPT Hoover: max steps per ``dyn.run()`` (overlap guard off, or per overlap sub-chunk).
# Long single segments can NaN the barostat/crystal and segfault in Fortran ``upimag``.
DEFAULT_CPT_DYNAMICS_CHUNK_NSTEP = 250

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.derivative_test import TestFirstConfig
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

import numpy as np

PathLike = Union[str, Path]


def _dynamics_io_fortran_path(
    path: PathLike,
    *,
    for_write: bool,
    append: bool = False,
) -> tuple[str, Any | None]:
    """Return ``(fortran_path, io_alias_or_none)`` for dynamics file open."""
    p = Path(path)
    try:
        from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_fortran_path

        fortran_path, alias = charmm_fortran_path(
            p,
            for_write=for_write,
            append=append,
        )
        return str(fortran_path), alias
    except ImportError:
        return str(p.expanduser().resolve()), None


def _apply_dynamics_io_setters(kw: dict[str, Any]) -> None:
    """Wire restart/trajectory paths via CHARMM dynamics C API (MPI-safe).

    ``DynamicsScript`` unit numbers do not always reach Fortran ``reawri`` under
    ``mpirun``; ``dynamics_set_iuncrd`` / ``dynamics_set_iunwri`` do.
    Integer unit values are left for ``CharmmFile`` + script ``iuncrd N`` paths.
    """
    try:
        import pycharmm.dynamics as charm_dyn
    except (ImportError, OSError):
        return

    iunrea = kw.get("iunrea")
    if isinstance(iunrea, str):
        if kw.get("restart"):
            if not charm_dyn.set_iunrea(iunrea):
                raise RuntimeError(f"dynamics iunrea open failed: {iunrea}")
        kw.pop("iunrea", None)

    for key, setter in (
        ("iunwri", charm_dyn.set_iunwri),
        ("iuncrd", charm_dyn.set_iuncrd),
    ):
        val = kw.get(key)
        if isinstance(val, str):
            if not setter(val):
                raise RuntimeError(f"dynamics {key} open failed: {val}")
            kw.pop(key, None)


def _emit_overlap_log(
    detail: str,
    *,
    context: str | None = None,
    quiet: bool = False,
) -> None:
    from mmml.utils.rich_report import emit_overlap_log

    emit_overlap_log(detail, context=context, quiet=quiet)


def _maybe_show_energy(show: bool) -> None:
    if not show:
        return
    from mmml.interfaces.pycharmmInterface.import_pycharmm import safe_energy_show

    safe_energy_show()


@dataclass
class CharmmTrajectoryFiles:
    """Unit numbers and paths for CHARMM restart / trajectory I/O."""

    restart_read: Optional[Path] = None
    restart_write: Optional[Path] = None
    trajectory: Optional[Path] = None
    pressure_tensor_log: Optional[Path] = None
    restart_read_unit: int = 3
    restart_write_unit: int = 2
    trajectory_unit: int = 1
    pressure_tensor_log_unit: int = 29

    def open_for_run(self) -> tuple[list[Any], dict[str, Any], list[Any]]:
        """Open dynamics I/O; returns ``(open_files, dynamics_io_kwargs, io_aliases)``."""
        open_files: list[Any] = []
        aliases: list[Any] = []
        kw: dict[str, Any] = {}
        if self.restart_read is not None:
            fortran_path, alias = _dynamics_io_fortran_path(
                self.restart_read,
                for_write=False,
            )
            if alias is not None:
                aliases.append(alias)
            kw["iunrea"] = fortran_path
        if self.restart_write is not None:
            p = Path(self.restart_write)
            p.parent.mkdir(parents=True, exist_ok=True)
            fortran_path, alias = _dynamics_io_fortran_path(p, for_write=True)
            if alias is not None:
                aliases.append(alias)
            kw["iunwri"] = fortran_path
        if self.trajectory is not None:
            p = Path(self.trajectory)
            p.parent.mkdir(parents=True, exist_ok=True)
            fortran_path, alias = _dynamics_io_fortran_path(p, for_write=True)
            if alias is not None:
                aliases.append(alias)
            kw["iuncrd"] = fortran_path
        if self.pressure_tensor_log is not None:
            import pycharmm

            p = Path(self.pressure_tensor_log)
            p.parent.mkdir(parents=True, exist_ok=True)
            f = pycharmm.CharmmFile(
                file_name=str(p),
                file_unit=self.pressure_tensor_log_unit,
                formatted=True,
                read_only=False,
            )
            open_files.append(f)
            kw["iupten"] = self.pressure_tensor_log_unit
        return open_files, kw, aliases

    def open_trajectory_for_run(
        self,
        *,
        append: bool = False,
    ) -> tuple[list[Any], dict[str, int]]:
        """Open the DCD once (for multi-chunk overlap runs; append across ``dyna`` calls).

        Pass ``append=True`` only when resuming an existing trajectory on disk.
        Overlap chunking passes ``iuncrd`` on the first ``dyna`` call only so
        PyCHARMM does not reopen/truncate the file on every restart chunk.
        """
        import pycharmm

        if self.trajectory is None:
            return [], {}
        p = Path(self.trajectory)
        p.parent.mkdir(parents=True, exist_ok=True)
        f = pycharmm.CharmmFile(
            file_name=str(p),
            file_unit=self.trajectory_unit,
            formatted=False,
            read_only=False,
            append=append,
        )
        return [f], {"iuncrd": self.trajectory_unit}


@dataclass
class CharmmMmMinimizeConfig:
    """CGENFF-only minimization (no MLpot) to relax Packmol / IC clashes before ML registration."""

    nstep_sd: int = 50
    nstep_abnr: int = 0
    nprint: int = 10
    tolenr: float = 1e-3
    tolgrd: float = 1e-3
    verbose: bool = True
    show_energy: bool = False
    reference_positions: Optional[np.ndarray] = None
    dcd_path: Optional[PathLike] = None
    dcd_nsavc: int = 1
    dcd_unit: int = 51
    save_crd_path: Optional[PathLike] = None
    save_pdb_path: Optional[PathLike] = None
    save_psf_path: Optional[PathLike] = None
    save_energy_json_path: Optional[PathLike] = None
    save_title: str = "CHARMM MM minimize"
    use_pbc: bool = False
    inbfrq: int | None = None
    ihbfrq: int | None = None


def minimize_charmm_mm_only(config: CharmmMmMinimizeConfig) -> None:
    """Run CHARMM SD (and optional ABNR) on the current PSF using MM terms only.

    Call **before** :func:`register_mlpot` so the PSF still has bonds and no ML model is loaded.
    """
    pycharmm, cons_fix, energy, minimize, *_ = _import_pycharmm_modules()
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    if config.reference_positions is not None:
        sync_charmm_positions(config.reference_positions)

    pbc_side_before_block: float | None = None
    if bool(config.use_pbc):
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            probe_charmm_cubic_box_side_A,
        )

        pbc_side_before_block, _ = probe_charmm_cubic_box_side_A()

    apply_charmm_mm_block()
    if bool(config.use_pbc):
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            restore_charmm_cubic_crystal_lattice,
        )

        if pbc_side_before_block is not None and float(pbc_side_before_block) > 0.0:
            restore_charmm_cubic_crystal_lattice(
                float(pbc_side_before_block),
                quiet=not config.verbose,
            )
    elif not bool(config.use_pbc):
        setup_default_nbonds()
    if config.nstep_sd <= 0 and config.nstep_abnr <= 0:
        return

    n_atoms = int(get_charmm_positions_array().shape[0])
    if n_atoms == 0:
        raise RuntimeError("CHARMM MM minimize: no atoms in PSF (coordinates not loaded?)")

    pycharmm.lingo.charmm_script("ENER")
    if config.verbose:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

        print(
            f"CHARMM MM minimize start: {n_atoms} atoms, GRMS={charmm_grms():.4f} kcal/mol/Å",
            flush=True,
        )

    dcd_file = None
    dcd_kw: dict[str, int] = {}
    if config.dcd_path is not None and config.dcd_nsavc > 0:
        dcd_file = open_minimize_dcd(config.dcd_path, unit=config.dcd_unit)
        dcd_kw = {"iuncrd": config.dcd_unit, "nsavc": max(1, int(config.dcd_nsavc))}

    base_inb = int(config.inbfrq if config.inbfrq is not None else 50)
    inbfrq = _prepare_charmm_mm_minimize_list_frequencies(
        pycharmm,
        use_pbc=bool(config.use_pbc),
        nstep=int(config.nstep_sd),
        inbfrq=base_inb,
    )
    ihbfrq = int(
        config.ihbfrq
        if config.ihbfrq is not None
        else (inbfrq if bool(config.use_pbc) else 0)
    )
    sd_kw = {
        "nstep": max(1, int(config.nstep_sd)),
        "nprint": max(1, int(config.nprint)),
        "tolenr": float(config.tolenr),
        "tolgrd": float(config.tolgrd),
        "inbfrq": inbfrq,
        "ihbfrq": ihbfrq,
        **dcd_kw,
    }
    try:
        if config.verbose and config.show_energy:
            _maybe_show_energy(True)
        if config.nstep_sd > 0:
            if config.verbose:
                print(f"CHARMM MM SD: nstep={config.nstep_sd}", flush=True)
            from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output

            with charmm_quiet_output():
                minimize.run_sd(**sd_kw)
        if config.nstep_abnr > 0:
            if config.verbose:
                print(f"CHARMM MM ABNR: nstep={config.nstep_abnr}", flush=True)
            from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output

            abnr_inb = _prepare_charmm_mm_minimize_list_frequencies(
                pycharmm,
                use_pbc=bool(config.use_pbc),
                nstep=int(config.nstep_abnr),
                inbfrq=base_inb,
            )
            with charmm_quiet_output():
                minimize.run_abnr(
                    nstep=int(config.nstep_abnr),
                    tolenr=float(config.tolenr),
                    tolgrd=float(config.tolgrd),
                    inbfrq=abnr_inb,
                    ihbfrq=abnr_inb if config.use_pbc else 0,
                    **dcd_kw,
                )
        pycharmm.lingo.charmm_script("ENER")
        if config.verbose:
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

            print(
                f"CHARMM MM minimize end: GRMS={charmm_grms():.4f} kcal/mol/Å",
                flush=True,
            )
        if config.verbose and config.show_energy:
            _maybe_show_energy(True)
        if any(
            p is not None
            for p in (
                config.save_crd_path,
                config.save_pdb_path,
                config.save_psf_path,
                config.save_energy_json_path,
            )
        ):
            save_minimization_results(
                pdb_path=config.save_pdb_path,
                crd_path=config.save_crd_path,
                psf_path=config.save_psf_path,
                energy_json_path=config.save_energy_json_path,
                title=config.save_title,
                show_energy=False,
            )
    finally:
        if dcd_file is not None:
            dcd_file.close()
        cons_fix.turn_off()


_BONDED_INTERNAL_TERM_KEYS = ("BOND", "ANGL", "ANGLE", "UREY", "UB", "DIHE", "IMPR", "CMAP")


def _charmm_eterm_value(name: str) -> float | None:
    """Read one CHARMM energy term by name (after ``ENER``)."""
    import pycharmm.energy as energy

    try:
        return float(energy.get_term_by_name(name.upper()))
    except ValueError:
        return None


def charmm_bonded_term_kcalmol(term: str) -> float | None:
    """Read one CHARMM bonded energy term (e.g. ``ANGL``) after ``ENER``."""
    return _charmm_eterm_value(term)


def charmm_internal_energy_kcalmol(*, require: bool = False) -> float | None:
    """CHARMM internal energy (kcal/mol): ``INTE`` if present, else sum of bonded terms."""
    bonded = 0.0
    for name in _BONDED_INTERNAL_TERM_KEYS:
        val = _charmm_eterm_value(name)
        if val is not None:
            bonded += val
    inte = _charmm_eterm_value("INTE")

    if inte is not None and abs(inte) > 1e-8:
        return float(inte)
    if abs(bonded) > 1e-8:
        return bonded
    if inte is not None:
        return float(inte)

    # Fallback when eterm lookup is empty (e.g. some MPI builds).
    terms = charmm_energy_terms()
    if terms:
        eterm = {str(k).strip().upper(): float(v) for k, v in terms.items()}
        bonded = sum(float(eterm.get(k, 0.0)) for k in _BONDED_INTERNAL_TERM_KEYS)
        inte = eterm.get("INTE")
        if inte is not None and abs(inte) > 1e-8:
            return float(inte)
        if abs(bonded) > 1e-8:
            return bonded
        if inte is not None:
            return float(inte)

    if require:
        raise RuntimeError(
            "CHARMM internal energy terms are all zero after ENER "
            "(bonded MM terms could not be read from CHARMM)"
        )
    return None


def _log_bonded_term_diagnostics(*, verbose: bool) -> None:
    """Verbose warning when CHARMM bonded terms read zero after MLpot detach."""
    if not verbose:
        return
    angl = charmm_bonded_term_kcalmol("ANGL")
    bond = charmm_bonded_term_kcalmol("BOND")
    if angl is not None and abs(angl) < 1e-8:
        terms = charmm_energy_terms()
        if terms:
            keys = ", ".join(sorted(terms.keys()))
            print(
                f"WARN: ANGL=0 after ENER (MM-only); energy terms: {keys}",
                flush=True,
            )
        user = _charmm_eterm_value("USER")
        if user is not None and abs(user) > 1e-8:
            print(
                f"WARN: USER={float(user):.4f} kcal/mol still active during MM-only work",
                flush=True,
            )
    if bond is not None and abs(bond) < 1e-8 and angl is not None and abs(angl) < 1e-8:
        print(
            "WARN: BOND and ANGL both zero — check PSF connectivity / BLOCK state",
            flush=True,
        )


def _with_mlpot_detached(ctx: "MlpotContext", fn):
    """Unset MLpot USER, run MM work, then reattach MLpot + hybrid BLOCK."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")
    ctx.unset()
    try:
        return fn()
    finally:
        ctx.reregister_mlpot(reregister_params=True)


@dataclass
class BondedMmMiniConfig:
    """Short bonded-only recovery mini (JAX FIRE and/or CHARMM SD)."""

    nstep_sd: int = 50
    nprint: int = 10
    tolenr: float = 1e-3
    tolgrd: float = 1e-3
    verbose: bool = True
    show_energy: bool = False
    backend: str = "auto"
    nstep_jax: int | None = None


def bonded_mm_mini_config_from_namespace(
    args: Any,
    *,
    nstep_sd: int | None = None,
    verbose: bool | None = None,
    **overrides: Any,
) -> BondedMmMiniConfig:
    """Build :class:`BondedMmMiniConfig` from md-system / dynamics CLI namespace."""
    nprint = max(
        1,
        int(
            getattr(args, "nprint", None)
            or getattr(args, "dyn_nprint", 10)
            or 10
        ),
    )
    kw: dict[str, Any] = dict(
        nstep_sd=int(
            nstep_sd
            if nstep_sd is not None
            else getattr(args, "bonded_mm_mini_steps", 50)
        ),
        nprint=nprint,
        verbose=not bool(getattr(args, "quiet", False))
        if verbose is None
        else bool(verbose),
        show_energy=bool(getattr(args, "show_energy", False)),
        backend=str(getattr(args, "bonded_recovery_backend", "auto")),
    )
    kw.update(overrides)
    return BondedMmMiniConfig(**kw)


def _with_mlpot_block_restored(ctx: "MlpotContext", fn):
    """Run ``fn`` with full/bonded MM registration, then restore hybrid MLpot MM-off."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        MlpotContext,
        _apply_mlpot_psf_mm_off_and_pbc,
    )

    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")
    if ctx.ml_selection is None:
        raise RuntimeError("MlpotContext missing ml_selection for BLOCK restore")
    try:
        return fn()
    finally:
        ctx.block_tag = _apply_mlpot_psf_mm_off_and_pbc(ctx)


def measure_mm_grms_with_full_block(ctx: "MlpotContext") -> float:
    """MM bonded strain proxy: GRMS (kcal/mol/Å) with full MM BLOCK, MLpot stays on."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    def _measure() -> float:
        apply_charmm_mm_block()
        run_charmm_script_quiet("ENER")
        return float(charmm_grms())

    return _with_mlpot_block_restored(ctx, _measure)


def _bonded_recovery_sd_kwargs(ctx: "MlpotContext", config: BondedMmMiniConfig) -> dict[str, Any]:
    """SD frequencies compatible with vacuum vs PBC (CHARMM rejects inbfrq=0 + imgfrq≠0)."""
    kw: dict[str, Any] = {
        "nstep": max(1, int(config.nstep_sd)),
        "nprint": max(1, int(config.nprint)),
        "tolenr": float(config.tolenr),
        "tolgrd": float(config.tolgrd),
        # Always use heuristic NB updates during rescue SD. Heat/production dyna sets
        # imgfrq=50 via _base_dyn_kwargs even for vacuum clusters; inbfrq=0 then
        # triggers FINCYC "INBFRQ is zero when IMGFRQ is not" at BOMLev -2.
        "inbfrq": 50,
        "ihbfrq": 50 if ctx.use_pbc else 0,
    }
    return kw


def _prepare_bonded_mm_rescue_environment(ctx: "MlpotContext") -> None:
    """Refresh pair lists with UPDATE only (no upinb) after MLpot detach."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext
    from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
        prepare_rescue_lists_safe,
    )

    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")
    prepare_rescue_lists_safe(ctx, context="Bonded-MM rescue")


def minimize_bonded_mm_recovery(
    ctx: "MlpotContext",
    config: BondedMmMiniConfig,
    *,
    topology_psf: PathLike | None = None,
) -> float | None:
    """Bonded-only recovery mini (JAX FIRE when enabled, else CHARMM SD with MLpot detached)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")

    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos_before = np.asarray(get_charmm_positions_array(), dtype=np.float64, copy=True)

    backend = str(getattr(config, "backend", "auto")).lower()
    if backend in ("auto", "jax"):
        from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
            minimize_bonded_jax_recovery,
        )

        try:
            grms = minimize_bonded_jax_recovery(
                ctx,
                config,
                topology_psf=topology_psf,
            )
            if grms is not None:
                _print_bonded_recovery_geometry_diff(
                    pos_before,
                    ctx,
                    topology_psf=topology_psf,
                    label="bonded-MM recovery (JAX)",
                )
                return grms
        except Exception as exc:
            if backend == "jax":
                raise
            if config.verbose:
                print(
                    f"Bonded JAX recovery skipped ({exc}); falling back to CHARMM SD",
                    flush=True,
                )

    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _mlpot_covers_all_atoms,
    )

    if _mlpot_covers_all_atoms(ctx):
        if config.verbose:
            print(
                "Bonded-MM CHARMM SD skipped: all-ML cluster "
                "(CGENFF bonded on ML atoms distorts PhysNet geometry; "
                "use MLpot SD recovery instead)",
                flush=True,
            )
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_grms_after_ener_force,
        )

        return charmm_grms_after_ener_force()

    grms = _minimize_bonded_charmm_recovery(ctx, config)
    _print_bonded_recovery_geometry_diff(
        pos_before,
        ctx,
        topology_psf=topology_psf,
        label="bonded-MM recovery (CHARMM SD)",
    )
    return grms


def _print_bonded_recovery_geometry_diff(
    pos_before: np.ndarray,
    ctx: "MlpotContext",
    *,
    topology_psf: PathLike | None,
    label: str,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint_diagnostics import (
        print_geometry_checkpoint_diff,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    print_geometry_checkpoint_diff(
        pos_before,
        get_charmm_positions_array(),
        step_label=label,
        mlpot_ctx=ctx,
        topology_psf=topology_psf,
    )


def _minimize_bonded_charmm_recovery(
    ctx: "MlpotContext",
    config: BondedMmMiniConfig,
) -> float | None:
    """Bonded-only CHARMM SD (BOND/ANGL/DIHE); MLpot detached for pure CHARMM minimization."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_bonded_mm_only_block
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        MlpotContext,
        get_charmm_positions_array,
    )

    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")

    def _run_sd() -> float | None:
        apply_bonded_mm_only_block()
        _prepare_bonded_mm_rescue_environment(ctx)
        pycharmm, cons_fix, *_ = _import_pycharmm_modules()
        minimize = _import_pycharmm_modules()[3]
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output

        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_grms_after_ener_force,
        )

        if config.nstep_sd <= 0:
            return charmm_grms_after_ener_force()

        grms_before = charmm_grms_after_ener_force()
        angl_before = charmm_bonded_term_kcalmol("ANGL")
        bond_before = charmm_bonded_term_kcalmol("BOND")
        _log_bonded_term_diagnostics(verbose=config.verbose)
        if config.verbose:
            msg = (
                f"Bonded-MM mini start: GRMS={grms_before:.4f} kcal/mol/Å "
                "(bonded terms only, MLpot detached)"
            )
            if angl_before is not None:
                msg += f", ANGL={angl_before:.4f} kcal/mol"
            if bond_before is not None:
                msg += f", BOND={bond_before:.4f} kcal/mol"
            print(msg, flush=True)
        sd_kw = _bonded_recovery_sd_kwargs(ctx, config)
        if config.verbose and config.show_energy:
            _maybe_show_energy(True)

        with charmm_quiet_output():
            minimize.run_sd(**sd_kw)
        grms = charmm_grms_after_ener_force()
        angl_after = charmm_bonded_term_kcalmol("ANGL")
        if config.verbose:
            internal_after = charmm_internal_energy_kcalmol()
            msg = f"Bonded-MM mini end: GRMS={grms:.4f} kcal/mol/Å"
            if angl_after is not None:
                msg += f", ANGL={angl_after:.4f} kcal/mol"
                if angl_before is not None:
                    msg += f" (Δ={angl_after - angl_before:+.4f})"
            if internal_after is not None:
                msg += f", internal={internal_after:.4f} kcal/mol"
            print(msg, flush=True)
        cons_fix.turn_off()
        _ = get_charmm_positions_array()
        return grms

    return _with_mlpot_detached(ctx, _run_sd)


def _prepare_overlap_rescue_lists(ctx: "MlpotContext") -> None:
    """Sync pair lists after ``apply_recovery_nbonds`` (UPDATE only; no second upinb)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext
    from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
        prepare_rescue_lists_safe,
    )

    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")
    prepare_rescue_lists_safe(ctx, context="overlap rescue")


def minimize_overlap_rescue(
    ctx: "MlpotContext",
    config: "OverlapRescueConfig",
) -> float | None:
    """Bonded+VDW rescue SD/ABNR (NBXMOD 2); MLpot detached so CHARMM VDW/BOND apply."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_vdw_recovery_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms_after_ener_force
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        MlpotContext,
        apply_recovery_nbonds,
        get_charmm_positions_array,
        restore_workflow_nbonds,
    )

    if not isinstance(config, OverlapRescueConfig):
        raise TypeError("config must be OverlapRescueConfig")
    if not isinstance(ctx, MlpotContext) and not hasattr(ctx, "mock_calls"):
        raise TypeError("ctx must be MlpotContext")

    def _run_rescue() -> float | None:
        apply_recovery_nbonds(ctx)
        apply_bonded_vdw_recovery_block()
        _prepare_overlap_rescue_lists(ctx)
        pycharmm, cons_fix, *_ = _import_pycharmm_modules()
        minimize = _import_pycharmm_modules()[3]
        grms_before = charmm_grms_after_ener_force(silent=not config.verbose)
        if config.verbose:
            print(
                f"Overlap rescue start: GRMS={grms_before:.4f} kcal/mol/Å",
                flush=True,
            )
        try:
            sd_kw = _bonded_recovery_sd_kwargs(
                ctx,
                BondedMmMiniConfig(
                    nstep_sd=config.nstep_sd,
                    nprint=config.nprint,
                    tolenr=config.tolenr,
                    tolgrd=config.tolgrd,
                    verbose=config.verbose,
                    show_energy=False,
                ),
            )
            if config.nstep_sd > 0:
                from mmml.interfaces.pycharmmInterface.charmm_levels import (
                    charmm_quiet_output,
                )

                with charmm_quiet_output():
                    minimize.run_sd(**sd_kw)
            if config.nstep_abnr > 0:
                from mmml.interfaces.pycharmmInterface.charmm_levels import (
                    charmm_quiet_output,
                )

                with charmm_quiet_output():
                    minimize.run_abnr(
                        nstep=int(config.nstep_abnr),
                        tolenr=float(config.tolenr),
                        tolgrd=float(config.tolgrd),
                    )
            grms = charmm_grms_after_ener_force(silent=not config.verbose)
            if config.verbose:
                print(
                    f"Overlap rescue end: GRMS={grms:.4f} kcal/mol/Å",
                    flush=True,
                )
            return grms
        finally:
            restore_workflow_nbonds(ctx)
            cons_fix.turn_off()
            _ = get_charmm_positions_array()

    return _with_mlpot_detached(ctx, _run_rescue)


@dataclass
class MinimizeWithMlpotConfig:
    """SD minimization while MLpot supplies the ML region energy."""

    fixed_ml_selection: Optional[Any] = None
    nstep: int = 500
    nprint: int = 10
    tolenr: float = 1e-5
    tolgrd: float = 1e-5
    save: bool = False
    pdb_path: Optional[PathLike] = None
    crd_path: Optional[PathLike] = None
    psf_path: Optional[PathLike] = None
    energy_json_path: Optional[PathLike] = None
    xyz_path: Optional[PathLike] = None
    dcd_path: Optional[PathLike] = None
    dcd_nsavc: int = 1
    dcd_unit: int = 51
    reference_positions: Optional[np.ndarray] = None
    pyCModel: Optional[Any] = None
    mlpot_ctx: Optional["MlpotContext"] = None
    title: str = "Mini SD"
    skip_if_crd_exists: bool = True
    show_energy: bool = False
    verbose: bool = False
    test_first: Optional["TestFirstConfig"] = None
    # ABNR steps after each SD pass. ``None`` → same as ``nstep``; ``0`` disables ABNR.
    nstep_abnr: Optional[int] = None
    # When the MLpot USER energy before SD exceeds this threshold (kcal/mol),
    # run a bonded-MM rescue pass (MLpot detached) before the main MLpot SD.
    # None disables the check; 1e5 is a safe default for catching severe clashes.
    pre_sd_bonded_recovery_energy_kcalmol: Optional[float] = None
    # Number of bonded-MM SD steps to use for the pre-SD recovery pass.
    pre_sd_bonded_recovery_nstep: int = 200
    # Also run bonded-MM rescue when MLpot GRMS exceeds this (kcal/mol/Å).
    pre_sd_bonded_recovery_grms_kcalmol_A: Optional[float] = None
    pre_sd_bonded_recovery_backend: str = "auto"
    # Chunk long SD/ABNR passes and CHARMM UPDATE between chunks (``inbfrq=0``).
    sd_chunk_nstep: Optional[int] = None
    # Stop SD/ABNR early when hybrid GRMS exceeds ``factor * initial`` or rises sharply.
    sd_grms_watchdog_factor: float = 2.5
    sd_abort_on_grms_increase: bool = True
    # Abort when hybrid GRMS plateaus (CHARMM SD ineffective on geometry_stress).
    sd_stall_patience_chunks: int = 3
    sd_stall_grms_abs_tol: float = 0.1
    # Skip remaining SD chunks when hybrid GRMS is already at/below this ceiling.
    sd_converged_grms_kcalmol_A: Optional[float] = None
    # ASE BFGS on hybrid calculator before MLpot SD (geometry_stress / high GRMS).
    calculator_pre_minimize: bool = True
    calculator_minimize_steps: int = 200
    calculator_minimize_fmax_ev_a: float = 0.05
    calculator_bfgs_maxstep: float = 0.05
    quiet_bfgs: bool = False
    calculator_fire_steps: int = 200
    calculator_fire_fmax_ev_a: float | None = None
    calculator_fire_maxstep: float = 0.2
    # Watchdog compares chunk GRMS to this baseline (post calculator mini if run).
    sd_watchdog_initial_grms: float | None = None


@dataclass(frozen=True)
class MlpotSdChunkResult:
    """Outcome of one chunked SD/ABNR pass under MLpot."""

    completed: bool
    rolled_back: bool = False
    stalled: bool = False
    last_grms: float | None = None
    rolled_back_chunk: int = 0


@dataclass(frozen=True)
class OverlapDynamicsResult:
    """Outcome of :func:`run_dynamics_with_io` (overlap or single-segment)."""

    last_dyn: Any
    integrated_step: int
    completed_full: bool
    salvaged_partial: bool = False
    abort_reason: str | None = None


def _overlap_dynamics_result(
    last_dyn: Any,
    *,
    integrated_step: int,
    completed_full: bool = True,
    salvaged_partial: bool = False,
    abort_reason: str | None = None,
) -> OverlapDynamicsResult:
    return OverlapDynamicsResult(
        last_dyn=last_dyn,
        integrated_step=max(0, int(integrated_step)),
        completed_full=bool(completed_full),
        salvaged_partial=bool(salvaged_partial),
        abort_reason=abort_reason,
    )


def _import_pycharmm_modules():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.cons_fix as cons_fix
    import pycharmm.energy as energy
    import pycharmm.minimize as minimize
    import pycharmm.read as read
    import pycharmm.write as write

    return pycharmm, cons_fix, energy, minimize, read, write


def _ensure_domdec_off_for_mlpot_energy(*, context: str) -> bool:
    """Mockable domdec-off hook before MLpot SD / dynamics."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        ensure_domdec_off_for_mlpot_energy,
    )

    return ensure_domdec_off_for_mlpot_energy(context=context)


def _non_pbc_dyn_freq_kwargs(*, inbfrq: int = 50) -> dict[str, int]:
    """Image list freqs for vacuum / free-space (no crystal updates)."""
    # inbfrq=-1: heuristic rebuild when the cluster moves (best list/force consistency).
    # inbfrq>0: fixed cadence (can leave lists stale between updates; mini uses inbfrq=0).
    return {
        "imgfrq": 0,
        "ihbfrq": 0,
        "ilbfrq": 0,
        "inbfrq": int(inbfrq),
    }


def sync_charmm_lists_after_mini(*, quiet: bool = False) -> None:
    """Refresh NB/MLpot pair lists from current coordinates after mini (``inbfrq=0``).

    MLpot SD keeps ``inbfrq=0`` to avoid per-step ``mlpot_update``; the first NVE
    ``UPDECI`` otherwise jumps from stale lists.  Uses CHARMM ``UPDATE`` (which still
    invokes ``upinb`` / ``UPIMNB`` on PBC builds — avoid calling this before the
    first MLpot SD step right after deferred JAX materialize).
    """
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    pycharmm = _import_pycharmm_modules()[0]
    with charmm_silent_command():
        pycharmm.lingo.charmm_script("ENER")
        pycharmm.lingo.charmm_script("UPDATE")
    if not quiet:
        print(
            "CHARMM UPDATE after mini (sync NB/MLpot lists before dyna)",
            flush=True,
        )


def invalidate_mlpot_calculator_caches(mlpot_ctx: Any | None) -> None:
    """Drop JAX MM pair-list caches after CHARMM UPDATE (positions moved)."""
    if mlpot_ctx is None:
        return
    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if pyCModel is None:
        return
    calc = None
    get_calc = getattr(pyCModel, "get_pycharmm_calculator", None)
    if callable(get_calc):
        try:
            calc = get_calc()
        except Exception:
            calc = None
    if calc is None:
        calc = getattr(pyCModel, "_registered_calculator", None)
    if calc is not None and hasattr(calc, "_cached_update_fn"):
        calc._cached_update_fn = None


def _mlpot_sd_chunk_nstep(config: MinimizeWithMlpotConfig) -> int:
    """Steps per SD/ABNR chunk before CHARMM UPDATE (PBC uses smaller chunks)."""
    if config.sd_chunk_nstep is not None:
        return max(1, int(config.sd_chunk_nstep))
    pbc = bool(getattr(config.mlpot_ctx, "use_pbc", False)) if config.mlpot_ctx else False
    return 25 if pbc else 500


def _effective_mlpot_sd_chunk_nstep(
    config: MinimizeWithMlpotConfig,
    *,
    previous_grms: float | None,
) -> int:
    """Shrink PBC chunks when hybrid GRMS is already low (stale-list risk)."""
    cap = _mlpot_sd_chunk_nstep(config)
    if previous_grms is None or not np.isfinite(previous_grms):
        return cap
    if previous_grms < 1.0:
        return max(1, min(cap, 10))
    if previous_grms < 5.0:
        return max(1, min(cap, 25))
    return cap


# Post-calculator-mini GRMS is often <1 kcal/mol/Å; SD chunk noise to ~2–3 is normal.
_SD_WATCHDOG_LOW_BASELINE_GRMS = 5.0


def _resolved_sd_converged_grms(config: MinimizeWithMlpotConfig) -> float | None:
    """GRMS ceiling below which chunked SD is treated as already converged."""
    if config.sd_converged_grms_kcalmol_A is not None:
        return float(config.sd_converged_grms_kcalmol_A)
    if config.pre_sd_bonded_recovery_grms_kcalmol_A is not None:
        return float(config.pre_sd_bonded_recovery_grms_kcalmol_A)
    return None


def _maybe_abort_sd_on_grms_stall(
    config: MinimizeWithMlpotConfig,
    *,
    previous_grms: float | None,
    current_grms: float,
    stagnant_chunks: int,
    pass_label: str,
    step_label: str,
) -> bool:
    """Return True when hybrid GRMS has plateaued and SD is making no progress."""
    if stagnant_chunks < int(config.sd_stall_patience_chunks):
        return False
    target = _resolved_sd_converged_grms(config)
    if target is not None and current_grms <= float(target):
        return False
    if config.verbose:
        from mmml.utils.prep_ladder_report import PrepMetrics, emit_sd_event

        prev_txt = f"{previous_grms:.4f}" if previous_grms is not None else "?"
        emit_sd_event(
            "stall",
            pass_label,
            metrics=PrepMetrics.from_mlpot(config.mlpot_ctx, hybrid_grms=current_grms),
            detail=(
                f"hybrid GRMS unchanged at {current_grms:.4f} kcal/mol/Å "
                f"(was {prev_txt}) for {stagnant_chunks} chunk(s); stopping early"
            ),
            quiet=False,
        )
    return True


def _maybe_abort_sd_on_grms(
    config: MinimizeWithMlpotConfig,
    *,
    initial_grms: float | None,
    previous_grms: float | None,
    current_grms: float,
    pass_label: str,
    step_label: str,
) -> bool:
    """Return True when hybrid GRMS watchdog should stop further SD/ABNR chunks."""
    if not config.sd_abort_on_grms_increase or config.mlpot_ctx is None:
        return False
    factor = float(config.sd_grms_watchdog_factor)
    max_allowed: float | None = None
    threshold = config.pre_sd_bonded_recovery_grms_kcalmol_A
    if threshold is not None:
        max_allowed = float(threshold)
    low_baseline = (
        initial_grms is not None
        and np.isfinite(initial_grms)
        and float(initial_grms) < _SD_WATCHDOG_LOW_BASELINE_GRMS
    )
    if initial_grms is not None and np.isfinite(initial_grms) and not low_baseline:
        cap = float(initial_grms) * factor
        max_allowed = cap if max_allowed is None else min(max_allowed, cap)
    if max_allowed is not None and current_grms > max_allowed:
        if config.verbose:
            print(
                f"MLpot SD watchdog ({pass_label}, {step_label}): "
                f"hybrid GRMS {current_grms:.4f} > {max_allowed:.4f} kcal/mol/Å; "
                "stopping minimization early",
                flush=True,
            )
        return True
    if (
        previous_grms is not None
        and np.isfinite(previous_grms)
        and previous_grms > 1.0e-6
        and current_grms > float(previous_grms) * factor
    ):
        if low_baseline and max_allowed is not None and current_grms <= max_allowed:
            return False
        if config.verbose:
            print(
                f"MLpot SD watchdog ({pass_label}, {step_label}): "
                f"GRMS rose {previous_grms:.4f} -> {current_grms:.4f} kcal/mol/Å; "
                "stopping minimization early",
                flush=True,
            )
        return True
    return False


def _sync_mlpot_lists_after_sd_chunk(
    config: MinimizeWithMlpotConfig,
    *,
    pass_label: str,
    chunk_index: int,
    n_chunks: int,
    method: str = "SD",
    nstep: int = 0,
) -> float | None:
    """CHARMM UPDATE + hybrid GRMS after one SD/ABNR chunk."""
    sync_charmm_lists_after_mini(quiet=True)
    invalidate_mlpot_calculator_caches(config.mlpot_ctx)
    if config.mlpot_ctx is None:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import refresh_mlpot_energy_and_grms

    context = ""
    grms = refresh_mlpot_energy_and_grms(
        config.mlpot_ctx,
        context=context,
        reregister=False,
        verbose=False,
    )
    if config.verbose and n_chunks > 1:
        from mmml.utils.prep_ladder_report import PrepMetrics, emit_sd_chunk_progress

        emit_sd_chunk_progress(
            method,
            pass_label,
            chunk_index=chunk_index + 1,
            n_chunks=n_chunks,
            nstep=nstep,
            metrics=PrepMetrics.from_mlpot(config.mlpot_ctx, hybrid_grms=grms),
            quiet=False,
        )
    return grms


def _rollback_mlpot_sd_chunk_geometry(
    config: MinimizeWithMlpotConfig,
    positions: np.ndarray,
    *,
    pass_label: str,
    chunk_index: int,
    bad_grms: float,
    good_grms: float | None,
) -> None:
    """Restore last good coordinates after an intra-chunk list/force blow-up."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(positions)
    sync_charmm_lists_after_mini(quiet=True)
    invalidate_mlpot_calculator_caches(config.mlpot_ctx)
    if config.mlpot_ctx is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import refresh_mlpot_energy_and_grms

        refresh_mlpot_energy_and_grms(
            config.mlpot_ctx,
            context="",
            reregister=False,
            verbose=False,
        )
    if config.verbose:
        good_txt = f"{good_grms:.4f}" if good_grms is not None else "?"
        print(
            f"MLpot SD rollback ({pass_label}): restored coords from chunk {chunk_index} "
            f"(GRMS {good_txt} kcal/mol/Å) after spike to {bad_grms:.4f}",
            flush=True,
        )


def _run_minimize_in_chunks(
    minimize: Any,
    pycharmm: Any,
    config: MinimizeWithMlpotConfig,
    base_kw: dict[str, Any],
    *,
    total_nstep: int,
    pass_label: str,
    method: str,
    run_attr: str,
) -> MlpotSdChunkResult:
    """Run ``run_sd`` or ``run_abnr`` in chunks with list sync between chunks."""
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_mlpot_grms_kcalmol_A
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    remaining = max(0, int(total_nstep))
    if remaining <= 0:
        return MlpotSdChunkResult(completed=True)

    run_fn = getattr(minimize, run_attr)
    chunk_cap = _mlpot_sd_chunk_nstep(config)
    min_chunk = min(chunk_cap, 10)
    n_chunks = max(1, (remaining + min_chunk - 1) // min_chunk)

    initial_grms: float | None = None
    previous_grms: float | None = None
    if config.sd_watchdog_initial_grms is not None and np.isfinite(
        config.sd_watchdog_initial_grms
    ):
        initial_grms = float(config.sd_watchdog_initial_grms)
        previous_grms = initial_grms
    elif config.mlpot_ctx is not None:
        initial_grms = resolve_mlpot_grms_kcalmol_A(config.mlpot_ctx, context="")
        previous_grms = initial_grms

    if config.verbose and n_chunks > 1:
        from mmml.utils.prep_ladder_report import PrepMetrics, emit_sd_pass_header

        baseline_metrics = None
        if config.mlpot_ctx is not None and initial_grms is not None and np.isfinite(initial_grms):
            baseline_metrics = PrepMetrics.from_mlpot(
                config.mlpot_ctx,
                hybrid_grms=float(initial_grms),
            )
        emit_sd_pass_header(
            method,
            pass_label,
            remaining=remaining,
            n_chunks=n_chunks,
            chunk_cap=chunk_cap,
            min_chunk=min_chunk,
            metrics=baseline_metrics,
            quiet=False,
        )

    last_good_positions: np.ndarray | None = None
    last_good_grms: float | None = None
    last_good_chunk = 0
    if initial_grms is not None and np.isfinite(initial_grms):
        last_good_positions = get_charmm_positions_array().copy()
        last_good_grms = float(initial_grms)
        last_good_chunk = 0

    chunk_index = 0
    stagnant_chunks = 0
    converged_grms = _resolved_sd_converged_grms(config)
    pbc_sd = bool(getattr(config.mlpot_ctx, "use_pbc", False)) if config.mlpot_ctx else False
    while remaining > 0:
        chunk_index += 1
        if chunk_index == 1 and pbc_sd:
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                assert_charmm_pbc_lattice_ready_for_mlpot,
            )

            box_side = getattr(config.mlpot_ctx, "cubic_box_side_A", None)
            assert_charmm_pbc_lattice_ready_for_mlpot(
                context=f"MLpot {method} {pass_label}",
                cubic_box_side_A=float(box_side) if box_side is not None else None,
            )
        if chunk_index == 1 and config.mlpot_ctx is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
                materialize_deferred_mlpot_jax_before_sd,
            )

            materialize_deferred_mlpot_jax_before_sd(
                config.mlpot_ctx,
                verbose=config.verbose,
                force_ener_probe=True,
            )
            from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                recover_mpi_for_charmm_after_jax,
            )

            recover_mpi_for_charmm_after_jax(phase="before MLpot SD chunk")
        step = min(_effective_mlpot_sd_chunk_nstep(config, previous_grms=previous_grms), remaining)
        kw = {**base_kw, "nstep": step}
        _prepare_mlpot_sd_list_frequencies(pycharmm, sd_kw=kw)
        if config.verbose:
            if n_chunks > 1:
                pass
            elif chunk_index == 1:
                from mmml.utils.prep_ladder_report import emit_tagged

                emit_tagged(
                    "SD",
                    f"{method} {pass_label}: nstep={remaining} nprint={config.nprint}",
                    tag_style="bold blue",
                    quiet=False,
                )
        with charmm_quiet_output():
            run_fn(**kw)
        remaining -= step
        if remaining <= 0:
            break
        grms = _sync_mlpot_lists_after_sd_chunk(
            config,
            pass_label=pass_label,
            chunk_index=chunk_index,
            n_chunks=n_chunks,
            method=method,
            nstep=step,
        )
        if grms is not None and converged_grms is not None and grms <= float(converged_grms):
            improved = (
                initial_grms is None
                or not np.isfinite(initial_grms)
                or float(grms) < float(initial_grms) - float(config.sd_stall_grms_abs_tol)
            )
            if improved:
                if config.verbose:
                    print(
                        f"MLpot SD converged ({pass_label}, after chunk {chunk_index}): "
                        f"hybrid GRMS {grms:.4f} <= {float(converged_grms):.4f} kcal/mol/Å",
                        flush=True,
                    )
                return MlpotSdChunkResult(completed=True, last_grms=grms)
        if grms is not None and previous_grms is not None:
            if abs(float(grms) - float(previous_grms)) <= float(config.sd_stall_grms_abs_tol):
                stagnant_chunks += 1
            else:
                stagnant_chunks = 0
        if grms is not None and _maybe_abort_sd_on_grms_stall(
            config,
            previous_grms=previous_grms,
            current_grms=grms,
            stagnant_chunks=stagnant_chunks,
            pass_label=pass_label,
            step_label=f"after chunk {chunk_index}",
        ):
            return MlpotSdChunkResult(
                completed=False,
                stalled=True,
                last_grms=grms,
            )
        if grms is not None and _maybe_abort_sd_on_grms(
            config,
            initial_grms=initial_grms,
            previous_grms=previous_grms,
            current_grms=grms,
            pass_label=pass_label,
            step_label=f"after chunk {chunk_index}",
        ):
            if last_good_positions is not None:
                _rollback_mlpot_sd_chunk_geometry(
                    config,
                    last_good_positions,
                    pass_label=pass_label,
                    chunk_index=last_good_chunk,
                    bad_grms=grms,
                    good_grms=last_good_grms,
                )
                return MlpotSdChunkResult(
                    completed=False,
                    rolled_back=True,
                    last_grms=last_good_grms,
                    rolled_back_chunk=last_good_chunk,
                )
            return MlpotSdChunkResult(completed=False)
        if grms is not None:
            previous_grms = grms
            last_good_positions = get_charmm_positions_array().copy()
            last_good_grms = grms
            last_good_chunk = chunk_index
    return MlpotSdChunkResult(completed=True, last_grms=previous_grms)


def apply_dyn_inbfrq_from_args(
    kw: dict[str, Any],
    args: Any,
    *,
    charmm_pbc: bool,
) -> None:
    """Override ``inbfrq`` (and vacuum image freqs) when ``--dyn-inbfrq`` is set."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_dyn_inbfrq

    inb = resolve_dyn_inbfrq(args)
    if inb is None:
        return
    kw["inbfrq"] = int(inb)
    if not charmm_pbc:
        kw["imgfrq"] = 0
        kw["ihbfrq"] = 0
        kw["ilbfrq"] = 0


def apply_dyn_imgfrq_from_args(
    kw: dict[str, Any],
    args: Any,
    *,
    charmm_pbc: bool,
) -> None:
    """Override PBC image/HB list cadence when ``--dyn-imgfrq`` is set."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_dyn_imgfrq

    if not charmm_pbc:
        return
    img = resolve_dyn_imgfrq(args)
    if img is None:
        return
    freq = int(img)
    kw["imgfrq"] = freq
    kw["ihbfrq"] = freq
    kw["ilbfrq"] = freq


def _strip_crystal_dyn_keywords(kw: dict[str, Any]) -> None:
    """Remove ``ixtfrq`` when no crystal is active (avoids DCNTRL extraneous-keyword warn)."""
    kw.pop("ixtfrq", None)


# Image/HB/extended list freqs disabled for loose-PBC (cluster in large box, no MIC).
# ``ntrfrq`` (COM rotation/translation stop) must also stay above ``nstep`` on short
# segments or it fires mid-run (FINCYC retune / spurious COM resets during heat).
_LOOSE_PBC_DISABLED_FREQ_KEYS = ("ixtfrq", "imgfrq", "ihbfrq", "ilbfrq", "ntrfrq")


def _ensure_ntrfrq_above_nstep(kw: dict[str, Any], nstep: int) -> None:
    """Lift ``ntrfrq`` above ``nstep`` when it would fire during this integration."""
    if "ntrfrq" not in kw:
        return
    n = max(1, int(nstep))
    val = int(kw["ntrfrq"])
    if val > 0 and val <= n:
        kw["ntrfrq"] = n + 1


def _ensure_ixtfrq_above_nstep(kw: dict[str, Any], nstep: int) -> None:
    """Skip crystal transform updates (``ixtfrq``) within this integration segment.

    Fixed-box NVT (scale heat, Hoover with ``pmass=0``) does not resize the cell;
    ``UPDECI`` at ``ixtfrq`` can segfault in ``mlpot_update`` on MPI-linked MLpot PBC.
    """
    if "ixtfrq" not in kw:
        return
    n = max(1, int(nstep))
    val = int(kw["ixtfrq"])
    if val > 0 and val <= n:
        kw["ixtfrq"] = n + 1


def _maybe_disable_ixtfrq_for_fixed_volume_chunk(
    chunk_kw: dict[str, Any],
    chunk_nstep: int,
) -> None:
    """Disable ``ixtfrq`` within a dynamics chunk unless this segment is NPT barostat."""
    pmass = chunk_kw.get("pmass")
    is_npt = bool(chunk_kw.get("cpt")) and pmass is not None and int(pmass) > 0
    if is_npt:
        return
    _ensure_ixtfrq_above_nstep(chunk_kw, chunk_nstep)


def apply_loose_pbc_dyn_freq_kwargs(kw: dict[str, Any], *, nstep: int) -> None:
    """Set crystal/image list rebuild freqs above ``nstep`` so they never fire.

    CHARMM skips updates when the frequency exceeds the integration length.
    Used for loose PBC (``--box-size`` without ``--mlpot-pbc``): crystal stays
    active for Hoover CPT, but periodic image lists are not needed physically.
    """
    n = max(1, int(nstep))
    disabled = n + 1
    for key in _LOOSE_PBC_DISABLED_FREQ_KEYS:
        if key in kw:
            kw[key] = disabled


def _base_dyn_kwargs(
    *,
    timestep: float,
    nstep: int,
    nsavc: int,
    inbfrq: int = 50,
    ihbfrq: int = 50,
    ilbfrq: int = 50,
    imgfrq: int = 50,
    ixtfrq: int = 1000,
    nprint: int = 50,
    iprfrq: int = 500,
    isvfrq: int = 50,
    ntrfrq: int = 0,
    echeck: float = 100.0,
) -> dict[str, Any]:
    return {
        "timestep": timestep,
        "nstep": nstep,
        "nsavc": nsavc,
        "inbfrq": inbfrq,
        "ihbfrq": ihbfrq,
        "ilbfrq": ilbfrq,
        "imgfrq": imgfrq,
        "ixtfrq": ixtfrq,
        "nprint": nprint,
        "iprfrq": iprfrq,
        "isvfrq": isvfrq,
        "ntrfrq": ntrfrq,
        "echeck": echeck,
    }


def ps_to_nsteps(timestep_ps: float, duration_ps: float) -> int:
    """Convert a timestep (ps) and total time (ps) to an integer step count."""
    return int(round(duration_ps / timestep_ps))


def nsavc_for_interval(timestep_ps: float, interval_ps: float) -> int:
    """Steps between trajectory saves."""
    return max(1, int(round(interval_ps / timestep_ps)))


def apply_heat_ramp_frequencies(
    kw: dict[str, Any],
    *,
    nstep: int,
    ihtfrq: int,
) -> None:
    """Update ``ihtfrq`` / ``TEMINC`` after the stage length or CLI cadence is known."""
    nstep = max(1, int(nstep))
    requested = max(1, int(ihtfrq))
    min_rescales = 2 if nstep >= 2 else 1
    iht = min(requested, nstep)
    # CHARMM disables IHTFRQ velocity scaling when ihtfrq >= nstep (no interior events).
    if iht >= nstep:
        iht = max(1, nstep // min_rescales)
    iht = _harmonize_dynamics_frequency(iht, nstep)
    firstt = float(kw.get("firstt", kw.get("finalt", 300.0)))
    finalt = float(kw.get("finalt", 300.0))
    heat_updates = max(1, int(nstep) // iht)
    kw["ihtfrq"] = iht
    kw["TEMINC"] = max(0.0, (finalt - firstt) / heat_updates)


def _apply_overlap_chunk_heat_ramp(
    chunk_kw: dict[str, Any],
    *,
    chunk_index: int,
    chunk_nstep: int,
    steps_done: int,
    ramp_spec: dict[str, float | int],
) -> None:
    """Recompute ``ihtfrq`` / ``TEMINC`` for one overlap chunk (stage cadence ≠ chunk ``nstep``)."""
    apply_heat_ramp_frequencies(
        chunk_kw,
        nstep=int(chunk_nstep),
        ihtfrq=int(ramp_spec["ihtfrq"]),
    )
    if int(chunk_index) <= 0:
        return
    apply_heat_ramp_overlap_chunk(
        chunk_kw,
        chunk_index=int(chunk_index),
        steps_done=int(steps_done),
        ramp_spec=ramp_spec,
    )


def heat_ramp_bath_target_K(
    *,
    firstt: float,
    finalt: float,
    teminc: float,
    ihtfrq: int,
    step: int,
) -> float:
    """Target bath temperature (K) after ``step`` integration steps with ``ihtfrq`` scaling."""
    if ihtfrq <= 0 or teminc <= 0.0:
        return float(firstt)
    rescales = max(0, int(step) // int(ihtfrq))
    target = float(firstt) + rescales * float(teminc)
    return min(target, float(finalt))


def hoover_cpt_heat_ramp_target_K(
    *,
    firstt: float,
    finalt: float,
    step: int,
    total_nstep: int,
    n_chunks: int = 1,
) -> float:
    """Linear Hoover CPT bath target (K) after ``step`` of ``total_nstep`` segment steps."""
    t0 = float(firstt)
    t1 = float(finalt)
    if t1 <= t0:
        return t1
    if int(n_chunks) <= 1:
        # One dyna segment: avoid hoover reft=finalt for the whole leg (overshoot).
        return 0.5 * (t0 + t1)
    if total_nstep <= 0:
        return t1
    frac = min(1.0, max(0.0, int(step) / int(total_nstep)))
    return t0 + frac * (t1 - t0)


def hoover_cpt_heat_ramp_spec_from_kw(
    kw: dict[str, Any],
) -> dict[str, float] | None:
    """Return segment-level Hoover CPT ramp endpoints, or ``None`` if inactive."""
    if not bool(kw.get("cpt")) or "hoover reft" not in kw:
        return None
    if int(kw.get("ihtfrq", 0) or 0) > 0:
        return None
    firstt = kw.get("firstt")
    finalt = kw.get("finalt")
    if firstt is None or finalt is None:
        return None
    t0 = float(firstt)
    t1 = float(finalt)
    if t1 <= t0:
        return None
    return {"firstt": t0, "finalt": t1}


def _cpt_npt_fresh_barostat_from_restart_kw(chunk_kw: dict[str, Any]) -> bool:
    """True when overlap must keep ``restart+start+iasvel=1`` for CPT barostat init.

    HEAT Hoover NVT uses ``pmass=0``; EQUI/PROD NPT uses ``pmass>0``.  A plain
    ``dyna restart`` without ``start`` leaves barostat pistons uninitialized
    (garbage ``PIXX`` / ``PRESSI`` at step 0).  ``staged_workflow`` sets
    ``restart+start`` via :func:`staged_workflow._configure_equi_dynamics_start`;
    overlap chunk prep must not clear those flags on chunk 0.
    """
    if not bool(chunk_kw.get("cpt")):
        return False
    pmass = chunk_kw.get("pmass")
    if pmass is None or int(pmass) <= 0:
        return False
    return bool(chunk_kw.get("start")) and int(chunk_kw.get("iasvel", 0) or 0) == 1


def _hoover_cpt_overlap_chunk_needs_cold_start(
    chunk_kw: dict[str, Any],
    *,
    chunk_index: int,
    has_restart_read: bool,
    steps_done: int,
) -> bool:
    """True when overlap chunk 0 needs single-dyna Boltzmann + CPT barostat init."""
    if int(chunk_index) != 0:
        return False
    if bool(chunk_kw.get("restart")) or has_restart_read:
        return False
    if int(steps_done) != 0:
        return False
    if not bool(chunk_kw.get("cpt")):
        return False
    # Post-nstep=0 Boltzmann assign (NPT / restart-file path): keep iasvel=0.
    if chunk_kw.get("start") is False and int(chunk_kw.get("iasvel", 1)) == 0:
        return False
    return True


def apply_hoover_cpt_heat_ramp_overlap_chunk(
    chunk_kw: dict[str, Any],
    *,
    chunk_index: int,
    steps_done: int,
    ramp_spec: dict[str, float],
    total_nstep: int,
    n_chunks: int = 1,
) -> None:
    """Set Hoover CPT bath target for overlap chunk ``chunk_index`` (including 0)."""
    target = hoover_cpt_heat_ramp_target_K(
        firstt=float(ramp_spec["firstt"]),
        finalt=float(ramp_spec["finalt"]),
        step=int(steps_done),
        total_nstep=int(total_nstep),
        n_chunks=int(n_chunks),
    )
    chunk_kw["firstt"] = target
    chunk_kw["finalt"] = float(ramp_spec["finalt"])
    chunk_kw["tbath"] = float(ramp_spec["finalt"])
    chunk_kw["hoover reft"] = target
    # Chunk 0 at segment start: single dyna with ``start`` + ``iasvel=1`` (see
    # ``_configure_heat_dynamics_start``).  Later chunks / retries keep RAM vel.
    if _hoover_cpt_overlap_chunk_needs_cold_start(
        chunk_kw,
        chunk_index=chunk_index,
        has_restart_read=False,
        steps_done=int(steps_done),
    ):
        chunk_kw["start"] = True
        chunk_kw["iasvel"] = 1
    else:
        chunk_kw["iasvel"] = 0
        chunk_kw["start"] = False


def heat_ramp_spec_from_kw(kw: dict[str, Any]) -> dict[str, float | int] | None:
    """Return stage-level velocity-scaling ramp parameters, or ``None`` if inactive."""
    ihtfrq = int(kw.get("ihtfrq", 0) or 0)
    if ihtfrq <= 0 or "hoover reft" in kw or bool(kw.get("cpt")):
        return None
    teminc = float(kw.get("TEMINC", 0) or 0)
    if teminc <= 0.0:
        return None
    return {
        "firstt": float(kw.get("firstt", kw.get("finalt", 300.0))),
        "finalt": float(kw.get("finalt", 300.0)),
        "teminc": teminc,
        "ihtfrq": ihtfrq,
    }


def apply_heat_ramp_overlap_chunk(
    chunk_kw: dict[str, Any],
    *,
    chunk_index: int,
    steps_done: int,
    ramp_spec: dict[str, float | int],
) -> None:
    """Continue a velocity-scaling heat ramp on overlap chunk ``chunk_index`` > 0.

    PyCHARMM omits ``start`` when ``start=False``, so CHARMM may keep START active
    from chunk 0.  With lingering START, ``iasvel=0`` reads comparison **coordinates**
    as velocities (``COMP_AND_HEATING.md``) — zero COMP → T≈0 and ``ihtfrq`` scaling
    from ``**********`` old temperature.  Use ``iasvel=1`` so lingering START falls back
    to Boltzmann at the chunk ``firstt`` target instead of COMP.
    """
    if chunk_index <= 0:
        return
    chunk_kw["firstt"] = heat_ramp_bath_target_K(
        firstt=float(ramp_spec["firstt"]),
        finalt=float(ramp_spec["finalt"]),
        teminc=float(ramp_spec["teminc"]),
        ihtfrq=int(ramp_spec["ihtfrq"]),
        step=int(steps_done),
    )
    chunk_kw["finalt"] = float(ramp_spec["finalt"])
    # ``ihtfrq`` / ``TEMINC`` for this chunk's ``nstep`` come from
    # ``apply_heat_ramp_frequencies`` (called immediately before this helper).
    chunk_kw["iasvel"] = 1
    chunk_kw["iasors"] = 0
    chunk_kw["start"] = False


_HEAT_FIN_FREQ_KEYS = ("ihtfrq", "iprfrq", "nprint", "isvfrq")


def apply_heat_segment_ramp_kwargs(
    kw: dict[str, Any],
    *,
    seg_index: int,
    n_segments: int,
    heat_firstt: float,
    heat_finalt: float,
    nstep: int,
    ihtfrq: int,
) -> None:
    """Set per-segment bath ramp for staged heating (``n_heat_segments`` > 1)."""
    if n_segments <= 1:
        return
    n_seg = max(1, int(n_segments))
    seg_i = max(0, min(int(seg_index), n_seg - 1))
    t0 = float(heat_firstt)
    t1 = float(heat_finalt)
    seg_firstt = t0 + (t1 - t0) * (seg_i / n_seg)
    seg_finalt = t0 + (t1 - t0) * ((seg_i + 1) / n_seg)
    kw["firstt"] = seg_firstt
    kw["finalt"] = seg_finalt
    kw["tbath"] = seg_finalt
    if "hoover reft" in kw:
        kw["hoover reft"] = seg_firstt
    iht = max(1, min(int(ihtfrq), max(1, int(nstep))))
    if int(kw.get("ihtfrq", 0) or 0) > 0:
        heat_updates = max(1, int(nstep) // iht)
        kw["ihtfrq"] = iht
        kw["TEMINC"] = max(0.0, (seg_finalt - seg_firstt) / heat_updates)


def finalize_heat_dynamics_frequencies(kw: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Harmonize heat thermostat/print freqs with ``nstep`` and refresh ``TEMINC``.

    CHARMM FINCYC retunes frequencies that do not divide ``nstep``.  When that
    happens to ``ihtfrq``, the ``TEMINC`` computed for the CLI value no longer
    matches the rescale cadence and the bath target can jump backward mid-ramp.
    """
    nstep = int(kw.get("nstep", 0))
    changes: dict[str, tuple[int, int]] = {}
    if nstep <= 0:
        return changes
    if heat_ramp_spec_from_kw(kw) is None:
        return changes
    for key in _HEAT_FIN_FREQ_KEYS:
        if key not in kw:
            continue
        old = int(kw[key])
        new = _harmonize_dynamics_frequency(old, nstep)
        if new != old:
            changes[key] = (old, new)
        kw[key] = new
    if "nsavc" in kw:
        ns = int(kw["nsavc"])
        for key in ("nprint", "iprfrq", "isvfrq"):
            if key in kw and int(kw[key]) != ns:
                changes[key] = (int(kw[key]), ns)
                kw[key] = ns
    apply_heat_ramp_frequencies(kw, nstep=nstep, ihtfrq=int(kw["ihtfrq"]))
    return changes


def _infer_heat_velocity_init_label(
    kw: dict[str, Any],
    *,
    heat_thermostat: str,
) -> str:
    """Human-readable label for how velocities / CPT state are initialized."""
    hoover_cpt = heat_thermostat == "hoover" and bool(kw.get("cpt"))
    start = bool(kw.get("start"))
    restart = bool(kw.get("restart"))
    iasvel = int(kw.get("iasvel", 0) or 0)

    if start and not restart and iasvel == 1:
        if hoover_cpt:
            return (
                "single dyna (start=True, restart=False, iasvel=1): Boltzmann at "
                "FIRSTT + Hoover CPT barostat init in one call (no separate nstep=0)"
            )
        return (
            "single dyna (start=True, restart=False, iasvel=1): Boltzmann at "
            "FIRSTT, then ihtfrq velocity scaling (iasors=0)"
        )
    if restart and start and iasvel == 1:
        if hoover_cpt:
            return (
                "single dyna (restart+start, iasvel=1): READYN coordinates + Boltzmann "
                "+ CPT barostat init (no separate nstep=0)"
            )
        return (
            "single dyna (restart+start, iasvel=1): READYN coordinates + Boltzmann "
            "at FIRSTT, then ihtfrq velocity scaling (iasors=0)"
        )
    if restart and not start:
        return (
            "dyna restart (start=False, iasvel=0): continue global step and "
            "thermostat/barostat state from READYN"
        )
    if not restart and not start and iasvel == 0:
        return (
            "in-process continuation (iasvel=0): keep in-memory velocities "
            "(overlap chunk / CPT sub-chunk / segment handoff)"
        )
    return f"custom flags (start={start}, restart={restart}, iasvel={iasvel})"


def _classify_heat_thermostat_mode(
    kw: dict[str, Any],
    *,
    heat_thermostat: str,
    use_pbc: bool,
) -> str:
    if heat_thermostat == "hoover" and use_pbc and bool(kw.get("cpt")):
        return "Hoover CPT NVT (constant volume: pmass=0, pint pconst)"
    if heat_thermostat == "hoover" and not use_pbc:
        return (
            "Hoover requested but vacuum/no PBC → scale fallback "
            "(ihtfrq/TEMINC, iasors=0; CPT requires CRYSTal)"
        )
    if int(kw.get("ihtfrq", 0) or 0) > 0:
        return "velocity scaling heat (ihtfrq + TEMINC ramp, iasors=0)"
    if heat_thermostat == "hoover":
        return "Hoover (no CPT keywords in dyna dict)"
    return f"heat thermostat={heat_thermostat!r} (no ihtfrq ramp active)"


def describe_heat_dynamics_setup(
    kw: dict[str, Any],
    *,
    heat_thermostat: str,
    use_pbc: bool = True,
    args: Any | None = None,
    segment_index: int | None = None,
    n_segments: int | None = None,
    use_memory: bool = False,
    restart_read: PathLike | None = None,
    restart_write: PathLike | None = None,
    stage_ps: float | None = None,
    timestep_ps: float | None = None,
    freq_harmonized: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Structured snapshot of resolved HEAT ``dyna`` keywords for debugging."""
    nstep = int(kw.get("nstep", 0) or 0)
    ts = float(timestep_ps if timestep_ps is not None else kw.get("timestep", 0.00025))
    duration_ps = stage_ps if stage_ps is not None else nstep * ts

    thermostat_requested = heat_thermostat
    thermostat_forced = False
    force_reason: str | None = None
    if args is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            _requested_heat_thermostat,
            heat_thermostat_requires_hoover_after_pretreat,
            resolve_heat_thermostat,
        )

        thermostat_requested = _requested_heat_thermostat(args)
        effective = resolve_heat_thermostat(args)
        heat_thermostat = effective
        if thermostat_requested != effective:
            thermostat_forced = True
            if heat_thermostat_requires_hoover_after_pretreat(args):
                force_reason = "CHARMM MM pretreat + PBC heat (scale/ihtfrq conflicts with Hoover)"

    ramp_updates = 0
    ihtfrq = int(kw.get("ihtfrq", 0) or 0)
    if ihtfrq > 0 and nstep > 0:
        ramp_updates = max(1, nstep // ihtfrq)

    info: dict[str, Any] = {
        "segment": (
            f"{segment_index + 1}/{n_segments}"
            if segment_index is not None and n_segments is not None
            else None
        ),
        "thermostat_requested": thermostat_requested,
        "thermostat_effective": heat_thermostat,
        "thermostat_forced": thermostat_forced,
        "thermostat_force_reason": force_reason,
        "mode": _classify_heat_thermostat_mode(
            kw, heat_thermostat=heat_thermostat, use_pbc=use_pbc
        ),
        "velocity_init": _infer_heat_velocity_init_label(
            kw, heat_thermostat=heat_thermostat
        ),
        "use_pbc": bool(use_pbc),
        "use_memory_handoff": bool(use_memory),
        "integration": {
            "nstep": nstep,
            "timestep_ps": ts,
            "duration_ps": duration_ps,
            "nsavc": kw.get("nsavc"),
            "dcd_interval_ps": kw.get("dcd_interval_ps"),
            "echeck": kw.get("echeck"),
            "inbfrq": kw.get("inbfrq"),
            "imgfrq": kw.get("imgfrq"),
            "ixtfrq": kw.get("ixtfrq"),
        },
        "temperature": {
            "firstt_K": kw.get("firstt"),
            "finalt_K": kw.get("finalt"),
            "tbath_K": kw.get("tbath"),
            "tstruct_K": kw.get("tstruct"),
            "hoover_reft_K": kw.get("hoover reft"),
            "treference_K": kw.get("treference"),
            "ihtfrq": ihtfrq,
            "TEMINC_K": kw.get("TEMINC"),
            "ieqfrq": kw.get("ieqfrq"),
            "ramp_rescale_events": ramp_updates if ihtfrq > 0 else 0,
        },
        "dyna_flags": {
            "start": bool(kw.get("start")),
            "restart": bool(kw.get("restart")),
            "new": bool(kw.get("new")),
            "iasvel": kw.get("iasvel"),
            "iasors": kw.get("iasors"),
            "iasors_meaning": (
                "Gaussian reassignment each ihtfrq (legacy)"
                if int(kw.get("iasors", 0) or 0) == 1
                else "velocity scaling at ihtfrq (preferred for ML heat)"
            ),
            "iscale": kw.get("iscale"),
            "iscvel": kw.get("iscvel"),
            "ichecw": kw.get("ichecw"),
            "iunrea": kw.get("iunrea"),
        },
        "cpt": (
            {
                "cpt": bool(kw.get("cpt")),
                "pmass": kw.get("pmass"),
                "tmass": kw.get("tmass"),
                "pgamma": kw.get("pgamma"),
                "pint": kw.get("pint"),
                "pref": kw.get("pref"),
                "leap": bool(kw.get("leap")),
            }
            if kw.get("cpt")
            else None
        ),
        "restart_io": {
            "restart_read": str(restart_read) if restart_read is not None else None,
            "restart_write": str(restart_write) if restart_write is not None else None,
        },
        "freq_harmonized": freq_harmonized or {},
    }
    if args is not None:
        info["cli"] = {
            "heat_firstt": getattr(args, "heat_firstt", None),
            "heat_finalt": getattr(args, "heat_finalt", None),
            "heat_ihtfrq": getattr(args, "heat_ihtfrq", None),
            "no_echeck_heat": bool(getattr(args, "no_echeck_heat", False)),
            "no_echeck": bool(getattr(args, "no_echeck", False)),
            "heat_comp_damp": bool(getattr(args, "heat_comp_damp", False)),
            "n_heat_segments": int(getattr(args, "n_heat_segments", 1) or 1),
        }
    return info


def format_heat_dynamics_diagnostics(info: dict[str, Any]) -> str:
    """Render :func:`describe_heat_dynamics_setup` as a multi-line debug block."""
    lines: list[str] = []
    title = "HEAT dynamics diagnostics"
    if info.get("segment"):
        title += f" (segment {info['segment']})"
    lines.append(f"=== {title} ===")

    req = info["thermostat_requested"]
    eff = info["thermostat_effective"]
    lines.append("Thermostat policy:")
    lines.append(f"  CLI --heat-thermostat: {req!r} → effective: {eff!r}")
    if info.get("thermostat_forced"):
        lines.append(
            f"  FORCED to {eff!r}: {info.get('thermostat_force_reason') or 'policy override'}"
        )
    lines.append(f"  Mode: {info['mode']}")
    lines.append(f"  PBC: {info['use_pbc']} | memory handoff: {info['use_memory_handoff']}")

    integ = info["integration"]
    lines.append("Integration:")
    lines.append(
        f"  nstep={integ['nstep']}  timestep={integ['timestep_ps']:.6g} ps  "
        f"duration≈{float(integ['duration_ps']):.6g} ps"
    )
    lines.append(
        f"  nsavc={integ['nsavc']}  dcd_interval_ps={integ['dcd_interval_ps']}  "
        f"echeck={integ['echeck']}"
    )
    lines.append(
        f"  inbfrq={integ['inbfrq']}  imgfrq={integ['imgfrq']}  ixtfrq={integ['ixtfrq']}"
    )

    temp = info["temperature"]
    lines.append("Temperature / bath:")
    lines.append(
        f"  firstt={temp['firstt_K']} K  finalt={temp['finalt_K']} K  "
        f"tbath={temp['tbath_K']} K"
    )
    if temp.get("hoover_reft_K") is not None:
        lines.append(f"  hoover reft={temp['hoover_reft_K']} K")
    if int(temp.get("ihtfrq") or 0) > 0:
        lines.append(
            f"  ihtfrq={temp['ihtfrq']}  TEMINC={temp['TEMINC_K']} K/step  "
            f"~{temp['ramp_rescale_events']} rescale(s) this segment"
        )
    else:
        lines.append("  ihtfrq=0 (no velocity-scaling ramp; Hoover or fixed bath)")
    lines.append(f"  ieqfrq={temp['ieqfrq']}")

    flags = info["dyna_flags"]
    lines.append("CHARMM dyna flags:")
    lines.append(
        f"  start={flags['start']}  restart={flags['restart']}  new={flags['new']}"
    )
    lines.append(
        f"  iasvel={flags['iasvel']}  iasors={flags['iasors']} "
        f"({flags['iasors_meaning']})"
    )
    lines.append(f"  iunrea={flags['iunrea']}")

    lines.append("Velocity / CPT initialization:")
    lines.append(f"  {info['velocity_init']}")

    cpt = info.get("cpt")
    if cpt is not None:
        lines.append("CPT barostat:")
        lines.append(
            f"  pmass={cpt['pmass']}  tmass={cpt['tmass']}  pgamma={cpt['pgamma']}  "
            f"pint={cpt['pint']!r}  pref={cpt['pref']}"
        )
        lines.append(f"  leap={cpt['leap']}")

    io_info = info["restart_io"]
    lines.append("Restart I/O:")
    lines.append(f"  restart_read: {io_info['restart_read'] or '(none / in-memory)'}")
    lines.append(f"  restart_write: {io_info['restart_write'] or '(none)'}")

    harmonized = info.get("freq_harmonized") or {}
    if harmonized:
        parts = ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(harmonized.items()))
        lines.append(f"Frequency harmonization (vs nstep): {parts}")

    cli = info.get("cli")
    if cli is not None:
        lines.append("CLI heat options:")
        lines.append(
            f"  --heat-firstt={cli['heat_firstt']}  --heat-finalt={cli['heat_finalt']}  "
            f"--heat-ihtfrq={cli['heat_ihtfrq']}"
        )
        lines.append(
            f"  no_echeck_heat={cli['no_echeck_heat']}  heat_comp_damp={cli['heat_comp_damp']}  "
            f"n_heat_segments={cli['n_heat_segments']}"
        )

    lines.append("=" * (len(title) + 8))
    return "\n".join(lines)


def build_heat_dynamics_dashboard_sections(
    info: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Rich dashboard sections for :func:`describe_heat_dynamics_setup`."""
    sections: list[tuple[str, dict[str, Any]]] = []

    therm: dict[str, Any] = {
        "CLI --heat-thermostat": f"{info['thermostat_requested']!r} → {info['thermostat_effective']!r}",
        "mode": info["mode"],
        "PBC": info["use_pbc"],
        "memory handoff": info["use_memory_handoff"],
    }
    if info.get("thermostat_forced"):
        therm["forced"] = info.get("thermostat_force_reason") or "policy override"
    sections.append(("Thermostat policy", therm))

    integ = info["integration"]
    dcd_iv = integ.get("dcd_interval_ps")
    sections.append(
        (
            "Integration",
            {
                "nstep": integ["nstep"],
                "timestep (ps)": f"{integ['timestep_ps']:.6g}",
                "duration (ps)": f"{float(integ['duration_ps']):.6g}",
                "nsavc": integ["nsavc"],
                "dcd interval (ps)": f"{dcd_iv:.6g}" if dcd_iv is not None else "—",
                "echeck": integ["echeck"],
                "inbfrq": integ["inbfrq"],
                "imgfrq": integ["imgfrq"],
                "ixtfrq": integ["ixtfrq"],
            },
        )
    )

    temp = info["temperature"]
    temp_map: dict[str, Any] = {
        "firstt (K)": temp["firstt_K"],
        "finalt (K)": temp["finalt_K"],
        "tbath (K)": temp["tbath_K"],
        "ieqfrq": temp["ieqfrq"],
    }
    if temp.get("hoover_reft_K") is not None:
        temp_map["hoover reft (K)"] = temp["hoover_reft_K"]
    if int(temp.get("ihtfrq") or 0) > 0:
        temp_map["ihtfrq"] = temp["ihtfrq"]
        temp_map["TEMINC (K/step)"] = temp["TEMINC_K"]
        temp_map["~rescales"] = temp["ramp_rescale_events"]
    else:
        temp_map["ihtfrq"] = "0 (no velocity-scaling ramp)"
    sections.append(("Temperature / bath", temp_map))

    flags = info["dyna_flags"]
    sections.append(
        (
            "CHARMM dyna flags",
            {
                "start": flags["start"],
                "restart": flags["restart"],
                "new": flags["new"],
                "iasvel": flags["iasvel"],
                "iasors": f"{flags['iasors']} ({flags['iasors_meaning']})",
                "iunrea": flags["iunrea"],
            },
        )
    )

    sections.append(("Velocity / CPT init", {"policy": info["velocity_init"]}))

    cpt = info.get("cpt")
    if cpt is not None:
        sections.append(
            (
                "CPT barostat",
                {
                    "pmass": cpt["pmass"],
                    "tmass": cpt["tmass"],
                    "pgamma": cpt["pgamma"],
                    "pint": cpt["pint"],
                    "pref": cpt["pref"],
                    "leap": cpt["leap"],
                },
            )
        )

    io_info = info["restart_io"]
    sections.append(
        (
            "Restart I/O",
            {
                "read": io_info["restart_read"] or "(none / in-memory)",
                "write": io_info["restart_write"] or "(none)",
            },
        )
    )

    harmonized = info.get("freq_harmonized") or {}
    if harmonized:
        sections.append(
            (
                "Frequency harmonization",
                {k: f"{a} → {b}" for k, (a, b) in sorted(harmonized.items())},
            )
        )

    cli = info.get("cli")
    if cli is not None:
        sections.append(
            (
                "CLI heat options",
                {
                    "--heat-firstt": cli["heat_firstt"],
                    "--heat-finalt": cli["heat_finalt"],
                    "--heat-ihtfrq": cli["heat_ihtfrq"],
                    "no_echeck_heat": cli["no_echeck_heat"],
                    "heat_comp_damp": cli["heat_comp_damp"],
                    "n_heat_segments": cli["n_heat_segments"],
                },
            )
        )

    return sections


def print_heat_dynamics_diagnostics(
    kw: dict[str, Any],
    *,
    heat_thermostat: str,
    use_pbc: bool = True,
    args: Any | None = None,
    segment_index: int | None = None,
    n_segments: int | None = None,
    use_memory: bool = False,
    restart_read: PathLike | None = None,
    restart_write: PathLike | None = None,
    stage_ps: float | None = None,
    timestep_ps: float | None = None,
    freq_harmonized: dict[str, tuple[int, int]] | None = None,
    quiet: bool = False,
) -> None:
    """Print a detail-rich HEAT setup summary immediately before ``dyna``."""
    if quiet:
        return
    info = describe_heat_dynamics_setup(
        kw,
        heat_thermostat=heat_thermostat,
        use_pbc=use_pbc,
        args=args,
        segment_index=segment_index,
        n_segments=n_segments,
        use_memory=use_memory,
        restart_read=restart_read,
        restart_write=restart_write,
        stage_ps=stage_ps,
        timestep_ps=timestep_ps,
        freq_harmonized=freq_harmonized,
    )
    from mmml.utils.rich_report import emit_dashboard

    title = "HEAT dynamics diagnostics"
    if info.get("segment"):
        title += f" (segment {info['segment']})"
    emit_dashboard(
        title,
        build_heat_dynamics_dashboard_sections(info),
        border_style="cyan",
        quiet=quiet,
    )


def build_heat_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 10.0,
    save_interval_ps: float = 0.1,
    temp: float = 300.0,
    firstt: float | None = None,
    finalt: float | None = None,
    echeck: float = 100.0,
    use_pbc: bool = True,
    ihtfrq: int = 50,
) -> dict[str, Any]:
    """NVT heating dict for ``DynamicsScript`` (CHARMM + MLpot)."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    heat_finalt = float(finalt if finalt is not None else temp)
    heat_firstt = float(firstt if firstt is not None else heat_finalt * 0.2)
    freq_kwargs = {} if use_pbc else _non_pbc_dyn_freq_kwargs()
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=50,
        echeck=echeck,
        **freq_kwargs,
    )
    if not use_pbc:
        _strip_crystal_dyn_keywords(kw)
    kw.update(
        {
            "verlet": True,
            "new": True,
            "start": True,
            "ieqfrq": 0,
            "iasors": 1,
            "iasvel": 1,
            "iscvel": 0,
            "ichecw": 0,
            "firstt": heat_firstt,
            "finalt": heat_finalt,
            "tbath": heat_finalt,
        }
    )
    apply_heat_ramp_frequencies(kw, nstep=nstep, ihtfrq=ihtfrq)
    if use_pbc:
        _ensure_ixtfrq_above_nstep(kw, nstep)
    return kw


def build_hoover_heat_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 10.0,
    save_interval_ps: float = 0.1,
    temp: float = 300.0,
    firstt: float | None = None,
    finalt: float | None = None,
    echeck: float = 100.0,
    use_pbc: bool = True,
    tmass: int | None = None,
    pgamma: float = 5.0,
    ihtfrq: int = 100,
) -> dict[str, Any]:
    """NVT heating with CHARMM Hoover (PBC) or velocity-scaling ramp (vacuum).

    With PBC + CRYSTal, Hoover constant-T uses CPT at constant volume
    (``pmass=0``, ``pint pconst``, ``hoover reft``).

    Vacuum/free-space runs cannot use CPT (CHARMM: "CRYStal must be used for
    constant pressure simulations").  For ``use_pbc=False`` this falls back to
    :func:`build_heat_dynamics` (``iasors=0`` ``ihtfrq`` scaling), which is the
    supported all-ML heat path in ``COMP_AND_HEATING.md``.

    On the first heat segment, Boltzmann assignment and CPT barostat init must
    happen in one ``dyna`` call (``start=True``, ``iasvel=1``); see
    :func:`staged_workflow._configure_heat_dynamics_start`.
    """
    heat_finalt = float(finalt if finalt is not None else temp)
    heat_firstt = float(firstt if firstt is not None else heat_finalt * 0.2)
    if not use_pbc:
        nstep = ps_to_nsteps(timestep_ps, duration_ps)
        kw = build_heat_dynamics(
            timestep_ps=timestep_ps,
            duration_ps=duration_ps,
            save_interval_ps=save_interval_ps,
            temp=temp,
            firstt=heat_firstt,
            finalt=heat_finalt,
            echeck=echeck,
            use_pbc=False,
            ihtfrq=ihtfrq,
        )
        kw["iasors"] = 0
        apply_heat_ramp_frequencies(kw, nstep=nstep, ihtfrq=ihtfrq)
        return kw
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=100,
        echeck=echeck,
    )
    kw.update(
        {
            "new": True,
            "start": True,
            "firstt": heat_firstt,
            "finalt": heat_finalt,
            "tbath": heat_finalt,
            "verlet": True,
        }
    )
    if tmass is None:
        _, tmass = compute_cpt_piston_masses()
        tmass = max(400, min(int(tmass), 1200))
    else:
        tmass = max(1, int(tmass))
    _apply_npt_cpt_kwargs(
        kw,
        temp=heat_finalt,
        thermostat="hoover",
        pref=1.0,
        pmass=0,
        tmass=tmass,
        pgamma=0.0,
        firstt=heat_firstt,
        hoover_reft=heat_firstt,
    )
    # Constant-volume Hoover NVT (pmass=0): skip crystal resize updates. They are
    # unnecessary for fixed-box heating and can segfault in libcharmm after MLpot
    # PBC image/NB rebuilds on MPI-linked builds (aco/dcm clusters @ ~6k steps).
    _ensure_ixtfrq_above_nstep(kw, nstep)
    return kw


def boltzmann_velocity_kwargs(temp: float = 300.0) -> dict[str, Any]:
    """Dynamics kwargs for Maxwell-Boltzmann velocity assignment (``iasvel`` / ``firstt``).

    CHARMM c47 does not accept a standalone ``velocity`` script command (it is parsed as
    ``VELO`` and fails). Velocities are assigned when ``dyna`` runs with ``iasvel`` 1.
    """
    t = float(temp)
    return {
        "iasors": 1,
        "iasvel": 1,
        "iscale": 0,
        "iscvel": 0,
        "ichecw": 0,
        "firstt": t,
        "finalt": t,
        "tbath": t,
        "tstruct": t,
    }


def assign_boltzmann_velocities(temp: float = 300.0) -> dict[str, Any]:
    """Return ``dyna`` kwargs for velocity init (alias of :func:`boltzmann_velocity_kwargs`)."""
    return boltzmann_velocity_kwargs(temp)


def assign_velocities_at_temperature(
    firstt: float,
    *,
    timestep_ps: float = 0.00025,
    restart_path: PathLike | None = None,
    use_pbc: bool = True,
    read_unit: int = 90,
) -> None:
    """Removed: Boltzmann assignment must happen in the main ``dyna`` call."""
    raise RuntimeError(
        "assign_velocities_at_temperature (nstep=0 Boltzmann assign) was removed; "
        "use start=True, iasvel=1, nstep>=1 on the main dynamics call instead "
        f"(firstt={float(firstt):.4g} K, restart_path={restart_path!r}, "
        f"timestep_ps={float(timestep_ps)}, use_pbc={use_pbc}, read_unit={read_unit})"
    )


def build_nve_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 50.0,
    save_interval_ps: float = 0.01,
    restart: bool = True,
    temp: float = 300.0,
    nprint: int = 100,
    iprfrq: int = 500,
    isvfrq: int = 500,
    echeck: float = 100.0,
    use_pbc: bool = True,
    dyn_inbfrq: int | None = None,
) -> dict[str, Any]:
    """NVE production-style dict (restart from heat)."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    if use_pbc:
        freq_kwargs: dict[str, int] = {}
    else:
        freq_kwargs = _non_pbc_dyn_freq_kwargs(
            inbfrq=50 if dyn_inbfrq is None else int(dyn_inbfrq)
        )
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=max(1, nprint),
        iprfrq=max(1, iprfrq),
        isvfrq=max(1, isvfrq),
        ntrfrq=100,
        echeck=echeck,
        **freq_kwargs,
    )
    if not use_pbc:
        _strip_crystal_dyn_keywords(kw)
    kw.update(
        {
            "leap": True,
            "verlet": True,
            "new": False,
            "start": False,
            "restart": restart,
            "ihtfrq": 0,
            "ieqfrq": 0,
        }
    )
    if restart:
        kw["iasvel"] = 0
    else:
        kw.update(boltzmann_velocity_kwargs(temp))
    return kw


def compute_cpt_piston_masses() -> tuple[int, int]:
    """Return ``(pmass, tmass)`` from total PSF mass (CHARMM CPT recipe).

    ``pmass = int(sum(mass) / 50)``, ``tmass = pmass * 10``.
    """
    import pycharmm.select as select

    pmass = int(np.sum(select.get_property("mass")) / 50.0)
    return pmass, int(pmass * 10)


def _apply_npt_cpt_kwargs(
    kw: dict[str, Any],
    *,
    temp: float,
    thermostat: NptThermostat = "hoover",
    pref: float = 1.0,
    pmass: int | None = None,
    tmass: int | None = None,
    pgamma: float = 5,
    firstt: float | None = None,
    hoover_reft: float | None = None,
    tcoupling: float = 5.0,
    pressure_tensor: Any | None = None,
) -> None:
    """Attach CPT barostat + temperature control keywords to a dynamics dict."""
    from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
        apply_npt_pressure_reference,
    )

    if pmass is None or tmass is None:
        pmass, tmass = compute_cpt_piston_masses()
    kw.update(
        {
            "leap": True,
            "cpt": True,
            "pgamma": pgamma,
            "pmass": pmass,
            "ihtfrq": 0,
            "ieqfrq": 0,
            "ichecw": 0,
        }
    )
    apply_npt_pressure_reference(kw, pref=pref, pressure_tensor=pressure_tensor)
    if thermostat == "hoover":
        kw["hoover reft"] = float(hoover_reft if hoover_reft is not None else temp)
        kw["tmass"] = tmass
        if firstt is not None:
            kw["firstt"] = firstt
    elif thermostat == "berendsen":
        kw["tcons"] = True
        kw["tcoupling"] = tcoupling
        kw["treference"] = temp
    else:
        raise ValueError(f"unknown NPT thermostat: {thermostat!r}")


def _apply_hoover_nvt_kwargs(
    kw: dict[str, Any],
    *,
    temp: float,
    tmass: int | None = None,
    firstt: float | None = None,
) -> None:
    """Hoover NVT for vacuum/free-space (no ``cpt`` / crystal required)."""
    if tmass is None:
        _, tmass = compute_cpt_piston_masses()
    kw.update(
        {
            "leap": True,
            "ihtfrq": 0,
            "ieqfrq": 0,
            "hoover reft": temp,
            "tmass": tmass,
            "imgfrq": 0,
            "ihbfrq": 0,
            "ilbfrq": 0,
        }
    )
    if firstt is not None:
        kw["firstt"] = firstt


def build_nvt_equilibration_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 50.0,
    save_interval_ps: float = 0.01,
    temp: float = 300.0,
    restart: bool = True,
    echeck: float = 500.0,
    tmass: int | None = None,
    include_firstt: bool = True,
) -> dict[str, Any]:
    """NVT equilibration for vacuum/free-space clusters.

    Free-space CHARMM runs are kept on the same velocity-scaling path as the
    heating stage.  Avoid Hoover/CPT-style controls here because there is no
    meaningful periodic volume or pressure for a vacuum cluster.
    """
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    firstt = temp if restart else temp * 0.2
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=100,
        echeck=echeck,
        imgfrq=0,
        ihbfrq=0,
        ilbfrq=0,
    )
    kw.update(
        {
            "leap": True,
            "verlet": True,
            "new": False,
            "start": False,
            "restart": restart,
            "ieqfrq": 0,
            "iasors": 1,
            "iasvel": 0 if restart else 1,
            "iscvel": 0,
            "ichecw": 0,
            "finalt": temp,
            "tbath": temp,
        }
    )
    if include_firstt and not restart:
        kw["firstt"] = firstt
        apply_heat_ramp_frequencies(kw, nstep=nstep, ihtfrq=50)
    else:
        kw["ihtfrq"] = 0
        kw["TEMINC"] = 0.0
    return kw


def build_nvt_production_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 100.0,
    save_interval_ps: float = 0.01,
    temp: float = 300.0,
    restart: bool = True,
    echeck: float = 500.0,
    tmass: int | None = None,
) -> dict[str, Any]:
    """NVT production (Hoover) for vacuum/free-space clusters."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=100,
        echeck=echeck,
        imgfrq=0,
        ihbfrq=0,
        ilbfrq=0,
    )
    kw.update(
        {
            "new": False,
            "start": False,
            "restart": restart,
        }
    )
    _apply_hoover_nvt_kwargs(kw, temp=temp, tmass=tmass, firstt=None)
    return kw


def build_cpt_equilibration_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 50.0,
    save_interval_ps: float = 0.01,
    temp: float = 300.0,
    restart: bool = True,
    echeck: float = 500.0,
    thermostat: NptThermostat = "hoover",
    pref: float = 1.0,
    pmass: int | None = None,
    tmass: int | None = None,
    pgamma: float = 5,
    include_firstt: bool = True,
    pressure_tensor: Any | None = None,
) -> dict[str, Any]:
    """NPT equilibration (CPT + Hoover by default); matches example mini-MD scripts."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=100,
        echeck=echeck,
    )
    kw.update(
        {
            "new": False,
            "start": False,
            "restart": restart,
        }
    )
    _apply_npt_cpt_kwargs(
        kw,
        temp=temp,
        thermostat=thermostat,
        pref=pref,
        pmass=pmass,
        tmass=tmass,
        pgamma=pgamma,
        firstt=temp if include_firstt else None,
        pressure_tensor=pressure_tensor,
    )
    return kw


def build_cpt_production_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 100.0,
    save_interval_ps: float = 0.01,
    temp: float = 300.0,
    restart: bool = True,
    echeck: float = 500.0,
    thermostat: NptThermostat = "hoover",
    pref: float = 1.0,
    pmass: int | None = None,
    tmass: int | None = None,
    pgamma: float = 5,
    pressure_tensor: Any | None = None,
) -> dict[str, Any]:
    """NPT production (CPT + Hoover by default; same barostat recipe as equilibration)."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=100,
        echeck=echeck,
    )
    kw.update(
        {
            "new": False,
            "start": False,
            "restart": restart,
        }
    )
    _apply_npt_cpt_kwargs(
        kw,
        temp=temp,
        thermostat=thermostat,
        pref=pref,
        pmass=pmass,
        tmass=tmass,
        pgamma=pgamma,
        pressure_tensor=pressure_tensor,
    )
    return kw


def npt_restart_chain(
    data_dir: PathLike,
    *,
    n_segments: int,
    prefix: str,
    initial_restart: PathLike | None = None,
) -> list[CharmmTrajectoryFiles]:
    """Build chained restart/trajectory I/O for repeated NPT segments.

    Segment 0 reads ``initial_restart`` (if given). Segment ``i>0`` reads
    ``{prefix}.{i-1}.res`` and writes ``{prefix}.{i}.res`` / ``.dcd``.
    """
    data_dir = Path(data_dir)
    chain: list[CharmmTrajectoryFiles] = []
    for ii in range(n_segments):
        if ii == 0:
            rread = Path(initial_restart) if initial_restart is not None else None
        else:
            rread = data_dir / f"{prefix}.{ii - 1}.res"
        chain.append(
            CharmmTrajectoryFiles(
                restart_read=rread,
                restart_write=data_dir / f"{prefix}.{ii}.res",
                trajectory=data_dir / f"{prefix}.{ii}.dcd",
            )
        )
    return chain


def final_npt_segment_restart(data_dir: PathLike, prefix: str, n_segments: int) -> Path:
    """Path to the last restart in a multi-segment NPT chain."""
    data_dir = Path(data_dir)
    if n_segments > 1:
        return data_dir / f"{prefix}.{n_segments - 1}.res"
    return data_dir / f"{prefix}.res"


def _release_charmm_dynamics_api_buffers() -> None:
    """Free CHARMM dynamics dataframe buffers left allocated by ``dynamc``.

    BLOCK-enabled builds call ``lambdata_init`` inside ``dynamc`` even when
    PyCHARMM does not request ``lambdata=True``.  A second ``dyna`` (e.g.
    ``nstep=0`` Boltzmann assign followed by HEAT) then hits ``dataframe_init``
    ``chmrealloc`` and can segfault.  Mirror PyCHARMM's post-run cleanup.
    """
    try:
        import pycharmm.lib as lib
    except ImportError:
        return
    for name in ("lambdata_del", "ktable_del", "velos_del", "msldata_del"):
        fn = getattr(lib.charmm, name, None)
        if fn is not None:
            fn()


def _strip_non_charmm_dynamics_keywords(kw: dict[str, Any]) -> None:
    """Remove MMML-only metadata before ``DynamicsScript`` (not CHARMM ``dyna`` keys)."""
    for key in list(kw):
        if key.startswith("_") or key == "dcd_interval_ps":
            kw.pop(key, None)


def apply_charmm_dynamics_echeck_kw(kw: dict[str, Any], echeck: float) -> None:
    """Apply ECHECK to dynamics kwargs and CHARMM global state.

    Pretreat/minimize legs can leave a positive ECHECK in Fortran; dynamics must
    reset it explicitly or short NPT legs abort at iprfrq cadence (e.g. step 240).
    """
    val = float(echeck)
    kw["echeck"] = val
    kw["ichecw"] = 0
    try:
        import pycharmm.dynamics as charm_dyn

        charm_dyn.set_echeck(val)
    except (ImportError, OSError):
        pass


def _normalize_dynamics_heat_ramp_kw(kw: dict[str, Any]) -> None:
    """Map scale-heat ramp keys to PyCHARMM script names and Fortran setters."""
    teminc = kw.pop("TEMINC", None)
    if teminc is not None and "teminc" not in kw:
        kw["teminc"] = float(teminc)
    iht = kw.get("ihtfrq")
    if iht is not None and int(iht) > 0:
        kw["ihtfrq"] = int(iht)
    try:
        import pycharmm.dynamics as charm_dyn

        if "firstt" in kw:
            charm_dyn.set_firstt(float(kw["firstt"]))
        if "finalt" in kw:
            charm_dyn.set_finalt(float(kw["finalt"]))
        if "teminc" in kw:
            charm_dyn.set_teminc(float(kw["teminc"]))
    except (ImportError, OSError):
        pass


def _merge_dynamics_script_append(base_script: str, append: str) -> str:
    """Join ``append`` onto the ``dynamics`` line (``CommandScript`` adds trailing ``\\n``)."""
    extra = (append or "").strip()
    if not extra:
        return base_script
    body = base_script.rstrip("\n").strip()
    return f"{body} {extra}\n"


def _execute_dynamics_script(dyn: Any, *, append: str = "") -> None:
    """Run ``DynamicsScript`` with optional keywords on the same ``dynamics`` line."""
    import pycharmm.lingo as lingo

    script = _merge_dynamics_script_append(dyn.create_script_string(), append)
    lingo.charmm_script(script)


def _dynamics_c_api_available() -> bool:
    """True when dynamics can run without the ``dynamics`` script command."""
    try:
        import pycharmm.dynamics as charm_dyn

        return bool(charm_dyn.dynamics_run_kw_available())
    except (ImportError, OSError):
        return False


def _run_dynamics_via_c_api(kw: dict[str, Any], *, heat_append: str = "") -> Any:
    """Run ``dynopt`` with a keyword line (KEY_LIBRARY; leap/hoover/cpt flags)."""
    import pycharmm
    import pycharmm.dynamics as charm_dyn

    dyn = pycharmm.DynamicsScript(**kw)
    script = _merge_dynamics_script_append(dyn.create_script_string(), heat_append)
    command_line = charm_dyn.flatten_dynamics_script(script)
    charm_dyn.run_with_command_line(command_line, **kw)
    return dyn


def _dynamics_script_append_for_heat_ramp(kw: dict[str, Any]) -> str:
    """Extra ``dyna`` keywords when scale-heat ramp must reach Fortran ``reawri``."""
    parts: list[str] = []
    iht = kw.get("ihtfrq")
    if iht is not None and int(iht) > 0:
        parts.append(f"ihtfrq {int(iht)}")
    for key in ("TEMINC", "teminc"):
        if key in kw:
            parts.append(f"teminc {float(kw[key])}")
            break
    return (" " + " ".join(parts)) if parts else ""


def run_dynamics(dynamics_kwargs: dict[str, Any]) -> Any:
    """Instantiate and run ``pycharmm.DynamicsScript``."""
    kw = dict(dynamics_kwargs)
    _strip_non_charmm_dynamics_keywords(kw)
    nstep = int(kw.get("nstep", 0) or 0)
    if nstep < 1:
        raise ValueError(
            f"run_dynamics: nstep must be >= 1 (got {nstep}); "
            "use start=True, iasvel=1 on the main dyna call for velocity assignment"
        )

    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        clear_comparison_coordinates,
    )

    import pycharmm
    # PyCHARMM omits ``start`` from the script when start=False, so CHARMM may keep
    # START active after a prior Boltzmann assign. With iasvel=0 that reads COMP
    # coordinates as velocities — zero COMP defensively.
    if not kw.get("start") and int(kw.get("iasvel", 1)) == 0:
        clear_comparison_coordinates()
    if "echeck" in kw:
        apply_charmm_dynamics_echeck_kw(kw, float(kw["echeck"]))
    if int(kw.get("ihtfrq", 0) or 0) > 0:
        apply_heat_ramp_frequencies(kw, nstep=nstep, ihtfrq=int(kw["ihtfrq"]))
    _normalize_dynamics_heat_ramp_kw(kw)
    _apply_dynamics_io_setters(kw)
    _prepare_dynamics_list_frequencies(kw, nstep=nstep)
    heat_append = _dynamics_script_append_for_heat_ramp(kw)
    _release_charmm_dynamics_api_buffers()
    if _dynamics_c_api_available():
        dyn = _run_dynamics_via_c_api(kw, heat_append=heat_append)
    else:
        dyn = pycharmm.DynamicsScript(**kw)
        _execute_dynamics_script(dyn, append=heat_append)
    _release_charmm_dynamics_api_buffers()
    return dyn


def _valid_restart_file(path: PathLike | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        return None

    # CHARMM can write coordinate-history files during failed dynamics and label
    # them as restart output, but those files explicitly cannot restart a run.
    try:
        head = p.read_bytes()[:8192]
    except OSError:
        return None
    first_line = head.splitlines()[0].decode("ascii", errors="ignore").split()
    if (
        len(first_line) >= 3
        and first_line[0].upper() == "REST"
        and first_line[2].lstrip("+-").isdigit()
        and int(first_line[2]) < 0
    ):
        print(
            f"overlap: ignoring non-restartable CHARMM scratch restart {p}",
            flush=True,
        )
        return None
    if b"C A N N O T" in head and b"RESTART A RUN" in head:
        print(
            f"overlap: ignoring non-restartable CHARMM scratch restart {p}",
            flush=True,
        )
        return None
    if p.suffix.lower() in (".crd", ".pdb", ".dcd", ".xyz"):
        return None
    text_start = head.decode("ascii", errors="ignore").lstrip()
    if not (
        text_start.upper().startswith("REST")
        or "!NATOM" in text_start
        or text_start.startswith("NATOM")
    ):
        return None
    return p


def _overlap_restart_slot_paths(final_restart: Path) -> tuple[Path, Path]:
    """Alternating scratch restarts so read and write are never the same file."""
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import overlap_restart_slot_paths

    return overlap_restart_slot_paths(final_restart)


def _is_overlap_scratch_restart(
    write_path: PathLike | None,
    final_restart: Path,
) -> bool:
    """True when ``write_path`` is an alternating overlap scratch, not the stage restart."""
    if write_path is None:
        return False
    p = Path(write_path)
    if p == final_restart:
        return False
    slot_a, slot_b = _overlap_restart_slot_paths(final_restart)
    return p in (slot_a, slot_b)


def _refresh_restart_write_after_chunk(
    write_path: PathLike | None,
    *,
    final_restart: Path | None,
) -> None:
    """Rewrite restart from in-memory CHARMM state after a dynamics chunk.

    CHARMM ``dyna`` often leaves coordinate-history ``.res`` files whose ``JHSTRT``
    field stays 0 even when the log shows ``WRIDYN: ... step 8000``.  Refreshing
    from memory fixes overlap step accounting and post-run validation.
    """
    if write_path is None:
        return
    if _valid_restart_file(write_path) is not None:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_from_current_state,
    )

    rewrite_dynamics_restart_from_current_state(write_path)


def _refresh_overlap_scratch_restart(
    write_path: PathLike | None,
    *,
    final_restart: Path,
) -> None:
    """Rewrite overlap scratch restart from in-memory CHARMM state.

    CHARMM dynamics can leave coordinate-history ``.res`` files that fail
    ``_valid_restart_file``; intra overlap rescue also updates coordinates without
    rewriting the previous chunk's scratch path.  Refresh after each chunk (and
    after overlap checks) so the next ``READYN`` handoff is valid.
    """
    if not _is_overlap_scratch_restart(write_path, final_restart):
        return
    _refresh_restart_write_after_chunk(write_path, final_restart=final_restart)


def _materialize_early_abort_restart_handoff(
    chunk_io: CharmmTrajectoryFiles,
    *,
    steps_before_chunk: int,
    overlap_context: str,
    chunk_kw: dict[str, Any] | None = None,
    mlpot_ctx: Optional["MlpotContext"] = None,
) -> CharmmTrajectoryFiles:
    """Write a valid READYN restart from in-memory state after a mid-chunk abort.

    CPT / Hoover overlap chunks segfault in ``readyn`` when dynamics is retried
    with ``iunrea=-1`` but CHARMM still expects barostat restart internals.
    Materialize scratch restart(s) and hand off via normal ``READYN``.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        patch_restart_global_step,
    )

    read_path = (
        Path(chunk_io.restart_read) if chunk_io.restart_read is not None else None
    )
    write_path = (
        Path(chunk_io.restart_write) if chunk_io.restart_write is not None else None
    )
    targets: list[Path] = []
    if write_path is not None:
        targets.append(write_path)
    if read_path is not None and read_path not in targets:
        targets.append(read_path)

    if not targets:
        raise RuntimeError(
            f"overlap ({overlap_context}): early-abort restart handoff has no "
            f"scratch restart path (read={read_path!r}, write={write_path!r})"
        )

    step = max(0, int(steps_before_chunk))
    for path in targets:
        if not rewrite_dynamics_restart_validated(path):
            raise RuntimeError(
                f"overlap ({overlap_context}): early-abort restart handoff failed "
                f"writing {path.name} from in-memory CHARMM state"
            )
        patch_restart_global_step(path, step)

    valid_read: Path | None = None
    for candidate in (write_path, read_path):
        if candidate is None:
            continue
        valid_read = _valid_overlap_chunk_restart_read(candidate)
        if valid_read is None:
            valid_read = _valid_restart_file(candidate)
        if valid_read is not None:
            break

    if valid_read is None:
        raise RuntimeError(
            f"overlap ({overlap_context}): early-abort restart handoff wrote "
            f"{', '.join(p.name for p in targets)} but none validate as READYN"
        )

    print(
        f"overlap ({overlap_context}): early-abort restart handoff from "
        f"{valid_read.name} at global step {step} (READYN)",
        flush=True,
    )
    return CharmmTrajectoryFiles(
        restart_read=valid_read,
        restart_write=chunk_io.restart_write,
        trajectory=chunk_io.trajectory,
        restart_read_unit=chunk_io.restart_read_unit,
        restart_write_unit=chunk_io.restart_write_unit,
        trajectory_unit=chunk_io.trajectory_unit,
    )


def _salvage_overlap_segment_progress(
    *,
    chunk_io: Optional[CharmmTrajectoryFiles],
    final_restart: Path | None,
    steps_before_chunk: int,
    overlap_context: str,
    mlpot_ctx: Optional["MlpotContext"],
    overlap_restart_read: PathLike | None = None,
    total_nstep: int | None = None,
) -> int | None:
    """Rewind to the last good overlap handoff instead of failing the stage."""
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        restore_geometry_from_ladder,
    )

    read_path = (
        Path(overlap_restart_read)
        if overlap_restart_read is not None
        else (
            Path(chunk_io.restart_read)
            if chunk_io is not None and chunk_io.restart_read is not None
            else None
        )
    )
    if read_path is None or _valid_restart_file(read_path) is None:
        return None
    try:
        restore_geometry_from_ladder(
            [read_path],
            label=f"overlap salvage ({overlap_context})",
            allow_in_memory=False,
        )
    except RuntimeError:
        return None

    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        patch_restart_global_step,
    )

    targets: list[Path] = []
    if final_restart is not None:
        targets.append(Path(final_restart))
    if chunk_io is not None and chunk_io.restart_write is not None:
        write = Path(chunk_io.restart_write)
        if write not in targets:
            targets.append(write)
    step = max(0, int(steps_before_chunk))
    wrote = False
    for path in targets:
        if rewrite_dynamics_restart_validated(path):
            patch_restart_global_step(path, step)
            wrote = True
    if not wrote:
        return None
    if mlpot_ctx is not None:
        _prepare_overlap_chunk_after_restart(mlpot_ctx, restart_read=read_path)
    total_txt = f"/{int(total_nstep)}" if total_nstep is not None else ""
    print(
        f"WARN: overlap ({overlap_context}): salvaged at global step {step}{total_txt} "
        f"from {read_path.name} (partial stage; remaining chunks skipped; equi/prod "
        "blocked unless --allow-incomplete-dynamics)",
        flush=True,
    )
    return step


def _restart_header_step_field(path: Path) -> str | None:
    """Return the third token on the ``REST`` line (step / history marker), if present."""
    try:
        first_line = path.read_text(errors="ignore").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if len(first_line) >= 3 and first_line[0].upper() == "REST":
        return first_line[2]
    return None


def _overlap_refresh_or_validate_scratch_restart(
    write_path: PathLike | None,
    *,
    final_restart: Path,
    chunk_index: int,
    n_chunks: int,
    overlap_context: str,
    mlpot_ctx: Optional["MlpotContext"],
    memory_handoff: bool = False,
) -> None:
    """Refresh scratch restart; validate only when the next chunk will ``READYN`` it."""
    if memory_handoff and mlpot_ctx is not None and n_chunks > 1:
        return
    _ensure_valid_overlap_scratch_restart(
        write_path,
        final_restart=final_restart,
        chunk_index=chunk_index,
        n_chunks=n_chunks,
        overlap_context=overlap_context,
    )


def _ensure_valid_overlap_scratch_restart(
    write_path: PathLike | None,
    *,
    final_restart: Path,
    chunk_index: int,
    n_chunks: int,
    overlap_context: str,
) -> None:
    """Refresh scratch restart from memory and fail fast if still not ``READYN``-able."""
    if not _is_overlap_scratch_restart(write_path, final_restart):
        return
    path = Path(write_path)
    if _valid_restart_file(path) is not None:
        return
    _refresh_overlap_scratch_restart(write_path, final_restart=final_restart)
    if _valid_restart_file(path) is not None:
        return
    rest_field = _restart_header_step_field(path)
    hint = (
        "CHARMM wrote a coordinate-history restart (REST third field -1) or dynamics "
        "was unstable. On MPI-linked CHARMM use in-memory overlap handoff (default "
        "under mpirun; or --dynamics-overlap-memory-handoff). For HEAT after mini use "
        "in-memory Boltzmann assignment; avoid dyna start without firstt on overlap "
        "chunk 0; try --heat-thermostat hoover and a single heat segment "
        "(--dynamics-overlap-check-interval >= heat nstep)."
    )
    raise RuntimeError(
        f"overlap ({overlap_context}): scratch restart {path.name} is not restartable "
        f"after chunk {chunk_index + 1}/{n_chunks} "
        f"(REST step field={rest_field!r}). {hint}"
    )


def _overlap_chunk_trajectory_path(trajectory: Path, chunk_index: int) -> Path:
    """Per-chunk DCD path for overlap-segmented dynamics."""
    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import overlap_chunk_trajectory_path

    return overlap_chunk_trajectory_path(trajectory, chunk_index)


def _overlap_chunk_restart_paths(
    io: CharmmTrajectoryFiles,
    *,
    chunk_index: int,
    n_chunks: int,
) -> tuple[Path | None, Path | None]:
    """``(restart_read, restart_write)`` for overlap chunk ``chunk_index``.

    Alternates ``.overlap_a/.overlap_b.res`` scratch files so CHARMM never reads
    and writes the same path in one step.  Chunk 0 always writes scratch (even
    without an external read) so chunk 1 can ``READYN`` without EOF.
    """
    if io.restart_write is None:
        return _valid_overlap_chunk_restart_read(io.restart_read), None
    final = Path(io.restart_write)
    if n_chunks <= 1:
        return _valid_overlap_chunk_restart_read(io.restart_read), final
    slot_a, slot_b = _overlap_restart_slot_paths(final)
    if chunk_index == 0:
        return _valid_overlap_chunk_restart_read(io.restart_read), slot_a
    if chunk_index == n_chunks - 1:
        prev = slot_a if (chunk_index % 2) == 1 else slot_b
        return _valid_overlap_chunk_restart_read(prev), final
    if chunk_index % 2 == 1:
        return _valid_overlap_chunk_restart_read(slot_a), slot_b
    return _valid_overlap_chunk_restart_read(slot_b), slot_a


def _overlap_chunk_uses_memory_handoff(
    mlpot_ctx: Optional["MlpotContext"],
    *,
    chunk_index: int,
    n_chunks: int,
    overlap: Optional["DynamicsOverlapConfig"] = None,
    cpt: bool = False,
) -> bool:
    """Continue overlap segments in-process when MLpot is active.

    Hoover CPT barostat piston state is **not** restored by plain
    ``write restart`` / ``READYN`` on scratch ``.overlap_*.res`` files.
    Default is in-memory handoff (same as CPT stability sub-chunks); set
    ``MMML_CPT_READYN_SUBCHUNK=1`` to opt into scratch READYN between overlap
    chunks (legacy / debugging only).
    """
    if mlpot_ctx is None or n_chunks <= 1:
        return False
    if cpt and not _cpt_subchunk_use_in_memory_handoff():
        return False
    if overlap is not None and overlap.memory_handoff:
        return True
    return False


def _valid_overlap_chunk_restart_read(path: PathLike | None) -> Path | None:
    """Valid restart for overlap chunk READYN; excludes handoff/pretreat seeds."""
    if path is None:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        is_handoff_seed_restart_path,
        is_pretreat_mm_restart_path,
    )

    p = Path(path)
    if is_handoff_seed_restart_path(p) or is_pretreat_mm_restart_path(p):
        return None
    return _valid_restart_file(p)


def overlap_first_chunk_skips_readyn(
    *,
    overlap: Optional["DynamicsOverlapConfig"],
    mlpot_ctx: Optional["MlpotContext"],
    nstep: int,
    nsavc: int | None = None,
    cpt: bool = False,
    restart_read: PathLike | None = None,
    memory_handoff_default: bool = False,
) -> bool:
    """True when overlap chunk 0 will run ``dyna`` without ``READYN``.

    Mirrors the chunk-0 decision in :func:`run_dynamics_with_io` so staged NPT can
    call :func:`_configure_npt_dynamics_start` before dynamics when stage-level
    ``use_memory`` is false but overlap uses in-process handoff on chunk 0.
    """
    if overlap is None or not (
        overlap.enabled or overlap.intra_enabled or overlap.extent_enabled
    ):
        return False
    total_nstep = max(0, int(nstep))
    if total_nstep <= 0:
        return False

    requested_interval = max(1, int(overlap.check_interval))
    interval = effective_overlap_check_interval(
        total_nstep,
        requested_interval,
        nsavc=nsavc,
        cpt=cpt,
    )
    n_chunks = total_nstep // interval
    if n_chunks <= 1:
        return False

    memory_handoff = bool(overlap.memory_handoff)
    if mlpot_ctx is not None and not memory_handoff and memory_handoff_default:
        from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import _truthy_env

        if not _truthy_env("MMML_NO_OVERLAP_MEMORY_HANDOFF"):
            memory_handoff = True

    if memory_handoff and mlpot_ctx is not None:
        return True

    if _valid_overlap_chunk_restart_read(restart_read) is None:
        return True

    return False


def _overlap_chunk_io(
    io: CharmmTrajectoryFiles,
    *,
    chunk_index: int,
    n_chunks: int,
    split_trajectory: bool = True,
    mlpot_ctx: Optional["MlpotContext"] = None,
    use_memory_handoff: bool = False,
) -> CharmmTrajectoryFiles:
    """Restart I/O for overlap chunking via alternating scratch restarts.

    When ``split_trajectory`` is true and ``n_chunks > 1``, each chunk writes its
    own ``*.chunk.NNNN.dcd`` (kept on disk; no stage-level merge).
    """
    rread, rwri = _overlap_chunk_restart_paths(
        io, chunk_index=chunk_index, n_chunks=n_chunks
    )
    if use_memory_handoff:
        rread = None
    traj: Path | None = None
    if io.trajectory is not None:
        if n_chunks <= 1:
            traj = io.trajectory
        elif split_trajectory:
            traj = _overlap_chunk_trajectory_path(Path(io.trajectory), chunk_index)
    return CharmmTrajectoryFiles(
        restart_read=rread,
        restart_write=rwri,
        trajectory=traj,
        restart_read_unit=io.restart_read_unit,
        restart_write_unit=io.restart_write_unit,
        trajectory_unit=io.trajectory_unit,
    )


def _drop_trajectory_io(io: Optional[CharmmTrajectoryFiles]) -> Optional[CharmmTrajectoryFiles]:
    """Return an equivalent I/O descriptor without opening a DCD trajectory."""
    if io is None or io.trajectory is None:
        return io
    return CharmmTrajectoryFiles(
        restart_read=io.restart_read,
        restart_write=io.restart_write,
        trajectory=None,
        pressure_tensor_log=io.pressure_tensor_log,
        restart_read_unit=io.restart_read_unit,
        restart_write_unit=io.restart_write_unit,
        trajectory_unit=io.trajectory_unit,
        pressure_tensor_log_unit=io.pressure_tensor_log_unit,
    )


# Per-chunk DCDs at nsavc=1 and overlap interval 2 create O(nstep) tiny files.
_OVERLAP_MAX_CHUNK_DCD_FILES = 64


def _overlap_should_split_trajectory(
    *,
    n_chunks: int,
    traj_nsavc: int | None,
) -> bool:
    """Write per-chunk ``*.chunk.NNNN.dcd`` files for multi-segment overlap runs."""
    if n_chunks <= 1:
        return False
    if traj_nsavc is not None and int(traj_nsavc) <= 1:
        if n_chunks > _OVERLAP_MAX_CHUNK_DCD_FILES:
            return False
        if n_chunks > 8:
            return False
    return True


def effective_overlap_check_interval(
    total_nstep: int,
    requested_interval: int,
    *,
    nsavc: int | None = None,
    cpt: bool = False,
) -> int:
    """Largest chunk size ≤ ``requested_interval`` that divides ``total_nstep`` evenly.

    Avoids a short final overlap chunk (e.g. 40 steps when 1640 total and
    requested 50), which triggers CHARMM FINCYC frequency retuning and PBC
    instability after scratch restart reads.

    ``nsavc`` is intentionally not allowed to inflate the geometry-check cadence.
    Chunks with ``nsavc >= nstep`` suppress trajectory I/O and keep a large
    harmless CHARMM ``NSAVC`` internally, so frequent geometry checks do not imply
    per-step DCD bookkeeping.
    """
    n = max(1, int(total_nstep))
    req = max(1, int(requested_interval))
    if n % req == 0:
        return req
    for d in range(min(req, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def _harmonize_dynamics_frequency(value: int, chunk_nstep: int) -> int:
    """Pick a CHARMM update frequency that divides ``chunk_nstep`` (FINCYC compatibility)."""
    n = max(1, int(chunk_nstep))
    val = int(value)
    if val <= 0:
        return val
    if val > n:
        return n
    if n % val == 0:
        return val
    for d in range(min(val, n), 0, -1):
        if n % d == 0:
            return d
    return n


def _harmonize_nsavc_frequency(value: int, chunk_nstep: int) -> int:
    """Backward-compatible alias; see :func:`harmonize_nsavc_frequency` in validation."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        harmonize_nsavc_frequency,
    )

    return harmonize_nsavc_frequency(value, chunk_nstep)


def _ensure_nsavc_below_nstep(kw: dict[str, Any]) -> None:
    """Clamp ``nsavc`` so CHARMM dynamics has ``nsavc < nstep``."""
    if "nsavc" not in kw or "nstep" not in kw:
        return
    nstep = int(kw["nstep"])
    if nstep <= 1:
        return
    old = int(kw["nsavc"])
    new = _harmonize_nsavc_frequency(old, nstep)
    if new != old:
        print(
            f"DCD nsavc {old} -> {new} (must be < nstep={nstep})",
            flush=True,
        )
    kw["nsavc"] = new


def _align_inbfrq_with_imgfrq(chunk_kw: dict[str, Any]) -> None:
    """CHARMM FINCYC: ``IMGFRQ`` must be a multiple of ``INBFRQ`` when ``INBFRQ > 0``."""
    if "inbfrq" not in chunk_kw or "imgfrq" not in chunk_kw:
        return
    inb = int(chunk_kw["inbfrq"])
    img = int(chunk_kw["imgfrq"])
    if inb <= 0:
        return
    if img <= 0:
        chunk_kw["imgfrq"] = inb
        if "ihbfrq" in chunk_kw:
            chunk_kw["ihbfrq"] = inb
        if "ilbfrq" in chunk_kw:
            chunk_kw["ilbfrq"] = inb
        return
    if img % inb == 0:
        return
    for d in range(min(inb, img), 0, -1):
        if img % d == 0:
            chunk_kw["inbfrq"] = d
            return


def _prepare_dynamics_list_frequencies(kw: dict[str, Any], *, nstep: int) -> None:
    """Push aligned ``inbfrq`` / ``imgfrq`` to Fortran before ``dynopt`` (C API path).

    ``dynamics.set_inbfrq`` updates ``contrl`` but not image-list cadence; a leftover
    ``imgfrq=-1`` with ``inbfrq>0`` triggers BOMLev -2 (IMGFRQ not a multiple of INBFRQ).
    """
    _align_inbfrq_with_imgfrq(kw)
    inb = int(kw.get("inbfrq", -1))
    if inb == 0:
        kw["imgfrq"] = 0
        kw["ihbfrq"] = 0
        kw["ilbfrq"] = 0
    try:
        import pycharmm.nbonds as nbonds
    except (ImportError, OSError):
        return
    if inb == 0:
        nbonds.set_inbfrq(0)
        nbonds.set_imgfrq(0)
        return
    if inb > 0:
        inb_eff = _harmonize_dynamics_frequency(inb, max(1, int(nstep)))
        if inb_eff != inb:
            kw["inbfrq"] = inb_eff
            inb = inb_eff
        nbonds.set_inbfrq(inb)
        img = int(kw.get("imgfrq", inb))
        if img <= 0 or img % inb != 0:
            img = inb
        kw["imgfrq"] = img
        kw["ihbfrq"] = img
        kw["ilbfrq"] = img
        nbonds.set_imgfrq(img)


_OVERLAP_CHUNK_FREQ_KEYS = (
    "inbfrq",
    "ihbfrq",
    "ilbfrq",
    "imgfrq",
    "iprfrq",
    "nprint",
    "isvfrq",
    "nsavc",
)


def _harmonize_overlap_chunk_frequencies(
    chunk_kw: dict[str, Any],
    chunk_nstep: int,
    *,
    loose_pbc: bool = False,
    global_step_start: int = 0,
    split_trajectory: bool = False,
) -> None:
    """Align list/image/HB update freqs with this chunk's ``nstep`` (avoids FINCYC retune)."""

    n = max(1, int(chunk_nstep))
    for key in _OVERLAP_CHUNK_FREQ_KEYS:
        if key not in chunk_kw:
            continue
        if key == "nsavc":
            old = int(chunk_kw[key])
            if old >= n:
                # Suppress trajectory I/O but keep an explicit large NSAVC.
                # If NSAVC is omitted, CHARMM defaults to NSAVC=1 even with
                # IUNCRD=-1, which creates expensive per-step trajectory
                # bookkeeping.  n-1 is accepted by CHARMM and avoids the slow
                # default while producing no DCD without an IUNCRD handle.
                chunk_kw["nsavc"] = max(1, n - 1)
                if split_trajectory:
                    # Per-chunk ``*.chunk.NNNN.dcd``: one frame at chunk end.
                    pass
                else:
                    chunk_kw["_suppress_trajectory"] = True
                    for k in ("nprint", "iprfrq", "isvfrq"):
                        chunk_kw.pop(k, None)
                    _emit_overlap_log(
                        f"chunk: skip DCD (target nsavc={old} >= nstep={n}; "
                        f"CHARMM nsavc={chunk_kw['nsavc']}; "
                        f"global step {int(global_step_start)}–{int(global_step_start) + n})",
                    )
                if split_trajectory:
                    _emit_overlap_log(
                        f"chunk: per-chunk DCD nsavc={chunk_kw['nsavc']} "
                        f"(global step {int(global_step_start)}–{int(global_step_start) + n})",
                    )
            else:
                for k in ("nprint", "iprfrq", "isvfrq"):
                    if k in chunk_kw:
                        chunk_kw[k] = old
        else:
            chunk_kw[key] = _harmonize_dynamics_frequency(int(chunk_kw[key]), n)
    _align_inbfrq_with_imgfrq(chunk_kw)
    if loose_pbc:
        apply_loose_pbc_dyn_freq_kwargs(chunk_kw, nstep=n)
    else:
        _ensure_ntrfrq_above_nstep(chunk_kw, n)
        _maybe_disable_ixtfrq_for_fixed_volume_chunk(chunk_kw, n)


def _mlpot_ctx_cubic_box_side_A(mlpot_ctx: Optional["MlpotContext"]) -> float | None:
    if mlpot_ctx is None:
        return None
    for attr in ("charmm_cubic_box_side_A", "cubic_box_side_A"):
        side = getattr(mlpot_ctx, attr, None)
        if side is not None and float(side) > 0.0:
            return float(side)
    return None


def _post_rescue_bath_target_K(chunk_kw: dict[str, Any]) -> float:
    """Bath temperature for post-rescue Boltzmann draw and CPT thermostat keywords."""
    if "hoover reft" in chunk_kw:
        return float(chunk_kw["hoover reft"])
    if "treference" in chunk_kw:
        return float(chunk_kw["treference"])
    return float(
        chunk_kw.get("tbath", chunk_kw.get("finalt", chunk_kw.get("firstt", 300.0)))
    )


def _prepare_post_rescue_bath_and_crystal(
    chunk_kw: dict[str, Any],
    *,
    mlpot_ctx: Optional["MlpotContext"],
) -> None:
    """Set bath keywords and CRYSTal before post-rescue dynamics (no ``nstep=0`` assign)."""
    bath = _post_rescue_bath_target_K(chunk_kw)
    if "hoover reft" in chunk_kw:
        chunk_kw["hoover reft"] = bath
    chunk_kw["firstt"] = bath
    chunk_kw["tbath"] = bath
    use_pbc = bool(chunk_kw.get("cpt")) or (
        mlpot_ctx is not None and bool(getattr(mlpot_ctx, "use_pbc", False))
    )
    if use_pbc and bool(chunk_kw.get("cpt")):
        side = _mlpot_ctx_cubic_box_side_A(mlpot_ctx)
        if side is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                ensure_charmm_crystal_for_cpt,
            )

            ensure_charmm_crystal_for_cpt(side, quiet=True)


def _assign_post_rescue_velocities_and_crystal(
    chunk_kw: dict[str, Any],
    *,
    mlpot_ctx: Optional["MlpotContext"],
) -> None:
    """Alias kept for overlap tests; bath/crystal only — velocities via ``start=True``."""
    _prepare_post_rescue_bath_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)


def _prepare_post_rescue_overlap_handoff(
    chunk_kw: dict[str, Any],
    *,
    mlpot_ctx: Optional["MlpotContext"],
) -> None:
    """Continue overlap dynamics in-process after geometry rescue (no ``READYN``).

    Required for Hoover CPT: static ``write restart`` lacks barostat piston
    internals, so ``READYN`` on scratch ``.overlap_*.res`` files fails with EOF.
    """
    if mlpot_ctx is not None and getattr(mlpot_ctx, "_overlap_post_rescue_cold_start", False):
        _prepare_post_rescue_cold_start_overlap_handoff(chunk_kw, mlpot_ctx=mlpot_ctx)
        return
    _prepare_post_rescue_bath_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
    chunk_kw["restart"] = False
    chunk_kw["new"] = False
    chunk_kw["start"] = True
    chunk_kw["iasvel"] = 1
    chunk_kw.pop("iunrea", None)
    chunk_kw["iunrea"] = -1
    _strip_stale_heat_ramp_keywords(chunk_kw)
    if int(chunk_kw.get("ihtfrq", 0) or 0) != 0:
        chunk_kw["ihtfrq"] = 0


def _prepare_post_rescue_cold_start_overlap_handoff(
    chunk_kw: dict[str, Any],
    *,
    mlpot_ctx: Optional["MlpotContext"],
) -> None:
    """Zero velocities and resume with ``iasvel=0`` (IHTFRQ heat ramp when configured)."""
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        prepare_comp_for_iasvel0,
    )

    _prepare_post_rescue_bath_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
    prepare_comp_for_iasvel0(zero_only=True)
    chunk_kw["restart"] = False
    chunk_kw["new"] = False
    chunk_kw["start"] = False
    chunk_kw["iasvel"] = 0
    chunk_kw["iasors"] = 0
    chunk_kw.pop("iunrea", None)
    chunk_kw["iunrea"] = -1
    if mlpot_ctx is not None:
        setattr(mlpot_ctx, "_overlap_post_rescue_cold_start", False)
    print(
        "overlap: post-rescue cold restart (iasvel=0; IHTFRQ ramp when configured)",
        flush=True,
    )


def _materialize_post_rescue_restart_handoff(
    chunk_io: CharmmTrajectoryFiles,
    chunk_kw: dict[str, Any],
    *,
    steps_done: int,
    mlpot_ctx: Optional["MlpotContext"],
    overlap: Optional["DynamicsOverlapConfig"],
    overlap_context: str,
) -> CharmmTrajectoryFiles:
    """Write rescued coords+velocities to scratch restart; next chunk uses READYN."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        patch_restart_global_step,
    )

    _assign_post_rescue_velocities_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
    cold_start = mlpot_ctx is not None and getattr(
        mlpot_ctx, "_overlap_post_rescue_cold_start", False
    )
    if cold_start:
        from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
            prepare_comp_for_iasvel0,
        )

        prepare_comp_for_iasvel0(zero_only=True)

    read_path = (
        Path(chunk_io.restart_read) if chunk_io.restart_read is not None else None
    )
    write_path = (
        Path(chunk_io.restart_write) if chunk_io.restart_write is not None else None
    )
    targets: list[Path] = []
    if write_path is not None:
        targets.append(write_path)
    if read_path is not None and read_path not in targets:
        targets.append(read_path)

    if not targets:
        return chunk_io

    step = max(0, int(steps_done))
    for path in targets:
        if not rewrite_dynamics_restart_validated(path):
            rescued_positions = None
            if overlap is not None:
                from mmml.interfaces.pycharmmInterface.mlpot.setup import (
                    get_charmm_positions_array,
                )

                rescued_positions = np.asarray(
                    get_charmm_positions_array(), dtype=float
                ).copy()
            rescue_crd, prior_restart = _overlap_rescue_restart_fallback_paths(overlap)
            from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                restore_post_rescue_coordinates,
            )

            source = restore_post_rescue_coordinates(
                rescued_positions=rescued_positions,
                rescue_crd=rescue_crd,
                prior_restart=prior_restart,
            )
            _assign_post_rescue_velocities_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
            if not rewrite_dynamics_restart_validated(path):
                raise RuntimeError(
                    f"overlap ({overlap_context}): post-rescue restart handoff failed "
                    f"writing {path.name} after fallback from {source}"
                )
        patch_restart_global_step(path, step)

    valid_read = (
        _valid_overlap_chunk_restart_read(write_path) if write_path is not None else None
    )
    if valid_read is None and write_path is not None:
        valid_read = _valid_restart_file(write_path)

    _prepare_post_rescue_bath_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
    chunk_kw["restart"] = True
    chunk_kw["new"] = False
    if cold_start:
        chunk_kw["start"] = False
        chunk_kw["iasvel"] = 0
        chunk_kw["iasors"] = 0
        if mlpot_ctx is not None:
            setattr(mlpot_ctx, "_overlap_post_rescue_cold_start", False)
    else:
        chunk_kw["start"] = True
        chunk_kw["iasvel"] = 1
        _strip_stale_heat_ramp_keywords(chunk_kw)
        if int(chunk_kw.get("ihtfrq", 0) or 0) != 0:
            chunk_kw["ihtfrq"] = 0
    chunk_kw.pop("iunrea", None)

    handoff_mode = "cold restart (iasvel=0)" if cold_start else "READYN; MLpot stabilized"
    print(
        f"overlap ({overlap_context}): post-rescue restart handoff from "
        f"{valid_read.name if valid_read is not None else targets[0].name} "
        f"at global step {step} ({handoff_mode})",
        flush=True,
    )

    return CharmmTrajectoryFiles(
        restart_read=valid_read,
        restart_write=chunk_io.restart_write,
        trajectory=chunk_io.trajectory,
        restart_read_unit=chunk_io.restart_read_unit,
        restart_write_unit=chunk_io.restart_write_unit,
        trajectory_unit=chunk_io.trajectory_unit,
    )


def _apply_post_rescue_overlap_handoff(
    chunk_io: CharmmTrajectoryFiles,
    chunk_kw: dict[str, Any],
    *,
    steps_done: int,
    mlpot_ctx: Optional["MlpotContext"],
    overlap: Optional["DynamicsOverlapConfig"],
    overlap_context: str,
) -> tuple[CharmmTrajectoryFiles, bool]:
    """Hand off after overlap rescue: in-memory for CPT, ``READYN`` otherwise.

    Returns ``(chunk_io, in_memory_handoff)``.
    """
    if bool(chunk_kw.get("cpt")):
        print(
            f"overlap ({overlap_context}): post-rescue in-memory handoff "
            f"scheduled at global step {max(0, int(steps_done))} for next overlap "
            f"chunk (CPT; no READYN on scratch restart — barostat state stays in RAM)",
            flush=True,
        )
        return chunk_io, True
    return (
        _materialize_post_rescue_restart_handoff(
            chunk_io,
            chunk_kw,
            steps_done=steps_done,
            mlpot_ctx=mlpot_ctx,
            overlap=overlap,
            overlap_context=overlap_context,
        ),
        False,
    )


def _overlap_rescue_restart_fallback_paths(
    overlap: Optional["DynamicsOverlapConfig"],
) -> tuple[Path | None, Path | None]:
    """Return ``(rescue_crd, prior_segment_restart)`` for post-rescue restart recovery."""
    if overlap is None:
        return None, None
    rescue_crd: Path | None = None
    registry = getattr(overlap, "artifact_registry", None)
    if registry is not None:
        rescue_crd = getattr(registry, "last_rescue_crd", None)
    prior = getattr(overlap, "prior_segment_restart", None)
    prior_path = Path(prior) if prior is not None else None
    return rescue_crd, prior_path


def _refresh_segment_restart_after_overlap_rescue(
    restart_path: PathLike | None,
    chunk_kw: dict[str, Any],
    *,
    mlpot_ctx: Optional["MlpotContext"],
    overlap: Optional["DynamicsOverlapConfig"] = None,
) -> None:
    """Rewrite the segment restart after last-chunk rescue (valid READYN if needed)."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_post_rescue_coordinates,
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    rescued_positions = np.asarray(get_charmm_positions_array(), dtype=float).copy()
    rescue_crd, prior_restart = _overlap_rescue_restart_fallback_paths(overlap)

    _assign_post_rescue_velocities_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
    if rewrite_dynamics_restart_validated(restart_path):
        return

    source = restore_post_rescue_coordinates(
        rescued_positions=rescued_positions,
        rescue_crd=rescue_crd,
        prior_restart=prior_restart,
    )
    print(
        f"WARN: post-rescue restart write had non-finite coordinates; "
        f"restored from {source}",
        flush=True,
    )
    _assign_post_rescue_velocities_and_crystal(chunk_kw, mlpot_ctx=mlpot_ctx)
    if not rewrite_dynamics_restart_validated(restart_path):
        raise RuntimeError(
            f"post-rescue restart {Path(restart_path).name} still has non-finite "
            f"coordinates after fallback from {source}"
        )
    print(
        f"post-rescue restart recovered from {source}",
        flush=True,
    )


def _prepare_overlap_chunk_after_restart(
    mlpot_ctx: Optional["MlpotContext"],
    *,
    restart_read: PathLike | None = None,
) -> None:
    """Stabilize CHARMM before the next overlap ``dyna`` chunk that will ``READYN``.

    With MLpot registered, do **not** call ``update_bnbnd`` (``upinb``) — ML exclusion
    lists are already established and rebuilding them mid-workflow segfaults after
    long MD (same as ``reregister_mlpot`` / ``refresh_nbonds_after_mlpot*``).
    ``READYN`` on the scratch restart restores coordinates, velocities, and lists.

    When MLpot is active, refresh the JAX MIC cell from the upcoming scratch restart
    (or live CHARMM pbound) so ``calculate_charmm`` does not keep using a stale
    pretreat ``.res`` path set at registration time.
    """
    if mlpot_ctx is not None:
        pyCModel = getattr(mlpot_ctx, "pyCModel", None)
        if pyCModel is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
                sync_mlpot_pbc_cell_from_charmm,
            )

            side = sync_mlpot_pbc_cell_from_charmm(
                pyCModel,
                fallback_side_A=getattr(mlpot_ctx, "cubic_box_side_A", None),
                restart_path=restart_read,
                verbose=False,
            )
            mlpot_ctx.cubic_box_side_A = float(side)
            mlpot_ctx.charmm_cubic_box_side_A = float(side)
        return

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.nbonds_config import vacuum_nbond_kwargs

    pycharmm = _import_pycharmm_modules()[0]
    with charmm_relaxed_bomlev():
        pycharmm.nbonds.update_bnbnd()
        pycharmm.UpdateNonBondedScript(**vacuum_nbond_kwargs(nbxmod=5)).run()
        pycharmm.lingo.charmm_script("ENER")
        pycharmm.lingo.charmm_script("UPDATE")


def _cpt_subchunk_restart_slot_paths(base: Path) -> tuple[Path, Path]:
    """Alternating scratch restarts for CPT stability sub-chunks within one segment."""
    parent = base.parent
    stem = base.stem
    return (
        parent / f"{stem}.cptsc_a.res",
        parent / f"{stem}.cptsc_b.res",
    )


def _cpt_subchunk_use_in_memory_handoff() -> bool:
    """Continue Hoover CPT sub-chunks in-process unless scratch READYN is requested.

    Plain ``write restart`` snapshots coordinates/velocities but **not** Hoover CPT
    barostat piston internals.  ``READYN`` on ``.cptsc_*.res`` between sub-chunks
    therefore yields garbage ``PIXX`` / ``PRESSI`` at dyna step 0.  Default is
    in-memory continuation; set ``MMML_CPT_READYN_SUBCHUNK=1`` to opt into scratch
    READYN (legacy / debugging only).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import _truthy_env

    if _truthy_env("MMML_CPT_READYN_SUBCHUNK"):
        return False
    return True


def _apply_cpt_in_memory_continuation_kw(kw: dict[str, Any]) -> None:
    """In-process ``dyna`` continuation for CPT sub-chunks (no ``READYN``).

    Preserves Hoover CPT barostat state in RAM between sub-chunks.  Plain
    ``write restart`` / ``READYN`` handoffs do not restore piston internals.
    """
    kw["restart"] = False
    kw["new"] = False
    kw["start"] = False
    kw["iasvel"] = 0
    kw.pop("firstt", None)
    kw.pop("iunrea", None)
    kw["iunrea"] = -1
    _strip_stale_heat_ramp_keywords(kw)
    if int(kw.get("ihtfrq", 0) or 0) != 0:
        kw["ihtfrq"] = 0


def _apply_cpt_restart_continuation_kw(kw: dict[str, Any]) -> None:
    """``READYN`` continuation for CPT sub-chunks (global step only; not barostat).

    Requires a dynamics restart file with full CPT internals.  MMML scratch
    ``write restart`` handoffs do **not** qualify — use in-memory sub-chunks instead.
    """
    kw["restart"] = True
    kw["new"] = False
    kw["start"] = False
    kw["iasvel"] = 0
    kw.pop("firstt", None)
    _strip_stale_heat_ramp_keywords(kw)
    if int(kw.get("ihtfrq", 0) or 0) != 0:
        kw["ihtfrq"] = 0


def _materialize_cpt_subchunk_restart_handoff(
    write_path: Path,
    *,
    global_step: int,
    overlap_context: str,
    mlpot_ctx: Optional["MlpotContext"] = None,
    chunk_kw: dict[str, Any] | None = None,
) -> Path:
    """Write a validated READYN restart from in-memory state after a CPT sub-chunk.

    Do **not** Boltzmann-assign (``nstep=0``) before the write: that re-initializes
    velocities but leaves Hoover CPT barostat pistons uninitialized, producing
    garbage ``PIXX`` / ``PRESSI`` on the next ``READYN`` (same failure mode as a
    separate assign before the first CPT ``dyna``).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        patch_restart_global_step,
    )

    if not rewrite_dynamics_restart_validated(write_path):
        raise RuntimeError(
            f"{overlap_context}: CPT sub-chunk restart handoff failed writing "
            f"{write_path.name} from in-memory CHARMM state"
        )
    patch_restart_global_step(write_path, global_step)
    valid = _valid_restart_file(write_path)
    if valid is None:
        raise RuntimeError(
            f"{overlap_context}: CPT sub-chunk restart handoff wrote "
            f"{write_path.name} but it does not validate as READYN"
        )
    return valid


def _strip_stale_heat_ramp_keywords(kw: dict[str, Any]) -> None:
    """Remove HEAT ramp keywords that confuse Hoover CPT on later overlap chunks."""
    for key in ("finalt", "TEMINC", "teminc"):
        kw.pop(key, None)


def _apply_overlap_chunk_dynamics_kw(
    chunk_kw: dict[str, Any],
    *,
    chunk_index: int,
    has_restart_read: bool,
) -> None:
    """Set ``restart`` / ``new`` / ``start`` for one overlap chunk (in-place)."""
    if has_restart_read:
        chunk_kw["new"] = False
        chunk_kw["restart"] = True
        if (
            int(chunk_index) == 0
            and _cpt_npt_fresh_barostat_from_restart_kw(chunk_kw)
        ):
            return
        if (
            int(chunk_index) > 0
            and int(chunk_kw.get("ihtfrq", 0) or 0) > 0
            and "hoover reft" not in chunk_kw
            and not bool(chunk_kw.get("cpt"))
        ):
            # Scale-heat ramp across READYN handoff (see COMP_AND_HEATING.md).
            chunk_kw["start"] = False
            chunk_kw["iasvel"] = 1
            chunk_kw["iasors"] = 0
            return
        chunk_kw["start"] = False
        chunk_kw["iasvel"] = 0
        if chunk_index > 0:
            chunk_kw.pop("firstt", None)
        return

    preserve_ihtfrq_heat_ramp = (
        chunk_index == 0
        and not has_restart_read
        and int(chunk_kw.get("ihtfrq", 0) or 0) > 0
        and "hoover reft" not in chunk_kw
        and not bool(chunk_kw.get("cpt"))
    )
    preserve_cold_start = (
        chunk_index == 0
        and not has_restart_read
        and (
            bool(chunk_kw.get("start"))
            or "hoover reft" in chunk_kw
            or bool(chunk_kw.get("cpt"))
            or preserve_ihtfrq_heat_ramp
        )
    )
    if not preserve_cold_start:
        # Chunk > 0 in-process continuation keeps velocities; HEAT ramp apply may
        # override iasvel=1 afterward when a stage ramp is active.
        chunk_kw["iasvel"] = 0 if chunk_index > 0 else 1
        chunk_kw["start"] = False
        if chunk_index > 0:
            chunk_kw.pop("firstt", None)
            _strip_stale_heat_ramp_keywords(chunk_kw)
    elif preserve_ihtfrq_heat_ramp:
        # Chunk 0 keeps ``start`` from ``_configure_heat_dynamics_start`` (cold Boltzmann).
        chunk_kw["iasvel"] = 1 if int(chunk_index) == 0 else 0
        chunk_kw["iasors"] = 0
        if int(chunk_index) > 0:
            chunk_kw["start"] = False
        # Hoover NVT: keep thermostat keywords; ensure scale-heat ramps stay off.
        if int(chunk_kw.get("ihtfrq", 0)) != 0 and "hoover reft" in chunk_kw:
            chunk_kw["ihtfrq"] = 0
    elif preserve_cold_start and (
        "hoover reft" in chunk_kw or bool(chunk_kw.get("cpt"))
    ):
        # Hoover CPT chunk 0: single-dyna cold start (heat/NPT barostat init).
        if _hoover_cpt_overlap_chunk_needs_cold_start(
            chunk_kw,
            chunk_index=chunk_index,
            has_restart_read=has_restart_read,
            steps_done=0,
        ):
            chunk_kw["iasvel"] = 1
            chunk_kw["start"] = True
        elif bool(chunk_kw.get("start")):
            chunk_kw["iasvel"] = 1
        elif int(chunk_kw.get("iasvel", 0)) == 1:
            chunk_kw["iasvel"] = 1
            chunk_kw["start"] = False
        else:
            chunk_kw["iasvel"] = 0
            chunk_kw["start"] = False
        if int(chunk_kw.get("ihtfrq", 0)) != 0:
            chunk_kw["ihtfrq"] = 0
    if chunk_index == 0 and not has_restart_read:
        chunk_kw["restart"] = False
        chunk_kw.pop("iunrea", None)
        chunk_kw["iunrea"] = -1
        return
    chunk_kw["new"] = False
    chunk_kw["start"] = False
    if has_restart_read:
        chunk_kw["restart"] = True
    else:
        chunk_kw["restart"] = False
        chunk_kw.pop("iunrea", None)
        chunk_kw["iunrea"] = -1


def _cleanup_overlap_restart_slots(io: Optional[CharmmTrajectoryFiles]) -> None:
    if io is None or io.restart_write is None:
        return
    final = Path(io.restart_write)
    slot_a, slot_b = _overlap_restart_slot_paths(final)
    cpt_a, cpt_b = _cpt_subchunk_restart_slot_paths(final)
    for path in (slot_a, slot_b, cpt_a, cpt_b):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _sync_dynamics_io_units(
    kw: dict[str, Any],
    iokw: dict[str, Any],
) -> None:
    """Drop restart/trajectory unit numbers not backed by opened CharmmFile handles."""
    for key in ("iunrea", "iunwri", "iuncrd", "iunvel"):
        if key in iokw:
            continue
        if key == "iunrea" and int(kw.get("iunrea", 0)) == -1:
            # Explicit no-read: keep in ``dyna`` so CHARMM does not reuse a stale unit.
            continue
        kw.pop(key, None)


def _refresh_charmm_dynamics_rng(*, base: int | None, salt: int) -> None:
    """Reseed CHARMM dynamics RNG (thermostat / velocity assignment noise)."""
    import pycharmm.dynamics as dyn

    nrand = max(1, int(dyn.get_nrand()))
    if base is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(base) + int(salt) * 1_000_003)
    seeds = [int(x) for x in rng.integers(1, 2**31, size=nrand)]
    dyn.set_rngseeds(seeds)


def _rng_salt_for_dynamics(
    *,
    overlap_context: str,
    chunk_index: int,
    steps_done: int,
    retry_count: int = 0,
) -> int:
    ctx_hash = abs(hash(str(overlap_context))) & 0x7FFF_FFFF
    return int(
        ctx_hash
        + chunk_index * 1_000_003
        + steps_done * 10_007
        + int(retry_count) * 7919
    )


def _integrated_step_from_restart(
    *,
    chunk_io: Optional[CharmmTrajectoryFiles],
    final_restart: Path | None,
    fallback_steps: int,
    steps_before_chunk: int = 0,
) -> int:
    """Read global dynamics step (``JHSTRT``) from the latest restart write."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_last_step,
    )

    fb = max(0, int(fallback_steps))
    for candidate in (
        getattr(chunk_io, "restart_write", None) if chunk_io is not None else None,
        final_restart,
    ):
        if candidate is None:
            continue
        step = read_restart_last_step(Path(candidate))
        if step is None:
            continue
        step = int(step)
        abs_step = abs(step)
        if abs_step <= 0:
            continue

        if abs_step < steps_before_chunk:
            # Memory-handoff scratch files often store a segment-local step (e.g. 100
            # or 500) while integrated accounting already advanced to ``fb``.  Negative
            # REST abort markers (e.g. REST 48 -344) must still add ``steps_before_chunk``.
            if step > 0 and fb >= steps_before_chunk + 1:
                return fb
            global_step = steps_before_chunk + abs_step
        else:
            global_step = abs_step

        if step > 0:
            if global_step >= fb - 1:
                return global_step
            # CHARMM coord-history / scratch restarts often leave NSTEP at the last
            # overlap sub-chunk size (e.g. 500) while the integrated segment ran longer
            # (e.g. 2500).  Trust ``fallback_steps`` when the header divides evenly.
            if fb > global_step and fb % global_step == 0 and fb // global_step >= 2:
                return fb
        
        return global_step
    return fb


def _run_dynamics_chunk(
    dynamics_kwargs: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles],
    *,
    extra_iokw: dict[str, int] | None = None,
    rng_base: int | None = None,
    rng_salt: int = 0,
) -> Any:
    _refresh_charmm_dynamics_rng(base=rng_base, salt=rng_salt)
    open_files: list[Any] = []
    io_aliases: list[Any] = []
    kw = dict(dynamics_kwargs)
    iokw: dict[str, Any] = {}
    if io is not None:
        open_files, iokw, io_aliases = io.open_for_run()
        kw.update(iokw)
    if extra_iokw:
        kw.update(extra_iokw)
        iokw = {**iokw, **extra_iokw}
    if kw.get("restart", False):
        kw.pop("iunrea", None)
        if "iunrea" in iokw:
            kw["iunrea"] = iokw["iunrea"]
    elif "iunrea" not in iokw:
        kw.pop("iunrea", None)
        kw["iunrea"] = -1
    _sync_dynamics_io_units(kw, iokw)
    try:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

        with charmm_relaxed_bomlev(level=-2):
            return run_dynamics(kw)
    finally:
        for f in open_files:
            f.close()
        for alias in io_aliases:
            try:
                alias.finalize()
            except Exception:
                pass


def _cpt_stability_chunk_nstep(kw: dict[str, Any], total_nstep: int) -> int | None:
    """Chunk size for NPT CPT when overlap guard is off; ``None`` = single ``dyn.run()``."""
    import os

    if not bool(kw.get("cpt")) or total_nstep <= 0:
        return None
    raw = os.environ.get("MMML_CPT_DYNAMICS_CHUNK_NSTEP")
    chunk = (
        int(raw)
        if raw is not None and str(raw).strip() != ""
        else DEFAULT_CPT_DYNAMICS_CHUNK_NSTEP
    )
    chunk = max(1, int(chunk))
    if total_nstep <= chunk:
        return None
    return chunk


def _dynamics_chunk_state_corrupt(
    *,
    overlap_context: str,
    restart_path: Path | None,
) -> bool:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        charmm_dynamics_state_is_finite,
        restart_has_nonfinite_coordinates,
    )

    if not charmm_dynamics_state_is_finite():
        print(
            f"WARN: {overlap_context}: non-finite CHARMM coordinates/energy after "
            "dynamics chunk",
            flush=True,
        )
        return True
    if restart_path is not None and restart_has_nonfinite_coordinates(restart_path):
        print(
            f"WARN: {overlap_context}: restart {Path(restart_path).name} has "
            "non-finite coordinates after dynamics chunk",
            flush=True,
        )
        return True
    return False


def _run_cpt_stability_subchunked(
    kw: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles],
    *,
    overlap_context: str,
    rng_base: int | None,
    chunk_nstep: int,
    total_nstep: int | None = None,
    extra_iokw: dict[str, int] | None = None,
    rng_salt_base: int = 0,
    loose_pbc: bool = False,
    log_banner: bool = True,
    global_step_offset: int = 0,
    mlpot_ctx: Optional["MlpotContext"] = None,
) -> Any:
    """Integrate CPT dynamics in short ``dyn.run()`` segments with state checks."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        validate_charmm_dynamics_state_after_chunk,
    )

    total = int(total_nstep if total_nstep is not None else kw.get("nstep", 0))
    steps_done = 0
    last_dyn = None
    n_subchunks = (total + chunk_nstep - 1) // chunk_nstep
    use_in_memory = _cpt_subchunk_use_in_memory_handoff()
    final_write = (
        Path(io.restart_write)
        if io is not None and io.restart_write is not None
        else None
    )
    use_restart_handoff = not use_in_memory and final_write is not None
    slot_a, slot_b = (
        _cpt_subchunk_restart_slot_paths(final_write)
        if final_write is not None
        else (None, None)
    )
    prev_write: Path | None = None
    if log_banner and n_subchunks > 1:
        mode = (
            "in-memory"
            if not use_restart_handoff
            else "READYN restart"
        )
        print(
            f"{overlap_context}: CPT stability chunking — {total} steps in "
            f"{n_subchunks} segment(s) of ≤{chunk_nstep} "
            f"({mode}; Hoover CPT barostat kept in RAM between sub-chunks)",
            flush=True,
        )
    while steps_done < total:
        n = min(chunk_nstep, total - steps_done)
        sub_kw = dict(kw)
        sub_kw["nstep"] = n
        is_continuation = steps_done > 0
        if is_continuation:
            if use_restart_handoff:
                _apply_cpt_restart_continuation_kw(sub_kw)
            else:
                _apply_cpt_in_memory_continuation_kw(sub_kw)
        _harmonize_overlap_chunk_frequencies(
            sub_kw,
            n,
            loose_pbc=loose_pbc,
            global_step_start=int(global_step_offset) + steps_done,
        )

        sub_io = io
        write_path: Path | None = None
        if (
            use_restart_handoff
            and n_subchunks > 1
            and slot_a is not None
            and slot_b is not None
        ):
            if is_continuation:
                read_path = prev_write
            else:
                read_path = io.restart_read if io is not None else None
            if steps_done + n >= total:
                write_path = final_write
            elif not is_continuation:
                write_path = slot_a
            else:
                write_path = slot_b if prev_write == slot_a else slot_a
            sub_io = CharmmTrajectoryFiles(
                restart_read=Path(read_path) if read_path is not None else None,
                restart_write=write_path,
                trajectory=io.trajectory if io is not None else None,
                pressure_tensor_log=(
                    io.pressure_tensor_log if io is not None else None
                ),
                restart_read_unit=io.restart_read_unit if io is not None else 3,
                restart_write_unit=io.restart_write_unit if io is not None else 2,
                trajectory_unit=io.trajectory_unit if io is not None else 1,
                pressure_tensor_log_unit=(
                    io.pressure_tensor_log_unit if io is not None else 29
                ),
            )
            if is_continuation and mlpot_ctx is not None and read_path is not None:
                _prepare_overlap_chunk_after_restart(
                    mlpot_ctx,
                    restart_read=read_path,
                )

        suppress_sub_traj = bool(sub_kw.get("_suppress_trajectory", False))
        if suppress_sub_traj or "nsavc" not in sub_kw:
            sub_io = _drop_trajectory_io(sub_io)
        sub_traj_iokw = (
            extra_iokw
            if (
                extra_iokw
                and steps_done == 0
                and "nsavc" in sub_kw
                and not suppress_sub_traj
            )
            else {}
        )
        last_dyn = _run_dynamics_chunk(
            sub_kw,
            sub_io,
            extra_iokw=sub_traj_iokw,
            rng_base=rng_base,
            rng_salt=rng_salt_base + steps_done,
        )
        restart_path = (
            write_path
            if write_path is not None
            else (
                Path(io.restart_write)
                if io is not None and io.restart_write is not None
                else None
            )
        )
        if _dynamics_chunk_state_corrupt(
            overlap_context=f"{overlap_context} CPT sub-chunk ending step {steps_done + n}",
            restart_path=restart_path,
        ):
            validate_charmm_dynamics_state_after_chunk(
                context=f"{overlap_context} at step {steps_done + n}",
            )
        chunk_end = steps_done + n
        if restart_path is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                read_restart_last_step,
            )

            actual_global = read_restart_last_step(Path(restart_path))
            if actual_global is not None:
                actual_in_segment = int(actual_global) - int(global_step_offset)
                if actual_in_segment < chunk_end:
                    steps_done = max(steps_done, actual_in_segment)
                    break
        global_step = max(0, int(global_step_offset) + chunk_end)
        if (
            use_restart_handoff
            and write_path is not None
            and n_subchunks > 1
        ):
            _materialize_cpt_subchunk_restart_handoff(
                write_path,
                global_step=global_step,
                overlap_context=overlap_context,
                mlpot_ctx=mlpot_ctx,
                chunk_kw=sub_kw,
            )
            prev_write = write_path
        steps_done += n
    return last_dyn


def _run_cpt_stability_chunked_dynamics(
    kw: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles],
    *,
    overlap_context: str,
    rng_base: int | None,
    chunk_nstep: int,
) -> Any:
    """Integrate NPT CPT in short segments with finite-state checks between chunks."""
    from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
        maybe_record_forces,
    )

    total = int(kw.get("nstep", 0))
    last_dyn = _run_cpt_stability_subchunked(
        kw,
        io,
        overlap_context=overlap_context,
        rng_base=rng_base,
        chunk_nstep=chunk_nstep,
        total_nstep=total,
        log_banner=True,
    )
    maybe_record_forces(total, ml_forces=None)
    return last_dyn


def run_dynamics_with_io(
    dynamics_kwargs: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles] = None,
    *,
    overlap: Optional["DynamicsOverlapConfig"] = None,
    overlap_context: str = "dynamics",
    mlpot_ctx: Optional["MlpotContext"] = None,
    rng_base: int | None = None,
    loose_pbc: bool = False,
    overlap_run_state_dir: Path | None = None,
    overlap_run_state_every_chunks: int = 0,
    segment_restart_read: Path | str | None = None,
) -> OverlapDynamicsResult:
    """Run dynamics and open/close CharmmFile units from ``io``.

    When ``overlap`` is enabled, integration runs in chunks of
    ``overlap.check_interval`` steps with inter-monomer distance checks
    between chunks.  Multi-chunk runs alternate scratch ``.overlap_a/.b.res``
    restarts (chunk 0 always writes scratch) so time/step counters advance
    correctly; the last chunk writes the final restart file.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        check_dynamics_overlap,
    )

    if mlpot_ctx is not None:
        _ensure_domdec_off_for_mlpot_energy(context=overlap_context)

    kw = dict(dynamics_kwargs)
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        install_target_dcd_metadata,
    )

    install_target_dcd_metadata(kw)
    _ensure_nsavc_below_nstep(kw)
    total_nstep = int(kw.get("nstep", 0))
    if loose_pbc and total_nstep > 0:
        apply_loose_pbc_dyn_freq_kwargs(kw, nstep=total_nstep)
    elif total_nstep > 0:
        _ensure_ntrfrq_above_nstep(kw, total_nstep)
    guard_active = (
        overlap is not None
        and isinstance(overlap, DynamicsOverlapConfig)
        and (overlap.enabled or overlap.intra_enabled or overlap.extent_enabled)
    )
    if (
        guard_active
        and overlap is not None
        and mlpot_ctx is not None
        and total_nstep > 0
        and not overlap.memory_handoff
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import _truthy_env

        if not _truthy_env("MMML_NO_OVERLAP_MEMORY_HANDOFF"):
            from dataclasses import replace

            overlap = replace(overlap, memory_handoff=True)
            if total_nstep > int(overlap.check_interval):
                cpt_note = (
                    " (Hoover CPT barostat kept in RAM between chunks)"
                    if bool(kw.get("cpt"))
                    else ""
                )
                print(
                    f"overlap ({overlap_context}): MLpot in-memory chunk handoff"
                    f"{cpt_note}",
                    flush=True,
                )
    if guard_active and overlap is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
            attach_prior_segment_restart,
        )

        overlap = attach_prior_segment_restart(
            overlap,
            segment_index=overlap.segment_index,
            prev_restart=(
                None
                if overlap.prior_segment_restart is not None
                and Path(overlap.prior_segment_restart).is_file()
                else (io.restart_read if io is not None else None)
            ),
            out_dir=overlap.segment_out_dir,
            restart_prefix=overlap.segment_restart_prefix,
            restart_write=io.restart_write if io is not None else None,
        )
    if not guard_active or total_nstep <= 0:
        cpt_chunk = _cpt_stability_chunk_nstep(kw, total_nstep)
        if cpt_chunk is not None:
            last_dyn = _run_cpt_stability_chunked_dynamics(
                kw,
                io,
                overlap_context=overlap_context,
                rng_base=rng_base,
                chunk_nstep=cpt_chunk,
            )
            integrated = int(total_nstep)
            completed = True
            if io is not None and io.restart_write is not None:
                from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                    resolve_integrated_restart_step,
                )

                restart_step = resolve_integrated_restart_step(
                    Path(io.restart_write),
                    expected_nstep=total_nstep,
                )
                if restart_step is not None:
                    integrated = int(restart_step)
                    completed = integrated >= max(1, int(total_nstep * 0.95))
            return _overlap_dynamics_result(
                last_dyn,
                integrated_step=integrated,
                completed_full=completed,
            )
        last_dyn = _run_dynamics_chunk(
            kw,
            io,
            rng_base=rng_base,
            rng_salt=_rng_salt_for_dynamics(
                overlap_context=overlap_context,
                chunk_index=0,
                steps_done=0,
            ),
        )
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
            validate_charmm_dynamics_state_after_chunk,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
            maybe_record_forces,
        )

        restart_path = (
            Path(io.restart_write)
            if io is not None and io.restart_write is not None
            else None
        )
        if _dynamics_chunk_state_corrupt(
            overlap_context=overlap_context,
            restart_path=restart_path,
        ):
            validate_charmm_dynamics_state_after_chunk(context=overlap_context)
        maybe_record_forces(int(kw.get("nstep", 0)), ml_forces=None)
        return _overlap_dynamics_result(
            last_dyn,
            integrated_step=int(kw.get("nstep", 0)),
            completed_full=True,
        )

    requested_interval = max(1, int(overlap.check_interval))
    traj_nsavc = int(kw["nsavc"]) if "nsavc" in kw else None
    cpt_overlap = bool(kw.get("cpt"))
    interval = effective_overlap_check_interval(
        total_nstep,
        requested_interval,
        nsavc=traj_nsavc,
        cpt=cpt_overlap,
    )
    min_for_nsavc = (traj_nsavc + 1) if traj_nsavc is not None and traj_nsavc > 0 else None
    if interval != requested_interval:
        reason = f"so {total_nstep} steps divide evenly ({total_nstep // interval} chunks)"
        if (
            min_for_nsavc is not None
            and requested_interval < min_for_nsavc
            and interval >= min_for_nsavc
        ):
            reason = (
                f"nsavc={traj_nsavc} requires chunk nstep >= {min_for_nsavc}; "
                + reason
            )
        _emit_overlap_log(
            f"check interval {requested_interval} -> {interval} steps {reason}",
            context=overlap_context,
        )
    _cleanup_overlap_restart_slots(io)
    check_dynamics_overlap(
        overlap,
        context=f"before {overlap_context}",
        step=0,
        mlpot_ctx=mlpot_ctx,
    )

    n_chunks = total_nstep // interval
    post_rescue_in_memory_mode = False
    post_rescue_handoff_applied = False
    any_post_rescue_restart_handoff = False
    early_abort_memory_handoff = False
    early_abort_restart_handoff = False
    if (
        n_chunks > 20
        and "heat" in str(overlap_context).lower()
        and hoover_cpt_heat_ramp_spec_from_kw(kw) is not None
    ):
        _emit_overlap_log(
            f"warning: {n_chunks} Hoover CPT heat restart chunks "
            f"(interval={interval}, total={total_nstep} steps). Prefer "
            f"--dynamics-overlap-check-interval >= {total_nstep} or fewer "
            f"--n-heat-segments to reduce scratch READYN handoffs.",
            context=overlap_context,
        )
    heat_ramp_spec = heat_ramp_spec_from_kw(kw)
    hoover_heat_ramp_spec = hoover_cpt_heat_ramp_spec_from_kw(kw)
    overlap_memory_handoff = bool(
        overlap is not None and getattr(overlap, "memory_handoff", False)
    )
    final_restart = Path(io.restart_write) if io is not None and io.restart_write else None
    last_dyn: Any = None
    steps_done = 0
    completed = False
    salvaged_partial = False
    abort_reason: str | None = None
    trajectory_files: list[Any] = []
    trajectory_iokw: dict[str, int] = {}
    chunk_dcd_paths: list[Path] = []
    logged_mem_handoff = False
    split_trajectory = (
        io is not None
        and io.trajectory is not None
        and _overlap_should_split_trajectory(n_chunks=n_chunks, traj_nsavc=traj_nsavc)
    )
    if (
        io is not None
        and io.trajectory is not None
        and n_chunks > 1
        and split_trajectory
    ):
        _emit_overlap_log(
            f"writing {n_chunks} per-chunk DCD(s) "
            f"({Path(io.trajectory).stem}.chunk.*{Path(io.trajectory).suffix})",
            context=overlap_context,
        )
    elif (
        io is not None
        and io.trajectory is not None
        and n_chunks > 1
        and not split_trajectory
    ):
        _emit_overlap_log(
            f"writing one DCD ({io.trajectory.name}) "
            f"across {n_chunks} chunks (nsavc=1 chunk explosion guard)",
            context=overlap_context,
        )
    try:
        if io is not None and n_chunks > 1 and io.trajectory is not None and not split_trajectory:
            trajectory_files, trajectory_iokw = io.open_trajectory_for_run()
        chunk_index = 0
        pending_readyn_chunk_io: Optional[CharmmTrajectoryFiles] = None
        _MAX_EARLY_ABORT_CHUNK_RETRIES = 3
        while chunk_index < n_chunks:
            chunk_retry_count = 0
            segment_aborted = False
            early_abort_memory_handoff = False
            early_abort_restart_handoff = False
            # One-chunk scope: rescue at end of chunk N arms memory handoff for N+1 only.
            post_rescue_memory_this_chunk = post_rescue_in_memory_mode
            post_rescue_in_memory_mode = False
            post_rescue_handoff_applied = False
            overlap_restart_read_for_chunk: Path | None = None
            if io is not None and n_chunks > 1:
                planned_read, _ = _overlap_chunk_restart_paths(
                    io, chunk_index=chunk_index, n_chunks=n_chunks
                )
                if planned_read is not None:
                    overlap_restart_read_for_chunk = Path(planned_read)
            rerun_chunk = True
            while rerun_chunk:
                rerun_chunk = False
                chunk_nstep = interval
                steps_before_chunk = steps_done
                chunk_kw = dict(kw)
                chunk_kw["nstep"] = chunk_nstep
                if pending_readyn_chunk_io is not None:
                    chunk_io = pending_readyn_chunk_io
                    pending_readyn_chunk_io = None
                elif io is None:
                    chunk_io = None
                elif n_chunks == 1:
                    chunk_io = io
                else:
                    chunk_io = _overlap_chunk_io(
                        io,
                        chunk_index=chunk_index,
                        n_chunks=n_chunks,
                        split_trajectory=split_trajectory,
                        mlpot_ctx=mlpot_ctx,
                        use_memory_handoff=(
                            post_rescue_memory_this_chunk
                            or early_abort_memory_handoff
                            or _overlap_chunk_uses_memory_handoff(
                                mlpot_ctx,
                                chunk_index=chunk_index,
                                n_chunks=n_chunks,
                                overlap=overlap,
                                cpt=bool(chunk_kw.get("cpt")),
                            )
                        ),
                    )
                if early_abort_restart_handoff and chunk_io is not None:
                    chunk_io = _materialize_early_abort_restart_handoff(
                        chunk_io,
                        steps_before_chunk=steps_before_chunk,
                        overlap_context=overlap_context,
                        chunk_kw=chunk_kw,
                        mlpot_ctx=mlpot_ctx,
                    )
                    if chunk_io.restart_read is None:
                        if bool(chunk_kw.get("cpt")):
                            raise RuntimeError(
                                f"overlap ({overlap_context}): CPT early-abort retry "
                                "requires a valid READYN scratch restart; refusing "
                                "in-memory continuation (readyn segfault risk)"
                            )
                        print(
                            f"overlap ({overlap_context}): early-abort restart "
                            "materialization failed; falling back to in-memory retry",
                            flush=True,
                        )
                        early_abort_restart_handoff = False
                        early_abort_memory_handoff = True
                using_readyn_chunk_io = (
                    chunk_io is not None
                    and getattr(chunk_io, "restart_read", None) is not None
                )
                mem_handoff = (
                    not using_readyn_chunk_io
                    and (
                        post_rescue_memory_this_chunk
                        or early_abort_memory_handoff
                        or _overlap_chunk_uses_memory_handoff(
                            mlpot_ctx,
                            chunk_index=chunk_index,
                            n_chunks=n_chunks,
                            overlap=overlap,
                            cpt=bool(chunk_kw.get("cpt")),
                        )
                    )
                )
                has_restart_read = using_readyn_chunk_io and not mem_handoff
                if (
                    has_restart_read
                    and chunk_io is not None
                    and chunk_io.restart_read is not None
                ):
                    filtered = _valid_overlap_chunk_restart_read(chunk_io.restart_read)
                    if filtered is None:
                        has_restart_read = False
                        chunk_io = CharmmTrajectoryFiles(
                            restart_read=None,
                            restart_write=chunk_io.restart_write,
                            trajectory=chunk_io.trajectory,
                            restart_read_unit=chunk_io.restart_read_unit,
                            restart_write_unit=chunk_io.restart_write_unit,
                            trajectory_unit=chunk_io.trajectory_unit,
                        )
                    elif filtered != chunk_io.restart_read:
                        chunk_io = CharmmTrajectoryFiles(
                            restart_read=filtered,
                            restart_write=chunk_io.restart_write,
                            trajectory=chunk_io.trajectory,
                            restart_read_unit=chunk_io.restart_read_unit,
                            restart_write_unit=chunk_io.restart_write_unit,
                            trajectory_unit=chunk_io.trajectory_unit,
                        )
                if mem_handoff and chunk_index == 1 and not logged_mem_handoff:
                    print(
                        f"overlap ({overlap_context}): in-memory handoff between "
                        f"{n_chunks} chunks",
                        flush=True,
                    )
                    logged_mem_handoff = True
                elif (
                    not mem_handoff
                    and chunk_index == 1
                    and n_chunks > 1
                    and mlpot_ctx is not None
                    and not logged_mem_handoff
                ):
                    print(
                        f"overlap ({overlap_context}): scratch restart handoff between "
                        f"{n_chunks} chunks (dyna restart on .overlap_a/.b.res)",
                        flush=True,
                    )
                    logged_mem_handoff = True
                if (
                    chunk_index > 0
                    and n_chunks > 1
                    and chunk_io is not None
                    and getattr(chunk_io, "restart_write", None) is not None
                    and not has_restart_read
                    and not mem_handoff
                ):
                    raise RuntimeError(
                        "overlap restart handoff failed: previous CHARMM chunk did not "
                        f"produce a restartable scratch file before {overlap_context} "
                        f"chunk {chunk_index + 1}/{n_chunks}. The dynamics segment is "
                        "unstable or CHARMM wrote a coordinate-history scratch file "
                        "instead of a restart; reduce the timestep, minimize longer, "
                        "or disable overlap chunking for this run."
                    )
                if (
                    chunk_index > 0
                    and overlap is not None
                    and chunk_io is not None
                    and getattr(chunk_io, "restart_read", None) is not None
                    and not mem_handoff
                ):
                    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
                        ensure_restartable_before_overlap_chunk,
                    )

                    ensure_restartable_before_overlap_chunk(
                        Path(chunk_io.restart_read),
                        overlap,
                        overlap_context=overlap_context,
                        overlap_run_state_dir=overlap_run_state_dir,
                    )
                if heat_ramp_spec is not None:
                    _apply_overlap_chunk_heat_ramp(
                        chunk_kw,
                        chunk_index=chunk_index,
                        chunk_nstep=chunk_nstep,
                        steps_done=steps_done,
                        ramp_spec=heat_ramp_spec,
                    )
                _apply_overlap_chunk_dynamics_kw(
                    chunk_kw,
                    chunk_index=chunk_index,
                    has_restart_read=has_restart_read,
                )
                if (
                    mem_handoff
                    and bool(chunk_kw.get("cpt"))
                    and chunk_index > 0
                ):
                    _apply_cpt_in_memory_continuation_kw(chunk_kw)
                if early_abort_memory_handoff:
                    chunk_kw["restart"] = False
                    chunk_kw["new"] = False
                    chunk_kw["start"] = False
                    chunk_kw["iasvel"] = 0
                    chunk_kw.pop("iunrea", None)
                    chunk_kw["iunrea"] = -1
                    _strip_stale_heat_ramp_keywords(chunk_kw)
                    if int(chunk_kw.get("ihtfrq", 0) or 0) != 0:
                        chunk_kw["ihtfrq"] = 0
                elif early_abort_restart_handoff:
                    chunk_kw["restart"] = True
                    chunk_kw["new"] = False
                    chunk_kw["start"] = False
                    chunk_kw["iasvel"] = 0
                    chunk_kw.pop("iunrea", None)
                elif hoover_heat_ramp_spec is not None:
                    apply_hoover_cpt_heat_ramp_overlap_chunk(
                        chunk_kw,
                        chunk_index=chunk_index,
                        steps_done=steps_done,
                        ramp_spec=hoover_heat_ramp_spec,
                        total_nstep=total_nstep,
                        n_chunks=n_chunks,
                    )
                if post_rescue_memory_this_chunk and not post_rescue_handoff_applied:
                    _prepare_post_rescue_overlap_handoff(
                        chunk_kw,
                        mlpot_ctx=mlpot_ctx,
                    )
                    post_rescue_handoff_applied = True
                elif early_abort_memory_handoff and mlpot_ctx is not None:
                    _prepare_overlap_chunk_after_restart(mlpot_ctx, restart_read=None)
                elif (
                    mem_handoff
                    and chunk_index > 0
                    and mlpot_ctx is not None
                ):
                    _prepare_overlap_chunk_after_restart(mlpot_ctx, restart_read=None)
                    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                        probe_and_light_resync_if_desync,
                    )

                    probe_and_light_resync_if_desync(
                        mlpot_ctx,
                        context=(
                            f"{overlap_context} chunk {chunk_index + 1}/{n_chunks}"
                        ),
                        silent_charmm=True,
                        verbose=False,
                    )
                if chunk_io is None or chunk_io.restart_write is None:
                    chunk_kw.pop("iunwri", None)

                _harmonize_overlap_chunk_frequencies(
                    chunk_kw,
                    chunk_nstep,
                    loose_pbc=loose_pbc,
                    global_step_start=steps_before_chunk,
                    split_trajectory=split_trajectory,
                )
                suppress_chunk_traj = bool(chunk_kw.get("_suppress_trajectory", False))
                if suppress_chunk_traj or "nsavc" not in chunk_kw:
                    chunk_io = _drop_trajectory_io(chunk_io)
                if (
                    split_trajectory
                    and chunk_io is not None
                    and chunk_io.trajectory is not None
                ):
                    chunk_dcd_paths.append(Path(chunk_io.trajectory))
                if has_restart_read:
                    _prepare_overlap_chunk_after_restart(
                        mlpot_ctx,
                        restart_read=(
                            getattr(chunk_io, "restart_read", None)
                            if chunk_io is not None
                            else None
                        ),
                    )

                chunk_traj_iokw = (
                    trajectory_iokw
                    if (
                        "nsavc" in chunk_kw
                        and not suppress_chunk_traj
                        and (
                            (not split_trajectory and chunk_index == 0)
                            or split_trajectory
                        )
                    )
                    else {}
                )
                cpt_sub_chunk = (
                    _cpt_stability_chunk_nstep(chunk_kw, chunk_nstep)
                    if bool(chunk_kw.get("cpt"))
                    else None
                )
                if cpt_sub_chunk is not None:
                    last_dyn = _run_cpt_stability_subchunked(
                        chunk_kw,
                        chunk_io,
                        overlap_context=(
                            f"overlap ({overlap_context}) chunk "
                            f"{chunk_index + 1}/{n_chunks}"
                        ),
                        rng_base=rng_base,
                        chunk_nstep=cpt_sub_chunk,
                        total_nstep=chunk_nstep,
                        extra_iokw=chunk_traj_iokw,
                        rng_salt_base=_rng_salt_for_dynamics(
                            overlap_context=overlap_context,
                            chunk_index=chunk_index,
                            steps_done=steps_done,
                            retry_count=chunk_retry_count,
                        ),
                        loose_pbc=loose_pbc,
                        log_banner=False,
                        global_step_offset=steps_before_chunk,
                        mlpot_ctx=mlpot_ctx,
                    )
                else:
                    last_dyn = _run_dynamics_chunk(
                        chunk_kw,
                        chunk_io,
                        extra_iokw=chunk_traj_iokw,
                        rng_base=rng_base,
                        rng_salt=_rng_salt_for_dynamics(
                            overlap_context=overlap_context,
                            chunk_index=chunk_index,
                            steps_done=steps_done,
                            retry_count=chunk_retry_count,
                        ),
                    )
                from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                    validate_charmm_dynamics_state_after_chunk,
                )

                validate_charmm_dynamics_state_after_chunk(
                    context=(
                        f"overlap ({overlap_context}) chunk "
                        f"{chunk_index + 1}/{n_chunks}"
                    ),
                )
                chunk_restart_path = (
                    Path(chunk_io.restart_write)
                    if chunk_io is not None
                    and getattr(chunk_io, "restart_write", None) is not None
                    else None
                )
                chunk_state_corrupt = _dynamics_chunk_state_corrupt(
                    overlap_context=(
                        f"overlap ({overlap_context}) chunk {chunk_index + 1}/{n_chunks}"
                    ),
                    restart_path=chunk_restart_path,
                )
                if chunk_io is not None and getattr(chunk_io, "restart_write", None) is not None:
                    _refresh_restart_write_after_chunk(
                        chunk_io.restart_write,
                        final_restart=final_restart,
                    )
                expected_after = steps_before_chunk + chunk_nstep
                reported_steps = _integrated_step_from_restart(
                    chunk_io=chunk_io,
                    final_restart=final_restart,
                    fallback_steps=expected_after,
                    steps_before_chunk=steps_before_chunk,
                )
                from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                    classify_chunk_outcome,
                    patch_restart_global_step,
                    read_restart_last_step,
                )

                chunk_outcome = classify_chunk_outcome(
                    steps_before_chunk=steps_before_chunk,
                    chunk_nstep=chunk_nstep,
                    reported_steps=reported_steps,
                    chunk_state_corrupt=chunk_state_corrupt,
                    restart_path=chunk_restart_path,
                    overlap_context=overlap_context,
                    chunk_index=chunk_index,
                    n_chunks=n_chunks,
                )
                steps_done = chunk_outcome.integrated_step
                if (
                    chunk_io is not None
                    and getattr(chunk_io, "restart_write", None) is not None
                    and steps_done >= expected_after - 1
                ):
                    restart_path = Path(chunk_io.restart_write)
                    if restart_path.is_file():
                        header_step = read_restart_last_step(restart_path)
                        if header_step is None or header_step < expected_after - 1:
                            patch_restart_global_step(restart_path, steps_done)
                if final_restart is not None and chunk_io is not None:
                    _overlap_refresh_or_validate_scratch_restart(
                        chunk_io.restart_write,
                        final_restart=final_restart,
                        chunk_index=chunk_index,
                        n_chunks=n_chunks,
                        overlap_context=overlap_context,
                        mlpot_ctx=mlpot_ctx,
                        memory_handoff=overlap_memory_handoff or post_rescue_memory_this_chunk,
                    )
                    for restart_path in {
                        Path(p)
                        for p in (chunk_io.restart_write, final_restart)
                        if p is not None
                    }:
                        if not restart_path.is_file():
                            continue
                        header_step = read_restart_last_step(restart_path)
                        if header_step is None or header_step < expected_after - 1:
                            patch_restart_global_step(restart_path, steps_done)
                if chunk_outcome.charmm_aborted:
                    print(
                        f"overlap ({overlap_context}): CHARMM abort at step "
                        f"{steps_done}/{total_nstep}",
                        flush=True,
                    )
                    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
                        attempt_overlap_early_abort_recovery,
                    )

                    recovery = attempt_overlap_early_abort_recovery(
                        overlap,
                        chunk_nstep=chunk_nstep,
                        steps_done=steps_done,
                        steps_before_chunk=steps_before_chunk,
                        overlap_context=overlap_context,
                        overlap_run_state_dir=overlap_run_state_dir,
                        overlap_restart_read=overlap_restart_read_for_chunk,
                        segment_restart_read=segment_restart_read,
                        mlpot_ctx=mlpot_ctx,
                        cpt=bool(chunk_kw.get("cpt")),
                        chunk_index=chunk_index,
                    )
                    if (
                        recovery.ok
                        and chunk_retry_count < _MAX_EARLY_ABORT_CHUNK_RETRIES
                    ):
                        chunk_retry_count += 1
                        early_abort_memory_handoff = False
                        early_abort_restart_handoff = False
                        rescued_overlap = False
                        geometry_violation = False
                        if overlap is not None:
                            from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
                                probe_dynamics_geometry_violation,
                            )

                            geometry_violation = probe_dynamics_geometry_violation(
                                overlap,
                                context=(
                                    f"{overlap_context} after early-abort recovery"
                                ),
                                step=steps_before_chunk,
                            )
                        if geometry_violation and mlpot_ctx is not None and overlap is not None:
                            from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                                finalize_overlap_rescue_for_dynamics,
                            )

                            finalize_overlap_rescue_for_dynamics(
                                mlpot_ctx,
                                overlap,
                                context=(
                                    f"{overlap_context} after early-abort recovery "
                                    f"(step {steps_before_chunk})"
                                ),
                            )
                            from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
                                save_stabilized_overlap_rescue_snapshot,
                            )

                            save_stabilized_overlap_rescue_snapshot(
                                mlpot_ctx,
                                overlap,
                                label=(
                                    f"{overlap_context} after early-abort recovery "
                                    f"(step {steps_before_chunk})"
                                ),
                            )
                            if recovery.source != "memory":
                                _, rescued_overlap = check_dynamics_overlap(
                                    overlap,
                                    context=(
                                        f"{overlap_context} after early-abort recovery"
                                    ),
                                    step=steps_before_chunk,
                                    mlpot_ctx=mlpot_ctx,
                                )
                        elif mlpot_ctx is not None:
                            mlpot_ctx.reregister_mlpot(verbose=False)
                        use_readyn_handoff = (
                            chunk_io is not None
                            and n_chunks > 1
                            and not bool(chunk_kw.get("cpt"))
                            and rescued_overlap
                        )
                        if use_readyn_handoff:
                            chunk_io = _materialize_post_rescue_restart_handoff(
                                chunk_io,
                                chunk_kw,
                                steps_done=steps_before_chunk,
                                mlpot_ctx=mlpot_ctx,
                                overlap=overlap,
                                overlap_context=overlap_context,
                            )
                            pending_readyn_chunk_io = chunk_io
                            post_rescue_handoff_applied = True
                        else:
                            if (
                                bool(chunk_kw.get("cpt"))
                                and recovery.source != "memory"
                                and mlpot_ctx is not None
                            ):
                                _prepare_post_rescue_overlap_handoff(
                                    chunk_kw,
                                    mlpot_ctx=mlpot_ctx,
                                )
                                post_rescue_handoff_applied = True
                            early_abort_memory_handoff = True
                        steps_done = steps_before_chunk
                        abort_reason = "charmm_aborted"
                        rerun_chunk = True
                        retry_reason = (
                            "geometry recovery"
                            if geometry_violation or rescued_overlap
                            else "state reload"
                        )
                        print(
                            f"overlap ({overlap_context}): retrying chunk "
                            f"{chunk_index + 1}/{n_chunks} after {retry_reason} "
                            f"(attempt {chunk_retry_count}/{_MAX_EARLY_ABORT_CHUNK_RETRIES})",
                            flush=True,
                        )
                        continue
                    print(
                        f"overlap ({overlap_context}): early-abort will salvage "
                        f"(recovery.ok={recovery.ok}, "
                        f"retries={chunk_retry_count}/{_MAX_EARLY_ABORT_CHUNK_RETRIES})",
                        flush=True,
                    )
                    salvaged = _salvage_overlap_segment_progress(
                        chunk_io=chunk_io,
                        final_restart=final_restart,
                        steps_before_chunk=steps_before_chunk,
                        overlap_context=overlap_context,
                        mlpot_ctx=mlpot_ctx,
                        overlap_restart_read=overlap_restart_read_for_chunk,
                        total_nstep=total_nstep,
                    )
                    if salvaged is not None:
                        steps_done = int(salvaged)
                        salvaged_partial = True
                        abort_reason = "charmm_aborted"
                        segment_aborted = False
                        rerun_chunk = False
                        chunk_index = n_chunks
                        break
                    segment_aborted = True
                else:
                    print(
                        f"overlap ({overlap_context}): geometry check at step {steps_done}",
                        flush=True,
                    )
                    _, rescued = check_dynamics_overlap(
                        overlap,
                        context=overlap_context,
                        step=steps_done,
                        mlpot_ctx=mlpot_ctx,
                    )
                    if not rescued and overlap is not None and overlap.extent_enabled:
                        from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
                            refresh_overlap_prior_segment_restart,
                        )

                        refresh_path = (
                            final_restart
                            if final_restart is not None
                            else (
                                chunk_io.restart_write
                                if chunk_io is not None
                                else (io.restart_write if io is not None else None)
                            )
                        )
                        overlap = refresh_overlap_prior_segment_restart(
                            overlap,
                            restart_path=refresh_path,
                        )
                    if rescued and chunk_io is not None and chunk_io.restart_write is not None:
                        from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
                            finalize_overlap_rescue_for_dynamics,
                        )
                        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                            patch_restart_global_step,
                        )

                        _refresh_restart_write_after_chunk(
                            chunk_io.restart_write,
                            final_restart=final_restart,
                        )
                        patch_restart_global_step(
                            Path(chunk_io.restart_write),
                            steps_done,
                        )
                        if mlpot_ctx is not None and overlap is not None:
                            finalize_overlap_rescue_for_dynamics(
                                mlpot_ctx,
                                overlap,
                                context=f"{overlap_context} at step {steps_done}",
                            )
                            from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
                                save_stabilized_overlap_rescue_snapshot,
                            )

                            save_stabilized_overlap_rescue_snapshot(
                                mlpot_ctx,
                                overlap,
                                label=f"{overlap_context} at step {steps_done}",
                            )
                        if chunk_index + 1 < n_chunks:
                            chunk_io, in_memory = _apply_post_rescue_overlap_handoff(
                                chunk_io,
                                chunk_kw,
                                steps_done=steps_done,
                                mlpot_ctx=mlpot_ctx,
                                overlap=overlap,
                                overlap_context=overlap_context,
                            )
                            if in_memory:
                                post_rescue_in_memory_mode = True
                            else:
                                pending_readyn_chunk_io = chunk_io
                            post_rescue_handoff_applied = True
                            if not in_memory:
                                any_post_rescue_restart_handoff = True
                        else:
                            _refresh_segment_restart_after_overlap_rescue(
                                chunk_io.restart_write,
                                chunk_kw,
                                mlpot_ctx=mlpot_ctx,
                                overlap=overlap,
                            )
                            print(
                                f"overlap ({overlap_context}): post-rescue restart refreshed "
                                f"at global step {steps_done} (fresh velocities; segment "
                                f"complete; no extra dyna)",
                                flush=True,
                            )
                ml_f = None
                if mlpot_ctx is not None:
                    py_model = getattr(mlpot_ctx, "pyCModel", None)
                    ml_f = getattr(py_model, "_last_ml_forces", None)
                from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
                    maybe_record_forces,
                )

                maybe_record_forces(steps_done, ml_forces=ml_f)
                if final_restart is not None and chunk_io is not None:
                    _overlap_refresh_or_validate_scratch_restart(
                        chunk_io.restart_write,
                        final_restart=final_restart,
                        chunk_index=chunk_index,
                        n_chunks=n_chunks,
                        overlap_context=overlap_context,
                        mlpot_ctx=mlpot_ctx,
                        memory_handoff=overlap_memory_handoff or post_rescue_memory_this_chunk,
                    )
                if (
                    not segment_aborted
                    and overlap_run_state_dir is not None
                    and overlap_run_state_every_chunks > 0
                    and (chunk_index + 1) % int(overlap_run_state_every_chunks) == 0
                ):
                    from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
                        maybe_save_overlap_run_state,
                    )

                    refresh_path = (
                        chunk_io.restart_write
                        if chunk_io is not None
                        else (io.restart_write if io is not None else None)
                    )
                    maybe_save_overlap_run_state(
                        overlap_run_state_dir,
                        step=steps_done,
                        segment=overlap_context,
                        chunk_index=chunk_index,
                        restart_path=refresh_path,
                    )
                if segment_aborted:
                    raise RuntimeError(
                        f"overlap ({overlap_context}): dynamics aborted after chunk "
                        f"{chunk_index + 1}/{n_chunks} at step {steps_done}/{total_nstep} "
                        "(geometry recovery failed or chunk too short)"
                    )
            chunk_index += 1
        completed = steps_done >= total_nstep - 1 and not salvaged_partial
        if (
            any_post_rescue_restart_handoff
            and io is not None
            and io.restart_write is not None
            and steps_done >= total_nstep - 1
        ):
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                patch_restart_global_step,
            )

            _refresh_restart_write_after_chunk(
                io.restart_write,
                final_restart=final_restart,
            )
            patch_restart_global_step(Path(io.restart_write), steps_done)
        if (
            io is not None
            and io.restart_write is not None
            and steps_done >= total_nstep - 1
        ):
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                patch_restart_global_step,
                read_restart_last_step,
            )

            restart_path = Path(io.restart_write)
            if restart_path.is_file():
                header_step = read_restart_last_step(restart_path)
                if header_step is None or header_step < total_nstep - 1:
                    patch_restart_global_step(restart_path, steps_done)
        if split_trajectory and chunk_dcd_paths:
            print(
                f"overlap ({overlap_context}): kept {len(chunk_dcd_paths)} "
                "per-chunk DCD file(s)",
                flush=True,
            )
    finally:
        if completed:
            _cleanup_overlap_restart_slots(io)
        elif io is not None and io.restart_write is not None:
            print(
                "overlap: preserving scratch restart files after failed overlap handling "
                f"for {io.restart_write}",
                flush=True,
            )
        for f in trajectory_files:
            f.close()
    return _overlap_dynamics_result(
        last_dyn,
        integrated_step=steps_done,
        completed_full=completed,
        salvaged_partial=salvaged_partial,
        abort_reason=abort_reason,
    )


def open_minimize_dcd(path: PathLike, *, unit: int = 51) -> Any:
    """Open a DCD file for minimization trajectory output (``iuncrd``)."""
    pycharmm, *_ = _import_pycharmm_modules()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return pycharmm.CharmmFile(
        file_name=str(p),
        file_unit=unit,
        formatted=False,
        read_only=False,
    )


def _prepare_charmm_mm_minimize_list_frequencies(
    pycharmm: Any,
    *,
    use_pbc: bool,
    nstep: int,
    inbfrq: int = 50,
) -> int:
    """Align NB/image list freqs before CGENFF SD (CHARMM FINCYC constraints).

    ``minimize.run_sd`` updates ``inbfrq`` via ``MinOpts`` but not ``imgfrq``. A
    leftover ``imgfrq=-1`` (CHARMM default) with ``inbfrq=50`` triggers BOMLev -2:
    "IMGFRQ is not a multiple of INBFRQ".
    """
    inb = _harmonize_dynamics_frequency(int(inbfrq), max(1, int(nstep)))
    if use_pbc:
        pycharmm.nbonds.set_inbfrq(inb)
        pycharmm.nbonds.set_imgfrq(inb)
    else:
        pycharmm.nbonds.set_inbfrq(inb)
        pycharmm.nbonds.set_imgfrq(0)
    return inb


def _prepare_mlpot_sd_list_frequencies(pycharmm: Any, *, sd_kw: dict[str, Any]) -> None:
    """Align image/HB list freqs with MLpot SD ``inbfrq=0`` (CHARMM FINCYC constraint).

    ``minimize.run_sd`` only sets ``inbfrq`` / ``ihbfrq`` via ``MinOpts``; a non-zero
    ``imgfrq`` left from production ``dyna`` kwargs triggers BOMLev -2:
    "INBFRQ is zero when IMGFRQ is not".
    """
    if int(sd_kw.get("inbfrq", 1)) != 0:
        return
    pycharmm.nbonds.set_imgfrq(0)
    pycharmm.nbonds.set_inbfrq(0)


def _sd_kwargs_from_config(config: MinimizeWithMlpotConfig) -> dict[str, Any]:
    """Common ``minimize.run_sd`` / ``run_abnr`` settings including optional DCD trajectory."""
    kw: dict[str, Any] = {
        "nstep": config.nstep,
        "nprint": config.nprint,
        "tolenr": config.tolenr,
        "tolgrd": config.tolgrd,
        # Do not rebuild nonbond/MLpot pair lists each mini step (mlpot_update can
        # segfault when all atoms are ML and the atom-pair list is empty). Sync once
        # after mini via ``sync_charmm_lists_after_mini`` (CHARMM UPDATE).
        "inbfrq": 0,
        "ihbfrq": 0,
    }
    if config.save and config.dcd_path is not None and config.dcd_nsavc > 0:
        kw["iuncrd"] = config.dcd_unit
        kw["nsavc"] = config.dcd_nsavc
    return kw


def _resolved_abnr_nstep(config: MinimizeWithMlpotConfig) -> int:
    if config.nstep_abnr is not None:
        return max(0, int(config.nstep_abnr))
    return max(0, int(config.nstep))


def _maybe_promote_mlpot_jax_after_sd(config: MinimizeWithMlpotConfig) -> None:
    """Promote deferred JAX to GPU after MLpot SD (MPI-linked CHARMM defer path)."""
    if config.mlpot_ctx is None:
        return
    pyCModel = getattr(config.mlpot_ctx, "pyCModel", None)
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotModel,
        maybe_warmup_deferred_decomposed_mlpot,
    )

    if not isinstance(pyCModel, DecomposedMlpotModel):
        return
    n_mono = int(
        getattr(pyCModel, "_n_monomers", 0)
        or getattr(config.mlpot_ctx, "n_monomers", 0)
        or 0
    )
    if n_mono <= 1:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    cell = getattr(config.mlpot_ctx, "cubic_box_side_A", None)
    maybe_warmup_deferred_decomposed_mlpot(
        pyCModel,
        get_charmm_positions_array(),
        cell=float(cell) if cell is not None else None,
        n_monomers=n_mono,
        verbose=config.verbose,
    )


def _run_mlpot_sd_then_abnr(
    minimize: Any,
    pycharmm: Any,
    config: MinimizeWithMlpotConfig,
    sd_kw: dict[str, Any],
    *,
    pass_label: str,
) -> MlpotSdChunkResult:
    """One SD pass followed by ABNR (same ``nstep`` by default) under MLpot."""
    sd_result = _run_minimize_in_chunks(
        minimize,
        pycharmm,
        config,
        sd_kw,
        total_nstep=int(config.nstep),
        pass_label=pass_label,
        method="SD",
        run_attr="run_sd",
    )
    if not sd_result.completed:
        return sd_result

    abnr_n = _resolved_abnr_nstep(config)
    if abnr_n <= 0:
        return sd_result

    abnr_kw = {
        "nprint": config.nprint,
        "tolenr": config.tolenr,
        "tolgrd": config.tolgrd,
        "inbfrq": int(sd_kw.get("inbfrq", 0)),
        "ihbfrq": int(sd_kw.get("ihbfrq", 0)),
    }
    return _run_minimize_in_chunks(
        minimize,
        pycharmm,
        config,
        abnr_kw,
        total_nstep=abnr_n,
        pass_label=pass_label,
        method="ABNR",
        run_attr="run_abnr",
    )


def minimize_with_mlpot(
    config: MinimizeWithMlpotConfig,
) -> bool:
    """Run SD minimization with optional fixed ML atoms; optional trajectory/output.

    When ``dcd_path`` is set and ``save=True``, frames are written during SD via
    CHARMM ``iuncrd`` / ``nsavc`` (see ``pycharmm.minimize.MinOpts``).

    Returns True if minimization ran, False if skipped because CRD exists.
    """
    pycharmm, cons_fix, energy, minimize, *_ = _import_pycharmm_modules()

    crd_path = Path(config.crd_path) if config.crd_path else None
    if config.skip_if_crd_exists and crd_path is not None and crd_path.exists():
        load_minimized_coordinates(crd_path)
        if config.show_energy:
            _maybe_show_energy(True)
        return False

    if config.reference_positions is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(config.reference_positions)

    _ensure_domdec_off_for_mlpot_energy(context="MLpot SD minimize")

    dcd_file = None
    if config.save and config.dcd_path is not None and config.dcd_nsavc > 0:
        dcd_file = open_minimize_dcd(config.dcd_path, unit=config.dcd_unit)

    sd_kw = _sd_kwargs_from_config(config)

    pyC_model = (
        getattr(config.mlpot_ctx, "pyCModel", None) if config.mlpot_ctx is not None else None
    )
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        charmm_mlpot_sd_jax_cpu_guard,
        materialize_deferred_mlpot_jax_before_sd,
    )

    try:
        if config.verbose and config.show_energy:
            print("CHARMM energy before minimization:")
            _maybe_show_energy(True)
        if config.mlpot_ctx is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                prepare_mlpot_hybrid_state_for_sd,
            )

            prepare_mlpot_hybrid_state_for_sd(
                config.mlpot_ctx,
                grms_limit=config.pre_sd_bonded_recovery_grms_kcalmol_A,
                energy_limit=config.pre_sd_bonded_recovery_energy_kcalmol,
                bonded_recovery_nstep=config.pre_sd_bonded_recovery_nstep,
                bonded_recovery_backend=config.pre_sd_bonded_recovery_backend,
                bonded_recovery_verbose=config.verbose,
                bonded_recovery_show_energy=config.show_energy,
                bonded_recovery_nprint=config.nprint,
                bonded_recovery_tolenr=config.tolenr,
                bonded_recovery_tolgrd=config.tolgrd,
                context_prefix="Pre-SD",
                verbose=config.verbose,
                calculator_minimize=config.calculator_pre_minimize,
                calculator_minimize_steps=config.calculator_minimize_steps,
                calculator_minimize_fmax_ev_a=config.calculator_minimize_fmax_ev_a,
                calculator_bfgs_maxstep=config.calculator_bfgs_maxstep,
                quiet_bfgs=config.quiet_bfgs,
                calculator_fire_steps=config.calculator_fire_steps,
                calculator_fire_fmax_ev_a=config.calculator_fire_fmax_ev_a,
                calculator_fire_maxstep=config.calculator_fire_maxstep,
            )
            baseline_raw = getattr(config.mlpot_ctx, "sd_watchdog_baseline_grms", None)
            baseline: float | None = None
            if baseline_raw is not None:
                try:
                    candidate = float(baseline_raw)
                except (TypeError, ValueError):
                    candidate = float("nan")
                if np.isfinite(candidate):
                    baseline = float(candidate)
            if baseline is not None:
                config.sd_watchdog_initial_grms = baseline
                if config.verbose:
                    from mmml.utils.prep_ladder_report import PrepMetrics, emit_prep_checkpoint

                    emit_prep_checkpoint(
                        "MLpot SD watchdog baseline",
                        PrepMetrics.from_mlpot(
                            config.mlpot_ctx,
                            hybrid_grms=baseline,
                        ),
                        note="post calculator pre-minimize",
                        quiet=False,
                    )
        if config.mlpot_ctx is not None:
            materialize_deferred_mlpot_jax_before_sd(
                config.mlpot_ctx,
                verbose=config.verbose,
                force_ener_probe=True,
            )
        with charmm_mlpot_sd_jax_cpu_guard(pyC_model):
            sd_result = _run_mlpot_sd_then_abnr(
                minimize,
                pycharmm,
                config,
                sd_kw,
                pass_label="pass 1 (free, all atoms)",
            )
            if not sd_result.completed:
                if sd_result.stalled:
                    grms_txt = (
                        f"{sd_result.last_grms:.4f}"
                        if sd_result.last_grms is not None
                        else "?"
                    )
                    print(
                        "MLpot SD pass 1 stalled: hybrid GRMS plateaued "
                        f"(GRMS≈{grms_txt} kcal/mol/Å); CHARMM SD ineffective on "
                        "geometry_stress — skipping remaining SD chunks",
                        flush=True,
                    )
                    if (
                        config.mlpot_ctx is not None
                        and config.pre_sd_bonded_recovery_grms_kcalmol_A is not None
                    ):
                        from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
                            maybe_run_density_prep_ladder_for_mlpot,
                        )

                        ladder_grms, ran_ladder = maybe_run_density_prep_ladder_for_mlpot(
                            config.mlpot_ctx,
                            max_grms=float(config.pre_sd_bonded_recovery_grms_kcalmol_A),
                            context="Post-SD-stall density prep ladder",
                            quiet=not config.verbose,
                        )
                        if ran_ladder and config.verbose:
                            print(
                                f"Post-SD-stall density prep ladder: GRMS→{ladder_grms:.4f} "
                                f"kcal/mol/Å (limit "
                                f"{float(config.pre_sd_bonded_recovery_grms_kcalmol_A):.4f})",
                                flush=True,
                            )
                elif sd_result.rolled_back:
                    grms_txt = (
                        f"{sd_result.last_grms:.4f}"
                        if sd_result.last_grms is not None
                        else "?"
                    )
                    print(
                        "MLpot SD pass 1 partial: watchdog stopped further chunks; "
                        f"geometry restored (GRMS≈{grms_txt} kcal/mol/Å)",
                        flush=True,
                    )
                else:
                    raise RuntimeError(
                        "MLpot SD pass 1 stopped early: hybrid GRMS watchdog triggered "
                        "(lists/forces diverged during inbfrq=0 minimization)"
                    )
            sync_charmm_lists_after_mini(quiet=not config.verbose)
            invalidate_mlpot_calculator_caches(config.mlpot_ctx)
            if config.verbose and config.mlpot_ctx is not None:
                from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                    refresh_mlpot_energy_and_grms,
                )

                refresh_mlpot_energy_and_grms(
                    config.mlpot_ctx,
                    context="Post MLpot SD pass 1",
                    reregister=False,
                )
            if config.verbose and config.show_energy:
                print("CHARMM energy after SD pass 1 (free):")
                _maybe_show_energy(True)

            if config.fixed_ml_selection is not None:
                n_fix = len(config.fixed_ml_selection.get_atom_indexes())
                cons_fix.setup(config.fixed_ml_selection)
                sd_result = _run_mlpot_sd_then_abnr(
                    minimize,
                    pycharmm,
                    config,
                    sd_kw,
                    pass_label=f"pass 2 (cons_fix, {n_fix} atoms)",
                )
                if not sd_result.completed:
                    if sd_result.rolled_back:
                        grms_txt = (
                            f"{sd_result.last_grms:.4f}"
                            if sd_result.last_grms is not None
                            else "?"
                        )
                        print(
                            f"MLpot SD pass 2 partial (cons_fix, {n_fix} atoms): "
                            "watchdog stopped further chunks; "
                            f"geometry restored (GRMS≈{grms_txt} kcal/mol/Å)",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(
                            f"MLpot SD pass 2 (cons_fix, {n_fix} atoms) stopped early: "
                            "hybrid GRMS watchdog triggered"
                        )
                sync_charmm_lists_after_mini(quiet=not config.verbose)
                invalidate_mlpot_calculator_caches(config.mlpot_ctx)
                if config.verbose and config.mlpot_ctx is not None:
                    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                        refresh_mlpot_energy_and_grms,
                    )

                    refresh_mlpot_energy_and_grms(
                        config.mlpot_ctx,
                        context="Post MLpot SD pass 2",
                        reregister=False,
                    )
                if config.verbose and config.show_energy:
                    print("CHARMM energy after SD pass 2 (constrained):")
                    _maybe_show_energy(True)
                cons_fix.turn_off()

        if config.mlpot_ctx is not None:
            invalidate_mlpot_calculator_caches(config.mlpot_ctx)

        _maybe_promote_mlpot_jax_after_sd(config)

        if config.save:
            from mmml.interfaces.pycharmmInterface.mlpot.setup import (
                resolve_export_positions,
            )

            export_pos = resolve_export_positions(
                pyCModel=config.pyCModel,
                reference_positions=config.reference_positions,
            )
            save_minimization_results(
                pdb_path=config.pdb_path,
                crd_path=config.crd_path,
                psf_path=config.psf_path,
                energy_json_path=config.energy_json_path,
                xyz_path=config.xyz_path,
                positions=export_pos,
                title=config.title,
                show_energy=config.show_energy,
            )
        elif config.show_energy:
            _maybe_show_energy(True)

        if config.test_first is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.derivative_test import (
                run_post_minimize_derivative_tests,
            )

            from mmml.interfaces.pycharmmInterface.mlpot.setup import (
                resolve_export_positions,
            )

            export_pos = resolve_export_positions(
                pyCModel=config.pyCModel,
                reference_positions=config.reference_positions,
            )
            run_post_minimize_derivative_tests(
                config.test_first,
                pyCModel=config.pyCModel,
                positions=export_pos,
            )
    finally:
        if dcd_file is not None:
            dcd_file.close()
    return True


def charmm_energy_terms() -> dict[str, float]:
    """Current CHARMM energy row as ``{term: value}`` (kcal/mol)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        should_skip_charmm_energy_show,
    )

    if should_skip_charmm_energy_show():
        return {}
    _, _, energy, *_ = _import_pycharmm_modules()
    df = energy.get_energy()
    row = df.iloc[0].to_dict()
    terms: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, np.floating)):
            terms[str(key)] = float(value)
    return terms


def save_minimization_results(
    *,
    pdb_path: Optional[PathLike] = None,
    crd_path: Optional[PathLike] = None,
    psf_path: Optional[PathLike] = None,
    energy_json_path: Optional[PathLike] = None,
    xyz_path: Optional[PathLike] = None,
    positions: Optional[np.ndarray] = None,
    title: str = "Mini SD",
    show_energy: bool = False,
) -> dict[str, Path]:
    """Write minimized coordinates and optional PSF / energy summary.

    If ``positions`` is given, sync to CHARMM before native PDB/CRD writes and use
    for XYZ. This avoids empty exports when only the main coordinate set is populated.

    Returns dict of output kind -> path for files that were written.
    """
    _, _, energy, _, _, write = _import_pycharmm_modules()
    written: dict[str, Path] = {}

    if positions is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(positions)

    if pdb_path is not None:
        p = Path(pdb_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        write.coor_pdb(str(p), title=title)
        written["pdb"] = p.resolve()

    if crd_path is not None:
        p = Path(crd_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        write.coor_card(str(p), title=title)
        written["crd"] = p.resolve()

    if psf_path is not None:
        import pycharmm.write as pywrite

        p = Path(psf_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pywrite.psf_card(str(p))
        written["psf"] = p.resolve()

    if energy_json_path is not None:
        p = Path(energy_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"title": title, "energy_kcal_mol": charmm_energy_terms()}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written["energy_json"] = p.resolve()

    if xyz_path is not None:
        try:
            import ase
            import ase.io
            from mmml.interfaces.pycharmmInterface.mlpot.setup import (
                get_charmm_positions_array,
            )
            from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

            pos = (
                np.asarray(positions, dtype=float)
                if positions is not None
                else get_charmm_positions_array()
            )
            numbers = np.asarray(get_Z_from_psf(), dtype=int)
            p = Path(xyz_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            ase.io.write(str(p), ase.Atoms(numbers=numbers, positions=pos))
            written["xyz"] = p.resolve()
        except Exception as exc:
            raise RuntimeError(f"Failed to write XYZ to {xyz_path}: {exc}") from exc

    if show_energy:
        _maybe_show_energy(True)
    return written


def write_minimized_coordinates(
    *,
    pdb_path: Optional[PathLike] = None,
    crd_path: Optional[PathLike] = None,
    title: str = "Mini SD",
) -> None:
    *_, write = _import_pycharmm_modules()
    if pdb_path is not None:
        write.coor_pdb(str(pdb_path), title=title)
    if crd_path is not None:
        write.coor_card(str(crd_path), title=title)


def load_minimized_coordinates(crd_path: PathLike) -> None:
    """Load optimized coords from a CRD card (preferred over PDB for ML exclusions)."""
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    *_, read, _write = _import_pycharmm_modules()
    path = Path(crd_path)
    if not path.exists():
        raise FileNotFoundError(f"CRD not found: {path}")
    with charmm_relaxed_bomlev():
        read.coor_card(str(path))
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        clear_comparison_coordinates,
    )

    clear_comparison_coordinates()


def production_restart_chain(
    data_dir: PathLike,
    *,
    n_segments: int = 10,
    prefix: str = "dyna",
    equi_restart: str = "equi.res",
) -> list[CharmmTrajectoryFiles]:
    """Build restart/trajectory file triples for chained production.

    Segment 0 reads ``equi_restart``; segment ``i>0`` reads ``{prefix}.{i-1}.res``.
    """
    return npt_restart_chain(
        data_dir,
        n_segments=n_segments,
        prefix=prefix,
        initial_restart=Path(data_dir) / equi_restart,
    )
