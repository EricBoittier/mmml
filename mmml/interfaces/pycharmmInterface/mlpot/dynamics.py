"""Minimization and MD workflows with MLpot active (PyCHARMM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

NptThermostat = Literal["hoover", "berendsen"]

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.derivative_test import TestFirstConfig
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        OverlapRescueConfig,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

import numpy as np

PathLike = Union[str, Path]


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
    restart_read_unit: int = 3
    restart_write_unit: int = 2
    trajectory_unit: int = 1

    def open_for_run(self) -> tuple[list[Any], dict[str, int]]:
        """Open CharmmFile handles; returns ``(open_files, dynamics_unit_kwargs)``."""
        import pycharmm

        open_files: list[Any] = []
        kw: dict[str, int] = {}
        if self.restart_read is not None:
            f = pycharmm.CharmmFile(
                file_name=str(self.restart_read),
                file_unit=self.restart_read_unit,
                formatted=True,
                read_only=True,
            )
            open_files.append(f)
            kw["iunrea"] = self.restart_read_unit
        if self.restart_write is not None:
            f = pycharmm.CharmmFile(
                file_name=str(self.restart_write),
                file_unit=self.restart_write_unit,
                formatted=True,
                read_only=False,
            )
            open_files.append(f)
            kw["iunwri"] = self.restart_write_unit
        if self.trajectory is not None:
            f = pycharmm.CharmmFile(
                file_name=str(self.trajectory),
                file_unit=self.trajectory_unit,
                formatted=False,
                read_only=False,
            )
            open_files.append(f)
            kw["iuncrd"] = self.trajectory_unit
        return open_files, kw


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

    apply_charmm_mm_block()
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

    sd_kw = {
        "nstep": max(1, int(config.nstep_sd)),
        "nprint": max(1, int(config.nprint)),
        "tolenr": float(config.tolenr),
        "tolgrd": float(config.tolgrd),
        "inbfrq": 50,
        "ihbfrq": 50,
    }
    if config.verbose and config.show_energy:
        _maybe_show_energy(True)
    if config.nstep_sd > 0:
        if config.verbose:
            print(f"CHARMM MM SD: nstep={config.nstep_sd}", flush=True)
        minimize.run_sd(**sd_kw)
    if config.nstep_abnr > 0:
        if config.verbose:
            print(f"CHARMM MM ABNR: nstep={config.nstep_abnr}", flush=True)
        minimize.run_abnr(
            nstep=int(config.nstep_abnr),
            tolenr=float(config.tolenr),
            tolgrd=float(config.tolgrd),
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
    """Unset MLpot USER, run MM work, then reregister MLpot + hybrid BLOCK."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    if not isinstance(ctx, MlpotContext):
        raise TypeError("ctx must be MlpotContext")
    ctx.unset()
    try:
        return fn()
    finally:
        ctx.reregister_mlpot()


@dataclass
class BondedMmMiniConfig:
    """Short bonded-only SD with MLpot temporarily detached (pure CHARMM bonded)."""

    nstep_sd: int = 50
    nprint: int = 10
    tolenr: float = 1e-3
    tolgrd: float = 1e-3
    verbose: bool = True
    show_energy: bool = False


def _with_mlpot_block_restored(ctx: "MlpotContext", fn):
    """Run ``fn`` with full/bonded MM BLOCK, then restore hybrid MLpot BLOCK."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_mlpot_energy_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    if not isinstance(ctx, MlpotContext):
        raise TypeError("ctx must be MlpotContext")
    if ctx.ml_selection is None:
        raise RuntimeError("MlpotContext missing ml_selection for BLOCK restore")
    try:
        return fn()
    finally:
        ctx.block_tag = apply_mlpot_energy_block(ctx.ml_selection)


def measure_mm_grms_with_full_block(ctx: "MlpotContext") -> float:
    """MM bonded strain proxy: GRMS (kcal/mol/Å) with full MM BLOCK, MLpot stays on."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms

    def _measure() -> float:
        apply_charmm_mm_block()
        pycharmm, *_ = _import_pycharmm_modules()
        pycharmm.lingo.charmm_script("ENER")
        return float(charmm_grms())

    return _with_mlpot_block_restored(ctx, _measure)


def _bonded_recovery_sd_kwargs(ctx: "MlpotContext", config: BondedMmMiniConfig) -> dict[str, Any]:
    """SD frequencies compatible with vacuum vs PBC (CHARMM rejects inbfrq=0 + imgfrq≠0)."""
    kw: dict[str, Any] = {
        "nstep": max(1, int(config.nstep_sd)),
        "nprint": max(1, int(config.nprint)),
        "tolenr": float(config.tolenr),
        "tolgrd": float(config.tolgrd),
    }
    if ctx.use_pbc:
        # PBC image list is active (imgfrq set by nbonds); inbfrq=0 is invalid — use heuristic.
        kw.update({"inbfrq": -1, "ihbfrq": 50})
    else:
        kw.update({"inbfrq": 0, "ihbfrq": 0})
    return kw


def minimize_bonded_mm_recovery(
    ctx: "MlpotContext",
    config: BondedMmMiniConfig,
) -> float | None:
    """Bonded-only rescue SD (BOND/ANGL/DIHE); MLpot detached for pure CHARMM minimization."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_bonded_mm_only_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        MlpotContext,
        get_charmm_positions_array,
    )

    if not isinstance(ctx, MlpotContext):
        raise TypeError("ctx must be MlpotContext")

    def _run_sd() -> float | None:
        apply_bonded_mm_only_block()
        pycharmm, cons_fix, *_ = _import_pycharmm_modules()
        minimize = _import_pycharmm_modules()[3]
        if config.nstep_sd <= 0:
            pycharmm.lingo.charmm_script("ENER")
            return float(charmm_grms())

        pycharmm.lingo.charmm_script("ENER")
        angl_before = charmm_bonded_term_kcalmol("ANGL")
        bond_before = charmm_bonded_term_kcalmol("BOND")
        grms_before = float(charmm_grms())
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
        minimize.run_sd(**sd_kw)
        pycharmm.lingo.charmm_script("ENER")
        grms = float(charmm_grms())
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


def minimize_overlap_rescue(
    ctx: "MlpotContext",
    config: "OverlapRescueConfig",
) -> float | None:
    """Bonded+VDW rescue SD/ABNR (NBXMOD 2); MLpot detached so CHARMM VDW/BOND apply."""
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_vdw_recovery_block,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
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
    if not isinstance(ctx, MlpotContext):
        raise TypeError("ctx must be MlpotContext")

    def _run_rescue() -> float | None:
        apply_bonded_vdw_recovery_block()
        apply_recovery_nbonds(ctx)
        pycharmm, cons_fix, *_ = _import_pycharmm_modules()
        minimize = _import_pycharmm_modules()[3]
        grms_before = float(charmm_grms())
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
                pycharmm.lingo.charmm_script("ENER")
                minimize.run_sd(**sd_kw)
            if config.nstep_abnr > 0:
                minimize.run_abnr(
                    nstep=int(config.nstep_abnr),
                    tolenr=float(config.tolenr),
                    tolgrd=float(config.tolgrd),
                )
            pycharmm.lingo.charmm_script("ENER")
            grms = float(charmm_grms())
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
    title: str = "Mini SD"
    skip_if_crd_exists: bool = True
    show_energy: bool = False
    verbose: bool = False
    test_first: Optional["TestFirstConfig"] = None


def _import_pycharmm_modules():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.cons_fix as cons_fix
    import pycharmm.energy as energy
    import pycharmm.minimize as minimize
    import pycharmm.read as read
    import pycharmm.write as write

    return pycharmm, cons_fix, energy, minimize, read, write


def _base_dyn_kwargs(
    *,
    timestep: float,
    nstep: int,
    nsavc: int,
    inbfrq: int = -1,
    ihbfrq: int = 50,
    ilbfrq: int = 50,
    imgfrq: int = 50,
    ixtfrq: int = 1000,
    nprint: int = 100,
    iprfrq: int = 500,
    isvfrq: int = 1000,
    ntrfrq: int = 1000,
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


def build_heat_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 10.0,
    save_interval_ps: float = 0.1,
    temp: float = 300.0,
    echeck: float = 100.0,
) -> dict[str, Any]:
    """NVT heating dict for ``DynamicsScript`` (CHARMM + MLpot)."""
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
            "verlet": True,
            "new": True,
            "start": True,
            "ihtfrq": 40,
            "TEMINC": 1,
            "ieqfrq": 1000,
            "firstt": temp / 2.0,
            "finalt": temp,
            "tbath": temp,
        }
    )
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
) -> dict[str, Any]:
    """NVE production-style dict (restart from heat)."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=max(1, nprint),
        iprfrq=max(1, iprfrq),
        isvfrq=max(1, isvfrq),
        ntrfrq=0,
        echeck=echeck,
    )
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
    if not restart:
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
    tcoupling: float = 5.0,
) -> None:
    """Attach CPT barostat + temperature control keywords to a dynamics dict."""
    if pmass is None or tmass is None:
        pmass, tmass = compute_cpt_piston_masses()
    kw.update(
        {
            "leap": True,
            "cpt": True,
            "pint pconst pref": pref,
            "pgamma": pgamma,
            "pmass": pmass,
            "ihtfrq": 0,
            "ieqfrq": 0,
        }
    )
    if thermostat == "hoover":
        kw["hoover reft"] = temp
        kw["tmass"] = tmass
        if firstt is not None:
            kw["firstt"] = firstt
    elif thermostat == "berendsen":
        kw["tcons"] = True
        kw["tcoupling"] = tcoupling
        kw["treference"] = temp
    else:
        raise ValueError(f"unknown NPT thermostat: {thermostat!r}")


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


def run_dynamics(dynamics_kwargs: dict[str, Any]) -> Any:
    """Instantiate and run ``pycharmm.DynamicsScript``."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import disable_charmm_domdec

    import pycharmm

    disable_charmm_domdec()
    dyn = pycharmm.DynamicsScript(**dynamics_kwargs)
    dyn.run()
    return dyn


def _valid_restart_file(path: PathLike | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_file() and p.stat().st_size > 0:
        return p
    return None


def _overlap_restart_slot_paths(final_restart: Path) -> tuple[Path, Path]:
    """Alternating scratch restarts so read and write are never the same file."""
    parent = final_restart.parent
    stem = final_restart.stem
    return (
        parent / f"{stem}.overlap_a.res",
        parent / f"{stem}.overlap_b.res",
    )


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
        return _valid_restart_file(io.restart_read), None
    final = Path(io.restart_write)
    if n_chunks <= 1:
        return _valid_restart_file(io.restart_read), final
    slot_a, slot_b = _overlap_restart_slot_paths(final)
    if chunk_index == 0:
        return _valid_restart_file(io.restart_read), slot_a
    if chunk_index == n_chunks - 1:
        prev = slot_a if (chunk_index % 2) == 1 else slot_b
        return prev, final
    if chunk_index % 2 == 1:
        return slot_a, slot_b
    return slot_b, slot_a


def _overlap_chunk_io(
    io: CharmmTrajectoryFiles,
    *,
    chunk_index: int,
    n_chunks: int,
) -> CharmmTrajectoryFiles:
    """Restart I/O for overlap chunking via alternating scratch restarts."""
    rread, rwri = _overlap_chunk_restart_paths(
        io, chunk_index=chunk_index, n_chunks=n_chunks
    )
    return CharmmTrajectoryFiles(
        restart_read=rread,
        restart_write=rwri,
        trajectory=io.trajectory,
        restart_read_unit=io.restart_read_unit,
        restart_write_unit=io.restart_write_unit,
        trajectory_unit=io.trajectory_unit,
    )


def _apply_overlap_chunk_dynamics_kw(
    chunk_kw: dict[str, Any],
    *,
    chunk_index: int,
    has_restart_read: bool,
) -> None:
    """Set ``restart`` / ``new`` / ``start`` for one overlap chunk (in-place)."""
    if chunk_index == 0 and not has_restart_read:
        chunk_kw.pop("iunrea", None)
        chunk_kw["iunrea"] = -1
        return
    chunk_kw["new"] = False
    chunk_kw["start"] = False
    chunk_kw.pop("firstt", None)
    if has_restart_read:
        chunk_kw["restart"] = True
    else:
        chunk_kw["restart"] = False
        chunk_kw.pop("iunrea", None)
        chunk_kw["iunrea"] = -1


def _cleanup_overlap_restart_slots(io: Optional[CharmmTrajectoryFiles]) -> None:
    if io is None or io.restart_write is None:
        return
    slot_a, slot_b = _overlap_restart_slot_paths(Path(io.restart_write))
    for path in (slot_a, slot_b):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _sync_dynamics_io_units(
    kw: dict[str, Any],
    iokw: dict[str, int],
) -> None:
    """Drop restart/trajectory unit numbers not backed by opened CharmmFile handles."""
    for key in ("iunrea", "iunwri", "iuncrd", "iunvel"):
        if key not in iokw:
            kw.pop(key, None)


def _run_dynamics_chunk(
    dynamics_kwargs: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles],
) -> Any:
    open_files: list[Any] = []
    kw = dict(dynamics_kwargs)
    iokw: dict[str, int] = {}
    if io is not None:
        open_files, iokw = io.open_for_run()
        kw.update(iokw)
    _sync_dynamics_io_units(kw, iokw)
    try:
        return run_dynamics(kw)
    finally:
        for f in open_files:
            f.close()


def run_dynamics_with_io(
    dynamics_kwargs: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles] = None,
    *,
    overlap: Optional["DynamicsOverlapConfig"] = None,
    overlap_context: str = "dynamics",
    mlpot_ctx: Optional["MlpotContext"] = None,
) -> Any:
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

    kw = dict(dynamics_kwargs)
    total_nstep = int(kw.get("nstep", 0))
    if (
        overlap is None
        or not isinstance(overlap, DynamicsOverlapConfig)
        or not overlap.enabled
        or total_nstep <= 0
    ):
        return _run_dynamics_chunk(kw, io)

    interval = max(1, int(overlap.check_interval))
    _cleanup_overlap_restart_slots(io)
    check_dynamics_overlap(
        overlap,
        context=f"before {overlap_context}",
        step=0,
        mlpot_ctx=mlpot_ctx,
    )

    n_chunks = (total_nstep + interval - 1) // interval
    last_dyn: Any = None
    steps_done = 0
    try:
        for chunk_index in range(n_chunks):
            chunk_nstep = min(interval, total_nstep - steps_done)
            chunk_kw = dict(kw)
            chunk_kw["nstep"] = chunk_nstep
            if io is None:
                chunk_io = None
            elif n_chunks == 1:
                chunk_io = io
            else:
                chunk_io = _overlap_chunk_io(
                    io, chunk_index=chunk_index, n_chunks=n_chunks
                )
            has_restart_read = (
                chunk_io is not None
                and getattr(chunk_io, "restart_read", None) is not None
            )
            if chunk_index == 0 and has_restart_read:
                chunk_kw["new"] = False
                chunk_kw["start"] = False
                chunk_kw["restart"] = True
            else:
                _apply_overlap_chunk_dynamics_kw(
                    chunk_kw,
                    chunk_index=chunk_index,
                    has_restart_read=has_restart_read,
                )
            if chunk_io is None or chunk_io.restart_write is None:
                chunk_kw.pop("iunwri", None)

            last_dyn = _run_dynamics_chunk(chunk_kw, chunk_io)
            steps_done += chunk_nstep
            check_dynamics_overlap(
                overlap,
                context=overlap_context,
                step=steps_done,
                mlpot_ctx=mlpot_ctx,
            )
    finally:
        _cleanup_overlap_restart_slots(io)
    return last_dyn


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


def _sd_kwargs_from_config(config: MinimizeWithMlpotConfig) -> dict[str, Any]:
    """Common ``minimize.run_sd`` settings including optional DCD trajectory."""
    kw: dict[str, Any] = {
        "nstep": config.nstep,
        "nprint": config.nprint,
        "tolenr": config.tolenr,
        "tolgrd": config.tolgrd,
        # Do not rebuild nonbond/MLpot pair lists each SD step (mlpot_update can
        # segfault when all atoms are ML and the atom-pair list is empty).
        "inbfrq": 0,
        "ihbfrq": 0,
    }
    if config.save and config.dcd_path is not None and config.dcd_nsavc > 0:
        kw["iuncrd"] = config.dcd_unit
        kw["nsavc"] = config.dcd_nsavc
    return kw


def minimize_with_mlpot(
    config: MinimizeWithMlpotConfig,
) -> bool:
    """Run SD minimization with optional fixed ML atoms; optional trajectory/output.

    When ``dcd_path`` is set and ``save=True``, frames are written during SD via
    CHARMM ``iuncrd`` / ``nsavc`` (see ``pycharmm.minimize.MinOpts``).

    Returns True if minimization ran, False if skipped because CRD exists.
    """
    pycharmm, cons_fix, energy, minimize, *_ = _import_pycharmm_modules()
    from mmml.interfaces.pycharmmInterface.charmm_mpi import recover_mpi_for_charmm_after_jax

    crd_path = Path(config.crd_path) if config.crd_path else None
    if config.skip_if_crd_exists and crd_path is not None and crd_path.exists():
        load_minimized_coordinates(crd_path)
        if config.show_energy:
            _maybe_show_energy(True)
        return False

    if config.reference_positions is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

        sync_charmm_positions(config.reference_positions)

    dcd_file = None
    if config.save and config.dcd_path is not None and config.dcd_nsavc > 0:
        dcd_file = open_minimize_dcd(config.dcd_path, unit=config.dcd_unit)

    sd_kw = _sd_kwargs_from_config(config)

    try:
        if config.verbose and config.show_energy:
            print("CHARMM energy before minimization:")
            _maybe_show_energy(True)
        recover_mpi_for_charmm_after_jax(phase="before MLpot SD minimize")
        if config.verbose:
            print(
                f"SD pass 1 (free, all atoms): nstep={config.nstep} nprint={config.nprint}"
            )
        minimize.run_sd(**sd_kw)
        if config.verbose and config.show_energy:
            print("CHARMM energy after SD pass 1 (free):")
            _maybe_show_energy(True)

        if config.fixed_ml_selection is not None:
            n_fix = len(config.fixed_ml_selection.get_atom_indexes())
            cons_fix.setup(config.fixed_ml_selection)
            if config.verbose:
                print(
                    f"SD pass 2 (cons_fix, {n_fix} atoms): "
                    f"nstep={config.nstep} nprint={config.nprint}"
                )
            minimize.run_sd(**sd_kw)
            if config.verbose and config.show_energy:
                print("CHARMM energy after SD pass 2 (constrained):")
                _maybe_show_energy(True)
            cons_fix.turn_off()

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
