"""Minimization and MD workflows with MLpot active (PyCHARMM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

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


def minimize_charmm_mm_only(config: CharmmMmMinimizeConfig) -> None:
    """Run CHARMM SD (and optional ABNR) on the current PSF using MM terms only.

    Call **before** :func:`register_mlpot` so the PSF still has bonds and no ML model is loaded.
    """
    _, cons_fix, energy, minimize, *_ = _import_pycharmm_modules()
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()
    if config.nstep_sd <= 0 and config.nstep_abnr <= 0:
        return

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
    _import_pycharmm_modules()[0].lingo.charmm_script("ENER")
    if config.verbose and config.show_energy:
        _maybe_show_energy(True)
    cons_fix.turn_off()


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


def _cpt_mass_kwargs(temp: float = 300.0) -> dict[str, Any]:
    import pycharmm.select as select

    pmass = int(np.sum(select.get_property("mass")) / 50.0)
    tmass = int(pmass * 10)
    return {
        "leap": True,
        "cpt": True,
        "pint pconst pref": 1,
        "pgamma": 5,
        "pmass": pmass,
        "hoover reft": temp,
        "tmass": tmass,
    }


def build_cpt_equilibration_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 50.0,
    save_interval_ps: float = 0.01,
    temp: float = 300.0,
    restart: bool = True,
    echeck: float = 500.0,
) -> dict[str, Any]:
    """NPT equilibration (CPT + Hoover); matches example mini-MD scripts."""
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
    kw.update(_cpt_mass_kwargs(temp))
    return kw


def build_cpt_production_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 100.0,
    save_interval_ps: float = 0.01,
    temp: float = 300.0,
    restart: bool = True,
) -> dict[str, Any]:
    """NPT production segment (same integrator as equilibration)."""
    return build_cpt_equilibration_dynamics(
        timestep_ps=timestep_ps,
        duration_ps=duration_ps,
        save_interval_ps=save_interval_ps,
        temp=temp,
        restart=restart,
    )


def run_dynamics(dynamics_kwargs: dict[str, Any]) -> Any:
    """Instantiate and run ``pycharmm.DynamicsScript``."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import disable_charmm_domdec

    import pycharmm

    disable_charmm_domdec()
    dyn = pycharmm.DynamicsScript(**dynamics_kwargs)
    dyn.run()
    return dyn


def run_dynamics_with_io(
    dynamics_kwargs: dict[str, Any],
    io: Optional[CharmmTrajectoryFiles] = None,
) -> Any:
    """Run dynamics and open/close CharmmFile units from ``io``."""
    open_files: list[Any] = []
    kw = dict(dynamics_kwargs)
    if io is not None:
        open_files, iokw = io.open_for_run()
        kw.update(iokw)
    try:
        return run_dynamics(kw)
    finally:
        for f in open_files:
            f.close()


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
    _, cons_fix, energy, minimize, *_ = _import_pycharmm_modules()

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
    *_, read, _write = _import_pycharmm_modules()
    path = Path(crd_path)
    if not path.exists():
        raise FileNotFoundError(f"CRD not found: {path}")
    read.coor_card(str(path))


def production_restart_chain(
    data_dir: PathLike,
    *,
    n_segments: int = 10,
    prefix: str = "dyna",
    equi_restart: str = "equi.res",
) -> list[CharmmTrajectoryFiles]:
    """Build restart/trajectory file triples for chained production (stub planner).

    Segment 0 reads ``equi.res``; segment ``i>0`` reads ``dyna.{i-1}.res``.
    """
    data_dir = Path(data_dir)
    chain: list[CharmmTrajectoryFiles] = []
    for ii in range(n_segments):
        if ii == 0:
            rread = data_dir / equi_restart
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
