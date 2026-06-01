"""Minimization and MD workflows with MLpot active (PyCHARMM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

PathLike = Union[str, Path]


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
class MinimizeWithMlpotConfig:
    """SD minimization while MLpot supplies the ML region energy."""

    fixed_ml_selection: Optional[Any] = None
    nstep: int = 500
    nprint: int = 10
    tolenr: float = 1e-5
    tolgrd: float = 1e-5
    pdb_path: Optional[PathLike] = None
    crd_path: Optional[PathLike] = None
    title: str = "Mini SD"
    skip_if_crd_exists: bool = True
    show_energy: bool = True


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
    echeck: int = -1,
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
) -> dict[str, Any]:
    """NVT heating dict for ``DynamicsScript`` (CHARMM + MLpot)."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(timestep=timestep_ps, nstep=nstep, nsavc=nsavc, nprint=100)
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


def build_nve_dynamics(
    *,
    timestep_ps: float = 0.00025,
    duration_ps: float = 50.0,
    save_interval_ps: float = 0.01,
    restart: bool = True,
) -> dict[str, Any]:
    """NVE production-style dict (restart from heat)."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(
        timestep=timestep_ps,
        nstep=nstep,
        nsavc=nsavc,
        nprint=10,
        ntrfrq=0,
    )
    kw.update(
        {
            "verlet": True,
            "new": False,
            "start": False,
            "restart": restart,
            "ihtfrq": 0,
            "ieqfrq": 0,
        }
    )
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
) -> dict[str, Any]:
    """NPT equilibration (CPT + Hoover); matches example mini-MD scripts."""
    nstep = ps_to_nsteps(timestep_ps, duration_ps)
    nsavc = nsavc_for_interval(timestep_ps, save_interval_ps)
    kw = _base_dyn_kwargs(timestep=timestep_ps, nstep=nstep, nsavc=nsavc, nprint=100)
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
    import pycharmm

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


def minimize_with_mlpot(
    config: MinimizeWithMlpotConfig,
) -> bool:
    """Run SD minimization with optional fixed ML atoms; write PDB/CRD.

    Returns True if minimization ran, False if skipped because CRD exists.
    """
    pycharmm, cons_fix, energy, minimize, read, write = _import_pycharmm_modules()

    crd_path = Path(config.crd_path) if config.crd_path else None
    if config.skip_if_crd_exists and crd_path is not None and crd_path.exists():
        load_minimized_coordinates(crd_path)
        if config.show_energy:
            energy.show()
        return False

    if config.fixed_ml_selection is not None:
        cons_fix.setup(config.fixed_ml_selection)
        minimize.run_sd(
            nstep=config.nstep,
            nprint=config.nprint,
            tolenr=config.tolenr,
            tolgrd=config.tolgrd,
        )
        cons_fix.turn_off()
    minimize.run_sd(
        nstep=config.nstep,
        nprint=config.nprint,
        tolenr=config.tolenr,
        tolgrd=config.tolgrd,
    )

    write_minimized_coordinates(
        pdb_path=config.pdb_path,
        crd_path=config.crd_path,
        title=config.title,
    )
    if config.show_energy:
        energy.show()
    return True


def write_minimized_coordinates(
    *,
    pdb_path: Optional[PathLike] = None,
    crd_path: Optional[PathLike] = None,
    title: str = "Mini SD",
) -> None:
    _, _, _, _, write = _import_pycharmm_modules()
    if pdb_path is not None:
        write.coor_pdb(str(pdb_path), title=title)
    if crd_path is not None:
        write.coor_card(str(crd_path), title=title)


def load_minimized_coordinates(crd_path: PathLike) -> None:
    """Load optimized coords from a CRD card (preferred over PDB for ML exclusions)."""
    _, _, _, _, read = _import_pycharmm_modules()
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
