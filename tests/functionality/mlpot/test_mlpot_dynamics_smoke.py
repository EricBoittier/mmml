"""Live PyCHARMM MLpot dynamics smoke: short NVE and heat with DCD/restart I/O."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLPOT_DIR = Path(__file__).resolve().parent

TIMESTEP_PS = 0.00025


def _can_import(name: str) -> bool:
	try:
		__import__(name)
		return True
	except Exception:
		return False


def _resolve_ckpt() -> Path | None:
	ckpt_env = os.environ.get("MMML_CKPT")
	candidates: list[Path] = []
	if ckpt_env:
		candidates.append(Path(ckpt_env))
	candidates.extend(
		[
			PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
			PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
		]
	)
	for p in candidates:
		if p.exists():
			return p.resolve()
	return None


def _setup_aco_dimer_mlpot(ckpt: Path):
	"""Build ACO dimer, register MLpot; return (ctx, z, n_atoms)."""
	if str(MLPOT_DIR) not in sys.path:
		sys.path.insert(0, str(MLPOT_DIR))

	import ase
	import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

	from _common import build_acetone_dimer_cluster
	from mmml.interfaces.pycharmmInterface.mlpot import (
		load_physnet_mlpot_bundle,
		register_mlpot,
		select_all_atoms,
		setup_default_nbonds,
		sync_charmm_positions,
	)

	z, r = build_acetone_dimer_cluster(4.0)
	n_atoms = len(z)
	setup_default_nbonds()
	sync_charmm_positions(r)

	atoms = ase.Atoms(numbers=z, positions=r)
	_, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
	ctx = register_mlpot(pyCModel, z, select_all_atoms())
	return ctx, z, n_atoms


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_mlpot_nve_writes_dcd_and_restart(tmp_path: Path):
	"""Short vacuum NVE through MLpot writes CHARMM restart + DCD."""
	ckpt = _resolve_ckpt()
	if ckpt is None:
		pytest.skip("No PhysNet checkpoint for MLpot dynamics test")

	from mmml.interfaces.pycharmmInterface.mlpot import (
		CharmmTrajectoryFiles,
		build_nve_dynamics,
		run_dynamics_with_io,
	)

	ctx, _z, _n = _setup_aco_dimer_mlpot(ckpt)
	res_path = tmp_path / "nve_smoke.res"
	dcd_path = tmp_path / "nve_smoke.dcd"
	nstep = 12
	try:
		kw = build_nve_dynamics(
			timestep_ps=TIMESTEP_PS,
			duration_ps=nstep * TIMESTEP_PS,
			save_interval_ps=TIMESTEP_PS,
			restart=False,
			temp=300.0,
			nprint=nstep,
			echeck=500.0,
			use_pbc=False,
		)
		kw.update(new=True, start=True, nstep=nstep, nsavc=2)
		run_dynamics_with_io(
			kw,
			CharmmTrajectoryFiles(restart_write=res_path, trajectory=dcd_path),
			overlap_context="pytest nve smoke",
		)
	finally:
		ctx.unset()

	assert res_path.is_file() and res_path.stat().st_size > 0
	assert dcd_path.is_file() and dcd_path.stat().st_size > 0


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_mlpot_heat_writes_dcd_and_restart(tmp_path: Path):
	"""Short vacuum heat (iasors=0 ramp) through MLpot writes restart + DCD."""
	ckpt = _resolve_ckpt()
	if ckpt is None:
		pytest.skip("No PhysNet checkpoint for MLpot dynamics test")

	from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
		CharmmTrajectoryFiles,
		build_heat_dynamics,
		run_dynamics_with_io,
	)
	from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
		_configure_heat_dynamics_start,
	)

	ctx, _z, _n = _setup_aco_dimer_mlpot(ckpt)
	res_path = tmp_path / "heat_smoke.res"
	dcd_path = tmp_path / "heat_smoke.dcd"
	nstep = 40
	try:
		kw = build_heat_dynamics(
			timestep_ps=TIMESTEP_PS,
			duration_ps=nstep * TIMESTEP_PS,
			save_interval_ps=TIMESTEP_PS * 4,
			temp=300.0,
			firstt=60.0,
			finalt=300.0,
			echeck=500.0,
			use_pbc=False,
			ihtfrq=10,
		)
		kw.update(nstep=nstep, nsavc=4, nprint=nstep)
		io = CharmmTrajectoryFiles(restart_write=res_path, trajectory=dcd_path)
		_configure_heat_dynamics_start(
			kw,
			io,
			coords_in_memory=True,
			restart_from_file=False,
			timestep_ps=TIMESTEP_PS,
			use_pbc=False,
			quiet=True,
			heat_thermostat="scale",
		)
		run_dynamics_with_io(
			kw,
			io,
			overlap_context="pytest heat smoke",
		)
	finally:
		ctx.unset()

	assert res_path.is_file() and res_path.stat().st_size > 0
	assert dcd_path.is_file() and dcd_path.stat().st_size > 0
