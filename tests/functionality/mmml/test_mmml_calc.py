import importlib.util
from pathlib import Path
import os
import pytest
import numpy as np
import ase


def test_ev2kcalmol_constant():
	# Ensure the EV->kcal/mol conversion used by calculators is reasonable
	from mmml.pycharmmInterface.mmml_calculator import ev2kcalmol
	assert abs(ev2kcalmol - 23.0605) < 0.05


@pytest.mark.skipif(
	importlib.util.find_spec("pycharmm") is None,
	reason="pycharmm not available in this environment",
)
def test_setup_calculator_factory_smoke():
	# Skip if no checkpoints are present
	ckpt_env = os.environ.get("MMML_CKPT")
	if ckpt_env:
		ckpt = Path(ckpt_env)
	else:
		ckpt = Path("mmml/physnetjax/ckpts")
	if not ckpt.exists():
		pytest.skip("No checkpoints present for ML model")

	from mmml.pycharmmInterface.mmml_calculator import setup_calculator

	# Create factory without touching MM (avoids CHARMM setup during smoke test)
	factory = setup_calculator(
		ATOMS_PER_MONOMER=2,
		N_MONOMERS=2,
		doML=True,
		doMM=False,
		model_restart_path=ckpt,
		MAX_ATOMS_PER_SYSTEM=8,
	)

	assert callable(factory)


