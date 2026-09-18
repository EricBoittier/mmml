"""Optional live PET-MAD / UPET smoke when MMML_METATOMIC_MODEL_DIR is set."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.interfaces.calculators.ase_fragment_hybrid import evaluate_fragment_hybrid
from mmml.interfaces.calculators.metatomic import have_metatomic, load_metatomic_calculator
from mmml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import build_metatomic_mlpot_model

_raw_dir = os.environ.get("MMML_METATOMIC_MODEL_DIR", "").strip()
MODEL_DIR = Path(_raw_dir) if _raw_dir else None
PET_MAD = (MODEL_DIR / "pet-mad-s-v1.0.2.pt") if MODEL_DIR is not None else None

pytestmark = pytest.mark.skipif(
    not have_metatomic() or PET_MAD is None or not PET_MAD.is_file(),
    reason="needs uv extra metatomic and MMML_METATOMIC_MODEL_DIR/pet-mad-s-v1.0.2.pt",
)


def test_pet_mad_loads_and_charmm_kcal_matches_fragment_energy() -> None:
    calc = load_metatomic_calculator(PET_MAD, device="cpu")
    a = molecule("H2O")
    b = molecule("H2O")
    b.positions += np.array([2.8, 0.0, 0.0])
    dimer = a + b
    z = dimer.get_atomic_numbers()
    r = dimer.get_positions()
    frag = evaluate_fragment_hybrid(calc, z, r, [3, 3])
    assert np.isfinite(frag.energy_ev)
    assert frag.n_dimers_evaluated == 1
    model = build_metatomic_mlpot_model(
        PET_MAD, z, [3, 3], 2, calculator=calc, do_mm=False, eval_mode="fragments"
    )
    charmm = model.get_pycharmm_calculator(ml_atom_indices=list(range(6)))
    dx = [0.0] * 6
    dy = [0.0] * 6
    dz = [0.0] * 6
    e_kcal = charmm.calculate_charmm(
        6,
        0,
        0,
        None,
        r[:, 0].tolist(),
        r[:, 1].tolist(),
        r[:, 2].tolist(),
        dx,
        dy,
        dz,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert e_kcal == pytest.approx(frag.energy_ev * EV_TO_KCAL_MOL, rel=1e-6, abs=1e-6)
    assert any(abs(v) > 0.0 for v in dx + dy + dz)
