"""Synthetic CGENFF MM params for serial warmup-mlpot-jax --do-mm."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    warmup_synthetic_mm_atom_params,
)


def test_warmup_synthetic_mm_atom_params_dcm_monomer():
    atc = ["CG301", "CG321", "HGA2", "CLGA1", "OT"]
    q = {"CG321": 0.1, "HGA2": 0.05, "CLGA1": -0.2, "CG301": 0.0, "OT": -0.3}
    z = np.array([6, 1, 1, 17, 1], dtype=int)
    charges, at_codes = warmup_synthetic_mm_atom_params(z, atc=atc, cgenff_params_dict_q=q)
    assert charges.shape == (5,)
    assert at_codes.shape == (5,)
    assert at_codes[0] == atc.index("CG321")
    assert at_codes[3] == atc.index("CLGA1")
    assert float(charges[3]) == -0.2
