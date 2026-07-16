"""E_MM must enter the hybrid total in eV, not kcal/mol.

cgenff_mm_energy returns kcal/mol (CGenFF epsilons are kcal/mol; COULOMB_CONSTANT
is the 332.06 kcal/mol form). The ML energy and the training targets are eV.
Summing them unconverted inflates E_MM by 23.06x -- the model absorbs the error
during training and then disagrees with the MD calculator, which converts at the
same boundary (`mm_E = mm_E * kcalmol2ev`).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.data.units import EV_TO_KCAL_MOL, KCAL_MOL_TO_EV

SIG = jnp.array([3.6527, 2.3876])
EPS = jnp.array([0.0780, 0.0240])
KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)


def test_hybrid_e_mm_is_the_kcal_energy_converted_to_ev():
    """e_mm reported by hybrid_forward == cgenff_mm_energy * KCAL_MOL_TO_EV."""
    from mmml.models.cgenff_mm import cgenff_mm_energy
    from mmml.models.hybrid_energy import hybrid_forward
    from tests.unit.test_hybrid_energy import _batch, _fake_model_apply

    b = _batch(9.0)                      # MM-tail: E_MM is non-zero here
    out = hybrid_forward(_fake_model_apply, {}, b, 1, SIG, EPS, **KW)
    e_mm_ev = float(np.asarray(out["e_mm"]).reshape(-1)[0])

    e_mm_kcal = float(
        cgenff_mm_energy(
            b["R"].reshape(5, 3), b["cgenff_type_idx"][0], b["mol_id"][0],
            b["cgenff_charge"][0], SIG, EPS,
            complementary_handoff=True, **KW,
        )
    )
    assert e_mm_kcal != pytest.approx(0.0, abs=1e-9), "fixture must exercise a live MM term"
    assert e_mm_ev == pytest.approx(e_mm_kcal * KCAL_MOL_TO_EV, rel=1e-6)
    # and it is NOT the raw kcal/mol number
    assert e_mm_ev != pytest.approx(e_mm_kcal, rel=1e-3)


def test_the_conversion_is_the_expected_magnitude():
    """Guards against a wrong-direction fix (multiplying by 23 instead)."""
    assert EV_TO_KCAL_MOL == pytest.approx(23.0605, abs=1e-3)
    assert KCAL_MOL_TO_EV == pytest.approx(1.0 / 23.0605, rel=1e-4)
    assert KCAL_MOL_TO_EV < 1.0


def test_matches_the_md_calculator_constant():
    """Training and MD must convert with the SAME factor (single source of truth)."""
    ase = pytest.importorskip("ase.units")
    md_factor = 1.0 / (1 / (ase.kcal / ase.mol))
    assert KCAL_MOL_TO_EV == pytest.approx(md_factor, rel=1e-6)
