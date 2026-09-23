"""MM charge scale (``mm_charge_scale`` in hybrid_mm.json) reaches the jax_mic MM path."""

from __future__ import annotations

import json

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")


def test_apply_mm_charge_scale_identity_and_validation():
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import apply_mm_charge_scale

    q = np.array([0.4, -0.8, 0.4])
    assert apply_mm_charge_scale(q, 1.0) is q
    assert apply_mm_charge_scale(q, None) is q
    assert np.allclose(apply_mm_charge_scale(q, 1.25), 1.25 * q)
    assert np.allclose(q, [0.4, -0.8, 0.4])  # input untouched
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            apply_mm_charge_scale(q, bad)


def test_jax_mic_coulomb_scales_with_charge_scale_squared():
    """E(s) = E_LJ + s^2 E_C, F likewise: check the quadratic on three scales."""
    pytest.importorskip("vesin")
    from tests.unit.test_mm_pair_list_completeness import _box, _build

    R = _box(5.3, seed=4)
    out = {}
    for s in (1.0, 1.2, 1.5):
        mm_fn, update = _build(R, charge_scale=s)
        pidx, pmask = update(R, force_rebuild=True)
        e, f = mm_fn(jnp.asarray(R), pidx, pmask)
        out[s] = (float(e), np.asarray(f))
    e1, f1 = out[1.0]
    ea, fa = out[1.2]
    eb, fb = out[1.5]
    e_c = (ea - e1) / (1.2**2 - 1.0)
    assert abs(e_c) > 1e-6  # the box has Coulomb in the MM window
    assert eb - e1 == pytest.approx((1.5**2 - 1.0) * e_c, rel=1e-6, abs=1e-9)
    f_c = (fa - f1) / (1.2**2 - 1.0)
    assert np.allclose(fb - f1, (1.5**2 - 1.0) * f_c, rtol=1e-5, atol=1e-8)


def test_resolve_md_charge_scale_from_tuner_sidecar(tmp_path):
    from mmml.models.mm_lj_scales import resolve_md_charge_scale
    from mmml.models.mm_nonbonded_tune import MonomerNonbonded, TuneParams, lj_sidecar_payload

    ff = MonomerNonbonded.from_arrays(
        ["CG331", "OG311"], [0.1, -0.1], {"CG331": 2.05, "OG311": 1.765},
        {"CG331": 0.078, "OG311": 0.1921},
    )
    side = tmp_path / "hybrid_mm.json"
    side.write_text(json.dumps(lj_sidecar_payload(ff, TuneParams(np.full(2, 1.5), np.ones(2), 1.219))))
    assert resolve_md_charge_scale(scales_file=side) == pytest.approx(1.219)
    # without a sidecar the scale is the identity
    empty = tmp_path / "elsewhere" / "ckpt"
    empty.mkdir(parents=True)
    assert resolve_md_charge_scale(scales_file=None, checkpoint=empty) == 1.0
    # an LJ-only sidecar (no mm_charge_scale key) also resolves to 1
    data = json.loads(side.read_text())
    data.pop("mm_charge_scale")
    side.write_text(json.dumps(data))
    assert resolve_md_charge_scale(scales_file=side) == 1.0
