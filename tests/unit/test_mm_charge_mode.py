"""Unit tests for MM charge mode taxonomy (fixed / latent / fixed+latent)."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.cgenff_mm import neutralize_per_monomer
from mmml.models.mm_charge_mode import (
    MMChargeMode,
    apply_mm_charge_mode,
    hybrid_mm_metadata_dict,
    mm_charge_mode_from_charge_correction,
    mm_charge_mode_needs_q_ml,
    parse_mm_charge_mode,
    require_charge_head_for_mode,
    resolve_hybrid_mm_charge_mode,
)
from mmml.interfaces.pycharmmInterface.mm_charge_correction import (
    assert_mm_charge_mode_dimer_supported,
    assert_mode_c_dimer_supported,
    load_hybrid_mm_metadata,
    map_padded_fragment_charges_to_global,
    mm_charges_fixed_plus_latent,
    resolve_mm_charge_mode_arg,
)


def test_parse_and_flag_mapping():
    assert parse_mm_charge_mode("fixed") is MMChargeMode.FIXED
    assert parse_mm_charge_mode("latent") is MMChargeMode.LATENT
    assert parse_mm_charge_mode("fixed_plus_latent") is MMChargeMode.FIXED_PLUS_LATENT
    assert parse_mm_charge_mode("charge_correction") is MMChargeMode.FIXED_PLUS_LATENT
    assert mm_charge_mode_from_charge_correction(False) is MMChargeMode.FIXED
    assert mm_charge_mode_from_charge_correction(True) is MMChargeMode.FIXED_PLUS_LATENT
    assert resolve_mm_charge_mode_arg(mm_charge_correction=True) is MMChargeMode.FIXED_PLUS_LATENT
    assert resolve_mm_charge_mode_arg(mm_charge_mode="latent") is MMChargeMode.LATENT
    assert resolve_hybrid_mm_charge_mode(mm_charge_mode="latent") is MMChargeMode.LATENT
    assert mm_charge_mode_needs_q_ml("latent")
    assert mm_charge_mode_needs_q_ml("fixed_plus_latent")
    assert not mm_charge_mode_needs_q_ml("fixed")


def test_conflicting_correction_and_mode_raises():
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_mm_charge_mode_arg(
            mm_charge_correction=True, mm_charge_mode="fixed"
        )
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_mm_charge_mode_arg(
            mm_charge_correction=True, mm_charge_mode="latent"
        )


def test_latent_mode_returns_neutralized_ml():
    q_c = jnp.array([0.1, -0.1, 0.2, -0.2])
    dq = jnp.array([0.4, -0.1, 0.25, 0.05])
    mid = jnp.array([0, 0, 1, 1])
    expected = neutralize_per_monomer(dq, mid)
    out = apply_mm_charge_mode("latent", q_c, dq, mid)
    assert np.allclose(np.asarray(out), np.asarray(expected), atol=1e-12)
    for sel in (slice(0, 2), slice(2, 4)):
        assert float(jnp.sum(out[sel])) == pytest.approx(0.0, abs=1e-12)


def test_fixed_mode_returns_cgenff():
    q = jnp.array([0.1, -0.1, 0.2, -0.2])
    mid = jnp.array([0, 0, 1, 1])
    out = apply_mm_charge_mode("fixed", q, jnp.ones_like(q), mid)
    assert np.allclose(np.asarray(out), np.asarray(q))


def test_fixed_plus_latent_matches_neutralize_formula():
    q_c = jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0])
    dq = jnp.array([0.4, -0.1, 0.25, 0.05, 7.0])
    mid = jnp.array([0, 0, 1, 1, -1])
    expected = q_c + neutralize_per_monomer(dq, mid)
    out = apply_mm_charge_mode("fixed_plus_latent", q_c, dq, mid)
    assert np.allclose(np.asarray(out), np.asarray(expected), atol=1e-12)
    for sel in (slice(0, 2), slice(2, 4)):
        assert float(jnp.sum(out[sel] - q_c[sel])) == pytest.approx(0.0, abs=1e-12)


def test_uniform_ml_charge_is_noop_on_correction():
    q_c = jnp.array([0.1, -0.1, 0.2, -0.2])
    mid = jnp.array([0, 0, 1, 1])
    out = apply_mm_charge_mode("fixed_plus_latent", q_c, jnp.full_like(q_c, 0.5), mid)
    assert np.allclose(np.asarray(out), np.asarray(q_c), atol=1e-12)


def test_require_charge_head():
    with pytest.raises(ValueError, match="charges=True"):
        require_charge_head_for_mode(MMChargeMode.FIXED_PLUS_LATENT, has_charges=False)
    with pytest.raises(ValueError, match="charges=True"):
        require_charge_head_for_mode(MMChargeMode.LATENT, has_charges=False)


@pytest.mark.parametrize("mode", [MMChargeMode.LATENT, MMChargeMode.FIXED_PLUS_LATENT])
def test_assert_mode_b_and_c_dimer_gates(mode):
    assert_mm_charge_mode_dimer_supported(
        MMChargeMode.FIXED,
        n_monomers=80,
        has_charges=False,
        lr_solver=None,
        doML=True,
        doML_dimer=True,
    )
    assert_mm_charge_mode_dimer_supported(
        mode,
        n_monomers=2,
        has_charges=True,
        lr_solver=None,
        doML=True,
        doML_dimer=True,
    )
    with pytest.raises(ValueError, match="dimer-only"):
        assert_mm_charge_mode_dimer_supported(
            mode,
            n_monomers=80,
            has_charges=True,
            lr_solver=None,
            doML=True,
            doML_dimer=True,
        )
    with pytest.raises(ValueError, match="JAX-PME"):
        assert_mm_charge_mode_dimer_supported(
            mode,
            n_monomers=2,
            has_charges=True,
            lr_solver="jax_pme",
            doML=True,
            doML_dimer=True,
        )
    with pytest.raises(ValueError, match="doML"):
        assert_mm_charge_mode_dimer_supported(
            mode,
            n_monomers=2,
            has_charges=True,
            lr_solver=None,
            doML=True,
            doML_dimer=False,
        )


def test_assert_mode_c_alias_still_works():
    assert_mode_c_dimer_supported(
        MMChargeMode.FIXED_PLUS_LATENT,
        n_monomers=2,
        has_charges=True,
        lr_solver=None,
        doML=True,
        doML_dimer=True,
    )


def test_map_padded_fragment_charges_excludes_padding():
    q_frag = jnp.array([0.1, 0.2, 0.3, 99.0])
    idx = jnp.array([0, 1, 2, 0])  # last slot filler points at 0
    mask = jnp.array([1.0, 1.0, 1.0, 0.0])
    out = map_padded_fragment_charges_to_global(q_frag, idx, mask, n_atoms=3)
    assert np.allclose(np.asarray(out), [0.1, 0.2, 0.3], atol=1e-12)


def test_mm_charges_fixed_plus_latent_helper():
    q_c = jnp.array([0.1, -0.1, 0.2, -0.2])
    dq = jnp.array([0.3, 0.1, -0.1, 0.1])
    mid = jnp.array([0, 0, 1, 1])
    out = mm_charges_fixed_plus_latent(q_c, dq, mid)
    assert float(jnp.sum(out[:2] - q_c[:2])) == pytest.approx(0.0, abs=1e-12)


def test_hybrid_mm_metadata_and_sidecar(tmp_path: Path):
    class _Cfg:
        mm_charge_mode = "fixed_plus_latent"
        charge_correction = True
        mm_switch_on = 8.0
        mm_switch_width = 5.0
        ml_switch_width = 1.5
        complementary_handoff = True

    meta = hybrid_mm_metadata_dict(_Cfg())
    assert meta["mm_charge_mode"] == "fixed_plus_latent"
    assert meta["charge_correction"] is True
    path = tmp_path / "hybrid_mm.json"
    path.write_text(json.dumps(meta))
    loaded = load_hybrid_mm_metadata(tmp_path)
    assert loaded["mm_charge_mode"] == "fixed_plus_latent"


def test_hybrid_mm_metadata_latent():
    class _Cfg:
        mm_charge_mode = "latent"
        mm_switch_on = 8.0
        mm_switch_width = 5.0
        ml_switch_width = 1.5
        complementary_handoff = True

    meta = hybrid_mm_metadata_dict(_Cfg())
    assert meta["mm_charge_mode"] == "latent"
    assert meta["charge_correction"] is False


# --- Mode D: latent_mean (precomputed template, liquid-compatible) --------


def test_parse_latent_mean_aliases():
    for alias in ("latent_mean", "latent-mean", "d", "mode_d"):
        assert parse_mm_charge_mode(alias) is MMChargeMode.LATENT_MEAN
    # Static-template mode needs no live q_ML.
    assert not mm_charge_mode_needs_q_ml("latent_mean")


def test_latent_mean_is_static_template_only():
    from mmml.models.mm_charge_mode import mm_charge_mode_is_static_template

    assert mm_charge_mode_is_static_template("latent_mean")
    assert not mm_charge_mode_is_static_template("fixed")
    assert not mm_charge_mode_is_static_template("latent")
    assert not mm_charge_mode_is_static_template("fixed_plus_latent")


def test_apply_mm_charge_mode_rejects_latent_mean():
    # latent_mean has no per-step composition -- it's injected directly by
    # the MD calculator as a precomputed mm_charges array, never routed
    # through apply_mm_charge_mode (training-time or live q_ML composition).
    q_c = jnp.array([0.1, -0.1, 0.2, -0.2])
    mid = jnp.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="precomputed template"):
        apply_mm_charge_mode("latent_mean", q_c, None, mid)


def test_latent_mean_bypasses_dimer_only_gate_for_liquids():
    # Unlike latent/fixed_plus_latent, latent_mean must work for any
    # n_monomers (a liquid box), with no charge head, no doML_dimer, and any
    # lr_solver -- it's a static charges array, not a live AB-dimer forward.
    assert_mm_charge_mode_dimer_supported(
        MMChargeMode.LATENT_MEAN,
        n_monomers=64,
        has_charges=False,
        lr_solver="ewald",
        doML=True,
        doML_dimer=False,
    )
    assert_mm_charge_mode_dimer_supported(
        MMChargeMode.LATENT_MEAN,
        n_monomers=1,
        has_charges=False,
        lr_solver="jax_pme",
        doML=False,
        doML_dimer=False,
    )
