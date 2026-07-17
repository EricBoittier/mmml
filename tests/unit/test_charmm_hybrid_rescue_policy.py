"""CHARMM MM rescue must not thrash a soft hybrid minimum."""

from __future__ import annotations

from mmml.cli.run.md_pbc_suite.jaxmd import (
    charmm_hybrid_rescue_accepted,
    should_skip_charmm_hybrid_rescue,
)


def test_skip_charmm_when_hybrid_fmax_already_soft():
    """Post-FIRE fmax≈0.28 with soft gate 1.0 must skip MM CHARMM rescue."""
    assert should_skip_charmm_hybrid_rescue(hybrid_fmax=0.28, soft_gate_eVA=1.0)
    assert should_skip_charmm_hybrid_rescue(hybrid_fmax=1.0, soft_gate_eVA=1.0)
    assert not should_skip_charmm_hybrid_rescue(hybrid_fmax=1.84, soft_gate_eVA=1.0)


def test_reject_charmm_when_hybrid_fmax_worsens():
    """Observed acetone thrash: CHARMM took hybrid fmax 0.28 → 1.83."""
    assert charmm_hybrid_rescue_accepted(1.84, 1.80)
    assert charmm_hybrid_rescue_accepted(1.84, 1.84)
    assert not charmm_hybrid_rescue_accepted(0.281703, 1.825092)


def test_jaxmd_wires_soft_skip_and_reject_into_pre_min():
    """Source contract: fire-first rescue path uses soft skip + reject-if-worse."""
    import inspect

    from mmml.cli.run.md_pbc_suite import jaxmd

    src = inspect.getsource(jaxmd)
    assert "should_skip_charmm_hybrid_rescue" in src
    assert "charmm_hybrid_rescue_accepted" in src
    assert "_maybe_charmm_then_hybrid" in src
    assert "rescue rejected" in src
    assert "skipped_soft" in src
