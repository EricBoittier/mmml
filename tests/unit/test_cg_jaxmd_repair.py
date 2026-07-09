"""Regression checks for the long-running CG JAX-MD example."""

from pathlib import Path


def _source() -> str:
    return Path("examples/cg_jaxmd.py").read_text(encoding="utf-8")


def test_md_reinitializers_consume_fresh_keys():
    source = _source()
    assert "md_key = jax.random.PRNGKey(SEED)" in source
    assert "def next_md_key()" in source
    assert "jax.random.split(md_key)" in source
    assert "_init_fn_nvt(key," not in source
    assert "_init_fn_nve(key," not in source
    assert source.count("next_md_key()") >= 6


def test_repair_branch_recomputes_bond_deviation_for_logged_state():
    source = _source()
    assert "Peptide bond deviation exceeded repair threshold" in source
    assert source.count("max_dev, mean_dev = get_peptide_bond_diagnostics") >= 5


def test_repair_sequence_stages_mm_ml_mmml_and_selects_candidate():
    source = _source()
    assert "def run_staged_repair_for_jaxmd" in source
    assert "MM(system)" in source
    assert "ML(peptide)" in source
    assert "MMML(system) FIRE" in source
    assert "_repair_candidate_metrics" in source
    assert "candidate ranking by max force, then energy" in source
    assert "min(finite_candidates, key=lambda item: (item[\"max_force\"], item[\"energy\"]))" in source
    assert source.count("run_staged_repair_for_jaxmd(scaled_pos") == 2
