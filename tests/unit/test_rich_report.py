"""Unit tests for shared Rich reporting helpers."""

from __future__ import annotations

import os

import pytest

from mmml.utils import rich_report


@pytest.fixture(autouse=True)
def _no_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMML_NO_RICH", "1")
    monkeypatch.delenv("MMML_QUIET", raising=False)
    rich_report._console.cache_clear()


def test_emit_tagged_plain(capsys) -> None:
    rich_report.emit_tagged("setup_calculator", "hello")
    out = capsys.readouterr().out
    assert "[setup_calculator] hello" in out


def test_emit_jax_compile_pass_plain(capsys) -> None:
    rich_report.emit_jax_compile_pass("test_kernel", 0, 1.23)
    out = capsys.readouterr().out
    assert "mmml: JAX compile timer [test_kernel] pass 1 (compile+run): 1.23s" in out


def test_emit_jax_compile_session_summary_plain(capsys) -> None:
    lines = [
        "mmml: JAX compile timers — estimated compile=1.00s, run=0.50s",
        "  test_kernel: compile≈1.00s, run≈0.50s (pass1=1.50s)",
    ]
    rich_report.emit_jax_compile_session_summary(lines)
    out = capsys.readouterr().out
    assert "estimated compile=1.00s" in out
    assert "test_kernel:" in out


def test_emit_charmm_block_plain(capsys) -> None:
    rich_report.emit_charmm_block(
        "MLpot all-ML (10 atoms, bonded/ELEC/VDW off)",
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "CHARMM BLOCK:" in out
    assert "MLpot all-ML" in out


def test_emit_charmm_block_suppressed_by_default(capsys) -> None:
    rich_report.emit_charmm_block("MLpot all-ML (10 atoms, bonded/ELEC/VDW off)")
    assert capsys.readouterr().out == ""


def test_emit_status_respects_quiet(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMML_QUIET", "1")
    rich_report.emit_status(True, "hidden")
    assert capsys.readouterr().out == ""


def test_model_attribute_rows_uses_display_labels() -> None:
    class _Model:
        features = 32
        natoms = 10
        n_res = 3
        num_iterations = 2
        use_pbc = False

    rows = dict(rich_report._model_attribute_rows(_Model()))
    assert "max_padded_atoms" in rows
    assert "n_refinement_blocks" in rows
    assert "message_passing_steps" in rows
    assert "natoms" not in rows
    assert "n_res" not in rows


def test_emit_model_loaded_runtime_max_padded_atoms(capsys) -> None:
    class _Model:
        features = 32
        natoms = 10

    rich_report.emit_model_loaded(_Model(), runtime_max_padded_atoms=34)
    out = capsys.readouterr().out
    assert "runtime_max_padded_atoms=34" in out


def test_model_attribute_rows_from_object() -> None:
    class _Model:
        features = 32
        natoms = 10
        use_pbc = False

    rows = rich_report._model_attribute_rows(_Model())
    assert ("features", 32) in rows
    assert ("class", "_Model") in rows


def test_emit_hybrid_ml_setup_plain(capsys) -> None:
    class _Model:
        features = 32
        natoms = 10
        cutoff = 12.0
        charges = False

    rich_report.emit_hybrid_ml_setup(
        system={"n_monomers": 2, "total_atoms": 10},
        handoff={"mm_switch_on_Å": "8.0"},
        neighbor_lists={"ml_sparse_dimers": True},
        model=_Model(),
        checkpoint={"epoch": 1000},
        runtime={"OMP_NUM_THREADS": "8", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"},
        ml_flags={"doML": True, "doMM": True},
        long_range={
            "lr_solver": "jax_pme",
            "jax_pme_method": "ewald",
            "jax_pme_sr_cutoff_Å": "6.0",
            "coulomb_mode": "jax-pme k-space + pair SR",
        },
    )
    out = capsys.readouterr().out
    assert "Hybrid ML/MM setup" in out
    assert "Long-range Coulomb" in out
    assert "jax_pme_method" in out
    assert "n_monomers" in out
    assert "Runtime threads" in out
    assert "XLA_FLAGS" in out
    assert "features" in out


def test_collect_psf_topology_mapping_without_charmm() -> None:
    assert rich_report.collect_psf_topology_mapping() is None


def test_psf_residue_summary_per_residue_names() -> None:
    class _Psf:
        def get_nres(self) -> int:
            return 10

        def get_res(self) -> list[str]:
            return ["DCM"] * 10

        def get_resid(self) -> list[str]:
            return [str(i) for i in range(1, 11)]

    n_res, label = rich_report._psf_residue_summary(_Psf(), n_atom=50, max_residue_rows=6)
    assert n_res == 10
    assert label == "DCM×10"


def test_psf_residue_summary_mixed_composition() -> None:
    class _Psf:
        def get_nres(self) -> int:
            return 4

        def get_res(self) -> list[str]:
            return ["MEOH", "MEOH", "ACET", "ACET"]

        def get_resid(self) -> list[str]:
            return ["1", "2", "3", "4"]

    n_res, label = rich_report._psf_residue_summary(_Psf(), n_atom=20, max_residue_rows=6)
    assert n_res == 4
    assert label == "MEOH×2, ACET×2"


def test_psf_residue_summary_per_atom_resids() -> None:
    class _Psf:
        def get_nres(self) -> int:
            return 2

        def get_res(self) -> list[str]:
            return ["DCM", "DCM"]

        def get_resid(self) -> list[str]:
            return ["1"] * 5 + ["2"] * 5

    n_res, label = rich_report._psf_residue_summary(_Psf(), n_atom=10, max_residue_rows=6)
    assert n_res == 2
    assert label == "DCM×2"
