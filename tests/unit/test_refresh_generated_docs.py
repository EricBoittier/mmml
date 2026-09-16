"""Tests for the generated-docs refresh entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.refresh_generated_docs import (
    CHECK_GENERATORS,
    DIFF_PATHS,
    WRITE_GENERATORS,
    fail_if_dirty,
    main,
    working_tree_drift,
)

REPO = Path(__file__).resolve().parents[2]


def test_write_generators_are_repo_scripts():
    names = [" ".join(cmd) for cmd in WRITE_GENERATORS]
    assert any("generate_cli_docs.py" in n for n in names)
    assert any("generate_package_architecture.py" in n for n in names)
    assert any("generate_crystal_lit_compare.py" in n for n in names)
    for cmd in WRITE_GENERATORS:
        assert (REPO / cmd[0]).is_file()


def test_check_generators_include_figures_and_audits():
    names = [" ".join(cmd) for cmd in CHECK_GENERATORS]
    assert any("generate_docs_figures.py --check" in n for n in names)
    assert any("check_evidence_registry.py" in n for n in names)
    assert any("audit_hardcoded_recommendations.py" in n for n in names)
    for cmd in CHECK_GENERATORS:
        assert (REPO / cmd[0]).is_file()


def test_diff_covers_docs_and_mkdocs_nav():
    assert DIFF_PATHS == ("docs", "mkdocs.yml")


def test_main_requires_a_mode():
    with pytest.raises(SystemExit):
        main([])


def test_fail_if_dirty_is_clean_on_an_untouched_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(
        "scripts.ci.refresh_generated_docs.PATCH", tmp_path / "generated-docs.patch"
    )
    # Architecture was regenerated in this change set; other tracked docs may
    # already be dirty in the agent worktree. Only assert the helper's empty
    # path here.
    monkeypatch.setattr(
        "scripts.ci.refresh_generated_docs.working_tree_drift",
        lambda: ("", ""),
    )
    assert fail_if_dirty() == 0
    assert not (tmp_path / "generated-docs.patch").is_file()


def test_fail_if_dirty_writes_a_patch(tmp_path, monkeypatch):
    patch = tmp_path / "generated-docs.patch"
    monkeypatch.setattr("scripts.ci.refresh_generated_docs.PATCH", patch)
    monkeypatch.setattr(
        "scripts.ci.refresh_generated_docs.working_tree_drift",
        lambda: ("diff --git a/docs/package-architecture.md\n", ""),
    )
    assert fail_if_dirty() == 1
    assert "package-architecture.md" in patch.read_text(encoding="utf-8")


def test_working_tree_drift_returns_strings():
    diff, untracked = working_tree_drift()
    assert isinstance(diff, str)
    assert isinstance(untracked, str)
