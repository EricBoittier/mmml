"""Coverage for mmml.validation.smoke_matrix internals not exercised by
test_smoke_matrix.py: manifest validation errors, per-requirement-type
blocking, command token resolution, hashing, FAIL status, and the CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mmml.validation.smoke_matrix import (
    SmokeCase,
    _blocked_reasons,
    _resolve_command,
    _sha256,
    load_smoke_manifest,
    main,
    run_smoke_matrix,
)


def _write_manifest(tmp_path: Path, body: str) -> Path:
    manifest_path = tmp_path / "matrix.yaml"
    manifest_path.write_text(body, encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# load_smoke_manifest: error paths
# ---------------------------------------------------------------------------


def test_load_smoke_manifest_rejects_wrong_schema_version(tmp_path):
    path = _write_manifest(tmp_path, "schema_version: 99\ncases: []\n")
    with pytest.raises(ValueError, match="schema_version"):
        load_smoke_manifest(path)


def test_load_smoke_manifest_rejects_empty_cases_list(tmp_path):
    path = _write_manifest(tmp_path, "schema_version: 1\ncases: []\n")
    with pytest.raises(ValueError, match="non-empty cases list"):
        load_smoke_manifest(path)


def test_load_smoke_manifest_rejects_missing_cases_key(tmp_path):
    path = _write_manifest(tmp_path, "schema_version: 1\n")
    with pytest.raises(ValueError, match="non-empty cases list"):
        load_smoke_manifest(path)


def test_load_smoke_manifest_rejects_duplicate_ids(tmp_path):
    path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - id: dup
    command: ["true"]
  - id: dup
    command: ["true"]
""",
    )
    with pytest.raises(ValueError, match="duplicated"):
        load_smoke_manifest(path)


def test_load_smoke_manifest_rejects_missing_command(tmp_path):
    path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - id: no_command
""",
    )
    with pytest.raises(ValueError, match="requires a command list"):
        load_smoke_manifest(path)


def test_load_smoke_manifest_rejects_non_mapping_case(tmp_path):
    path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - "not-a-mapping"
""",
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        load_smoke_manifest(path)


def test_load_smoke_manifest_defaults_optional_fields(tmp_path):
    path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - id: minimal
    command: ["true"]
""",
    )
    manifest = load_smoke_manifest(path)
    (case,) = manifest.cases
    assert case.category == "other"
    assert case.description == ""
    assert case.requires_env == ()
    assert case.requires_commands == ()
    assert case.requires_modules == ()
    assert case.requires_any_modules == ()
    assert case.expected_artifacts == ()
    assert case.tags == ()


# ---------------------------------------------------------------------------
# _blocked_reasons
# ---------------------------------------------------------------------------


def test_blocked_reasons_empty_for_unconstrained_case():
    case = SmokeCase(id="x", category="c", description="d", command=("true",))
    assert _blocked_reasons(case, env={}) == []


def test_blocked_reasons_env_unset(monkeypatch):
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_env=("MMML_TEST_UNSET_VAR",),
    )
    reasons = _blocked_reasons(case, env={})
    assert any("MMML_TEST_UNSET_VAR is unset" in r for r in reasons)


def test_blocked_reasons_env_path_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_env=("MMML_TEST_PATH_VAR",),
    )
    reasons = _blocked_reasons(case, env={"MMML_TEST_PATH_VAR": str(missing)})
    assert any("does not exist" in r for r in reasons)


def test_blocked_reasons_env_path_exists_not_blocked(tmp_path):
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_env=("MMML_TEST_PATH_VAR",),
    )
    reasons = _blocked_reasons(case, env={"MMML_TEST_PATH_VAR": str(tmp_path)})
    assert reasons == []


def test_blocked_reasons_requires_commands_unavailable():
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_commands=("definitely-not-a-real-binary-xyz",),
    )
    reasons = _blocked_reasons(case, env={})
    assert any("command" in r and "unavailable" in r for r in reasons)


def test_blocked_reasons_requires_commands_available():
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_commands=("sh",),
    )
    reasons = _blocked_reasons(case, env={})
    assert reasons == []


def test_blocked_reasons_requires_modules_unavailable():
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_modules=("definitely_not_a_real_module_xyz",),
    )
    reasons = _blocked_reasons(case, env={})
    assert any("Python module" in r for r in reasons)


def test_blocked_reasons_requires_modules_available():
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_modules=("os",),
    )
    reasons = _blocked_reasons(case, env={})
    assert reasons == []


def test_blocked_reasons_requires_any_modules_none_available():
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_any_modules=("no_such_module_a", "no_such_module_b"),
    )
    reasons = _blocked_reasons(case, env={})
    assert any("none of the Python modules are available" in r for r in reasons)


def test_blocked_reasons_requires_any_modules_one_available():
    case = SmokeCase(
        id="x", category="c", description="d", command=("true",),
        requires_any_modules=("no_such_module_a", "os"),
    )
    reasons = _blocked_reasons(case, env={})
    assert reasons == []


# ---------------------------------------------------------------------------
# _resolve_command
# ---------------------------------------------------------------------------


def test_resolve_command_substitutes_known_tokens(tmp_path):
    case = SmokeCase(
        id="x", category="c", description="d",
        command=("{python}", "--repo={repo}", "--out={output_dir}/x"),
    )
    repo = tmp_path / "repo"
    output_dir = tmp_path / "out"
    resolved = _resolve_command(case, repo=repo, output_dir=output_dir)
    assert resolved == [sys.executable, f"--repo={repo}", f"--out={output_dir}/x"]


def test_resolve_command_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("MMML_TEST_RESOLVE_VAR", "resolved-value")
    case = SmokeCase(id="x", category="c", description="d", command=("$MMML_TEST_RESOLVE_VAR",))
    resolved = _resolve_command(case, repo=tmp_path, output_dir=tmp_path)
    assert resolved == ["resolved-value"]


def test_resolve_command_raises_on_unresolved_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("MMML_TEST_UNRESOLVED_VAR", raising=False)
    case = SmokeCase(id="x", category="c", description="d", command=("$MMML_TEST_UNRESOLVED_VAR",))
    with pytest.raises(ValueError, match="unresolved environment variable"):
        _resolve_command(case, repo=tmp_path, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------


def test_sha256_missing_file_returns_none(tmp_path):
    assert _sha256(tmp_path / "nope.txt") is None


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    assert _sha256(f) == hashlib.sha256(b"hello world").hexdigest()


# ---------------------------------------------------------------------------
# run_smoke_matrix: FAIL path + unknown selection
# ---------------------------------------------------------------------------


def test_run_smoke_matrix_records_fail_on_nonzero_exit(tmp_path):
    manifest = load_smoke_manifest(
        _write_manifest(
            tmp_path,
            """
schema_version: 1
cases:
  - id: failing
    category: calculator
    description: always fails
    command: ["{python}", "-c", "raise SystemExit(3)"]
""",
        )
    )
    summary = run_smoke_matrix(manifest, output_root=tmp_path / "out", repo=tmp_path)
    assert summary["counts"] == {"PASS": 0, "FAIL": 1, "BLOCKED": 0}
    status = json.loads((tmp_path / "out/failing/status.json").read_text())
    assert status["status"] == "FAIL"
    assert status["returncode"] == 3


def test_run_smoke_matrix_fails_on_missing_expected_artifact(tmp_path):
    manifest = load_smoke_manifest(
        _write_manifest(
            tmp_path,
            """
schema_version: 1
cases:
  - id: no_artifact
    category: calculator
    description: exits 0 but never writes the expected artifact
    command: ["{python}", "-c", "pass"]
    expected_artifacts: ["result.json"]
""",
        )
    )
    summary = run_smoke_matrix(manifest, output_root=tmp_path / "out", repo=tmp_path)
    assert summary["counts"]["FAIL"] == 1
    status = json.loads((tmp_path / "out/no_artifact/status.json").read_text())
    assert status["status"] == "FAIL"
    assert any("missing expected artifact" in r for r in status["reasons"])


def test_run_smoke_matrix_rejects_unknown_case_selection(tmp_path):
    manifest = load_smoke_manifest(
        _write_manifest(
            tmp_path,
            """
schema_version: 1
cases:
  - id: only_case
    command: ["true"]
""",
        )
    )
    with pytest.raises(ValueError, match="unknown smoke cases"):
        run_smoke_matrix(manifest, output_root=tmp_path / "out", repo=tmp_path, selected=["nope"])


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


def test_main_list_prints_cases_and_returns_zero(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - id: only_case
    category: calculator
    description: a case
    command: ["true"]
""",
    )
    rc = main([str(manifest_path), "--output-root", str(tmp_path / "out"), "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "only_case" in out
    assert not (tmp_path / "out").exists()


def test_main_returns_nonzero_when_a_case_fails(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - id: failing
    category: calculator
    description: always fails
    command: ["{python}", "-c", "raise SystemExit(1)"]
""",
    )
    rc = main([str(manifest_path), "--output-root", str(tmp_path / "out")])
    assert rc == 1
    out = capsys.readouterr().out
    assert "1 fail" in out


def test_main_strict_blocked_fails_on_blocked_case(tmp_path, monkeypatch):
    monkeypatch.delenv("MMML_TEST_STRICT_BLOCKED_VAR", raising=False)
    manifest_path = _write_manifest(
        tmp_path,
        """
schema_version: 1
cases:
  - id: blocked
    category: calculator
    description: missing env
    requires_env: [MMML_TEST_STRICT_BLOCKED_VAR]
    command: ["true"]
""",
    )
    rc_lenient = main([str(manifest_path), "--output-root", str(tmp_path / "out1")])
    rc_strict = main(
        [str(manifest_path), "--output-root", str(tmp_path / "out2"), "--strict-blocked"]
    )
    assert rc_lenient == 0
    assert rc_strict == 1
