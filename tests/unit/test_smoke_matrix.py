from __future__ import annotations

import json
from pathlib import Path

from mmml.validation.smoke_matrix import load_smoke_manifest, run_smoke_matrix


def test_pcstudix_manifest_covers_calculators_backends_and_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_smoke_manifest(
        root / "workflows/validation_campaign/pcstudix_smoke_matrix.yaml"
    )
    ids = {case.id for case in manifest.cases}
    assert {
        "physnet_joint_neutral",
        "spookynet",
        "learned_mbd",
        "learned_multipoles",
        "efield",
        "xtb_tblite",
        "pyscf_hf",
        "dftb3_d4",
        "jax_cgenff_spoof",
        "pycharmm_live",
        "jaxmd_min",
        "jaxmd_nve",
        "jaxmd_nvt",
        "jaxmd_npt",
        "rigid_mc",
        "charge_and_lr_modes",
    }.issubset(ids)
    assert {case.category for case in manifest.cases} == {
        "calculator",
        "backend",
        "policy",
    }


def test_smoke_matrix_records_pass_and_blocked_cases(tmp_path, monkeypatch) -> None:
    artifact_code = (
        "from pathlib import Path; "
        "p=Path(__import__('sys').argv[1]); p.mkdir(parents=True); "
        "(p/'result.json').write_text('{}')"
    )
    manifest_path = tmp_path / "matrix.yaml"
    manifest_path.write_text(
        """
schema_version: 1
cases:
  - id: runnable
    category: calculator
    description: tiny isolated calculation
    command: ["{python}", "-c", "%s", "{output_dir}/artifact"]
    expected_artifacts: ["artifact/result.json"]
  - id: unavailable
    category: backend
    description: explicitly blocked dependency
    requires_env: [MMML_SMOKE_MISSING_TEST_PATH]
    command: ["{python}", "-c", "raise SystemExit(99)"]
"""
        % artifact_code,
        encoding="utf-8",
    )
    monkeypatch.delenv("MMML_SMOKE_MISSING_TEST_PATH", raising=False)

    summary = run_smoke_matrix(
        load_smoke_manifest(manifest_path),
        output_root=tmp_path / "output",
        repo=tmp_path,
    )

    assert summary["counts"] == {"PASS": 1, "FAIL": 0, "BLOCKED": 1}
    runnable = json.loads((tmp_path / "output/runnable/status.json").read_text())
    blocked = json.loads((tmp_path / "output/unavailable/status.json").read_text())
    assert runnable["status"] == "PASS"
    assert blocked["status"] == "BLOCKED"
    assert blocked["returncode"] is None
