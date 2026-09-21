"""End-to-end smoke: mock reference, no expensive QC, splits stay isolated."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mmml.acquisition.pipeline import (
    load_config,
    load_manifest,
    run_all,
    stage_extract,
    stage_fingerprint_models,
    stage_fit_pca,
    stage_label,
    stage_prepare_pool,
    stage_select,
)
from mmml.acquisition.splits import SPLIT_CANDIDATE, SPLIT_TEST, SPLIT_VALID, assert_split_isolation

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "workflows" / "label_acquisition" / "config.smoke.yaml"


@pytest.fixture
def smoke_cfg():
    return load_config(SMOKE)


def test_smoke_pipeline_writes_report_and_reuses_labels(tmp_path: Path, smoke_cfg):
    out = tmp_path / "run"
    report = run_all(smoke_cfg, out)
    assert report.is_file()
    text = report.read_text()
    assert "mock" in text.lower() or "not scientific" in text.lower()
    summary = json.loads((out / "report" / "summary.json").read_text())
    assert summary["label_accounting"]["unique_calculations"] >= 1
    # Overlap: unique calcs <= eval labels + sum of training selections
    acc = summary["label_accounting"]
    assert acc["unique_calculations"] <= acc["eval_labels"] + acc["sum_nominal_budgets"]
    # Selection manifests are immutable ID lists with the requested budget
    man = json.loads(
        next((out / "selections" / "activation_fps").rglob("manifest.json")).read_text()
    )
    assert man["n_selected"] == man["budget"] == 4
    assert man["immutable"] is True
    # Held-out IDs never appear in a training selection
    pool = load_manifest(out / "pool" / "manifest.json")
    assert_split_isolation(pool.records)
    held = {r.structure_id for r in pool.records if r.split in (SPLIT_VALID, SPLIT_TEST)}
    for path in (out / "selections").rglob("manifest.json"):
        ids = set(json.loads(path.read_text())["structure_ids"])
        assert ids.isdisjoint(held)
    # Unmodified student is a baseline row
    methods = {row["method"] for row in summary["results"]}
    assert "unmodified_student" in methods
    assert "stratified_random" in methods
    # Linear readout equivalence reported
    eq = summary["activation_energy_alignment"]
    assert eq["mean_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_pca_fit_ids_are_candidates_only(tmp_path: Path, smoke_cfg):
    out = tmp_path / "run"
    stage_prepare_pool(smoke_cfg, out)
    stage_fingerprint_models(smoke_cfg, out)
    stage_extract(smoke_cfg, out)
    stage_fit_pca(smoke_cfg, out)
    pool = load_manifest(out / "pool" / "manifest.json")
    held = {r.structure_id for r in pool.records if r.split in (SPLIT_VALID, SPLIT_TEST)}
    cand = {r.structure_id for r in pool.by_split(SPLIT_CANDIDATE)}
    pca = json.loads((out / "pca" / "activations_pca.json").read_text())
    fit_ids = set(pca["fit_ids"])
    assert fit_ids == cand
    assert fit_ids.isdisjoint(held)


def test_label_stage_refuses_to_run_before_selection(tmp_path: Path, smoke_cfg):
    out = tmp_path / "run"
    stage_prepare_pool(smoke_cfg, out)
    with pytest.raises(RuntimeError, match="not finalized"):
        stage_label(smoke_cfg, out)


def test_reference_labels_cannot_enter_extract(tmp_path: Path, smoke_cfg):
    out = tmp_path / "run"
    stage_prepare_pool(smoke_cfg, out)
    stage_fingerprint_models(smoke_cfg, out)
    man_path = out / "pool" / "manifest.json"
    payload = json.loads(man_path.read_text())
    payload["records"][0]["extra"] = {"reference_energy": 1.0}
    # pipeline load_manifest does not currently round-trip extra; inject via records_from extract guard
    from mmml.acquisition.linear_student import init_linear_student
    from mmml.acquisition.representations import extract_activations
    from mmml.acquisition.splits import StructureRecord

    rec = StructureRecord(
        index=0,
        structure_id="x",
        geometry_fingerprint="x",
        composition="H2O1",
        stratum="s",
        group="g",
        n_atoms=3,
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.zeros((3, 3)),
        extra={"reference_energy": 1.23},
    )
    with pytest.raises(ValueError, match="reference labels leaked"):
        extract_activations(init_linear_student(), [rec])


def test_cli_help_does_not_import_jax():
    script = """
import sys
from mmml.cli.parser_utils import get_subcommand_parser
p = get_subcommand_parser("label-acquire")
assert p is not None
assert "jax" not in sys.modules
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_snakemake_smoke_dry_run():
    workflow = REPO / "workflows" / "label_acquisition"
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "snakemake",
            "snakemake",
            "-n",
            "--quiet",
            "--profile",
            "profiles/local",
            "--configfile",
            "config.smoke.yaml",
        ],
        cwd=workflow,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "MMML_WORKFLOW_CONFIG": str(workflow / "config.smoke.yaml")},
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
