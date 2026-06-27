"""Unit tests for post-run md-system advice."""

from __future__ import annotations

import json
from pathlib import Path

from mmml.cli.run.md_run_advice import (
    RestartCandidate,
    build_run_advice,
    collect_restart_candidates,
    select_restart_candidate,
    write_run_advice_files,
)


def _write_restart(path: Path, *, natom: int = 10) -> None:
    path.write_text(f"REST\n NATOM {natom}\n", encoding="ascii")


def test_select_restart_prefers_lowest_grms_on_failure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.res"
    heat = tmp_path / "heat.res"
    _write_restart(baseline)
    _write_restart(heat)
    candidates = [
        RestartCandidate(
            path=baseline,
            label="baseline",
            leg="baseline",
            hybrid_grms=120.0,
            source="test",
            mtime=1.0,
            is_restart=True,
        ),
        RestartCandidate(
            path=heat,
            label="heat",
            leg="heat",
            hybrid_grms=45.0,
            source="test",
            mtime=2.0,
            is_restart=True,
        ),
    ]
    picked = select_restart_candidate(candidates, failed=True)
    assert picked is not None
    assert picked.path == heat
    assert picked.hybrid_grms == 45.0


def test_select_restart_tiebreaks_by_mtime_at_similar_grms(tmp_path: Path) -> None:
    from mmml.cli.run.md_run_advice import RestartCandidate

    older = tmp_path / "baseline.res"
    newer = tmp_path / "heat.res"
    _write_restart(older)
    _write_restart(newer)
    candidates = [
        RestartCandidate(
            path=older,
            label="baseline",
            leg="baseline",
            hybrid_grms=50.0,
            source="test",
            mtime=1.0,
            is_restart=True,
        ),
        RestartCandidate(
            path=newer,
            label="heat seg",
            leg="heat",
            hybrid_grms=50.2,
            source="test",
            mtime=5.0,
            is_restart=True,
        ),
    ]
    picked = select_restart_candidate(candidates, failed=True)
    assert picked is not None
    assert picked.path == newer


def test_collect_restart_candidates_from_journals(tmp_path: Path) -> None:
    prep = tmp_path / "prep_ladder"
    prep.mkdir()
    crd = prep / "001_initial.crd"
    crd.write_text("dummy crd\n", encoding="ascii")
    journal = {
        "steps": [
            {
                "label": "initial",
                "stem": "001_initial",
                "hybrid_grms_kcalmol_A": 88.5,
                "files": {"crd": str(crd)},
            }
        ]
    }
    (prep / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    _write_restart(tmp_path / "baseline.res")

    candidates = collect_restart_candidates(tmp_path)
    grms_vals = [c.hybrid_grms for c in candidates if c.hybrid_grms is not None]
    assert 88.5 in grms_vals
    assert any(c.path.name == "baseline.res" for c in candidates)


def test_build_run_advice_failure_suggests_resume(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    baseline = out / "baseline.res"
    _write_restart(baseline)
    (out / "cleanup" / "journal.json").parent.mkdir(parents=True)
    (out / "cleanup" / "journal.json").write_text(
        json.dumps({"steps": [{"label": "flyoff", "stem": "001_flyoff"}]}),
        encoding="utf-8",
    )
    summary = {
        "stages": [
            {"stage": "mini", "status": "complete"},
            {"stage": "heat", "status": "error"},
            {"stage": "equi", "status": "planned"},
        ]
    }
    (out / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "job_name": "dcm103_equil",
        "backend": "pycharmm",
        "exit_code": 1,
        "args": {
            "config": "mmml/cli/run/md_system.dcm103_equil.example.yaml",
            "md_stages": "mini,heat,equi",
            "output_dir": str(out),
            "no_echeck_heat": False,
            "heat_thermostat": "hoover",
        },
    }
    advice = build_run_advice(
        manifest=manifest,
        output_dir=out,
        exit_code=1,
        repo_root=tmp_path,
    )
    assert advice is not None
    assert advice.exit_code == 1
    assert advice.restart is not None
    assert advice.restart.path == baseline.resolve()
    assert "heat,equi" in (advice.md_stages or "")
    assert "presets/dynamics-flyoff-strict.yaml" in advice.include_presets
    assert "--restart-from" in advice.command
    assert "--job-id" not in advice.command
    assert "no_echeck_heat" in advice.config_yaml

    paths = write_run_advice_files(advice, out)
    assert paths["yaml"].is_file()
    assert paths["sh"].is_file()
    assert paths["json"].is_file()


def test_build_run_advice_success_continues_remaining_stages(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    _write_restart(out / "heat.res")
    summary = {
        "stages": [
            {"stage": "mini", "status": "complete"},
            {"stage": "heat", "status": "complete"},
            {"stage": "equi", "status": "planned"},
        ]
    }
    (out / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "job_name": "dcm52_equil",
        "backend": "pycharmm",
        "exit_code": 0,
        "args": {
            "md_stages": "mini,heat,equi",
            "output_dir": str(out),
        },
    }
    advice = build_run_advice(
        manifest=manifest,
        output_dir=out,
        exit_code=0,
        repo_root=tmp_path,
    )
    assert advice is not None
    assert advice.md_stages == "equi"
    assert advice.exit_code == 0


def test_select_restart_skips_pretreat_mm_on_failure(tmp_path: Path) -> None:
    pretreat = tmp_path / "pretreat" / "01_mm.crd"
    pretreat.parent.mkdir(parents=True)
    pretreat.write_text("crd\n", encoding="ascii")
    prep = tmp_path / "prep_ladder" / "006_pre_mlpot.crd"
    prep.parent.mkdir(parents=True)
    prep.write_text("crd\n", encoding="ascii")
    candidates = [
        RestartCandidate(
            path=pretreat,
            label="mini: CHARMM CGENFF SD/ABNR before MLpot (MM only)",
            leg="mini",
            hybrid_grms=0.28,
            source="test",
            mtime=1.0,
            is_restart=False,
        ),
        RestartCandidate(
            path=prep,
            label="prep_ladder: pre_mlpot_lattice_full",
            leg="prep_ladder",
            hybrid_grms=88.0,
            source="test",
            mtime=2.0,
            is_restart=False,
        ),
    ]
    picked = select_restart_candidate(candidates, failed=True)
    assert picked is not None
    assert picked.path == prep


def test_select_restart_skips_pretreat_on_success(tmp_path: Path) -> None:
    pretreat = tmp_path / "pretreat" / "01_mm.crd"
    pretreat.parent.mkdir(parents=True)
    pretreat.write_text("crd\n", encoding="ascii")
    heat = tmp_path / "heat.res"
    _write_restart(heat)
    candidates = [
        RestartCandidate(
            path=pretreat,
            label="mini: CHARMM CGENFF SD/ABNR before MLpot (MM only)",
            leg="mini",
            hybrid_grms=1.02,
            source="test",
            mtime=5.0,
            is_restart=False,
        ),
        RestartCandidate(
            path=heat,
            label="heat stage restart",
            leg="heat",
            hybrid_grms=None,
            source="test",
            mtime=3.0,
            is_restart=True,
        ),
    ]
    picked = select_restart_candidate(candidates, failed=False)
    assert picked is not None
    assert picked.path == heat


def test_build_run_advice_full_success_has_no_resume_command(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "mini.crd").write_text("crd\n", encoding="ascii")
    _write_restart(out / "heat.res")
    _write_restart(out / "equi.res")
    summary = {
        "stages": [
            {"stage": "mini", "status": "planned"},
            {"stage": "heat", "status": "complete"},
            {"stage": "equi", "status": "complete"},
        ]
    }
    (out / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "job_name": "dcm52_equil",
        "backend": "pycharmm",
        "exit_code": 0,
        "args": {
            "md_stages": "mini,heat,equi",
            "output_dir": str(out),
        },
    }
    advice = build_run_advice(
        manifest=manifest,
        output_dir=out,
        exit_code=0,
        repo_root=tmp_path,
    )
    assert advice is not None
    assert advice.command == ""
    assert "complete" in advice.headline.lower()
    paths = write_run_advice_files(advice, out)
    sh_text = paths["sh"].read_text(encoding="utf-8")
    assert "CMD=(" in sh_text
    assert "exec " in sh_text or "no resume" in sh_text


def test_shell_script_uses_bash_array(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    _write_restart(out / "baseline.res")
    manifest = {
        "job_name": "dcm52_equil",
        "backend": "pycharmm",
        "exit_code": 1,
        "args": {
            "config": "heat.conf",
            "md_stages": "mini,heat,equi",
            "output_dir": str(out),
        },
    }
    advice = build_run_advice(
        manifest=manifest,
        output_dir=out,
        exit_code=1,
        repo_root=tmp_path,
    )
    assert advice is not None
    assert "\n" not in advice.command
    assert "--md-stages" in advice.command
    assert advice.shell_script.count("CMD=(") == 1
    assert 'exec "${CMD[@]}"' in advice.shell_script
    paths = write_run_advice_files(advice, out)
    assert paths["command"].read_text(encoding="utf-8").strip() == advice.command

