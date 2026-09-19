"""Unit tests for the ASV GPU benchmark runner (no GPU, no asv run)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.gpu_bench_lib import (
    CheckResult,
    TimingRow,
    all_blocking_checks_passed,
    build_asv_publish_argv,
    build_asv_run_argv,
    check_jax_gpu_device,
    extract_timing_rows,
    finite_array_report,
    force_energy_relative_error,
    format_bench_value,
    latest_asv_result_files,
    prepare_bench_env,
    render_gpu_report_html,
    render_gpu_report_json,
    resolve_asv_command,
    run_gpu_benchmark,
)


def test_check_jax_gpu_device_accepts_gpu():
    devices = [SimpleNamespace(platform="gpu")]
    result = check_jax_gpu_device(devices)
    assert result.passed
    assert result.status == "pass"
    assert result.name == "jax_gpu"


def test_check_jax_gpu_device_rejects_cpu():
    devices = [SimpleNamespace(platform="cpu")]
    result = check_jax_gpu_device(devices)
    assert not result.passed
    assert result.status == "fail"
    assert "cpu" in result.detail.lower()


def test_check_jax_gpu_device_rejects_empty():
    result = check_jax_gpu_device([])
    assert not result.passed
    assert "no JAX devices" in result.detail


def test_finite_array_report_passes():
    info = finite_array_report("energy", np.array([1.25, -0.5]))
    assert info["finite"] is True
    assert info["abs_max"] == pytest.approx(1.25)


def test_finite_array_report_rejects_nan():
    info = finite_array_report("forces", np.array([1.0, np.nan]))
    assert info["finite"] is False


def test_force_energy_relative_error_quadratic():
    # E = 0.5 x^2 at x=2 → F = -2. Central difference with eps=1e-4.
    x = 2.0
    eps = 1e-4
    e0 = 0.5 * x * x
    e_plus = 0.5 * (x + eps) ** 2
    e_minus = 0.5 * (x - eps) ** 2
    err = force_energy_relative_error(e0, e_plus, e_minus, force_component=-x, eps=eps)
    assert err < 1e-6


def test_force_energy_relative_error_detects_mismatch():
    err = force_energy_relative_error(0.0, 1.0, -1.0, force_component=0.0, eps=0.1)
    assert err > 0.5


def test_all_blocking_checks_passed_skips_do_not_block():
    checks = [
        CheckResult("gpu", "pass", "ok"),
        CheckResult("physnet", "skip", "unavailable"),
    ]
    assert all_blocking_checks_passed(checks)


def test_all_blocking_checks_passed_fails_on_fail():
    checks = [
        CheckResult("gpu", "pass", "ok"),
        CheckResult("physnet", "fail", "nan energy"),
    ]
    assert not all_blocking_checks_passed(checks)


def test_build_asv_run_argv_requires_commit_hash():
    argv = build_asv_run_argv(
        commit="abc123def",
        bench="MDSystemSize",
        append_samples=True,
        machine="gpu-a100",
    )
    assert argv[:3] == ["run", "--set-commit-hash", "abc123def"]
    assert "--bench" in argv
    assert argv[argv.index("--bench") + 1] == "MDSystemSize"
    assert "--append-samples" in argv
    assert argv[argv.index("--machine") + 1] == "gpu-a100"


def test_build_asv_publish_argv():
    assert build_asv_publish_argv() == ["publish"]


def test_resolve_asv_command_prefers_venv(tmp_path: Path):
    asv = tmp_path / ".venv" / "bin" / "asv"
    asv.parent.mkdir(parents=True)
    asv.write_text("#!/bin/sh\n")
    asv.chmod(0o755)
    assert resolve_asv_command(tmp_path) == [str(asv)]


def test_resolve_asv_command_falls_back_to_uv(tmp_path: Path):
    assert resolve_asv_command(tmp_path) == ["uv", "run", "asv"]


def test_prepare_bench_env_sets_gpu_defaults(monkeypatch, tmp_path: Path):
    for key in (
        "MMML_BENCH_X64",
        "JAX_ENABLE_X64",
        "OMP_NUM_THREADS",
        "JAX_PLATFORMS",
        "MMML_CKPT",
    ):
        monkeypatch.delenv(key, raising=False)
    ckpt = tmp_path / "examples" / "ckpts_json" / "DESdimers_params.json"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("{}")
    env = prepare_bench_env(tmp_path, allow_cpu=False)
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["MMML_BENCH_X64"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MMML_CKPT"].endswith("DESdimers_params.json")


def test_prepare_bench_env_respects_allow_cpu(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    env = prepare_bench_env(tmp_path, allow_cpu=True)
    assert "JAX_PLATFORMS" not in env or env.get("JAX_PLATFORMS") != "cuda"


def test_latest_asv_result_files_skips_metadata(tmp_path: Path):
    machine = tmp_path / "host"
    machine.mkdir()
    (machine / "machine.json").write_text("{}")
    (tmp_path / "benchmarks.json").write_text("{}")
    keep = machine / "deadbeef-existing-py_same.json"
    keep.write_text("{}")
    found = latest_asv_result_files(tmp_path)
    assert found == [keep]


def test_extract_timing_rows_asv_v2_columns():
    payload = {
        "commit_hash": "abc123",
        "date": 1_700_000_000_000,
        "result_columns": ["result"],
        "results": {
            "bench_ml_physnet.PhysNetSystemSize.time_energy": {
                "result": [0.012, 0.020],
                "params": [["20", "40"]],
                "param_names": ["n_atoms"],
            }
        },
    }
    rows = extract_timing_rows(payload)
    assert [r.name for r in rows] == [
        "bench_ml_physnet.PhysNetSystemSize.time_energy",
        "bench_ml_physnet.PhysNetSystemSize.time_energy",
    ]
    assert rows[0].params == {"n_atoms": "20"}
    assert rows[0].value == pytest.approx(0.012)
    assert rows[1].params == {"n_atoms": "40"}


def test_extract_timing_rows_simple_mapping():
    payload = {
        "commit_hash": "fff",
        "results": {
            "bench_md_driver.MDSystemSize.track_ns_per_day": 42.5,
        },
    }
    rows = extract_timing_rows(payload)
    assert len(rows) == 1
    assert rows[0].value == pytest.approx(42.5)


def test_format_bench_value_units():
    assert "ms" in format_bench_value("bench.time_energy", 0.012)
    assert "ns/day" in format_bench_value("bench.track_ns_per_day", 12.3)
    assert format_bench_value("bench.track_pairs", 128.0) == "128"


def test_render_html_includes_correctness_timings_and_publish_note():
    checks = [
        CheckResult("jax_gpu", "pass", "gpu:0", values={"platform": "gpu"}),
        CheckResult("physnet", "fail", "energy was nan"),
    ]
    timings = [
        TimingRow(
            name="bench_md_driver.MDSystemSize.track_ns_per_day",
            params={"n_waters": "512"},
            value=18.4,
        )
    ]
    html = render_gpu_report_html(
        meta={
            "commit": "abc1234deadbeef",
            "hostname": "gpu-node",
            "gpu": "NVIDIA A100",
            "dirty": True,
        },
        checks=checks,
        timings=timings,
        asv_html_href="index.html",
    )
    assert "<!DOCTYPE html>" in html
    assert "FAIL" in html
    assert "jax_gpu" in html
    assert "physnet" in html
    assert "energy was nan" in html
    assert "ns/day" in html
    assert "index.html" in html
    assert "asv publish" in html
    assert "not uploaded from CI" in html
    # user-controlled strings must be escaped
    sneaky = render_gpu_report_html(
        meta={"hostname": "<script>alert(1)</script>"},
        checks=[CheckResult("x", "pass", "<img>")],
        timings=[],
    )
    assert "<script>alert(1)</script>" not in sneaky
    assert "<img>" not in sneaky


def test_render_json_round_trip():
    checks = [CheckResult("jax_gpu", "pass", "ok")]
    payload = json.loads(
        render_gpu_report_json(
            meta={"commit": "abc"},
            checks=checks,
            timings=[TimingRow("a.time_energy", {"n": "1"}, 0.01)],
        )
    )
    assert payload["meta"]["commit"] == "abc"
    assert payload["checks"][0]["name"] == "jax_gpu"
    assert payload["timings"][0]["value"] == pytest.approx(0.01)
    assert payload["ok"] is True


def test_run_gpu_benchmark_skips_asv_when_correctness_fails(
    monkeypatch, tmp_path: Path
):
    called: list[str] = []
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.collect_run_meta",
        lambda **_k: {"commit": "abc", "dirty": False, "hostname": "t"},
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.run_correctness_checks",
        lambda **_k: [CheckResult("jax_gpu", "fail", "cpu backend")],
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.run_asv",
        lambda **_k: called.append("asv"),
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.publish_asv",
        lambda **_k: called.append("publish"),
    )
    out = tmp_path / "html"
    rc = run_gpu_benchmark(
        Namespace(
            bench=None,
            append_samples=False,
            checks_only=False,
            skip_checks=False,
            force=False,
            allow_cpu=False,
            publish=True,
            output_dir=out,
            repo_root=tmp_path,
        )
    )
    assert rc == 1
    assert called == []
    report = out / "gpu-report.html"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "cpu backend" in text
    assert (out / "gpu-report.json").is_file()


def test_run_gpu_benchmark_times_after_checks_pass(monkeypatch, tmp_path: Path):
    called: list[str] = []
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.collect_run_meta",
        lambda **_k: {"commit": "abc123", "dirty": False, "hostname": "t"},
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.run_correctness_checks",
        lambda **_k: [CheckResult("jax_gpu", "pass", "gpu")],
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.run_asv",
        lambda **_k: called.append("asv"),
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.publish_asv",
        lambda **_k: called.append("publish"),
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.load_latest_timings",
        lambda **_k: [TimingRow("bench.time_energy", {}, 0.02)],
    )
    out = tmp_path / "html"
    rc = run_gpu_benchmark(
        Namespace(
            bench="bench_ml_physnet",
            append_samples=False,
            checks_only=False,
            skip_checks=False,
            force=False,
            allow_cpu=False,
            publish=True,
            output_dir=out,
            repo_root=tmp_path,
        )
    )
    assert rc == 0
    assert called == ["asv", "publish"]
    html = (out / "gpu-report.html").read_text(encoding="utf-8")
    assert "time_energy" in html


def test_run_gpu_benchmark_checks_only_skips_timing(monkeypatch, tmp_path: Path):
    called: list[str] = []
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.collect_run_meta",
        lambda **_k: {"commit": "abc", "dirty": False, "hostname": "t"},
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.run_correctness_checks",
        lambda **_k: [CheckResult("jax_gpu", "pass", "gpu")],
    )
    monkeypatch.setattr(
        "benchmarks.gpu_bench_lib.run_asv",
        lambda **_k: called.append("asv"),
    )
    out = tmp_path / "html"
    rc = run_gpu_benchmark(
        Namespace(
            bench=None,
            append_samples=False,
            checks_only=True,
            skip_checks=False,
            force=False,
            allow_cpu=True,
            publish=True,
            output_dir=out,
            repo_root=tmp_path,
        )
    )
    assert rc == 0
    assert called == []
