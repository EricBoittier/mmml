"""Unit tests for ``mmml warmup-mlpot-jax``."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mmml.cli.run import warmup_mlpot_jax as wm


def test_warmup_mlpot_jax_refuses_under_mpirun(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    args = wm.parse_args(["--checkpoint", "/tmp/ckpt.json"])
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        assert wm.run_warmup_mlpot_jax(args) == 2


def test_warmup_mlpot_jax_dry_run(tmp_path, monkeypatch):
    ckpt = tmp_path / "params.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    args = wm.parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--n-monomers",
            "4",
            "--dry-run",
        ]
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ):
        assert wm.run_warmup_mlpot_jax(args) == 0


def test_warmup_mlpot_jax_missing_checkpoint_exits():
    with pytest.raises(SystemExit):
        wm._resolve_checkpoint(None)


def test_warmup_mlpot_jax_resolve_checkpoint_dir(tmp_path):
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    params = ckpt_dir / "params.json"
    params.write_text("{}", encoding="utf-8")
    resolved = wm._resolve_checkpoint(ckpt_dir)
    assert resolved == params.resolve()


def test_warmup_mlpot_jax_run_mocks_build_and_warmup(tmp_path, monkeypatch, capsys):
    ckpt = tmp_path / "DESdimers_params.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)

    model = object()
    build_calls: list[dict] = []
    warmup_calls: list[tuple] = []

    def _build(*_a, **kw):
        build_calls.append(kw)
        return model

    def _warmup(m, pos, **kw):
        warmup_calls.append((m, pos, kw))

    args = wm.parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--n-monomers",
            "2",
            "--box-side",
            "20",
            "--ml-batch-size",
            "32",
            "--do-mm",
            "--quiet",
        ]
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.apply_mlpot_jax_platform_env",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_compile_threads.apply_jax_compile_xla_flags",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_compile_threads.resolve_jax_compile_thread_count",
        return_value=4,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_compilation_cache_dir",
        return_value=Path("/tmp/jax-cache"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.build_decomposed_mlpot_model",
        side_effect=_build,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.warmup_decomposed_mlpot",
        side_effect=_warmup,
    ), mock.patch(
        "mmml.utils.jax_gpu_warmup.maybe_log_jax_compile_timers",
    ):
        assert wm.run_warmup_mlpot_jax(args) == 0

    assert build_calls
    assert build_calls[0]["args"].include_mm is True
    assert build_calls[0]["ml_max_active_dimers"] is None
    assert warmup_calls and warmup_calls[0][0] is model
    assert warmup_calls[0][2]["cell"] == 20.0
    out = capsys.readouterr().out
    assert "warmup-mlpot-jax: done" in out


def test_warmup_mlpot_jax_dry_run_resolves_dimer_cap_and_cutoffs(tmp_path, monkeypatch, capsys):
    ckpt = tmp_path / "params.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    args = wm.parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--n-monomers",
            "60",
            "--atoms-per-monomer",
            "5",
            "--box-side",
            "32",
            "--ml-max-active-dimers",
            "1770",
            "--mm-switch-on",
            "6.0",
            "--mm-switch-width",
            "4.0",
            "--ml-switch-width",
            "1.0",
            "--dry-run",
        ]
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ):
        assert wm.run_warmup_mlpot_jax(args) == 0
    out = capsys.readouterr().out
    assert "ml_max_active_dimers: 1770 -> cap 1770" in out
    assert "handoff: mm_switch_on=6.0 mm_switch_width=4.0 ml_switch_width=1.0" in out


def test_warmup_mlpot_jax_passes_cutoffs_and_dimer_cap_to_build(tmp_path, monkeypatch):
    ckpt = tmp_path / "params.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    build_calls: list[dict] = []

    def _build(*_a, **kw):
        build_calls.append(kw)
        return object()

    args = wm.parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--n-monomers",
            "4",
            "--box-side",
            "20",
            "--ml-max-active-dimers",
            "12",
            "--mm-switch-on",
            "6.0",
            "--do-mm",
            "--quiet",
        ]
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.apply_mlpot_jax_platform_env",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_compile_threads.apply_jax_compile_xla_flags",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_compile_threads.resolve_jax_compile_thread_count",
        return_value=4,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_compilation_cache_dir",
        return_value=Path("/tmp/jax-cache"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.build_decomposed_mlpot_model",
        side_effect=_build,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.warmup_decomposed_mlpot",
    ), mock.patch(
        "mmml.utils.jax_gpu_warmup.maybe_log_jax_compile_timers",
    ):
        assert wm.run_warmup_mlpot_jax(args) == 0

    assert build_calls
    assert build_calls[0]["ml_max_active_dimers"] == 12
    build_args = build_calls[0]["args"]
    assert build_args.mm_switch_on == 6.0
    assert build_args.include_mm is True


def test_cli_dispatch_warmup_mlpot_jax(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["mmml", "warmup-mlpot-jax", "--dry-run", "--checkpoint", "/x/params.json"],
    )
    with mock.patch(
        "mmml.cli.run.warmup_mlpot_jax.main",
        return_value=0,
    ) as mock_main:
        from mmml.cli import __main__ as cli_main

        assert cli_main.main() == 0
    mock_main.assert_called_once()


def test_serial_vs_mpirun_dry_run_without_checkpoint():
    import subprocess
    import sys

    mod_path = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/08_serial_vs_mpirun_md_system.py"
    )
    proc = subprocess.run(
        [sys.executable, str(mod_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Serial (MMML_NO_MPI_RERUN=1" in proc.stdout
    assert "MMML_MPI_NP=1" in proc.stdout


def test_serial_vs_mpirun_environment_snapshot(monkeypatch):
    import importlib.util
    import sys

    mod_path = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/08_serial_vs_mpirun_md_system.py"
    )
    spec = importlib.util.spec_from_file_location("serial_probe_env", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["serial_probe_env"] = mod
    spec.loader.exec_module(mod)

    monkeypatch.delenv("MMML_MLPOT_DEVICE", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    snap = mod._environment_snapshot(overrides={"OMP_NUM_THREADS": "1", "MMML_MPI_NP": "1"})
    assert snap["OMP_NUM_THREADS"] == "1"
    assert snap["MMML_MPI_NP"] == "1"
    assert "MMML_MLPOT_DEVICE" in snap
