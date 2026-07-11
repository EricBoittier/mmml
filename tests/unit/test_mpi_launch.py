"""Tests for orthogonal OpenMPI/JAX launch policy construction."""

from pathlib import Path
import os
import sys

import pytest

from mmml.cli.run.mpi_launch import build_launch_plan, main


def _plan(command=("md-system", "--config", "run.yaml"), **kwargs):
    return build_launch_plan(
        command,
        environ={"MMML_ALLOCATED_CPUS": "16"},
        python="/tmp/uv-env/bin/python",
        wrapper=Path("/repo/scripts/mmml-charmm-mpirun.sh"),
        **kwargs,
    )


def test_uv_interpreter_is_forwarded_to_wrapper():
    plan = _plan()
    assert plan.argv == (
        "/repo/scripts/mmml-charmm-mpirun.sh",
        "md-system",
        "--config",
        "run.yaml",
    )
    assert plan.env["MMML_PYTHON"] == "/tmp/uv-env/bin/python"
    assert plan.env["MMML_MPI_NP"] == "1"
    assert plan.env["MMML_JAX_MODE"] == "gpu-single"


def test_cpu_threaded_policy_is_independent_of_charmm_threads():
    plan = _plan(
        mpi_ranks=2,
        jax_mode="cpu-threaded",
        jax_cpu_threads=6,
        charmm_omp_threads=2,
    )
    assert plan.env["JAX_PLATFORMS"] == "cpu"
    assert plan.env["MMML_JAX_CPU_THREADS"] == "6"
    assert plan.env["MMML_CHARMM_OMP_THREADS"] == "2"
    assert plan.env["OMP_NUM_THREADS"] == "2"
    assert "intra_op_parallelism_threads=6" in plan.env["XLA_FLAGS"]
    assert not plan.warnings


def test_cpu_threads_default_to_allocation_divided_by_ranks():
    plan = _plan(mpi_ranks=4, jax_mode="cpu-threaded")
    assert plan.env["MMML_JAX_CPU_THREADS"] == "4"


def test_spatial_and_rank0_policies_set_distinct_flags():
    spatial = _plan(mpi_ranks=2, jax_mode="spatial")
    assert spatial.env["MMML_MLPOT_SPATIAL_MPI"] == "1"
    assert spatial.env["MMML_MPI_PIN_GPU_PER_RANK"] == "1"

    rank0 = _plan(mpi_ranks=4, jax_mode="rank0")
    assert rank0.env["MMML_MLPOT_RANK0_BRIDGE"] == "1"
    assert rank0.env["MMML_MPI_PIN_GPU_PER_RANK"] == "0"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mpi_ranks": 0}, "mpi_ranks"),
        ({"mpi_ranks": 2, "jax_mode": "gpu-single"}, "exactly one"),
        ({"mpi_ranks": 1, "jax_mode": "spatial"}, "at least two"),
        ({"jax_cpu_threads": 0}, "jax_cpu_threads"),
        ({"charmm_omp_threads": 0}, "charmm_omp_threads"),
        ({"preset": "spatial"}, "requires --mpi-ranks"),
    ],
)
def test_invalid_policy_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _plan(**kwargs)


def test_oversubscription_warns_or_fails_strictly():
    plan = _plan(mpi_ranks=4, jax_mode="cpu-threaded", jax_cpu_threads=8)
    assert "32 CPU threads" in plan.warnings[0]
    with pytest.raises(ValueError, match="32 CPU threads"):
        _plan(
            mpi_ranks=4,
            jax_mode="cpu-threaded",
            jax_cpu_threads=8,
            strict_resources=True,
        )


def test_presets_are_convenience_aliases():
    cpu = _plan(preset="cpu", mpi_ranks=7, jax_mode="gpu-per-rank")
    assert cpu.env["MMML_MPI_NP"] == "1"
    assert cpu.env["MMML_JAX_MODE"] == "cpu-threaded"

    spatial = _plan(preset="spatial", mpi_ranks=3)
    assert spatial.env["MMML_JAX_MODE"] == "spatial"


def test_dry_run_does_not_execute_wrapper(monkeypatch, capsys):
    monkeypatch.setattr(sys, "executable", "/tmp/uv-env/bin/python")
    assert main(["--preset", "cpu", "--dry-run", "--", "md-system"]) == 0
    output = capsys.readouterr().out
    assert "MMML_JAX_MODE=cpu-threaded" in output
    assert "MMML_PYTHON=/tmp/uv-env/bin/python" in output
    assert "mmml-charmm-mpirun.sh md-system" in output


def test_gpu_per_rank_policy_pins_without_enabling_spatial(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        pin_cuda_for_spatial_mpi,
    )

    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    monkeypatch.setenv("MMML_JAX_MODE", "gpu-per-rank")
    monkeypatch.setenv("MMML_MPI_PIN_GPU_PER_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2")
    assert pin_cuda_for_spatial_mpi()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_cpu_policy_survives_charmm_openmp_pinning(monkeypatch):
    from mmml.interfaces.pycharmmInterface import charmm_mpi

    monkeypatch.setattr(charmm_mpi, "charmm_lib_links_mpi", lambda: True)
    monkeypatch.setenv("MMML_JAX_MODE", "cpu-threaded")
    monkeypatch.setenv("MMML_NO_JAX_COMPILE_THREADS", "1")
    monkeypatch.setenv("MMML_CHARMM_OMP_THREADS", "3")
    charmm_mpi._pin_charmm_openmp_for_serial_mlpot()
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MMML_NO_JAX_COMPILE_THREADS"] == "1"
