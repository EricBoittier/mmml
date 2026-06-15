"""Unit tests for ORCA external-tool GPU batching."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mmml.interfaces.orca_external.batch_inference import (
    OrcaStructureJob,
    _build_padded_batch,
    evaluate_structures_batched,
)
from mmml.interfaces.orca_external.runner import (
    OrcaPreparedJob,
    clear_calculator_cache,
    prepare_orca_job,
    run_prepared_jobs,
)
from mmml.interfaces.orca_external.server import MmmlOrcaServer
from mmml.interfaces.orca_external.settings import MmmlOrcaSettings


def _write_orca_job(tmp_path: Path, basename: str, positions: list[list[float]]) -> Path:
    lines = [str(len(positions)), basename, *positions]
    xyz_path = tmp_path / f"{basename}.xyz"
    xyz_path.write_text("\n".join(lines) + "\n")
    extinp_path = tmp_path / f"{basename}.extinp.tmp"
    extinp_path.write_text(
        "\n".join(
            [
                f"{basename}.xyz",
                "0",
                "1",
                "1",
                "1",
            ]
        )
    )
    return extinp_path


def test_build_padded_batch_masks_padding() -> None:
    jobs = [
        OrcaStructureJob(
            atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
            do_gradient=True,
        ),
        OrcaStructureJob(
            atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            do_gradient=True,
        ),
    ]
    flat_z, flat_r, dst_idx, src_idx, batch_segments, batch_mask, atom_mask, batch_size = (
        _build_padded_batch(jobs, natoms=4, cutoff=None)
    )

    assert batch_size == 2
    assert flat_z.shape == (8,)
    assert batch_segments.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert atom_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert len(dst_idx) == len(src_idx) == len(batch_mask)


def test_run_prepared_jobs_calls_batch_evaluator(tmp_path: Path, monkeypatch) -> None:
    clear_calculator_cache()
    checkpoint = tmp_path / "dummy.pkl"
    checkpoint.write_bytes(b"")
    ext_a = _write_orca_job(tmp_path, "a_EXT", ["O 0 0 0", "H 0.96 0 0"])
    ext_b = _write_orca_job(tmp_path, "b_EXT", ["O 0 0 0", "H 0.96 0 0"])
    settings = MmmlOrcaSettings(checkpoint=checkpoint)

    job_a = prepare_orca_job(ext_a, settings=settings)
    job_b = prepare_orca_job(ext_b, settings=settings)

    calls: list[int] = []

    def _fake_batch(calculator, jobs):
        calls.append(len(jobs))
        return [(-0.5, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for _ in jobs]

    class _MockCalc:
        pass

    monkeypatch.setattr(
        "mmml.interfaces.orca_external.runner.get_calculator",
        lambda settings: _MockCalc(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.orca_external.batch_inference.evaluate_structures_batched",
        _fake_batch,
    )

    paths = run_prepared_jobs([job_a, job_b])
    assert calls == [2]
    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
    clear_calculator_cache()


def test_server_batches_concurrent_requests(tmp_path: Path, monkeypatch) -> None:
    clear_calculator_cache()
    checkpoint = tmp_path / "dummy.pkl"
    checkpoint.write_bytes(b"")
    ext_a = _write_orca_job(tmp_path, "w0_EXT", ["O 0 0 0", "H 0.96 0 0"])
    ext_b = _write_orca_job(tmp_path, "w1_EXT", ["O 0 0 0", "H 0.96 0 0"])
    settings = MmmlOrcaSettings(checkpoint=checkpoint)

    batch_sizes: list[int] = []
    gate = threading.Event()

    def _fake_run_prepared_jobs(jobs: list[OrcaPreparedJob]) -> list[Path]:
        batch_sizes.append(len(jobs))
        gate.set()
        return [job.input_path.parent / f"{job.extinp.xyz_path.stem}.engrad" for job in jobs]

    monkeypatch.setattr(
        "mmml.interfaces.orca_external.server.run_prepared_jobs",
        _fake_run_prepared_jobs,
    )
    monkeypatch.setattr(
        "mmml.interfaces.orca_external.server.prepare_orca_job_from_arguments",
        lambda arguments, directory, default_settings=None: prepare_orca_job(
            Path(directory) / arguments[0],
            settings=settings,
        ),
    )

    server = MmmlOrcaServer(default_settings=settings, max_batch_size=8, batch_wait_ms=50)

    results: list[dict] = []
    errors: list[Exception] = []

    def _submit(arguments: list[str]) -> None:
        try:
            results.append(server.handle(arguments, str(tmp_path)))
        except Exception as exc:
            errors.append(exc)

    t0 = threading.Thread(target=_submit, args=([ext_a.name],))
    t1 = threading.Thread(target=_submit, args=([ext_b.name],))
    t0.start()
    t1.start()
    t0.join(timeout=5.0)
    t1.join(timeout=5.0)
    gate.wait(timeout=5.0)
    server.shutdown()

    assert not errors
    assert results and all(item["status"] == "Success" for item in results)
    assert batch_sizes == [2]
    clear_calculator_cache()


def test_evaluate_structures_batched_falls_back_sequentially() -> None:
    class _MockCalc:
        def calculate(self, atoms=None, properties=None, system_changes=None):
            self.results = {
                "energy": -27.211386,
                "forces": np.zeros((len(atoms), 3)),
            }

        def get_potential_energy(self, atoms=None):
            return self.results["energy"]

        def get_forces(self, atoms=None):
            return self.results["forces"]

    jobs = [
        OrcaStructureJob(atoms=Atoms("H"), do_gradient=True),
        OrcaStructureJob(atoms=Atoms("H"), do_gradient=True),
    ]
    results = evaluate_structures_batched(_MockCalc(), jobs)
    assert len(results) == 2
    assert all(len(item[1]) == 3 for item in results)
