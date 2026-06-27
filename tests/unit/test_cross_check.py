"""Unit tests for supplementary QC cross-check mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mmml.interfaces.qc_backends.factory import build_backend
from mmml.interfaces.qc_backends.ml_backend import MLBackend
from mmml.interfaces.qc_backends.npz_output import (
    infer_target_units,
    normalize_backend_npz,
    stack_frame_results,
)
from mmml.interfaces.qc_backends.orca_qm import (
    OrcaQMBackend,
    parse_orca_out_energy,
    read_orca_engrad,
    render_orca_input,
)
from mmml.interfaces.qc_backends.protocol import BackendSpec
from mmml.interfaces.qc_backends.pyscf_backend import PySCFBackend
from mmml.interfaces.qc_backends.runner import CrossCheckConfig, CrossCheckRunner
from mmml.interfaces.qc_backends.structures import load_structures_npz


@pytest.fixture
def water_npz(tmp_path: Path) -> Path:
    r = np.array(
        [
            [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            [[0.0, 0.0, 0.1200], [0.0, 0.7600, -0.4700], [0.0, -0.7600, -0.4700]],
        ],
        dtype=np.float64,
    )
    z = np.array([8, 1, 1], dtype=np.int32)
    e = np.array([-76.0, -76.01], dtype=np.float64)
    f = np.zeros((2, 3, 3), dtype=np.float64)
    path = tmp_path / "water.npz"
    np.savez(path, R=r, Z=z, N=np.array([3, 3], dtype=np.int32), E=e, F=f)
    return path


def test_stack_frame_results():
    frames_z = [np.array([8, 1, 1]), np.array([8, 1, 1])]
    frames_r = [np.zeros((3, 3)), np.ones((3, 3))]
    out = stack_frame_results(
        energies=[-1.0, -2.0],
        forces=[np.ones((3, 3)), np.zeros((3, 3))],
        dipoles=None,
        frames_z=frames_z,
        frames_r=frames_r,
    )
    assert out["E"].shape == (2,)
    assert out["F"].shape == (2, 3, 3)
    assert out["N"].tolist() == [3, 3]


def test_infer_target_units_hartree():
    ref = {"E": np.array([-76.0, -76.1])}
    e_unit, f_unit = infer_target_units(ref)
    assert e_unit == "hartree"
    assert f_unit == "hartree_bohr"


def test_load_structures_npz(water_npz: Path):
    frames, arrays = load_structures_npz(water_npz, max_frames=1, stride=1)
    assert len(frames) == 1
    assert len(frames[0]) == 3
    assert arrays["E"].shape[0] == 1


def test_pyscf_backend_mock():
    def fake_compute(r, z, **kwargs):
        return {
            "energy": np.array(-76.4),
            "gradient": np.zeros((len(z), 3)),
        }

    backend = PySCFBackend(compute_fn=fake_compute)
    frames = [Atoms("H2O")]
    out = backend.evaluate_batch(frames, properties=frozenset({"energy", "forces"}))
    assert out["E"].shape == (1,)
    assert out["F"].shape == (1, 3, 3)


def test_ml_backend_mock(tmp_path: Path):
    class FakeCalc:
        model = type("M", (), {"natoms": 3})()

        def get_potential_energy(self, atoms=None):
            return -100.0

        def get_forces(self, atoms=None):
            return np.zeros((3, 3))

        def get_dipole_moment(self):
            return np.zeros(3)

    def factory(_checkpoint):
        return FakeCalc()

    backend = MLBackend(checkpoint=tmp_path / "ckpt.json", calculator_factory=factory)
    out = backend.evaluate_batch([Atoms("H2O")], properties=frozenset({"energy", "forces"}))
    assert out["E"][0] == pytest.approx(-100.0)


def test_cross_check_runner_with_reference_npz(water_npz: Path, tmp_path: Path):
    ref_npz = water_npz
    pred_e = np.array([-75.9, -75.95], dtype=np.float64)
    pred_f = np.zeros((2, 3, 3), dtype=np.float64)

    def fake_compute(r, z, **kwargs):
        idx = 0
        return {
            "energy": np.array(pred_e[idx]),
            "gradient": -np.zeros((len(z), 3)),
        }

    class ShiftingPySCF(PySCFBackend):
        def evaluate_batch(self, frames, *, properties):
            energies = []
            forces = []
            frames_z = []
            frames_r = []
            for i, atoms in enumerate(frames):
                z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
                r = np.asarray(atoms.get_positions(), dtype=np.float64)
                frames_z.append(z)
                frames_r.append(r)
                energies.append(float(pred_e[i]))
                forces.append(pred_f[i])
            return stack_frame_results(
                energies=energies,
                forces=forces,
                dipoles=None,
                frames_z=frames_z,
                frames_r=frames_r,
            )

    config = CrossCheckConfig(
        structures=water_npz,
        output_dir=tmp_path / "out",
        reference_npz=ref_npz,
        backends=[BackendSpec(name="pyscf", options={})],
        max_frames=2,
        no_plots=True,
    )

    runner = CrossCheckRunner(config)
    original_build = build_backend

    def patched_build(spec):
        if spec.name == "pyscf":
            return ShiftingPySCF(compute_fn=fake_compute)
        return original_build(spec)

    import mmml.interfaces.qc_backends.runner as runner_mod

    runner_mod.build_backend = patched_build
    try:
        summary = runner.run()
    finally:
        runner_mod.build_backend = original_build

    assert summary["n_frames"] == 2
    assert "pyscf" in summary["backends"]
    metrics = summary["backends"]["pyscf"]["metrics"]
    assert metrics["energy"]["mae"] == pytest.approx(0.1, abs=0.05)


def test_read_orca_engrad(tmp_path: Path):
    text = """#
# Number of atoms
#
3
#
# Total energy [Eh]
#
-1.234567890000e+02
#
# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...
#
 1.000000000000e-03
 2.000000000000e-03
 3.000000000000e-03
 4.000000000000e-03
 5.000000000000e-03
 6.000000000000e-03
 7.000000000000e-03
 8.000000000000e-03
 9.000000000000e-03
"""
    path = tmp_path / "job.engrad"
    path.write_text(text)
    energy, grad = read_orca_engrad(path)
    assert energy == pytest.approx(-123.456789)
    assert grad is not None
    assert grad.shape == (3, 3)


def test_parse_orca_out_energy():
    text = "Some output\nFINAL SINGLE POINT ENERGY      -76.123456789\n"
    assert parse_orca_out_energy(text) == pytest.approx(-76.123456789)


def test_render_orca_input_template():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    text = render_orca_input(
        atoms=atoms,
        charge=0,
        multiplicity=1,
        template="! {method} {basis}\n{xyz}\n",
        method="PBE",
        basis="def2-SVP",
        pal=1,
    )
    assert "PBE" in text
    assert "O" in text


def test_orca_backend_mock_run():
    def fake_run(atoms, workdir, backend):
        engrad = workdir / "job.engrad"
        engrad.write_text("3\n-76.0\n" + "\n".join(["0.0"] * 9) + "\n")
        return read_orca_engrad(engrad)

    backend = OrcaQMBackend(run_orca=fake_run)
    out = backend.evaluate_batch([Atoms("H2O")], properties=frozenset({"energy", "forces"}))
    assert out["E"][0] == pytest.approx(-76.0)


def test_molpro_backend_mock():
    from mmml.interfaces.qc_backends.molpro import MolproBackend

    def fake_run(atoms, workdir, backend):
        return {
            "E": np.array([-76.0]),
            "F": np.zeros((1, 3, 3)),
        }

    backend = MolproBackend(run_molpro=fake_run, converter=None)
    out = backend.evaluate_batch([Atoms("H2O")], properties=frozenset({"energy", "forces"}))
    assert out["E"][0] == pytest.approx(-76.0)


def test_normalize_backend_npz_roundtrip():
    data = {"E": np.array([1.0]), "F": np.array([[[0.1, 0.0, 0.0]]])}
    out = normalize_backend_npz(
        data,
        backend="pyscf",
        target_energy_unit="ev",
        target_force_unit="ev_angstrom",
    )
    assert out["E"][0] > 20.0
