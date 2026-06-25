"""Unit tests for md-system --evaluate-npz."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.cli.run.md_evaluate_npz import (
    EvaluateNpzPayload,
    _reference_com_dist_A,
    classify_ml_regime,
    compare_evaluate_to_reference_npz,
    load_evaluate_npz,
    permute_handoff_array_to_evaluator_z,
    positions_for_evaluator_z,
    resolve_evaluate_use_pbc,
    resolve_evaluate_max_frames,
    resolve_reference_units,
    run_evaluate_npz,
    save_evaluate_compare_diagnostics,
    save_evaluate_extxyz,
    save_evaluate_extxyz_multi,
    save_evaluate_trajectory_npz,
    save_evaluate_trajectory_npz_multi,
    should_evaluate_reference_trajectory,
)
from mmml.cli.run.md_handoff import MdHandoffState


def test_load_evaluate_npz_optional_mm_fields(tmp_path: Path) -> None:
    pos = np.random.default_rng(0).random((6, 3))
    z = np.array([6, 1, 1, 6, 1, 1], dtype=np.int32)
    charges = np.linspace(-0.3, 0.3, 6)
    iac = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
    path = tmp_path / "geom.npz"
    np.savez_compressed(
        path,
        positions=pos,
        atomic_numbers=z,
        pbc=False,
        metadata=json.dumps({"source": "test"}),
        charges=charges,
        iac=iac,
    )

    payload = load_evaluate_npz(path)
    assert isinstance(payload, EvaluateNpzPayload)
    assert payload.charges is not None
    assert payload.at_codes is not None
    assert int(payload.at_codes.min()) == 0
    np.testing.assert_allclose(payload.charges, charges)
    np.testing.assert_allclose(payload.handoff.positions, pos)


def test_load_evaluate_npz_positions_only(tmp_path: Path) -> None:
    pos = np.random.default_rng(0).random((10, 3))
    path = tmp_path / "geom.npz"
    np.savez_compressed(path, positions=pos, pbc=False, metadata="{}")

    payload = load_evaluate_npz(path)
    np.testing.assert_allclose(payload.handoff.positions, pos)
    assert payload.handoff.atomic_numbers.size == 0


def test_evaluate_int_arg_treats_none_as_default() -> None:
    from mmml.cli.run.md_evaluate_npz import _evaluate_int_arg

    args = Namespace(max_pairs=None, jax_md_update_interval=None)
    assert _evaluate_int_arg(args, "max_pairs", 20_000) == 20_000
    assert _evaluate_int_arg(args, "jax_md_update_interval", 1) == 1
    assert _evaluate_int_arg(Namespace(), "max_pairs", 20_000) == 20_000


def test_save_evaluate_trajectory_npz_roundtrip(tmp_path: Path) -> None:
    z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    r = np.random.default_rng(0).random((5, 3))
    f = np.random.default_rng(1).random((5, 3))
    path = tmp_path / "evaluate.npz"
    save_evaluate_trajectory_npz(
        path,
        atomic_numbers=z,
        positions=r,
        energy_eV=-1.5,
        forces_eV_A=f,
    )
    with np.load(path) as data:
        assert int(data["N"][0]) == 5
        np.testing.assert_array_equal(data["Z"][0], z)
        np.testing.assert_allclose(data["R"][0], r)
        np.testing.assert_allclose(data["F"][0], f)
        assert data["E_eV"][0] == pytest.approx(-1.5)


def test_compare_evaluate_to_reference_npz(tmp_path: Path) -> None:
    z = np.array([6, 1, 1, 17, 17, 6, 1, 1, 17, 17], dtype=np.int32)
    r = np.random.default_rng(2).random((10, 3))
    ref_e_hartree = -1.6 / 27.211386245988
    ref_f = np.random.default_rng(3).random((1, 10, 3))
    ref_path = tmp_path / "ref.npz"
    np.savez_compressed(
        ref_path,
        N=np.array([10], dtype=int),
        Z=z.reshape(1, 10),
        R=r.reshape(1, 10, 3),
        E=np.array([ref_e_hartree], dtype=np.float64),
        F=ref_f,
        source_indices=np.array([41950], dtype=int),
    )
    pred_e = -1.6
    pred_f = ref_f[0] + 0.01
    cmp = compare_evaluate_to_reference_npz(
        ref_path,
        frame=0,
        atomic_numbers=z,
        positions=r,
        energy_eV=pred_e,
        forces_eV_A=pred_f,
        reference_energy_unit="hartree",
        reference_force_unit="hartree_bohr",
    )
    assert cmp["delta_energy_eV"] == pytest.approx(0.0, abs=1e-6)
    assert cmp["reference_source_index"] == 41950
    assert "force_rmse_eV_A" in cmp


def test_compare_evaluate_reorders_psf_order_reference(tmp_path: Path) -> None:
    perm = np.array([0, 3, 4, 1, 2], dtype=int)
    charmm_z = np.array([6, 1, 1, 17, 17, 6, 1, 1, 17, 17], dtype=np.int32)
    charmm_r = np.random.default_rng(4).random((10, 3))
    psf_z = charmm_z.reshape(2, 5)[:, perm].reshape(-1)
    psf_r = charmm_r.reshape(2, 5, 3)[:, perm, :].reshape(10, 3)
    ref_e_hartree = -1.6 / 27.211386245988
    ref_path = tmp_path / "ref_psf_order.npz"
    np.savez_compressed(
        ref_path,
        N=np.array([10], dtype=int),
        Z=psf_z.reshape(1, 10),
        R=psf_r.reshape(1, 10, 3),
        E=np.array([ref_e_hartree], dtype=np.float64),
        F=np.random.default_rng(5).random((1, 10, 3)),
    )
    cmp = compare_evaluate_to_reference_npz(
        ref_path,
        frame=0,
        atomic_numbers=charmm_z,
        positions=charmm_r,
        energy_eV=-1.6,
        forces_eV_A=np.ones((10, 3)),
        reference_energy_unit="hartree",
        reference_force_unit="hartree_bohr",
    )
    assert cmp["reference_reordered"] is True
    assert cmp["position_rmsd_A"] == pytest.approx(0.0, abs=1e-12)
    assert cmp["delta_energy_eV"] == pytest.approx(0.0, abs=1e-6)


def test_positions_for_evaluator_z_reorders_psf_order_dcm_like() -> None:
    perm = np.array([0, 3, 4, 1, 2], dtype=int)
    charmm_z = np.array([6, 1, 1, 17, 17, 6, 1, 1, 17, 17], dtype=np.int32)
    charmm_r = np.random.default_rng(7).random((10, 3))
    psf_z = charmm_z.reshape(2, 5)[:, perm].reshape(-1)
    psf_r = charmm_r.reshape(2, 5, 3)[:, perm, :].reshape(10, 3)
    aligned, meta = positions_for_evaluator_z(
        psf_r,
        psf_z,
        charmm_z,
        geometry_hint=charmm_r,
        atoms_per_list=[5, 5],
    )
    assert meta["geometry_reordered"] is True
    np.testing.assert_allclose(aligned, charmm_r, atol=1e-12)


def test_positions_for_evaluator_z_identity_for_psf_reference() -> None:
    perm = np.array([0, 3, 4, 1, 2], dtype=int)
    charmm_z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    charmm_r = np.random.default_rng(9).random((5, 3))
    psf_z = charmm_z[perm]
    psf_r = charmm_r[perm]
    aligned, meta = positions_for_evaluator_z(
        psf_r,
        psf_z,
        psf_z,
        atoms_per_list=[5],
    )
    assert meta["geometry_reordered"] is False
    np.testing.assert_allclose(aligned, psf_r, atol=1e-12)


def test_permute_handoff_charges_to_psf_order() -> None:
    perm = np.array([0, 3, 4, 1, 2], dtype=int)
    charmm_z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    charmm_r = np.random.default_rng(8).random((5, 3))
    psf_z = charmm_z[perm]
    charmm_q = np.linspace(-0.2, 0.2, 5)
    psf_q = charmm_q[perm]
    out = permute_handoff_array_to_evaluator_z(
        charmm_q,
        handoff_positions=charmm_r,
        handoff_numbers=charmm_z,
        evaluator_numbers=psf_z,
        atoms_per_list=[5],
    )
    np.testing.assert_allclose(out, psf_q)


def test_resolve_evaluate_max_frames_defaults_to_one() -> None:
    assert resolve_evaluate_max_frames(Namespace(max_frames=None)) == 1
    assert resolve_evaluate_max_frames(Namespace(max_frames=20)) == 20


def test_should_evaluate_reference_trajectory() -> None:
    assert should_evaluate_reference_trajectory(Namespace(evaluate_reference_npz=None)) is False
    assert (
        should_evaluate_reference_trajectory(
            Namespace(evaluate_reference_npz="/tmp/ref.npz", max_frames=None)
        )
        is False
    )
    assert (
        should_evaluate_reference_trajectory(
            Namespace(
                evaluate_reference_npz="/tmp/ref.npz",
                evaluate_npz="/tmp/ref.npz",
                evaluate_frame=16566,
                max_frames=None,
            )
        )
        is True
    )
    assert (
        should_evaluate_reference_trajectory(
            Namespace(
                evaluate_reference_npz="/tmp/ref.npz",
                evaluate_npz="/tmp/other.npz",
                evaluate_reference_frame=16566,
                max_frames=1,
            )
        )
        is True
    )
    assert (
        should_evaluate_reference_trajectory(
            Namespace(evaluate_reference_npz="/tmp/ref.npz", max_frames=20)
        )
        is True
    )


def test_resolve_reference_units_respects_manifest_when_cli_unset(tmp_path: Path) -> None:
    from mmml.data.units import UnitsManifestV2

    manifest = UnitsManifestV2(
        arrays={"E": "ev", "F": "ev_angstrom", "R": "angstrom"},
    )
    (tmp_path / "units_manifest.json").write_text(
        json.dumps(manifest.to_dict()),
        encoding="utf-8",
    )
    ref = tmp_path / "ref.npz"
    np.savez_compressed(
        ref,
        E=np.array([-43.0]),
        F=np.ones((1, 10, 3)),
        R=np.zeros((1, 10, 3)),
        Z=np.ones((1, 10), dtype=int),
        N=np.array([10], dtype=int),
    )
    e_unit, f_unit = resolve_reference_units(
        ref,
        Namespace(
            evaluate_reference_energy_unit=None,
            evaluate_reference_force_unit=None,
        ),
    )
    assert e_unit == "ev"
    assert f_unit == "ev_angstrom"


def test_save_evaluate_trajectory_npz_multi_roundtrip(tmp_path: Path) -> None:
    z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    r = np.random.default_rng(0).random((3, 5, 3))
    f = np.random.default_rng(1).random((3, 5, 3))
    e = np.array([-1.0, -1.1, -1.2])
    path = tmp_path / "evaluate_traj.npz"
    save_evaluate_trajectory_npz_multi(
        path,
        atomic_numbers=z,
        positions=r,
        energies_eV=e,
        forces_eV_A=f,
        frame_indices=np.array([0, 5, 10], dtype=int),
    )
    with np.load(path) as data:
        assert data["R"].shape == (3, 5, 3)
        assert data["E_eV"].shape == (3,)
        np.testing.assert_allclose(data["F"][1], f[1])
        np.testing.assert_array_equal(data["source_indices"], [0, 5, 10])


def test_save_evaluate_extxyz_multi_writes_all_frames(tmp_path: Path) -> None:
    from ase.io import read

    z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    r = np.zeros((4, 5, 3))
    f = np.ones((4, 5, 3)) * 0.25
    e = np.linspace(-2.0, -1.7, 4)
    path = tmp_path / "evaluate.extxyz"
    save_evaluate_extxyz_multi(
        path,
        atomic_numbers=z,
        positions=r,
        energies_eV=e,
        forces_eV_A=f,
    )
    frames = read(str(path), index=":")
    assert len(frames) == 4
    assert frames[2].get_potential_energy() == pytest.approx(float(e[2]))


def test_save_evaluate_extxyz_includes_forces(tmp_path: Path) -> None:
    from ase.io import read

    z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    r = np.zeros((5, 3))
    f = np.ones((5, 3)) * 0.5
    path = tmp_path / "evaluate.extxyz"
    save_evaluate_extxyz(
        path,
        atomic_numbers=z,
        positions=r,
        energy_eV=-2.0,
        forces_eV_A=f,
    )
    atoms = read(str(path), format="extxyz")
    assert atoms.calc is not None
    np.testing.assert_allclose(atoms.get_forces(), f)
    assert atoms.get_potential_energy() == pytest.approx(-2.0)


def test_center_positions_at_com_moves_mass_weighted_origin() -> None:
    from mmml.cli.run.md_evaluate_npz import center_positions_at_com

    z = np.array([6, 1, 1], dtype=int)
    pos = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    centered = center_positions_at_com(pos, z)
    from ase.data import atomic_masses

    masses = atomic_masses[z]
    np.testing.assert_allclose(np.average(centered, axis=0, weights=masses), 0.0, atol=1e-12)


def test_save_evaluate_compare_extxyz_trajectories_writes_three_layers(tmp_path: Path) -> None:
    from ase.io import read

    from mmml.cli.run.md_evaluate_npz import save_evaluate_compare_extxyz_trajectories

    z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    r = np.array([[[1.0, 2.0, 3.0]] * 5])
    model_f = np.array([[[1.0, 0.0, 0.0]] * 5])
    ref_f = np.array([[[0.5, 0.0, 0.0]] * 5])
    artifacts = save_evaluate_compare_extxyz_trajectories(
        tmp_path,
        atomic_numbers=z,
        positions=r,
        model_energies_eV=np.array([-1.0]),
        model_forces_eV_A=model_f,
        reference_energies_eV=np.array([-1.2]),
        reference_forces_eV_A=ref_f,
        prefix="evaluate",
    )
    assert (tmp_path / "evaluate_ground_truth.extxyz").is_file()
    assert (tmp_path / "evaluate_model.extxyz").is_file()
    assert (tmp_path / "evaluate_difference.extxyz").is_file()
    assert (tmp_path / "evaluate.extxyz").is_file()

    gt = read(str(tmp_path / "evaluate_ground_truth.extxyz"), index=0)
    model = read(str(tmp_path / "evaluate_model.extxyz"), index=0)
    diff = read(str(tmp_path / "evaluate_difference.extxyz"), index=0)
    np.testing.assert_allclose(gt.get_positions().mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(model.get_forces(), model_f[0])
    np.testing.assert_allclose(gt.get_forces(), ref_f[0])
    np.testing.assert_allclose(diff.get_forces(), model_f[0] - ref_f[0])
    assert diff.get_potential_energy() == pytest.approx(-1.0 - (-1.2))
    assert artifacts["extxyz_difference"] == str(tmp_path / "evaluate_difference.extxyz")


def test_resolve_evaluate_use_pbc_from_setup() -> None:
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.ones(3, dtype=int),
        pbc=False,
    )
    assert resolve_evaluate_use_pbc(Namespace(setup="free_nve"), handoff) is False
    assert resolve_evaluate_use_pbc(Namespace(setup="pbc_nvt"), handoff) is True


def test_md_system_parser_accepts_evaluate_npz() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    args = parse_md_system_args(
        [
            "--evaluate-npz",
            "/tmp/geom.npz",
            "--composition",
            "DCM:2",
            "--backend",
            "ase",
            "--setup",
            "free_nve",
            "--evaluate-frame",
            "2",
        ]
    )
    assert Path(args.evaluate_npz) == Path("/tmp/geom.npz")
    assert args.evaluate_frame == 2


def test_md_system_parser_accepts_optimize_cutoffs() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    args = parse_md_system_args(
        [
            "--optimize-cutoffs",
            "--reference-npz",
            "/tmp/traj.npz",
            "--composition",
            "DCM:2",
            "--ml-cutoff-grid",
            "1.5,2.0",
            "--max-frames",
            "10",
        ]
    )
    assert args.optimize_cutoffs is True
    assert Path(args.reference_npz) == Path("/tmp/traj.npz")
    assert args.ml_switch_width_grid == "1.5,2.0"


def test_run_evaluate_npz_ase_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    pos = np.zeros((4, 3))
    z = np.array([6, 1, 1, 1], dtype=np.int32)
    npz = tmp_path / "state.npz"
    np.savez_compressed(
        npz,
        positions=pos,
        atomic_numbers=z,
        pbc=False,
        metadata="{}",
    )
    out_dir = tmp_path / "out"
    args = Namespace(
        evaluate_npz=npz,
        evaluate_output=None,
        output_dir=out_dir,
        composition="ACO:1",
        n_molecules=1,
        checkpoint=None,
        backend="ase",
        setup="free_nve",
        box_size=None,
        handoff_require_cell=False,
        quiet=True,
        ml_cutoff=0.1,
        jax_md_capacity_multiplier=1.25,
        jax_md_capacity_growth_factor=1.5,
        jax_md_max_overflow_retries=4,
        jax_md_disable_fallback=False,
        jax_md_update_interval=1,
        jax_md_skin_distance=0.0,
        max_pairs=1000,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        min_com_restraint_distance=None,
        min_com_restraint_k=1.0,
        ml_batch_size=None,
        ml_max_active_dimers=None,
        ml_compute_dtype=None,
        verbose_calc=False,
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        mm_cutoff=5.0,
    )

    fake_atoms = MagicMock()
    fake_atoms.get_positions.return_value = pos
    fake_ase_mod = MagicMock()
    fake_ase_mod._cubic_box_length.return_value = 40.0
    fake_ase_mod._parse_composition.return_value = [("ACO", 1)]

    with (
        patch.dict(sys.modules, {"mmml.cli.run.md_pbc_suite.ase": fake_ase_mod}),
        patch(
            "mmml.cli.run.md_evaluate_npz.cluster_geometry_from_handoff",
            return_value=(z, pos, [4], ["ACO"], {"ACO": 1}),
        ),
        patch("mmml.cli.run.md_evaluate_npz.ensure_psf_for_handoff_cluster"),
        patch(
            "mmml.cli.run.md_evaluate_npz._build_atoms_for_evaluate",
            return_value=fake_atoms,
        ),
        patch(
            "mmml.cli.run.md_evaluate_npz._evaluate_ase_mmml",
            return_value={
                "energy_eV": -1.25,
                "forces_eV_A": np.zeros((4, 3)).tolist(),
                "max_force_eV_A": 0.01,
                "rms_force_eV_A": 0.01,
                "n_atoms": 4,
                "n_monomers": 1,
                "pbc": False,
                "box_A": None,
            },
        ),
        patch(
            "mmml.cli.base.resolve_checkpoint_paths",
            return_value=(Path("/tmp/ckpt"), Path("/tmp/ckpt")),
        ),
    ):
        code = run_evaluate_npz(args)

    assert code == 0
    result_path = out_dir / "evaluate.json"
    assert result_path.is_file()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "ase"
    assert payload["metrics"]["energy_eV"] == pytest.approx(-1.25)
    assert (out_dir / "evaluate.npz").is_file()
    assert (out_dir / "evaluate.extxyz").is_file()


def test_md_system_evaluate_npz_manifest_exit_code_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from mmml.cli.run import md_system

    out_dir = tmp_path / "eval_smoke"
    out_dir.mkdir()
    npz = tmp_path / "geom.npz"
    np.savez_compressed(npz, positions=np.zeros((10, 3)), pbc=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mmml",
            "--evaluate-npz",
            str(npz),
            "--composition",
            "DCM:2",
            "--backend",
            "ase",
            "--setup",
            "free_nve",
            "--output-dir",
            str(out_dir),
        ],
    )
    monkeypatch.setattr("mmml.cli.run.md_evaluate_npz.run_evaluate_npz", lambda _args: 0)

    assert md_system.main() == 0

    manifest_path = tmp_path / "artifacts" / "md_system" / "jobs" / "eval_smoke.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["exit_code"] == 0
    assert manifest["backend"] == "ase"


def test_normalize_metrics_converts_hartree_labeled_ev() -> None:
    from mmml.cli.run.md_evaluate_npz import normalize_metrics_to_ev

    metrics = normalize_metrics_to_ev(
        {
            "energy_eV": -43.31,
            "energy_unit": "hartree",
            "forces_eV_A": np.array([[0.01, 0.0, 0.0]]),
            "force_unit": "hartree_bohr",
        }
    )
    assert metrics["energy_eV"] == pytest.approx(-43.31 * 27.211386, rel=1e-6)
    assert metrics["units"]["energy"] == "eV"


def test_save_evaluate_npz_e_and_e_ev_consistent(tmp_path: Path) -> None:
    from mmml.data.units import EV_TO_HARTREE

    z = np.array([6, 1, 1], dtype=np.int32)
    r = np.zeros((3, 3))
    f = np.zeros((3, 3))
    path = tmp_path / "eval.npz"
    save_evaluate_trajectory_npz(
        path,
        atomic_numbers=z,
        positions=r,
        energy_eV=-27.211386,
        forces_eV_A=f,
    )
    with np.load(path, allow_pickle=True) as data:
        assert data["E_eV"][0] == pytest.approx(-27.211386)
        assert data["E"][0] == pytest.approx(-1.0, rel=1e-6)
        assert float(data["E"][0]) == pytest.approx(float(data["E_eV"][0]) * EV_TO_HARTREE)


def test_compare_autodetects_ev_reference(tmp_path: Path) -> None:
    z = np.array([1, 1], dtype=np.int32)
    r = np.zeros((2, 3))
    meta = json.dumps({"E": "ev", "F": "ev_angstrom", "R": "angstrom"})
    ref_path = tmp_path / "ref_ev.npz"
    np.savez_compressed(
        ref_path,
        N=np.array([2], dtype=int),
        Z=z.reshape(1, 2),
        R=r.reshape(1, 2, 3),
        E=np.array([-10.0], dtype=np.float64),
        _mmml_units=np.array(meta),
    )
    cmp = compare_evaluate_to_reference_npz(
        ref_path,
        frame=0,
        atomic_numbers=z,
        positions=r,
        energy_eV=-10.01,
        forces_eV_A=None,
    )
    assert cmp["delta_energy_eV"] == pytest.approx(-0.01, abs=1e-6)
    assert cmp["reference_energy_unit"] == "ev"


def test_apply_charmm_output_from_args_defaults_missing_bomlev() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_charmm_output_from_args,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.apply_charmm_verbosity"
    ) as mock_apply:
        nprint = apply_charmm_output_from_args(Namespace(quiet=True))
    assert nprint == 100
    mock_apply.assert_called_once_with(prnlev=0, warnlev=0, bomlev=-2)


def test_evaluate_jaxmd_uses_cutoff_parameters_from_cutoffs_module() -> None:
    import mmml.cli.run.md_evaluate_npz as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters" in source
    assert (
        "from mmml.interfaces.pycharmmInterface.calculator_utils import CutoffParameters"
        not in source
    )


def test_charmm_total_forces_ev_angstrom_converts_kcal_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_lib_available

    if not charmm_lib_available():
        pytest.skip("CHARMM runtime not available")
    import pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_total_forces_ev_angstrom,
        charmm_total_forces_kcalmol_A,
    )

    class _ForceFrame:
        def __init__(self, dx: float, dy: float, dz: float) -> None:
            self._vals = {"dx": dx, "dy": dy, "dz": dz}

        def __getitem__(self, key: str):
            val = np.array([self._vals[key]], dtype=float)

            class _Series:
                @staticmethod
                def to_numpy(dtype=float):
                    return val.astype(dtype)

            return _Series()

    monkeypatch.setattr(
        "pycharmm.coor.get_forces",
        lambda: _ForceFrame(-23.060548867, 0.0, 0.0),
    )
    kcal = charmm_total_forces_kcalmol_A()
    assert kcal[0, 0] == pytest.approx(23.060548867)
    forces = charmm_total_forces_ev_angstrom()
    assert forces.shape == (1, 3)
    assert forces[0, 0] == pytest.approx(1.0)


def test_evaluate_pycharmm_returns_forces_ev_angstrom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mmml.cli.run.md_evaluate_npz import _evaluate_pycharmm
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

    forces_kcal = np.array([[23.060548867, 0.0, 0.0], [0.0, 46.121097734, 0.0]])
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.apply_charmm_output_from_args",
        lambda _args: 50,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.setup_default_nbonds",
        lambda: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_checkpoint",
        lambda _ckpt: Path("/tmp/ckpt"),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow._register_mlpot_context",
        lambda *a, **k: (object(), object()),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        lambda _pos: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.refresh_mlpot_energy_and_grms",
        lambda *_a, **_k: 0.1,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_energy_row",
        lambda: {"ENER": float(ev2kcalmol)},
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_evaluate_forces_ev_angstrom",
        lambda _calc, *, natom, positions=None, use_pbc=False, box_A=None: (
            np.asarray(forces_kcal[: int(natom)], dtype=np.float64) / float(ev2kcalmol),
            "spherical_fn",
        ),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.collect_evaluate_force_sources_ev_angstrom",
        lambda *_a, **_k: {},
    )

    metrics = _evaluate_pycharmm(
        Namespace(quiet=True, checkpoint="/tmp/ckpt"),
        z=np.array([6, 1], dtype=int),
        positions=np.zeros((2, 3)),
        n_monomers=1,
        use_pbc=False,
        L=None,
    )
    assert metrics["energy_eV"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.asarray(metrics["forces_eV_A"]),
        forces_kcal / float(ev2kcalmol),
    )
    assert metrics["max_force_eV_A"] == pytest.approx(2.0)
    assert metrics.get("force_source") == "spherical_fn"


def test_mlpot_hybrid_forces_ev_angstrom_reads_last_ml_forces() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        mlpot_hybrid_forces_ev_angstrom,
    )
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

    class _Calc:
        last_ml_forces = np.array([[23.060548867, 0.0, 0.0]], dtype=np.float64)

    class _Model:
        _registered_calculator = _Calc()

    forces = mlpot_hybrid_forces_ev_angstrom(_Model(), natom=1)
    assert forces is not None
    assert forces[0, 0] == pytest.approx(1.0)
    assert forces.shape == (1, 3)

    class _ModelDirect:
        _last_ml_forces = np.array([[float(ev2kcalmol), 0.0, 0.0]], dtype=np.float64)

    forces_direct = mlpot_hybrid_forces_ev_angstrom(_ModelDirect(), natom=1)
    assert forces_direct is not None
    assert forces_direct[0, 0] == pytest.approx(1.0)


def test_reference_com_dist_A_from_trajectory() -> None:
    class _Ref:
        com_distances = np.array([2.5, 4.0, 6.5], dtype=np.float64)

    assert _reference_com_dist_A(_Ref(), 1) == pytest.approx(4.0)
    assert _reference_com_dist_A(_Ref(), 99) is None
    assert _reference_com_dist_A(object(), 0) is None


def test_build_atoms_for_evaluate_preserves_r0_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mmml.cli.run.md_evaluate_npz import _build_atoms_for_evaluate

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        lambda _pos: None,
    )

    perm = np.array([0, 3, 4, 1, 2], dtype=int)
    charmm_z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    charmm_r = np.random.default_rng(11).random((5, 3))
    psf_r = charmm_r[perm]
    handoff = MdHandoffState(
        positions=psf_r,
        atomic_numbers=charmm_z[perm],
        pbc=False,
    )
    monomer_offsets = np.array([0, 5], dtype=int)
    aligned_r = charmm_r.copy()

    corrupted = _build_atoms_for_evaluate(
        z=charmm_z,
        r0=aligned_r,
        handoff=handoff,
        monomer_offsets=monomer_offsets,
        use_pbc=False,
        L=None,
        preserve_r0=False,
    )
    preserved = _build_atoms_for_evaluate(
        z=charmm_z,
        r0=aligned_r,
        handoff=handoff,
        monomer_offsets=monomer_offsets,
        use_pbc=False,
        L=None,
        preserve_r0=True,
    )
    assert not np.allclose(corrupted.get_positions(), aligned_r, atol=1e-12)
    np.testing.assert_allclose(preserved.get_positions(), aligned_r, atol=1e-12)


def test_classify_ml_regime() -> None:
    assert classify_ml_regime(5.0, ml_switch_width=0.1, mm_switch_on=6.0) == "pure_ml"
    assert classify_ml_regime(5.95, ml_switch_width=0.1, mm_switch_on=6.0) == "handoff"
    assert classify_ml_regime(8.65, ml_switch_width=0.1, mm_switch_on=6.0) == "beyond_switch"


def test_save_evaluate_compare_diagnostics_writes_csv_and_summary(tmp_path: Path) -> None:
    per_frame = [
        {
            "status": "ok",
            "reference_frame": 10,
            "reference_source_index": 100,
            "com_dist_A": 3.0,
            "delta_energy_eV": 0.01,
            "abs_delta_energy_eV": 0.01,
            "force_rmse_eV_A": 0.25,
            "force_mae_eV_A": 0.10,
            "force_max_abs_eV_A": 0.73,
        },
        {
            "status": "ok",
            "reference_frame": 20,
            "reference_source_index": 200,
            "com_dist_A": 5.95,
            "delta_energy_eV": -0.002,
            "abs_delta_energy_eV": 0.002,
            "force_rmse_eV_A": 0.003,
            "force_mae_eV_A": 0.002,
            "force_max_abs_eV_A": 0.007,
        },
        {
            "status": "error",
            "reference_frame": 30,
            "error": "boom",
        },
    ]
    artifacts, summary = save_evaluate_compare_diagnostics(
        tmp_path,
        per_frame,
        backend="pycharmm",
        ml_switch_width=0.1,
        mm_switch_on=6.0,
        mm_switch_width=3.0,
    )
    csv_path = tmp_path / "evaluate_compare_diagnostics.csv"
    assert artifacts["compare_diagnostics_csv"] == str(csv_path)
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "ml_regime" in lines[0]
    assert "pure_ml" in lines[1]
    assert "handoff" in lines[2]
    assert summary["n_frames"] == 2
    assert summary["com_dist_A_min"] == pytest.approx(3.0)
    assert summary["com_dist_A_max"] == pytest.approx(5.95)
    assert summary["regime_counts"] == {"pure_ml": 1, "handoff": 1, "beyond_switch": 0}
    assert summary["cutoffs"]["ml_handoff_A"] == pytest.approx([5.9, 6.0])
    assert summary["cutoffs"]["mm_outer_taper_A"] == pytest.approx([6.0, 9.0])
    assert summary["mean_force_rmse_eV_A"] == pytest.approx((0.25 + 0.003) / 2)
    assert summary["corr_com_force_rmse"] == pytest.approx(-1.0, abs=1e-12)
    assert summary["mean_force_rmse_eV_A_pure_ml"] == pytest.approx(0.25)
    assert summary["mean_force_rmse_eV_A_handoff"] == pytest.approx(0.003)


def test_enrich_compare_with_force_sources() -> None:
    from mmml.cli.run.md_evaluate_npz import enrich_compare_with_force_sources

    cmp: dict = {"force_rmse_eV_A": 0.1}
    ref_f = np.zeros((2, 3), dtype=np.float64)
    sources = {
        "spherical_fn": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64),
        "charmm_total": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
    }
    enrich_compare_with_force_sources(
        cmp,
        reference_forces_ev=ref_f,
        force_sources=sources,
        charmm_energy_terms_kcal_mol={"USER": 12.5, "ENER": 12.5},
    )
    assert cmp["charmm_energy_terms_kcal_mol"]["USER"] == pytest.approx(12.5)
    assert cmp["force_rmse_spherical_fn_eV_A"] == pytest.approx(
        float(np.sqrt(np.mean(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]) ** 2)))
    )
    assert cmp["force_rmse_charmm_total_eV_A"] == pytest.approx(
        float(np.sqrt(np.mean(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) ** 2)))
    )
    assert "force_source_compare" in cmp


def test_metrics_for_json_export_strips_force_sources() -> None:
    from mmml.cli.run.md_evaluate_npz import metrics_for_json_export

    metrics = {
        "energy_eV": 1.0,
        "force_sources_all": {"charmm_total": np.zeros((2, 3))},
    }
    exported = metrics_for_json_export(metrics)
    assert "force_sources_all" not in exported
    assert exported["energy_eV"] == 1.0


def test_compare_force_sources_to_reference() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        compare_force_sources_to_reference,
        cross_lane_force_rmse_ev_angstrom,
        force_error_metrics_ev_angstrom,
    )

    ref = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    pred = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    m = force_error_metrics_ev_angstrom(pred, ref)
    assert m["force_rmse_eV_A"] == pytest.approx(float(np.sqrt(0.01 / 6.0)))
    lanes = compare_force_sources_to_reference(
        {"spherical_fn": pred, "charmm_total": ref},
        ref,
    )
    assert lanes["spherical_fn"]["force_rmse_eV_A"] == pytest.approx(float(np.sqrt(0.01 / 6.0)))
    assert lanes["charmm_total"]["force_rmse_eV_A"] == pytest.approx(0.0)
    cross = cross_lane_force_rmse_ev_angstrom(
        {"spherical_fn": pred, "charmm_total": ref},
        baseline="spherical_fn",
    )
    assert cross["force_rmse_vs_spherical_fn_eV_A"] == pytest.approx(float(np.sqrt(0.01 / 6.0)))


def test_evaluate_pycharmm_reuses_mlpot_state_without_reregister() -> None:
    """Multi-frame evaluate must not call register_mlpot per frame (upinb segfault)."""
    from mmml.cli.run.md_evaluate_npz import _evaluate_pycharmm

    z = np.array([6, 1, 1, 1, 1, 6, 1, 1, 1, 1], dtype=np.int32)
    pos = np.random.default_rng(0).random((10, 3))
    ctx = object()
    calc = object()
    args = Namespace(quiet=True)

    with patch(
        "mmml.cli.run.md_evaluate_npz.setup_pycharmm_eval_mlpot",
    ) as setup_mock, patch(
        "mmml.cli.run.md_evaluate_npz._pycharmm_eval_metrics",
        return_value={"energy_eV": 1.0, "forces_eV_A": np.zeros((10, 3))},
    ) as metrics_mock, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
    ) as sync_mock:
        _evaluate_pycharmm(
            args,
            z=z,
            positions=pos,
            n_monomers=2,
            use_pbc=False,
            L=None,
            mlpot_state=(ctx, calc),
        )

    setup_mock.assert_not_called()
    sync_mock.assert_called_once()
    metrics_mock.assert_called_once()
    assert metrics_mock.call_args.args[0] is ctx
    assert metrics_mock.call_args.args[1] is calc
