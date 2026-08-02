"""Tests for the post-minimize monomer internal-geometry check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "mmml"
    / "utils"
    / "monomer_internal_geometry.py"
)
_spec = importlib.util.spec_from_file_location("_test_monomer_internal_geom", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
_mig = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mig
_spec.loader.exec_module(_mig)

covalent_skeleton_pairs = _mig.covalent_skeleton_pairs
scan_monomer_internal_geometry = _mig.scan_monomer_internal_geometry
assert_monomer_internal_geometry = _mig.assert_monomer_internal_geometry
resolve_max_monomer_internal_deviation_A = _mig.resolve_max_monomer_internal_deviation_A

# Methanol (CH3OH) in CHARMM CGenFF atom order: C, H1, H2, H3, O, HG.
MEOH_NAMES = ["C", "H1", "H2", "H3", "O", "HG"]
MEOH_Z = np.array([6, 1, 1, 1, 8, 1], dtype=int)


def _methanol_template() -> np.ndarray:
    tet = np.array(
        [
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
        dtype=float,
    )
    c = np.zeros(3)
    o = np.array([1.42, 0.0, 0.0])
    # O-H at ~108 deg from the O->C direction, in the xy plane.
    hg = o + 0.96 * np.array([np.cos(np.deg2rad(72.0)), np.sin(np.deg2rad(72.0)), 0.0])
    return np.vstack([c, c + 1.09 * tet / np.linalg.norm(tet, axis=1)[:, None], o, hg])


def _templates() -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    return {"MEOH": (_methanol_template(), list(MEOH_NAMES), MEOH_Z)}


def _cluster(n_monomers: int, *, spacing: float = 8.0) -> tuple[np.ndarray, list[int], list[str]]:
    template = _methanol_template()
    chunks = [template + np.array([spacing * i, 0.0, 0.0]) for i in range(n_monomers)]
    return (
        np.vstack(chunks),
        [len(template)] * n_monomers,
        ["MEOH"] * n_monomers,
    )


def _rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(float(angle_deg))
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def test_skeleton_pairs_cover_bonds_and_angles_but_not_torsions():
    pairs = {
        (int(i), int(j))
        for i, j in covalent_skeleton_pairs(_methanol_template(), MEOH_Z)
    }
    # 1-2: C-O, C-H1/2/3, O-HG
    assert (0, 4) in pairs
    assert (0, 1) in pairs
    assert (4, 5) in pairs
    # 1-3: H-C-H, H-C-O, C-O-HG
    assert (1, 2) in pairs
    assert (1, 4) in pairs
    assert (0, 5) in pairs
    # 1-4 (torsion-dependent) must be excluded
    assert (1, 5) not in pairs
    assert (2, 5) not in pairs
    assert (3, 5) not in pairs


def test_monatomic_template_has_no_pairs():
    assert covalent_skeleton_pairs(np.zeros((1, 3)), np.array([11])).size == 0


def test_rigid_placement_has_zero_deviation():
    positions, atoms_per, residues = _cluster(4)
    rotated = positions.copy()
    template_len = atoms_per[0]
    for mi in range(len(atoms_per)):
        s = mi * template_len
        chunk = rotated[s : s + template_len]
        com = chunk.mean(axis=0)
        rotated[s : s + template_len] = (
            (chunk - com) @ _rotation(np.array([0.3, 1.0, -0.2]), 40.0 * mi).T
            + com
            + np.array([0.5, -1.5, 2.0])
        )
    deviations, report = scan_monomer_internal_geometry(
        rotated,
        atoms_per,
        residue_names=residues,
        templates=_templates(),
    )
    assert report.n_monomers_checked == 4
    assert report.n_monomers_skipped == 0
    assert report.max_deviation_A < 1.0e-9
    assert np.all(np.isfinite(deviations))


def test_hydroxyl_torsion_rotation_is_not_flagged():
    """A legitimate minimization move: rotate the O-H torsion by 120 deg."""
    positions, atoms_per, residues = _cluster(2)
    n_atoms = atoms_per[0]
    for mi in range(len(atoms_per)):
        s = mi * n_atoms
        c_pos, o_pos = positions[s + 0], positions[s + 4]
        rot = _rotation(o_pos - c_pos, 120.0)
        positions[s + 5] = o_pos + (positions[s + 5] - o_pos) @ rot.T
    _dev, report = scan_monomer_internal_geometry(
        positions,
        atoms_per,
        residue_names=residues,
        templates=_templates(),
    )
    assert report.max_deviation_A < 1.0e-6


def test_bond_and_angle_relaxation_passes_default_threshold():
    positions, atoms_per, residues = _cluster(3)
    rng = np.random.default_rng(7)
    # ~0.03 A of per-atom relaxation noise: bonds/angles move, skeleton intact.
    positions = positions + rng.normal(scale=0.03, size=positions.shape)
    report = assert_monomer_internal_geometry(
        positions,
        atoms_per,
        residue_names=residues,
        templates=_templates(),
        max_deviation_A=_mig.DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A,
    )
    assert report.max_deviation_A < _mig.DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A


def test_scrambled_monomer_is_rejected():
    """The observed failure: one monomer's atom lands on another monomer's atom."""
    positions, atoms_per, residues = _cluster(3)
    positions[1 * 6 + 4] = positions[2 * 6 + 5]  # monomer 2 "O" == monomer 3 "HG"
    with pytest.raises(RuntimeError) as excinfo:
        assert_monomer_internal_geometry(
            positions,
            atoms_per,
            residue_names=residues,
            templates=_templates(),
            max_deviation_A=0.35,
        )
    message = str(excinfo.value)
    assert "distorted covalent skeleton" in message
    assert "monomer 2 (MEOH)" in message


def test_non_finite_coordinates_are_rejected():
    positions, atoms_per, residues = _cluster(2)
    positions[3, 1] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_monomer_internal_geometry(
            positions,
            atoms_per,
            residue_names=residues,
            templates=_templates(),
        )


def test_zero_threshold_measures_without_enforcing():
    positions, atoms_per, residues = _cluster(2)
    positions[7] += 3.0
    report = assert_monomer_internal_geometry(
        positions,
        atoms_per,
        residue_names=residues,
        templates=_templates(),
        max_deviation_A=0.0,
    )
    assert report.max_deviation_A > 1.0


def test_unknown_residue_and_atom_count_mismatch_are_skipped():
    positions, atoms_per, residues = _cluster(2)
    _dev, report = scan_monomer_internal_geometry(
        positions,
        atoms_per,
        residue_names=["MEOH", "TIP3"],
        templates=_templates(),
    )
    assert report.n_monomers_checked == 1
    assert report.n_monomers_skipped == 1


def test_minimize_report_flags_only_exact_zero_start_grms():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmMmMinimizeReport

    broken = CharmmMmMinimizeReport(n_atoms=1962, ran=True, start_grms_kcalmol_A=0.0)
    assert broken.start_grms_is_exactly_zero
    healthy = CharmmMmMinimizeReport(n_atoms=1962, ran=True, start_grms_kcalmol_A=54.1)
    assert not healthy.start_grms_is_exactly_zero
    # A single atom genuinely has no gradient, and an unmeasured run is not evidence.
    assert not CharmmMmMinimizeReport(
        n_atoms=1, ran=True, start_grms_kcalmol_A=0.0
    ).start_grms_is_exactly_zero
    assert not CharmmMmMinimizeReport(n_atoms=1962, ran=False).start_grms_is_exactly_zero


def test_cluster_guard_warns_but_does_not_fail_on_zero_start_grms(capsys):
    """Healthy KEY_LIBRARY CHARMM builds report GRMS 0.0; that cannot gate a build."""
    from mmml.cli.run.md_pbc_suite.cluster import assert_packmol_cluster_minimize_sane
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmMmMinimizeReport

    positions, atoms_per, residues = _cluster(3)
    assert_packmol_cluster_minimize_sane(
        positions,
        atoms_per_list=atoms_per,
        residue_names=residues,
        residue_geometries=_templates(),
        minimize_report=CharmmMmMinimizeReport(
            n_atoms=len(positions), ran=True, start_grms_kcalmol_A=0.0
        ),
        verbose=False,
    )
    assert "WARNING" in capsys.readouterr().out


def test_cluster_guard_rejects_distorted_monomers():
    from mmml.cli.run.md_pbc_suite.cluster import assert_packmol_cluster_minimize_sane
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmMmMinimizeReport

    positions, atoms_per, residues = _cluster(3)
    positions[1 * 6 + 4] = positions[2 * 6 + 5]
    healthy = CharmmMmMinimizeReport(
        n_atoms=len(positions), ran=True, start_grms_kcalmol_A=42.0, end_grms_kcalmol_A=0.8
    )
    with pytest.raises(RuntimeError, match="distorted covalent skeleton"):
        assert_packmol_cluster_minimize_sane(
            positions,
            atoms_per_list=atoms_per,
            residue_names=residues,
            residue_geometries=_templates(),
            minimize_report=healthy,
            verbose=False,
        )


def test_cluster_guard_accepts_a_relaxed_cluster():
    from mmml.cli.run.md_pbc_suite.cluster import assert_packmol_cluster_minimize_sane
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmMmMinimizeReport

    positions, atoms_per, residues = _cluster(5)
    rng = np.random.default_rng(3)
    positions = positions + rng.normal(scale=0.03, size=positions.shape)
    report = assert_packmol_cluster_minimize_sane(
        positions,
        atoms_per_list=atoms_per,
        residue_names=residues,
        residue_geometries=_templates(),
        minimize_report=CharmmMmMinimizeReport(
            n_atoms=len(positions), ran=True, start_grms_kcalmol_A=87.3, end_grms_kcalmol_A=1.2
        ),
        verbose=False,
    )
    assert report.n_monomers_checked == 5
    assert report.max_deviation_A < _mig.DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A


def test_cluster_guard_warns_when_no_template_matches(capsys):
    """A check that silently covers nothing is the failure mode being prevented."""
    from mmml.cli.run.md_pbc_suite.cluster import assert_packmol_cluster_minimize_sane

    positions, atoms_per, residues = _cluster(2)
    report = assert_packmol_cluster_minimize_sane(
        positions,
        atoms_per_list=atoms_per,
        residue_names=["DCM", "DCM"],
        residue_geometries=_templates(),
        verbose=False,
    )
    assert report.n_monomers_checked == 0
    assert "covered 0 of 2" in capsys.readouterr().out


def _write_cache_entry(entry, positions, atoms_per, residues):
    from mmml.interfaces.pycharmmInterface import packmol_cache

    packmol_cache.save_packmol_cluster_cache(
        entry,
        manifest={
            "version": packmol_cache.CACHE_VERSION,
            "cache_key": entry.name,
            "composition": [["MEOH", len(atoms_per)]],
        },
        z=np.tile(MEOH_Z, len(atoms_per)),
        positions=positions,
        atoms_per_list=atoms_per,
        residue_names=residues,
        residue_geometries={"MEOH": (_methanol_template(), list(MEOH_NAMES), MEOH_Z)},
    )
    cached = packmol_cache.load_packmol_cluster_cache(entry)
    assert cached is not None
    return cached


def test_corrupted_cache_entry_is_rejected_on_load(tmp_path):
    """A cache written by a broken build must not be handed downstream."""
    from mmml.cli.run.md_pbc_suite.cluster import assert_packmol_cluster_minimize_sane

    positions, atoms_per, residues = _cluster(3)
    positions[1 * 6 + 4] = positions[2 * 6 + 5]
    cached = _write_cache_entry(tmp_path / "badkey", positions, atoms_per, residues)

    with pytest.raises(RuntimeError, match="distorted covalent skeleton"):
        assert_packmol_cluster_minimize_sane(
            cached["positions"],
            atoms_per_list=cached["atoms_per_list"],
            residue_names=cached["residue_names"],
            residue_geometries=cached["residue_geometries"],
            verbose=False,
        )


def test_healthy_cache_entry_round_trips(tmp_path):
    from mmml.cli.run.md_pbc_suite.cluster import assert_packmol_cluster_minimize_sane

    positions, atoms_per, residues = _cluster(3)
    rng = np.random.default_rng(11)
    positions = positions + rng.normal(scale=0.03, size=positions.shape)
    cached = _write_cache_entry(tmp_path / "goodkey", positions, atoms_per, residues)

    report = assert_packmol_cluster_minimize_sane(
        cached["positions"],
        atoms_per_list=cached["atoms_per_list"],
        residue_names=cached["residue_names"],
        residue_geometries=cached["residue_geometries"],
        verbose=False,
    )
    assert report.n_monomers_checked == 3


def test_threshold_resolution_from_env(monkeypatch):
    monkeypatch.delenv(_mig.MONOMER_INTERNAL_DEVIATION_ENV, raising=False)
    assert resolve_max_monomer_internal_deviation_A() == pytest.approx(
        _mig.DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A
    )
    monkeypatch.setenv(_mig.MONOMER_INTERNAL_DEVIATION_ENV, "0.75")
    assert resolve_max_monomer_internal_deviation_A() == pytest.approx(0.75)
    monkeypatch.setenv(_mig.MONOMER_INTERNAL_DEVIATION_ENV, "0")
    assert resolve_max_monomer_internal_deviation_A() == 0.0
    monkeypatch.setenv(_mig.MONOMER_INTERNAL_DEVIATION_ENV, "nonsense")
    assert resolve_max_monomer_internal_deviation_A() == pytest.approx(
        _mig.DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A
    )
    # An explicit argument always wins over the environment.
    assert resolve_max_monomer_internal_deviation_A(0.2) == pytest.approx(0.2)
