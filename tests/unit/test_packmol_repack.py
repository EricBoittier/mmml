"""Unit tests for Packmol-backed monomer repack during MD recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _two_monomer_system() -> tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    return pos, offsets


def test_packmol_selective_repack_writes_fixed_and_movable_blocks(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_repack

    captured: dict[str, str] = {}
    pos, offsets = _two_monomer_system()
    z = np.array([6, 1, 6, 1], dtype=int)

    def fake_execute(packmol_input: str, inp_path: Path) -> None:
        captured["input"] = packmol_input
        captured["inp_path"] = str(inp_path)
        out = tmp_path / "repack.pdb"
        out.write_text(
            "\n".join(
                [
                    "ATOM      1  C   F000A   1       0.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      2  H1  F000A   1       1.000   0.000   0.000  1.00  0.00           H",
                    "ATOM      3  C   M001A   2       5.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      4  H1  M001A   2       6.000   0.000   0.000  1.00  0.00           H",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(packmol_repack, "execute_packmol_script", fake_execute)
    monkeypatch.setattr(
        packmol_repack,
        "_charmm_atom_metadata",
        lambda _n: (["C", "H1", "C", "H1"], z),
    )

    out = packmol_repack.repack_selected_monomers_clear_overlap(
        pos,
        offsets,
        [1],
        min_distance=1.5,
        spacing=4.0,
        seed=7,
        cell=np.diag([10.0, 10.0, 10.0]),
        scratch_dir=tmp_path,
        atomic_numbers=z,
    )

    assert "fixed 0.5 0.0 0.0 0. 0. 0." in captured["input"]
    assert "number 1" in captured["input"]
    assert "number 1\n  inside cube" in captured["input"] or "inside cube" in captured["input"]
    assert "seed 7" in captured["input"]
    assert out[2, 0] == pytest.approx(5.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_packmol_repack_resolves_structure_paths_when_scratch_dir_relative(
    tmp_path, monkeypatch
):
    """Packmol cwd is scratch_dir; relative structure paths double-resolve and fail."""
    from mmml.interfaces.pycharmmInterface import packmol_repack

    captured: dict[str, str] = {}
    pos, offsets = _two_monomer_system()
    z = np.array([6, 1, 6, 1], dtype=int)
    rel_scratch = Path("packmol_repack_scratch")

    def fake_execute(packmol_input: str, inp_path: Path) -> None:
        captured["input"] = packmol_input
        out = inp_path.parent / "repack.pdb"
        out.write_text(
            "\n".join(
                [
                    "ATOM      1  C   F000A   1       0.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      2  H1  F000A   1       1.000   0.000   0.000  1.00  0.00           H",
                    "ATOM      3  C   M001A   2       5.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      4  H1  M001A   2       6.000   0.000   0.000  1.00  0.00           H",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(packmol_repack, "execute_packmol_script", fake_execute)
    monkeypatch.setattr(
        packmol_repack,
        "_charmm_atom_metadata",
        lambda _n: (["C", "H1", "C", "H1"], z),
    )

    packmol_repack.repack_selected_monomers_clear_overlap(
        pos,
        offsets,
        [1],
        min_distance=1.5,
        spacing=4.0,
        seed=7,
        cell=np.diag([10.0, 10.0, 10.0]),
        scratch_dir=rel_scratch,
        atomic_numbers=z,
    )

    resolved_scratch = (tmp_path / rel_scratch).resolve()
    assert f"structure {resolved_scratch / 'fixed_0000.pdb'}" in captured["input"]
    assert f"output {resolved_scratch / 'repack.pdb'}" in captured["input"]
    assert "structure packmol_repack_scratch/" not in captured["input"]


def test_read_packmol_monomer_coords_splits_sequential_atoms_not_residue_ids(tmp_path):
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        _read_packmol_monomer_coords,
    )

    out = tmp_path / "repack.pdb"
    # Three 2-atom monomers; Packmol often leaves all fixed atoms on residue 1.
    lines = []
    serial = 1
    for atom_idx in range(6):
        x = float(atom_idx + 1)
        lines.append(
            f"ATOM      {serial}  C   FIX A   1       {x:.3f}   0.000   0.000  1.00  0.00           C"
        )
        serial += 1
    lines.append("END")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    packed = _read_packmol_monomer_coords(out, expected_atoms_per_monomer=[2, 2, 2])
    assert len(packed) == 3
    assert packed[0].shape == (2, 3)
    assert packed[1][0, 0] == pytest.approx(3.0)
    assert packed[2][1, 0] == pytest.approx(6.0)


def test_packmol_repack_falls_back_to_grid_when_binary_missing(monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_repack

    pos, offsets = _two_monomer_system()
    z = np.array([6, 1, 6, 1], dtype=int)

    def boom(*_a, **_k):
        raise FileNotFoundError("packmol missing")

    monkeypatch.setattr(packmol_repack, "execute_packmol_script", boom)
    monkeypatch.setattr(
        packmol_repack,
        "_charmm_atom_metadata",
        lambda _n: (["C", "H1", "C", "H1"], z),
    )

    out = packmol_repack.repack_selected_monomers_clear_overlap(
        pos,
        offsets,
        [1],
        min_distance=1.5,
        spacing=4.0,
        seed=11,
        cell=None,
        atomic_numbers=z,
    )
    assert out.shape == pos.shape
    assert np.linalg.norm(out[2:] - pos[2:]) > 0.1


def test_resolve_packmol_tolerance_honors_large_config_value():
    from mmml.interfaces.pycharmmInterface.packmol_repack import _resolve_packmol_tolerance

    tol = _resolve_packmol_tolerance(
        min_distance=2.3,
        spacing=5.0,
        packmol_tolerance=5.0,
    )
    assert tol == pytest.approx(5.0)


@pytest.mark.parametrize("spacing", [None, 3.11, 4.07, 5.0, 12.0])
def test_resolve_packmol_tolerance_ignores_com_spacing(spacing):
    """Spacing is a COM pitch; as a contact tolerance it is unsatisfiable.

    A tolerance equal to the mean centre-to-centre separation forbids liquid
    density outright (methanol packs O···O at 2.8 Å with COMs 4.1 Å apart), and
    Packmol then grinds for hours without converging.
    """
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        PACKMOL_DEFAULT_TOLERANCE_A,
        _resolve_packmol_tolerance,
    )

    tol = _resolve_packmol_tolerance(min_distance=0.45, spacing=spacing)
    assert tol == pytest.approx(PACKMOL_DEFAULT_TOLERANCE_A)


def test_resolve_packmol_tolerance_defaults_and_overlap_floor():
    from mmml.interfaces.pycharmmInterface.packmol_repack import _resolve_packmol_tolerance

    # Prep-ladder floor (0.45 Å) is below Packmol's default and stays inert.
    assert _resolve_packmol_tolerance(min_distance=0.45) == pytest.approx(2.0)
    # A genuine contact floor above the default still raises the tolerance.
    assert _resolve_packmol_tolerance(min_distance=2.4) == pytest.approx(2.4)
    # An explicit request wins, including a tighter-than-default one.
    assert (
        _resolve_packmol_tolerance(min_distance=0.45, packmol_tolerance=1.5)
        == pytest.approx(1.5)
    )


def _methanol_template() -> np.ndarray:
    """COM-centred methanol-like geometry (CB OG HG1 HB1 HB2 HB3)."""
    xyz = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.430, 0.000, 0.000],
            [1.750, 0.900, 0.000],
            [-0.360, -1.030, 0.000],
            [-0.360, 0.510, 0.900],
            [-0.360, 0.510, -0.900],
        ],
        dtype=float,
    )
    return xyz - xyz.mean(axis=0)


def _random_rotation(seed: int) -> np.ndarray:
    """A proper rotation — Packmol never reflects a structure."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    assert np.linalg.det(q) == pytest.approx(1.0)
    return q


def test_group_identical_templates_collapses_rotated_copies():
    """Packmol randomises orientation, so raw coords never match; distances do."""
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        _group_identical_templates,
    )

    template = _methanol_template()
    n_mono = 8
    templates = [template @ _random_rotation(i).T for i in range(n_mono)]
    offsets = np.arange(n_mono + 1, dtype=int) * 6
    names = ["CB", "OG", "HG1", "HB1", "HB2", "HB3"] * n_mono
    z = np.array([6, 8, 1, 1, 1, 1] * n_mono, dtype=int)

    groups = _group_identical_templates(
        list(range(n_mono)),
        templates,
        offsets,
        atom_names=names,
        atomic_numbers=z,
    )

    assert groups == [list(range(n_mono))]


def test_group_identical_templates_keeps_distinct_conformers_and_species():
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        _group_identical_templates,
    )

    template = _methanol_template()
    stretched = template.copy()
    stretched[1, 0] += 0.5  # different conformer: C-O stretched by 0.5 Å
    templates = [template, template @ _random_rotation(3).T, stretched]
    offsets = np.arange(len(templates) + 1, dtype=int) * 6
    names = ["CB", "OG", "HG1", "HB1", "HB2", "HB3"] * len(templates)
    z = np.array([6, 8, 1, 1, 1, 1] * len(templates), dtype=int)

    groups = _group_identical_templates(
        list(range(len(templates))),
        templates,
        offsets,
        atom_names=names,
        atomic_numbers=z,
    )

    # Only the rotated duplicate merges; the stretched conformer keeps its own
    # block, since a group is packed from the representative's geometry.
    assert sorted(groups) == [[0, 1], [2]]


def test_group_identical_templates_does_not_merge_enantiomers():
    """Distances are reflection-blind; merging mirror images would flip a hand."""
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        _group_identical_templates,
    )

    # CHFClBr: four distinct substituents around one carbon.
    xyz = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.09, 0.00, 0.00],
            [-0.36, 1.35, 0.00],
            [-0.36, -0.70, 1.55],
            [-0.36, -0.70, -1.55],
        ],
        dtype=float,
    )
    left = xyz - xyz.mean(axis=0)
    right = left * np.array([1.0, 1.0, -1.0])  # enantiomer
    templates = [left, left @ _random_rotation(5).T, right]
    offsets = np.arange(len(templates) + 1, dtype=int) * 5
    names = ["C", "H", "F", "CL", "BR"] * len(templates)
    z = np.array([6, 1, 9, 17, 35] * len(templates), dtype=int)

    # The mirror image is a perfect distance-matrix match by construction.
    np.testing.assert_allclose(
        np.linalg.norm(left[:, None, :] - left[None, :, :], axis=-1),
        np.linalg.norm(right[:, None, :] - right[None, :, :], axis=-1),
    )

    groups = _group_identical_templates(
        list(range(len(templates))),
        templates,
        offsets,
        atom_names=names,
        atomic_numbers=z,
    )

    assert sorted(groups) == [[0, 1], [2]]


def test_group_identical_templates_separates_species_with_equal_geometry():
    """Same coordinates, different atoms — must not share a structure block."""
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        _group_identical_templates,
    )

    template = _methanol_template()
    templates = [template, template.copy()]
    offsets = np.array([0, 6, 12], dtype=int)
    names = ["CB", "OG", "HG1", "HB1", "HB2", "HB3"] + [
        "CB", "NG", "HG1", "HB1", "HB2", "HB3"
    ]
    z = np.array([6, 8, 1, 1, 1, 1] + [6, 7, 1, 1, 1, 1], dtype=int)

    groups = _group_identical_templates(
        [0, 1],
        templates,
        offsets,
        atom_names=names,
        atomic_numbers=z,
    )

    assert sorted(groups) == [[0], [1]]


def test_packmol_repack_emits_one_block_for_identical_movables(tmp_path, monkeypatch):
    """327 methanols must not become 327 Packmol molecule types."""
    from mmml.interfaces.pycharmmInterface import packmol_repack

    captured: dict[str, str] = {}
    n_mono = 6
    template = _methanol_template()
    coms = np.array([[3.0 * i, 0.0, 0.0] for i in range(n_mono)], dtype=float)
    pos = np.concatenate(
        [template @ _random_rotation(i).T + coms[i] for i in range(n_mono)]
    )
    offsets = np.arange(n_mono + 1, dtype=int) * 6
    names = ["CB", "OG", "HG1", "HB1", "HB2", "HB3"] * n_mono
    z = np.array([6, 8, 1, 1, 1, 1] * n_mono, dtype=int)

    def fake_execute(packmol_input: str, inp_path: Path) -> None:
        captured["input"] = packmol_input
        lines = []
        for atom_idx in range(6 * n_mono):
            x, y, zc = pos[atom_idx] + 0.25
            lines.append(
                f"ATOM  {atom_idx + 1:5d}  C   MOL A   1    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00           C"
            )
        lines.append("END")
        (inp_path.parent / "repack.pdb").write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(packmol_repack, "execute_packmol_script", fake_execute)

    packmol_repack.repack_monomers_clear_overlap(
        pos,
        offsets,
        min_distance=0.45,
        spacing=5.0,
        seed=3,
        cell=np.diag([30.0, 30.0, 30.0]),
        scratch_dir=tmp_path,
        atom_names=names,
        atomic_numbers=z,
    )

    assert captured["input"].count("structure ") == 1
    assert f"number {n_mono}" in captured["input"]
    assert "tolerance 2.0" in captured["input"]



def test_overlap_guard_repack_fn_uses_packmol_module():
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        _repack_monomers_clear_overlap_fn,
    )
    from mmml.interfaces.pycharmmInterface import packmol_repack

    assert _repack_monomers_clear_overlap_fn() is packmol_repack.repack_monomers_clear_overlap


def test_apply_overlap_repack_uses_psf_monomer_offsets(monkeypatch):
    """Repack must slice atoms by PSF counts, not uniform n_atoms/n_monomers."""
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        apply_overlap_repack_last_resort,
    )

    pos = np.arange(30, dtype=float).reshape(10, 3)
    captured: dict[str, np.ndarray] = {}

    class _Ctx:
        atoms_per_monomer = [3, 7]

    def fake_get_pos():
        return pos.copy()

    def fake_repack(positions, offsets, **kwargs):
        captured["offsets"] = np.asarray(offsets, dtype=int)
        return positions

    def fake_sync(_pos):
        return None

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        fake_get_pos,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        fake_sync,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._find_worst_intermonomer_overlap_fn",
        lambda: lambda _pos, _off, **kw: (1.0, None),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._repack_monomers_clear_overlap_fn",
        lambda: fake_repack,
    )
    monkeypatch.setattr(
        "mmml.utils.monomer_force_diag.resolve_selective_repack_monomers",
        lambda *_a, **_k: None,
    )

    cfg = DynamicsOverlapConfig(action="rescue", n_monomers=2, min_distance_A=2.0)
    apply_overlap_repack_last_resort(cfg, mlpot_ctx=_Ctx())

    np.testing.assert_array_equal(captured["offsets"], [0, 3, 10])
