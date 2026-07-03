"""MIC pipeline probes and Packmol inner-cube repack margins."""

from __future__ import annotations

import argparse

import numpy as np
import pytest


def test_probe_pre_mlpot_mic_contacts_logs_without_abort(capsys):
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        probe_pre_mlpot_mic_contacts,
    )

    args = argparse.Namespace(
        quiet=False,
        pre_mlpot_overlap_min_distance=2.3,
        pre_mlpot_h_heavy_min_distance=2.4,
        pre_mlpot_heavy_heavy_min_distance=2.9,
        solvents=["DCM"],
        dynamics_overlap_min_distance=1.5,
    )
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    worst = probe_pre_mlpot_mic_contacts(
        args,
        positions=pos,
        atoms_per_list=[2, 2],
        box_side=20.0,
        charmm_pbc=True,
        atomic_numbers=np.array([6, 1, 6, 1], dtype=int),
        context="MIC probe test",
        abort=False,
    )
    out = capsys.readouterr().out
    assert "MIC probe test:" in out
    assert worst > 0.0


def test_probe_pre_mlpot_mic_contacts_aborts_on_violation():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        probe_pre_mlpot_mic_contacts,
    )

    args = argparse.Namespace(
        quiet=True,
        pre_mlpot_overlap_min_distance=2.3,
        pre_mlpot_h_heavy_min_distance=2.4,
        pre_mlpot_heavy_heavy_min_distance=2.9,
        solvents=["DCM"],
    )
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0], [5.4, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="MIC probe abort"):
        probe_pre_mlpot_mic_contacts(
            args,
            positions=pos,
            atoms_per_list=[2, 2],
            box_side=20.0,
            charmm_pbc=True,
            atomic_numbers=np.array([17, 1, 17, 1], dtype=int),
            context="MIC probe abort",
            abort=True,
        )


def test_packmol_repack_uses_inner_cube_margin(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_repack

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    offsets = np.array([0, 2, 4], dtype=int)
    z = np.array([6, 1, 6, 1], dtype=int)
    captured: dict[str, str] = {}

    def fake_execute(packmol_input: str, inp_path):
        captured["input"] = packmol_input
        out = tmp_path / "repack.pdb"
        out.write_text(
            "\n".join(
                [
                    "ATOM      1  C   F000A   1       3.000   3.000   3.000  1.00  0.00           C",
                    "ATOM      2  H1  F000A   1       4.000   3.000   3.000  1.00  0.00           H",
                    "ATOM      3  C   M001A   2       6.000   3.000   3.000  1.00  0.00           C",
                    "ATOM      4  H1  M001A   2       7.000   3.000   3.000  1.00  0.00           H",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(packmol_repack, "execute_packmol_script", fake_execute)
    monkeypatch.setattr(packmol_repack, "_charmm_atom_metadata", lambda _n: (["C", "H1", "C", "H1"], z))

    packmol_repack.repack_monomers_clear_overlap(
        pos,
        offsets,
        min_distance=2.0,
        seed=3,
        cell=np.diag([10.0, 10.0, 10.0]),
        scratch_dir=tmp_path,
        atomic_numbers=z,
        packmol_margin_A=2.5,
    )

    # L=10, margin=2.5 → inner side 5, origin (2.5, 2.5, 2.5)
    assert "inside cube 2.5 2.5 2.5 5.0" in captured["input"]


def test_apply_density_prep_resilient_defaults_disables_pre_mlpot_lattice_abnr():
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_density_prep_resilient_defaults,
    )

    args = argparse.Namespace(liquid_prep=True, composition="DCM:52", box_size=28.0)
    apply_density_prep_resilient_defaults(args)
    assert int(getattr(args, "density_prep_lattice_abnr_steps", -1)) == 0
