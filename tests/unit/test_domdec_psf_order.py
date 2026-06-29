from __future__ import annotations

from pathlib import Path

from mmml.utils.domdec_psf_order import find_domdec_hydrogen_order_issues


def _write_psf(path: Path, atom_lines: list[str], bonds: list[tuple[int, int]]) -> None:
    bond_values = " ".join(f"{a:8d}{b:8d}" for a, b in bonds)
    path.write_text(
        "\n".join(
            [
                "PSF EXT",
                "",
                f"{len(atom_lines):10d} !NATOM",
                *atom_lines,
                "",
                f"{len(bonds):10d} !NBOND: bonds",
                bond_values,
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_domdec_order_accepts_adjacent_hydrogens(tmp_path: Path) -> None:
    psf = tmp_path / "dcm_like.psf"
    _write_psf(
        psf,
        [
            "1 SEG 1 DCM C CG321 0.0 12.011",
            "2 SEG 1 DCM H1 HGA2 0.0 1.008",
            "3 SEG 1 DCM H2 HGA2 0.0 1.008",
            "4 SEG 1 DCM CL1 CLGA1 0.0 35.45",
            "5 SEG 1 DCM CL2 CLGA1 0.0 35.45",
        ],
        [(1, 2), (1, 3), (1, 4), (1, 5)],
    )
    assert find_domdec_hydrogen_order_issues(psf) == []


def test_domdec_order_flags_nonadjacent_hydrogens(tmp_path: Path) -> None:
    psf = tmp_path / "aco_like.psf"
    _write_psf(
        psf,
        [
            "1 SEG 1 ACO O1 OG2D3 0.0 15.999",
            "2 SEG 1 ACO C1 CG2O5 0.0 12.011",
            "3 SEG 1 ACO C2 CG331 0.0 12.011",
            "4 SEG 1 ACO C3 CG331 0.0 12.011",
            "5 SEG 1 ACO H21 HGA3 0.0 1.008",
            "6 SEG 1 ACO H22 HGA3 0.0 1.008",
            "7 SEG 1 ACO H23 HGA3 0.0 1.008",
            "8 SEG 1 ACO H31 HGA3 0.0 1.008",
            "9 SEG 1 ACO H32 HGA3 0.0 1.008",
            "10 SEG 1 ACO H33 HGA3 0.0 1.008",
        ],
        [(2, 3), (2, 4), (2, 1), (3, 5), (3, 6), (3, 7), (4, 8), (4, 9), (4, 10)],
    )
    issues = find_domdec_hydrogen_order_issues(psf)
    assert [issue.heavy_name for issue in issues] == ["C2", "C3"]
    assert "expected adjacent" in issues[0].format()
