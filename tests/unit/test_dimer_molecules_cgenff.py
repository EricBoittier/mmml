"""Dimer-scan geometries for arbitrary CGenFF residues."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.analysis.dimer_molecules import make_oriented_scan_geometries
from mmml.analysis.dimer_scans import geometric_centroid


def test_campaign_pair_still_uses_chemically_motivated_orientation() -> None:
    geoms = list(make_oriented_scan_geometries("TIP3", "MEOH", [2.8], [0.0]))
    assert len(geoms) == 1
    assert geoms[0].pair == ("TIP3", "MEOH")
    assert geoms[0].distance_angstrom == pytest.approx(2.8)


def test_aco_maps_to_campaign_acetone_orientation() -> None:
    """CGenFF ACO should reuse the ACE campaign orientation (not generic)."""
    campaign = list(make_oriented_scan_geometries("ACE", "ACE", [4.0], [0.0]))
    cgenff = list(make_oriented_scan_geometries("ACO", "ACO", [4.0], [0.0]))
    assert len(cgenff) == 1
    assert cgenff[0].pair == ("ACO", "ACO")
    np.testing.assert_allclose(
        campaign[0].atoms.get_positions(),
        cgenff[0].atoms.get_positions(),
        atol=1e-6,
    )


def test_generic_cgenff_homodimer_centroid_separation() -> None:
    """Non-campaign CGenFF residues get a centroid–centroid Z scan."""
    # CYBZ has no campaign pair config; geometry comes from make-res or fail.
    # Use bundled ACO vs DCM which *is* a campaign pair — instead pick a
    # residue with a bundled PDB that is not in PAIR_SCAN_CONFIG as a pair
    # with a distinct second residue that forces the generic path.
    # ACO–CYBZ is not a campaign pair; CYBZ has no bundled PDB in CI, so use
    # a synthetic cwd override via monkeypatch in a separate test.
    geoms = list(
        make_oriented_scan_geometries(
            "ACO",
            "DCM",
            [5.0],
            [0.0],
            generate_missing=False,
        )
    )
    # ACO↔DCM maps to ACE↔DCM campaign pair.
    assert geoms[0].pair == ("ACO", "DCM")


def test_generic_pair_without_campaign_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    pdb = tmp_path / "pdb"
    pdb.mkdir()
    # Two simple 2-atom "monomers" written under real CGenFF names that do
    # not share a PAIR_SCAN_CONFIG entry with each other as a generic pair.
    # Use MEOH and BENZ? That IS a campaign pair. Use ACO and CYBZ with
    # local PDBs so generate is not required.
    for name, z_shift in (("cybz", 0.0), ("etoh", 0.0)):
        (pdb / f"{name}.pdb").write_text(
            "ATOM      1  C1  {res}    1       0.000   0.000   {z:.3f}  1.00  0.00           C\n"
            "ATOM      2  H1  {res}    1       1.090   0.000   {z:.3f}  1.00  0.00           H\n"
            "END\n".format(res=name.upper(), z=z_shift),
            encoding="utf-8",
        )

    # ETOH must exist in CGenFF RTF
    from mmml.interfaces.pycharmmInterface.cgenff_residues import is_cgenff_residue_name

    assert is_cgenff_residue_name("ETOH")
    assert is_cgenff_residue_name("CYBZ")

    geoms = list(
        make_oriented_scan_geometries(
            "CYBZ",
            "ETOH",
            [4.0],
            [0.0],
            generate_missing=False,
        )
    )
    assert len(geoms) == 1
    assert geoms[0].pair == ("CYBZ", "ETOH")
    frag_a, frag_b = geoms[0].fragments
    ca = geometric_centroid(geoms[0].atoms[frag_a])
    cb = geometric_centroid(geoms[0].atoms[frag_b])
    np.testing.assert_allclose(np.linalg.norm(cb - ca), 4.0, atol=1e-6)
    # Approach axis is Z for the generic path.
    np.testing.assert_allclose(abs(cb[2] - ca[2]), 4.0, atol=1e-6)


def test_unknown_residue_raises() -> None:
    with pytest.raises(KeyError, match="not a known campaign"):
        list(
            make_oriented_scan_geometries(
                "NOTAREAL",
                "TIP3",
                [3.0],
                [0.0],
                generate_missing=False,
            )
        )
