"""Close-contact filter: geometry, frame selection, and what must NOT be filtered.

The dangerous failure is silent: filtering a per-type table (cgenff_master_*) by
a per-frame mask, or dropping the wrong frames, produces a dataset that still
loads and trains but means something different.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.cli.misc.filter_close_contacts import main, min_intermolecular_distance


def _frames(distances):
    """Dimers of one atom each, separated along x by the given distances."""
    n = len(distances)
    R = np.zeros((n, 2, 3))
    for i, d in enumerate(distances):
        R[i, 1, 0] = d
    Z = np.ones((n, 2), dtype=int)
    mol_id = np.tile(np.array([0, 1]), (n, 1))
    return R, Z, mol_id


def test_distance_is_between_monomers_not_within():
    R, Z, mol_id = _frames([3.0])
    # Add a second atom to monomer 0, very close to the first -- intramolecular,
    # so it must NOT count as a contact.
    R = np.concatenate([R, np.array([[[0.1, 0.0, 0.0]]])], axis=1)
    Z = np.concatenate([Z, np.ones((1, 1), dtype=int)], axis=1)
    mol_id = np.concatenate([mol_id, np.zeros((1, 1), dtype=int)], axis=1)

    d = min_intermolecular_distance(R, Z, mol_id)
    assert d[0] == pytest.approx(2.9), "should measure 3.0-0.1 across monomers"


def test_padding_atoms_are_ignored():
    R, Z, mol_id = _frames([4.0])
    # Padding at the origin (Z=0) would otherwise read as a 0 A contact.
    R = np.concatenate([R, np.zeros((1, 1, 3))], axis=1)
    Z = np.concatenate([Z, np.zeros((1, 1), dtype=int)], axis=1)
    mol_id = np.concatenate([mol_id, -np.ones((1, 1), dtype=int)], axis=1)

    d = min_intermolecular_distance(R, Z, mol_id)
    assert d[0] == pytest.approx(4.0)


def test_monomer_only_frame_is_infinite_not_zero():
    R = np.zeros((1, 2, 3))
    R[0, 1, 0] = 2.0
    Z = np.ones((1, 2), dtype=int)
    mol_id = np.zeros((1, 2), dtype=int)  # both atoms in monomer 0
    d = min_intermolecular_distance(R, Z, mol_id)
    assert np.isinf(d[0]), "no second monomer -> undefined, must not read as 0"


def _write_npz(tmp_path, distances):
    R, Z, mol_id = _frames(distances)
    n = len(distances)
    path = tmp_path / "in.npz"
    np.savez(
        path,
        R=R, Z=Z, mol_id=mol_id,
        E=np.arange(n, dtype=float).reshape(n, 1),
        F=np.zeros((n, 2, 3)),
        N=np.full((n, 1), 2),
        cgenff_master_sigmas=np.arange(185, dtype=float),
        cgenff_master_epsilons=np.arange(185, dtype=float),
    )
    return path


def test_drops_only_frames_below_the_cut(tmp_path):
    inp = _write_npz(tmp_path, [0.9, 1.2, 1.6, 3.0, 5.0])
    out = tmp_path / "out.npz"
    assert main(["--in", str(inp), "--out", str(out), "--min-contact", "1.5"]) == 0

    d = np.load(out)
    assert len(d["E"]) == 3
    # E was the frame index, so surviving E identifies which frames were kept.
    assert d["E"].ravel().tolist() == [2.0, 3.0, 4.0]


def test_per_type_tables_are_not_filtered(tmp_path):
    """cgenff_master_* are indexed by type, not frame -- must pass through whole."""
    inp = _write_npz(tmp_path, [0.9, 3.0, 5.0])
    out = tmp_path / "out.npz"
    main(["--in", str(inp), "--out", str(out), "--min-contact", "1.5"])

    d = np.load(out)
    assert len(d["E"]) == 2, "frames were filtered"
    assert len(d["cgenff_master_sigmas"]) == 185, "per-type table must be untouched"
    assert len(d["cgenff_master_epsilons"]) == 185


def test_refuses_to_overwrite_input(tmp_path):
    inp = _write_npz(tmp_path, [3.0, 4.0])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--in", str(inp), "--out", str(inp)])


def test_refuses_an_existing_output(tmp_path):
    inp = _write_npz(tmp_path, [3.0, 4.0])
    out = tmp_path / "out.npz"
    out.write_text("existing")
    with pytest.raises(SystemExit, match="already exists"):
        main(["--in", str(inp), "--out", str(out)])


def test_refuses_to_delete_most_of_the_dataset(tmp_path):
    """A cut that removes >50% is a mistake, not a cleanup."""
    inp = _write_npz(tmp_path, [1.0, 1.1, 1.2, 5.0])
    out = tmp_path / "out.npz"
    with pytest.raises(SystemExit, match="Refusing to write"):
        main(["--in", str(inp), "--out", str(out), "--min-contact", "2.0"])


def test_dry_run_writes_nothing(tmp_path):
    inp = _write_npz(tmp_path, [0.9, 3.0, 5.0])
    out = tmp_path / "out.npz"
    assert main(["--in", str(inp), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_hydrogen_bonds_survive_the_default_cut(tmp_path):
    """1.5 A must keep H-bonds (H...O ~1.6-1.8 A), or we delete real physics."""
    inp = _write_npz(tmp_path, [1.65, 1.8, 2.8])
    out = tmp_path / "out.npz"
    main(["--in", str(inp), "--out", str(out), "--min-contact", "1.5"])
    assert len(np.load(out)["E"]) == 3
