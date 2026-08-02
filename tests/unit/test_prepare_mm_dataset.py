"""CGenFF NPZ enrichment (``mmml prepare-mm-dataset``).

Covers the dense-NPZ driver end to end on a synthetic water dimer set: the
per-atom fields the hybrid ML/MM trainer needs, the ``-1`` padding convention,
per-monomer charge conservation, dropping of non-dimer frames, and the shared
core against :mod:`mmml.data.cgenff_dataset`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.data.cgenff_dataset import assign_frame_cgenff, load_reference

# One water in the (O, H, H) atom order the TIP3 template expects.
_WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.927, 0.0]])
_WATER_Z = np.array([8, 1, 1])


def _water_dimer(separation: float = 3.0):
    r = np.concatenate([_WATER, _WATER + np.array([separation, 0.0, 0.0])], axis=0)
    z = np.concatenate([_WATER_Z, _WATER_Z])
    return z, r


def _padded_npz(path, n_frames=4, pad_to=8):
    R, Z, N = [], [], []
    for k in range(n_frames):
        z, r = _water_dimer(3.0 + 0.1 * k)
        rp = np.zeros((pad_to, 3))
        rp[: len(z)] = r
        zp = np.zeros(pad_to, dtype=np.int64)
        zp[: len(z)] = z
        R.append(rp)
        Z.append(zp)
        N.append(len(z))
    np.savez_compressed(
        path,
        R=np.asarray(R),
        Z=np.asarray(Z),
        N=np.asarray(N),
        E=np.arange(n_frames, dtype=np.float64).reshape(-1, 1),
    )


def test_assign_frame_water_dimer():
    ref = load_reference()
    z, r = _water_dimer()
    assignment, reason = assign_frame_cgenff(z, r, ref)

    assert reason is None
    assert assignment.res_names == ("TIP3", "TIP3")
    np.testing.assert_array_equal(assignment.mol_id, [0, 0, 0, 1, 1, 1])
    # O carries the negative charge; each monomer is net-neutral.
    assert assignment.cgenff_charge[0] < 0 < assignment.cgenff_charge[1]
    assert abs(assignment.cgenff_charge[:3].sum()) < 1e-9
    assert abs(assignment.cgenff_charge[3:].sum()) < 1e-9
    assert assignment.f_cgenff_mm.shape == (6, 3)


def test_assign_frame_rejects_non_dimer():
    ref = load_reference()
    # A lone water is one covalent component, not a dimer.
    assignment, reason = assign_frame_cgenff(_WATER_Z, _WATER, ref)
    assert assignment is None
    assert "non-dimer" in reason


def test_enrich_npz_adds_hybrid_fields(tmp_path):
    from mmml.cli.misc.prepare_mm_dataset import enrich_npz
    from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS

    inp = tmp_path / "in.npz"
    out = tmp_path / "out.npz"
    _padded_npz(inp, n_frames=4, pad_to=8)

    summary = enrich_npz(inp, out, quiet=True)
    assert summary["n_kept"] == 4

    data = dict(np.load(out, allow_pickle=True))

    # Every field the --hybrid-mm trainer requires is present.
    for key in (*HYBRID_MM_BATCH_KEYS, "cgenff_master_sigmas", "cgenff_master_epsilons"):
        assert key in data, f"missing {key}"

    # Dense per-atom shape, padding marked with -1 (type/mol) and 0 (charge).
    assert data["cgenff_type_idx"].shape == (4, 8)
    assert np.all(data["cgenff_type_idx"][:, 6:] == -1)
    assert np.all(data["mol_id"][:, 6:] == -1)
    assert np.all(data["cgenff_charge"][:, 6:] == 0.0)

    # Real atoms mapped: mol_id 0/1, valid type indices, charge conserved.
    assert np.all(data["mol_id"][:, :3] == 0)
    assert np.all(data["mol_id"][:, 3:6] == 1)
    assert np.all(data["cgenff_type_idx"][:, :6] >= 0)
    np.testing.assert_allclose(data["cgenff_charge"][:, :3].sum(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(data["cgenff_charge"][:, 3:6].sum(axis=1), 0.0, atol=1e-9)

    # Master tables are per-type (not per-sample) and index-compatible.
    n_types = data["cgenff_master_sigmas"].shape[0]
    assert data["cgenff_master_epsilons"].shape[0] == n_types
    assert int(data["cgenff_type_idx"][:, :6].max()) < n_types

    # Original per-sample arrays are carried through, MM baseline added.
    np.testing.assert_array_equal(data["E"].ravel(), np.arange(4))
    assert data["E_cgenff_mm"].shape == (4, 1)
    assert data["F_cgenff_mm"].shape == (4, 8, 3)
    # Inter-monomer force lives only on real atoms.
    assert np.all(data["F_cgenff_mm"][:, 6:] == 0.0)


def test_enrich_npz_drops_non_dimer_frames(tmp_path):
    from mmml.cli.misc.prepare_mm_dataset import enrich_npz

    inp = tmp_path / "in.npz"
    out = tmp_path / "out.npz"

    # Two water-dimer frames + one lone-water frame (dropped).
    z, r = _water_dimer()
    rp = np.zeros((3, 6, 3))
    zp = np.zeros((3, 6), dtype=np.int64)
    rp[0, :6] = r
    zp[0, :6] = z
    rp[1, :6] = r
    zp[1, :6] = z
    rp[2, :3] = _WATER  # single monomer
    zp[2, :3] = _WATER_Z
    np.savez_compressed(inp, R=rp, Z=zp, N=np.array([6, 6, 3]))

    summary = enrich_npz(inp, out, quiet=True)
    assert summary["n_kept"] == 2
    assert summary["n_dropped"] == 1
    assert dict(np.load(out))["cgenff_type_idx"].shape[0] == 2


def test_enrich_npz_strict_raises(tmp_path):
    from mmml.cli.misc.prepare_mm_dataset import enrich_npz

    inp = tmp_path / "in.npz"
    out = tmp_path / "out.npz"
    zp = np.zeros((1, 3), dtype=np.int64)
    rp = np.zeros((1, 3, 3))
    rp[0] = _WATER
    zp[0] = _WATER_Z
    np.savez_compressed(inp, R=rp, Z=zp, N=np.array([3]))

    with pytest.raises(ValueError):
        enrich_npz(inp, out, strict=True, quiet=True)
