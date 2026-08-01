"""Auto-unpadding must trim the hybrid ML/MM per-atom arrays too.

`_maybe_unpad_dataset` used to trim only R/Z/F. A CGenFF-enriched NPZ padded
wider than its own maximum -- which is what `scripts/des_h5_to_npz.py --pad 34`
produces for any residue-filtered subset -- then reached `hybrid_forward` with
`atom_mask` at the trimmed width and the `mol_id`-derived keep mask at the
original width, failing with

    TypeError: mul got incompatible shapes for broadcasting: (116,), (136,)

which names neither the field nor the file responsible.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.cli.make.make_training import _maybe_unpad_dataset

PAD = 34
MAX_N = 29
N = 6


def _write_padded_enriched(path) -> None:
    rng = np.random.default_rng(0)
    n_real = np.full(N, MAX_N, dtype=np.int64)
    Z = np.zeros((N, PAD), dtype=np.int32)
    Z[:, :MAX_N] = 6
    np.savez_compressed(
        path,
        R=rng.normal(size=(N, PAD, 3)),
        Z=Z,
        F=rng.normal(size=(N, PAD, 3)),
        N=n_real,
        E=rng.normal(size=N),
        # hybrid ML/MM fields written by `mmml prepare-mm-dataset`
        cgenff_type_idx=np.full((N, PAD), -1, dtype=np.int32),
        mol_id=np.full((N, PAD), -1, dtype=np.int32),
        cgenff_charge=np.zeros((N, PAD)),
        F_cgenff_mm=np.zeros((N, PAD, 3)),
        E_cgenff_mm=np.zeros((N, 1)),
        # shared master tables -- (n_types,), must pass through untouched
        cgenff_master_sigmas=np.linspace(1.0, 4.0, 164),
        cgenff_master_epsilons=np.linspace(0.01, 0.5, 164),
        cgenff_res_name=np.array([["TIP3", "MEOH"]] * N, dtype="<U8"),
    )


def test_unpad_trims_every_per_atom_array(tmp_path):
    src = tmp_path / "enriched.npz"
    _write_padded_enriched(src)

    out_path, natoms = _maybe_unpad_dataset(str(src), None)

    assert natoms == MAX_N
    assert out_path != str(src), "expected an unpadded copy to be written"

    out = np.load(out_path, allow_pickle=True)
    for key in ("R", "Z", "F", "cgenff_type_idx", "mol_id", "cgenff_charge",
                "F_cgenff_mm"):
        assert out[key].shape[1] == MAX_N, f"{key} was not unpadded"

    # Non-atom-axis arrays must survive unchanged.
    assert out["E_cgenff_mm"].shape == (N, 1)
    assert out["cgenff_master_sigmas"].shape == (164,)
    assert out["cgenff_res_name"].shape == (N, 2)
    assert out["N"].tolist() == [MAX_N] * N


def test_unpad_is_skipped_when_natoms_is_pinned(tmp_path):
    """--num-atoms 34 is the escape hatch, and must not rewrite the file."""
    src = tmp_path / "enriched.npz"
    _write_padded_enriched(src)

    out_path, natoms = _maybe_unpad_dataset(str(src), PAD)

    assert natoms == PAD
    assert out_path == str(src)


def test_unpad_noop_when_already_tight(tmp_path):
    src = tmp_path / "tight.npz"
    rng = np.random.default_rng(1)
    np.savez_compressed(
        src,
        R=rng.normal(size=(N, 10, 3)),
        Z=np.full((N, 10), 6, dtype=np.int32),
        N=np.full(N, 10, dtype=np.int64),
        E=rng.normal(size=N),
    )

    out_path, natoms = _maybe_unpad_dataset(str(src), None)

    assert natoms == 10
    assert out_path == str(src), "a tight dataset must not be rewritten"
