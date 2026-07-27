"""Unit tests for umbrella packed pair-index layout."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.umbrella.energy import build_packed_graph, pack_positions


def test_pack_positions_tiles():
    r = np.arange(6, dtype=np.float64).reshape(2, 3)
    packed = pack_positions(r, 3)
    assert packed.shape == (6, 3)
    np.testing.assert_allclose(packed[0:2], r)
    np.testing.assert_allclose(packed[2:4], r)
    np.testing.assert_allclose(packed[4:6], r)


def test_build_packed_graph_offsets_match_physnet_md_layout():
    e3x = pytest.importorskip("e3x")

    n_atoms = 4
    n_windows = 3
    graph = build_packed_graph(n_atoms, n_windows)

    dst_single, src_single = e3x.ops.sparse_pairwise_indices(n_atoms)
    dst_single = np.asarray(dst_single, dtype=np.int32)
    src_single = np.asarray(src_single, dtype=np.int32)
    n_pairs = len(dst_single)

    dst = np.asarray(graph["dst_idx"])
    src = np.asarray(graph["src_idx"])
    assert dst.shape == (n_windows * n_pairs,)
    assert src.shape == (n_windows * n_pairs,)

    for k in range(n_windows):
        off = k * n_atoms
        sl = slice(k * n_pairs, (k + 1) * n_pairs)
        np.testing.assert_array_equal(dst[sl], dst_single + off)
        np.testing.assert_array_equal(src[sl], src_single + off)

    segments = np.asarray(graph["batch_segments"])
    assert segments.shape == (n_windows * n_atoms,)
    expected = np.repeat(np.arange(n_windows, dtype=np.int32), n_atoms)
    np.testing.assert_array_equal(segments, expected)
    assert graph["batch_size"] == n_windows
    assert float(np.asarray(graph["atom_mask"]).sum()) == pytest.approx(
        float(n_windows * n_atoms)
    )


def test_build_packed_graph_rejects_bad_sizes():
    with pytest.raises(ValueError):
        build_packed_graph(0, 2)
    with pytest.raises(ValueError):
        build_packed_graph(2, 0)
