from __future__ import annotations

import numpy as np

from mmml.data.orbax_shards import (
    iter_restored_shards,
    read_manifest,
    write_orbax_shards,
)


def test_orbax_shards_bound_preprocessing_chunks(tmp_path) -> None:
    seen_sizes = []

    def preprocess(records):
        seen_sizes.append(len(records))
        return {"R": np.zeros((len(records), 3, 3), dtype=np.float32)}

    manifest_path = write_orbax_shards(
        range(5),
        tmp_path,
        preprocess,
        shard_size=2,
        dataset_kind="test",
    )

    manifest = read_manifest(tmp_path)
    restored = list(iter_restored_shards(tmp_path))
    assert manifest_path == tmp_path / "manifest.json"
    assert seen_sizes == [2, 2, 1]
    assert manifest["num_structures"] == 5
    assert [shard["R"].shape[0] for shard in restored] == [2, 2, 1]
