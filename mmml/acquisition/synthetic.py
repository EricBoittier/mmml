"""Synthetic unlabeled pools for smoke tests (not scientific data)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.acquisition.ids import composition_key


def make_smoke_pool(
    *,
    n_traj_per_stratum: int = 2,
    n_frames: int = 6,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Tiny H2O / CH4 pool with two temperatures and grouped trajectories."""
    rng = np.random.default_rng(int(seed))
    molecules = {
        "H2O": np.array([8, 1, 1], dtype=np.int32),
        "CH4": np.array([6, 1, 1, 1, 1], dtype=np.int32),
    }
    templates = {
        "H2O": np.array(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=np.float64,
        ),
        "CH4": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [0.63, -0.63, -0.63],
                [-0.63, 0.63, -0.63],
                [-0.63, -0.63, 0.63],
            ],
            dtype=np.float64,
        )
        * 1.09,
    }
    temperatures = (300.0, 400.0)
    pad = 5
    rows_R = []
    rows_Z = []
    rows_N = []
    rows_T = []
    rows_seed = []
    rows_frame = []
    rows_comp = []
    traj = 0
    for mol_name, z in molecules.items():
        templ = templates[mol_name]
        n_at = len(z)
        for T in temperatures:
            amp = 0.04 + 0.0001 * (T - 300.0)
            for _local in range(int(n_traj_per_stratum)):
                gseed = 1000 + traj
                for frame in range(int(n_frames)):
                    # Non-rigid per-atom jitter so COM-centering does not collapse frames.
                    jitter = rng.normal(0.0, amp, size=templ.shape)
                    shift = rng.normal(0.0, 0.2, size=(1, 3))
                    r = templ * (1.0 + 0.01 * frame) + jitter + shift
                    R = np.zeros((pad, 3), dtype=np.float64)
                    Z = np.zeros((pad,), dtype=np.int32)
                    R[:n_at] = r
                    Z[:n_at] = z
                    rows_R.append(R)
                    rows_Z.append(Z)
                    rows_N.append(n_at)
                    rows_T.append(T)
                    rows_seed.append(gseed)
                    rows_frame.append(frame)
                    rows_comp.append(composition_key(z, n_at))
                traj += 1
    n = len(rows_R)
    return {
        "R": np.stack(rows_R),
        "Z": np.stack(rows_Z),
        "N": np.asarray(rows_N, dtype=np.int32),
        "E": np.full(n, np.nan, dtype=np.float64),
        "F": np.full((n, pad, 3), np.nan, dtype=np.float64),
        "temperature": np.asarray(rows_T, dtype=np.float64),
        "group_seed": np.asarray(rows_seed, dtype=np.int32),
        "frame_index": np.asarray(rows_frame, dtype=np.int32),
        "composition": np.asarray(rows_comp, dtype=object),
        "phase": np.asarray(["gas"] * n, dtype=object),
    }


def default_seed_groups(pool: dict[str, Any], *, n_groups: int = 2) -> list[str]:
    seeds = np.unique(np.asarray(pool["group_seed"]))
    return [f"group_seed:{int(s)}" for s in seeds[:n_groups]]
