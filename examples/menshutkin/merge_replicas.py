#!/usr/bin/env python3
"""Combine independent umbrella replicas into one MBAR input, dropping failures.

Why replicas rather than one long run
-------------------------------------
This checkpoint sustains roughly 8 ps per window before some window finds a
spurious well in the fitted surface. Measured: a 0.5 ps/window run and an
8 ps/window run were clean, while a 55 ps/window run left three windows (of
thirty) resetting 106, 84 and 42 times while the other 27 sampled perfectly.

Short independent replicas match that. The same total sampling is obtained, a
window that hits a well costs one replica instead of the campaign, the replicas
are genuinely uncorrelated -- which is better for error bars than one long
trajectory -- and nothing has to be recovered mid-flight.

A window is dropped *for the replica in which it failed only*; the same window
from other replicas is kept. Coverage is reported per window so a window that
lost most of its replicas is visible rather than silently under-sampled.

    python merge_replicas.py <run_dir_1> ... -o <merged_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--min-replicas", type=int, default=2,
                   help="Warn if a window survives in fewer replicas than this")
    args = p.parse_args()

    loaded = []
    for d in args.run_dirs:
        f = d / "umbrella_snapshots.npz"
        if not f.exists():
            print(f"  skip {d.name}: no umbrella_snapshots.npz")
            continue
        loaded.append((d, np.load(f, allow_pickle=True)))
    if not loaded:
        raise SystemExit("no replicas with snapshots")

    ref = loaded[0][1]
    xi0 = np.asarray(ref["xi0"])
    n_win = len(xi0)
    for d, z in loaded[1:]:
        if not np.allclose(np.asarray(z["xi0"]), xi0):
            raise SystemExit(f"{d.name} has a different window ladder")

    print(f"{len(loaded)} replicas, {n_win} windows\n")
    print(f"{'replica':>22s} {'frames/win':>11s} {'failed windows':>40s}")
    kept_per_window = np.zeros(n_win, dtype=int)
    pos_parts: list[list[np.ndarray]] = [[] for _ in range(n_win)]
    cv_parts: list[list[np.ndarray]] = [[] for _ in range(n_win)]
    for d, z in loaded:
        failed = set(np.asarray(z["failed_windows"]).tolist()) if "failed_windows" in z.files else set()
        pos = np.asarray(z["positions"])
        cv = np.asarray(z["cv_traj"])
        print(f"{d.name:>22s} {pos.shape[1]:11d} "
              f"{(sorted(failed) if failed else 'none')!s:>40s}")
        for w in range(n_win):
            if w in failed:
                continue
            pos_parts[w].append(pos[w])
            cv_parts[w].append(cv[w])
            kept_per_window[w] += 1

    thin = [w for w in range(n_win) if kept_per_window[w] < args.min_replicas]
    n_frames = min(sum(a.shape[0] for a in parts) for parts in pos_parts if parts)
    print(f"\n{'w':>3s} {'xi0':>7s} {'replicas kept':>14s} {'frames':>8s}")
    for w in range(n_win):
        mark = "  <-- THIN" if w in thin else ""
        print(f"{w:3d} {xi0[w]:+7.2f} {kept_per_window[w]:14d} "
              f"{sum(a.shape[0] for a in pos_parts[w]):8d}{mark}")
    if not all(kept_per_window):
        raise SystemExit(
            "some windows failed in every replica: "
            f"{[w for w in range(n_win) if not kept_per_window[w]]}. "
            "Add replicas, or drop those windows from the ladder."
        )

    # MBAR needs equal frame counts per window, so truncate to the smallest.
    positions = np.stack(
        [np.concatenate(pos_parts[w], axis=0)[:n_frames] for w in range(n_win)]
    )
    cv_traj = np.stack(
        [np.concatenate(cv_parts[w], axis=0)[:n_frames] for w in range(n_win)]
    )
    print(f"\nmerged to {n_frames} frames per window "
          f"(truncated to the thinnest window)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(ref[k]) for k in ref.files
           if k not in ("positions", "cv_traj", "energies_ev",
                        "reset_counts", "failed_windows",
                        "bin_minima_frame_idx", "bin_minima_energy_ev")}
    out["positions"] = positions
    out["cv_traj"] = cv_traj
    out["n_replicas"] = np.asarray(kept_per_window, dtype=np.int32)
    np.savez_compressed(args.output_dir / "umbrella_snapshots.npz", **out)
    print(f"wrote {args.output_dir / 'umbrella_snapshots.npz'}")
    if thin:
        print(f"WARNING: windows {thin} survived in fewer than "
              f"{args.min_replicas} replicas; their statistics are weak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
