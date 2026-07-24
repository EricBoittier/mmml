"""Load NH3–CH3Cl geometries from the bundled filtered NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ = EXAMPLE_DIR / "nh3_ch3cl_filtered.npz"


def load_dimer_frame(
    npz_path: Path | str | None = None,
    *,
    index: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Z, R)`` for one N=9 dimer frame (Å).

    Atom order in the dataset is ``Cl, N, C, H×3(N), H×3(C)`` — not monomer-blocked.
    For ML-only smokes treat the complex as a single 9-atom system.
    """
    path = Path(npz_path) if npz_path is not None else DEFAULT_NPZ
    data = np.load(path, allow_pickle=True)
    n = np.asarray(data["N"])
    dimer_idx = np.flatnonzero(n == 9)
    if dimer_idx.size == 0:
        raise ValueError(f"No N=9 frames in {path}")
    if index is None:
        rng = np.random.default_rng(seed)
        index = int(rng.choice(dimer_idx))
    else:
        index = int(index)
        if index not in set(int(i) for i in dimer_idx):
            # Allow absolute indices into the full array if they are dimers.
            if not (0 <= index < len(n) and int(n[index]) == 9):
                raise ValueError(f"index={index} is not an N=9 frame in {path}")
    z = np.asarray(data["Z"][index], dtype=np.int32)
    r = np.asarray(data["R"][index], dtype=np.float64)
    mask = z > 0
    return z[mask], r[mask]


def write_evaluate_npz(
    out_path: Path | str,
    npz_path: Path | str | None = None,
    *,
    index: int | None = None,
    seed: int = 0,
) -> Path:
    """Write a single-frame NPZ for ``mmml md-system --evaluate-npz``."""
    z, r = load_dimer_frame(npz_path, index=index, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        positions=r.astype(np.float64),
        atomic_numbers=z.astype(np.int32),
        Z=z.astype(np.int32),
        R=r.astype(np.float64),
        N=np.array([len(z)], dtype=np.int32),
    )
    return out
