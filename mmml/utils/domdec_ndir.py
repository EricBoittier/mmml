"""CHARMM DOMDEC NDIR helpers for MPI rank counts."""

from __future__ import annotations


def suggest_domdec_ndir(n_ranks: int) -> tuple[int, int, int]:
    """Return ``(nx, ny, nz)`` for ``ENERGY DOMDEC NDIR nx ny nz``.

    c47 site builds reject a Y split of 2–7 nodes (Y must be 1 or >= 8).
    Prefer a 1-D decomposition along X or Z when possible.
    """
    n = int(n_ranks)
    if n <= 1:
        return (1, 1, 1)

    candidates: list[tuple[int, int, int, int, int]] = []
    for nx in range(1, n + 1):
        if n % nx:
            continue
        rest = n // nx
        for ny in range(1, rest + 1):
            if rest % ny:
                continue
            nz = rest // ny
            if ny not in (1,) and ny < 8:
                continue
            # sort key: avoid multi-axis splits, prefer X over Z, then larger nx
            score = (
                0 if ny == 1 else 1,
                0 if (nx == n or nz == n) else 1,
                0 if nx >= nz else 1,
                -nx,
            )
            candidates.append((score[0], score[1], score[2], score[3], nx, ny, nz))

    if not candidates:
        raise ValueError(f"no valid DOMDEC NDIR for n_ranks={n}")

    candidates.sort()
    _, _, _, _, nx, ny, nz = candidates[0]
    return (nx, ny, nz)


def format_domdec_ndir(n_ranks: int) -> str:
    nx, ny, nz = suggest_domdec_ndir(n_ranks)
    return f"{nx} {ny} {nz}"
