"""Pick the 10 % transfer-learning subset and save it as one self-contained npz.

    python select_subset.py                    # writes tl_subset.npz
    python select_subset.py --fraction 0.05
    python select_subset.py --strategy proportional

Run once. `submit_tl.sh` consumes the npz it writes, so the selection is frozen
and reproducible rather than re-drawn on every submission.

Two things this has to get right
--------------------------------
**The dataset contains TWO atom orderings.** 9577 of the 11572 nine-atom
complexes are (Cl, N, C, H x6); the other 1995 are (C, Cl, H, H, H, N, H, H, H).
Computing xi = r(C-Cl) - r(C-N) with fixed indices would silently mangle 17 % of
them -- and mangle them into *plausible* numbers, which is worse. Cl, N and C are
therefore located from `Z` in every frame.

**The set is not just complexes.** It also holds 1624 CH3Cl and 1604 NH3
fragments. Those set the dissociation asymptote, so dropping them would train a
correction that is only defined near the complex. They are kept in proportion.

Selection strategy
------------------
`uniform-xi` (default) splits the complexes into equal-width xi bins and draws
the same number from each. The raw distribution is heavily peaked -- its deciles
run -10.2, -5.2, -3.4, -2.2, -1.4, -1.2, -0.9, +1.0, +3.5, +7.2, +11.5, so half
the data sits in |xi| < 2.2 and the transition-state region is thin. A
proportional draw would reproduce that imbalance in the expensive calculation,
which is the opposite of what a reactive correction needs.

`proportional` reproduces the original distribution instead. Use it if the
correction is meant to be accurate where the base model is already sampled
most, rather than uniformly along the reaction coordinate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_SRC = REPO / "splits/extended_dataset/energies_forces_dipoles_train_fixed5892.npz"


def reaction_coordinate(R, Z, N):
    """xi = r(C-Cl) - r(C-N) per frame; NaN where the frame is not a complex."""
    xi = np.full(len(R), np.nan)
    for i, (r, z, n) in enumerate(zip(R, Z, N)):
        z = z[:n]
        cl, nn, c = np.flatnonzero(z == 17), np.flatnonzero(z == 7), np.flatnonzero(z == 6)
        if cl.size != 1 or nn.size != 1 or c.size != 1:
            continue                       # fragment, or ambiguous
        xi[i] = (np.linalg.norm(r[c[0]] - r[cl[0]])
                 - np.linalg.norm(r[c[0]] - r[nn[0]]))
    return xi


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=HERE / "tl_subset.npz")
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--strategy", choices=("uniform-xi", "proportional"),
                   default="uniform-xi")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--seed", type=int, default=314159)
    a = p.parse_args()

    d = np.load(a.src)
    R, Z, N = d["R"], d["Z"], d["N"]
    rng = np.random.default_rng(a.seed)
    n_target = int(round(len(R) * a.fraction))

    xi = reaction_coordinate(R, Z, N)
    is_complex = np.isfinite(xi)
    frag = np.flatnonzero(~is_complex)
    comp = np.flatnonzero(is_complex)

    # keep the complex/fragment ratio of the source
    n_comp = int(round(n_target * len(comp) / len(R)))
    n_frag = n_target - n_comp

    if a.strategy == "uniform-xi":
        edges = np.linspace(xi[comp].min(), xi[comp].max(), a.bins + 1)
        which = np.clip(np.digitize(xi[comp], edges) - 1, 0, a.bins - 1)
        per = int(np.ceil(n_comp / a.bins))
        picked = []
        for b in range(a.bins):
            pool = comp[which == b]
            if pool.size:
                picked.append(rng.choice(pool, size=min(per, pool.size),
                                         replace=False))
        sel_c = np.concatenate(picked)
        if sel_c.size > n_comp:                    # trim evenly, keep coverage
            sel_c = rng.choice(sel_c, size=n_comp, replace=False)
    else:
        sel_c = rng.choice(comp, size=n_comp, replace=False)

    # fragments split between the two species in their source proportion
    sel_f = rng.choice(frag, size=min(n_frag, frag.size), replace=False)
    sel = np.sort(np.concatenate([sel_c, sel_f]))

    out = {k: d[k][sel] for k in d.files}
    out["source_index"] = sel                     # row in the source npz
    out["xi"] = xi[sel]
    np.savez_compressed(a.out, **out)

    print(f"source     {a.src}")
    print(f"           {len(R)} frames  ({len(comp)} complex, {len(frag)} fragment)")
    print(f"strategy   {a.strategy}, seed {a.seed}")
    print(f"selected   {len(sel)} frames "
          f"({len(sel_c)} complex, {len(sel_f)} fragment) = "
          f"{100 * len(sel) / len(R):.1f} %")
    x = xi[sel_c]
    print(f"xi range   {x.min():+.2f} .. {x.max():+.2f}")
    h, e = np.histogram(x, bins=8)
    print("xi coverage (8 bins):")
    for c, lo, hi in zip(h, e[:-1], e[1:]):
        print(f"   {lo:+6.2f}..{hi:+6.2f}  {c:4d}  {'#' * int(40 * c / max(h.max(), 1))}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
