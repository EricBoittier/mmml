#!/usr/bin/env python3
"""Exhaustive orientation scan: how much of a dimer's PES has spurious minima?

A single rigid cut proves nothing general. Two rigid monomers have 6 relative
DOF: separation ``r``, the approach direction on A's sphere (2), and B's own
orientation (3). This sweeps S^2 x SO(3) and runs a 1D ``r`` scan along each
ray, then counts rays whose profile has more than one minimum -- which a rigid
scan does not admit.

Why this is affordable: ``hybrid_forward`` agrees with the deployed MD
calculator to 7.6e-4 eV (scripts/check_hybrid_train_md_parity.py), is batched,
and needs no CHARMM. So ~1e6 evaluations take minutes rather than hours, and
--validate spot-checks that licence rather than assuming it.

On symmetry: both DCM and acetone are nominally C2v (order 4), so S^2 x SO(3)
could be folded ~32x (4 x 4 x homodimer exchange). It is deliberately NOT done.
The monomers here are MD snapshots -- thermally distorted, so C2v holds only
approximately, and folding configurations that are not actually equivalent would
be wrong. Since the batched path makes the redundancy cost minutes, the
symmetric duplicates are kept and used as a free consistency control instead.

Sampling is low-discrepancy (Fibonacci on S^2, super-Fibonacci on SO(3)) rather
than an Euler grid, which clusters at the poles.

    python scripts/scan_dimer_orientations.py --checkpoint CKPT --data D.npz \\
        --resid ACO --n-directions 10 --n-orientations 24 --out orient_ACO
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EV_TO_KCAL = 23.0605


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform directions on S^2 (no pole clustering)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def super_fibonacci(n: int) -> np.ndarray:
    """``n`` near-uniform unit quaternions on SO(3) (Alexa, CVPR 2022)."""
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    s = i / n
    t = s * n / phi
    d = 2.0 * np.pi * (t - np.floor(t))
    r = np.sqrt(s)
    R = np.sqrt(1.0 - s)
    t2 = i / psi
    a = 2.0 * np.pi * (t2 - np.floor(t2))
    return np.stack([r * np.sin(d), r * np.cos(d), R * np.sin(a), R * np.cos(a)], axis=1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def find_minima(e: np.ndarray, prominence: float) -> list[int]:
    out = []
    for i in range(1, len(e) - 1):
        if e[i] < e[i - 1] and e[i] < e[i + 1]:
            left = np.max(e[:i])
            right = np.max(e[i + 1:])
            if min(left, right) - e[i] >= prominence:
                out.append(i)
    return out


def main() -> int:
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--resid", default="ACO")
    p.add_argument("--n-directions", type=int, default=10)
    p.add_argument("--n-orientations", type=int, default=24)
    p.add_argument("--r-min", type=float, default=3.0)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--n-r", type=int, default=36)
    p.add_argument(
        "--min-contact",
        type=float,
        default=None,
        help="Skip r-points with intermolecular atom–atom dmin below this (Å). "
        "Default: mmml.analysis.dimer_scans.DEFAULT_ORIENT_MIN_CONTACT_A (2.0). "
        "COM–COM r alone is not steric for DCM — clash points invent deep wells.",
    )
    # Default = kT at 150 K. A sub-thermal threshold counts ripple your dynamics
    # cannot resolve and inverts the verdict: at 0.023 kcal/mol the 8.0 model
    # looks WORSE than the 6.0 (71% vs 62% of rays); at kT it is 34.6% vs 14.6%.
    p.add_argument("--prominence", type=float, default=0.0129,
                   help="min well depth to count (eV; default 0.0129 = kT at 150 K)")
    p.add_argument("--ml-switch-width", type=float, default=DEFAULT_ML_SWITCH_WIDTH)
    p.add_argument("--mm-switch-on", type=float, default=DEFAULT_MM_SWITCH_ON)
    p.add_argument("--mm-switch-width", type=float, default=DEFAULT_MM_SWITCH_WIDTH)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="orient_scan")
    p.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gate the checkpoint's EMA params (default: on). "
        "Use --no-use-ema for the live training weights.",
    )
    args = p.parse_args()

    import jax
    import jax.numpy as jnp

    from mmml.analysis.dimer_scans import (
        DEFAULT_ORIENT_MIN_CONTACT_A,
        intermolecular_min_distance,
    )
    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS, hybrid_forward
    from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit
    from mmml.models.short_range_wall import inter_monomer_wall_energy

    min_contact = (
        DEFAULT_ORIENT_MIN_CONTACT_A if args.min_contact is None else float(args.min_contact)
    )

    raw = dict(np.load(args.data, allow_pickle=True))
    res = np.array([str(x) for x in raw["res_name"]])
    idx = np.where(res == args.resid)[0]
    if len(idx) == 0:
        print(f"no {args.resid} monomer in {args.data}", file=sys.stderr)
        return 1
    k = int(idx[0])
    n_mono = int(raw["N"][k])
    # Dataset order + dataset charges is self-consistent for hybrid_forward
    # (training never sees the PSF), so no PSF reindexing is needed here --
    # unlike the MD calculator, which indexes MM by PSF position.
    Z1 = np.asarray(raw["Z"][k])[:n_mono]
    R1 = np.asarray(raw["R"][k])[:n_mono]
    R1 = R1 - R1.mean(axis=0)
    t1 = np.asarray(raw["cgenff_type_idx"][k])[:n_mono]
    q1 = np.asarray(raw["cgenff_charge"][k])[:n_mono]

    n_at = 2 * n_mono
    pad = int(np.asarray(raw["Z"]).shape[1])
    if n_at > pad:
        print(f"dimer needs {n_at} atoms > padding {pad}", file=sys.stderr)
        return 1

    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = np.linspace(args.r_min, args.r_max, args.n_r)
    n_rays = len(dirs) * len(quats)
    n_tot = n_rays * len(rs)
    print(f"{args.resid}: {len(dirs)} directions x {len(quats)} orientations = "
          f"{n_rays} rays x {len(rs)} r-points = {n_tot} evaluations")
    print(f"handoff: ML on 0-{args.mm_switch_on - args.ml_switch_width:g}, "
          f"blend to {args.mm_switch_on:g}, MM tail to "
          f"{args.mm_switch_on + args.mm_switch_width:g} A")
    print(f"min intermolecular contact for well metrics: {min_contact:g} Å")

    # --- assemble every geometry up front ---------------------------------
    R_all = np.zeros((n_tot, pad, 3), dtype=np.float64)
    Z_all = np.zeros((n_tot, pad), dtype=np.int32)
    T_all = np.full((n_tot, pad), -1, dtype=np.int32)
    Q_all = np.zeros((n_tot, pad), dtype=np.float64)
    M_all = np.full((n_tot, pad), -1, dtype=np.int32)
    ray_of = np.zeros(n_tot, dtype=np.int32)
    ir_of = np.zeros(n_tot, dtype=np.int32)
    dmin_of = np.zeros(n_tot, dtype=np.float64)

    n = 0
    for di, d in enumerate(dirs):
        for qi, q in enumerate(quats):
            Rb0 = R1 @ quat_to_matrix(q).T
            for ri, r in enumerate(rs):
                Ra = R1 - 0.5 * r * d
                Rb = Rb0 + 0.5 * r * d
                R_all[n, :n_mono] = Ra
                R_all[n, n_mono:n_at] = Rb
                Z_all[n, :n_mono] = Z1
                Z_all[n, n_mono:n_at] = Z1
                T_all[n, :n_mono] = t1
                T_all[n, n_mono:n_at] = t1
                Q_all[n, :n_mono] = q1
                Q_all[n, n_mono:n_at] = q1
                M_all[n, :n_mono] = 0
                M_all[n, n_mono:n_at] = 1
                ray_of[n] = di * len(quats) + qi
                ir_of[n] = ri
                dmin_of[n] = intermolecular_min_distance(Ra, Rb)
                n += 1

    # prepare_batches_jit shuffles and DROPS the remainder, which would delete
    # random points and gut whole rays (seen: 48 dropped -> 35 of 72 rays lost).
    # Pad up to a whole number of batches with copies; ids keep them separable.
    n_pad = -(-n_tot // args.batch_size) * args.batch_size
    if n_pad > n_tot:
        extra = n_pad - n_tot
        R_all = np.concatenate([R_all, np.repeat(R_all[:1], extra, 0)])
        Z_all = np.concatenate([Z_all, np.repeat(Z_all[:1], extra, 0)])
        T_all = np.concatenate([T_all, np.repeat(T_all[:1], extra, 0)])
        Q_all = np.concatenate([Q_all, np.repeat(Q_all[:1], extra, 0)])
        M_all = np.concatenate([M_all, np.repeat(M_all[:1], extra, 0)])

    _, params, model = _load_physnet_checkpoint(Path(args.checkpoint), pad, use_ema=args.use_ema)
    sig = jnp.asarray(raw["cgenff_master_sigmas"])
    eps = jnp.asarray(raw["cgenff_master_epsilons"])

    d = {
        "R": jnp.asarray(R_all), "Z": jnp.asarray(Z_all),
        "F": jnp.zeros_like(jnp.asarray(R_all)), "E": jnp.zeros((n_pad, 1)),
        "N": jnp.full((n_pad,), n_at), "D": jnp.zeros((n_pad, 3)),
        "cgenff_type_idx": jnp.asarray(T_all), "cgenff_charge": jnp.asarray(Q_all),
        "mol_id": jnp.asarray(M_all), "id": jnp.arange(n_pad),
    }
    KEYS = ["R", "Z", "F", "E", "N", "D", "dst_idx", "src_idx", "batch_segments",
            "id"] + list(HYBRID_MM_BATCH_KEYS)
    # The trainer's own batcher: a hand-rolled one silently produced 30x-wrong
    # energies once already.
    batches = prepare_batches_jit(
        jax.random.PRNGKey(0), d, args.batch_size, num_atoms=pad,
        data_keys=KEYS, include_id=True,
    )
    covered = len(batches) * args.batch_size
    assert covered >= n_tot, f"batcher covers {covered} < {n_tot} real points"
    print(f"{len(batches)} batches of {args.batch_size} covering all {n_tot} real points "
          f"(+{n_pad - n_tot} padding)")

    KW = dict(mm_switch_on=args.mm_switch_on, mm_switch_width=args.mm_switch_width,
              ml_switch_width=args.ml_switch_width)
    fwd = jax.jit(lambda b: hybrid_forward(model.apply, params, b, args.batch_size,
                                           sig, eps, **KW))
    wall_fn = jax.jit(jax.vmap(inter_monomer_wall_energy))

    E = np.full(n_pad, np.nan)
    W = np.full(n_pad, np.nan)
    for bi, b in enumerate(batches):
        out = fwd(b)
        e = np.asarray(out["energy"]).reshape(args.batch_size)
        w = np.asarray(wall_fn(b["R"].reshape(args.batch_size, pad, 3), b["mol_id"]))
        ids = np.asarray(b["id"])
        E[ids] = e
        W[ids] = w
        if bi % 50 == 0:
            print(f"  batch {bi}/{len(batches)}", flush=True)

    # --- per-ray minima ----------------------------------------------------
    rows = []
    n_bad = n_ok = n_skip = 0
    for ray in range(n_rays):
        sel = np.where(ray_of == ray)[0]
        order = np.argsort(ir_of[sel])
        sel = sel[order]
        e = E[sel]
        w = W[sel]
        dmin = dmin_of[sel]
        if np.isnan(e).any():
            n_skip += 1
            continue
        e = e - e[-1]
        # Clash points (atom overlap) invent deep wells — exclude from minima.
        # Find minima on the contact-ok subsequence so neighbors of a clash
        # do not become artificial turning points.
        contact_ok = dmin >= min_contact
        idx_ok = np.flatnonzero(contact_ok)
        if idx_ok.size >= 3:
            mins = [int(idx_ok[i]) for i in find_minima(e[idx_ok], args.prominence)]
        else:
            mins = []
        # A minimum where the wall is live is geometry, not a model defect:
        # report it separately or the count is dominated by close contacts.
        wall_live = [i for i in mins if w[i] > 1e-3]
        clean = [i for i in mins if w[i] <= 1e-3]
        if mins:
            i_deep = int(mins[int(np.argmin(e[mins]))])
            e_min_kcal = float(e[i_deep] * EV_TO_KCAL)
            r_at_min = float(rs[i_deep])
            dmin_at_min = float(dmin[i_deep])
        else:
            e_min_kcal = 0.0
            r_at_min = float("nan")
            dmin_at_min = float("nan")
        rows.append({
            "ray": ray, "direction": int(ray // len(quats)), "orientation": int(ray % len(quats)),
            "n_min": len(mins), "n_min_ml": len(clean), "n_min_wall": len(wall_live),
            "e_min_kcal": e_min_kcal,
            "r_at_min": r_at_min,
            "dmin_at_min": dmin_at_min,
            "n_contact_ok": int(contact_ok.sum()),
            "r_safe_min": float(rs[contact_ok][0]) if contact_ok.any() else float("nan"),
        })
        if len(clean) > 1:
            n_bad += 1
        else:
            n_ok += 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rays.csv").open("w") as fh:
        keys = list(rows[0].keys())
        fh.write(",".join(keys) + "\n")
        for r_ in rows:
            fh.write(",".join(str(r_[k]) for k in keys) + "\n")

    frac = n_bad / max(n_ok + n_bad, 1)
    contact_rows = [r_ for r_ in rows if np.isfinite(r_["r_at_min"])]
    summary = {
        "resid": args.resid, "checkpoint": args.checkpoint, "use_ema": args.use_ema,
        "mm_switch_on": args.mm_switch_on, "ml_switch_width": args.ml_switch_width,
        "min_contact_A": min_contact,
        "n_rays": n_rays, "n_evaluated": n_ok + n_bad, "n_dropped": n_skip,
        "n_rays_spurious": n_bad, "frac_rays_spurious": frac,
        "n_rays_with_contact_ok_min": len(contact_rows),
        "mean_min_kcal": float(np.mean([r_["e_min_kcal"] for r_ in contact_rows])) if contact_rows else None,
        "deepest_kcal": float(np.min([r_["e_min_kcal"] for r_ in contact_rows])) if contact_rows else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.resid} @ mm_switch_on={args.mm_switch_on:g} ===")
    print(f"  rays evaluated : {n_ok + n_bad} ({n_skip} dropped by the batcher)")
    print(f"  SPURIOUS (>1 ML minimum) : {n_bad}  ({frac * 100:.1f}% of rays)")
    print(f"  deepest well  : {summary['deepest_kcal']:.2f} kcal/mol")
    print(f"  mean well     : {summary['mean_min_kcal']:.2f} kcal/mol")
    print(f"  -> {out_dir}/rays.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
