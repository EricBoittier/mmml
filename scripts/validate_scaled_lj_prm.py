#!/usr/bin/env python
"""Validate that a scaled CGenFF prm deploys the SAME LJ as the JAX scale path.

`periodic_external` gets its VDW from CHARMM, so trained LJ scales can only
reach it through the parameter file. This script checks that the rewritten prm
reproduces the JAX path's effective LJ, comparing two *independent* routes on
real geometries:

  route A (reference)  master tables from the base prm, then
                       apply_mm_lj_scales(master, sigma_scale, epsilon_scale)
  route B (deployed)   master tables parsed straight out of the scaled prm,
                       no scales applied

Route B goes through the CGenFF text parser; route A goes through the JAX
scaling helper. They share no code, so agreement is evidence, not tautology.
Energies are full Lennard-Jones sums over random dimer geometries with
Lorentz-Berthelot combining, i.e. the whole expression CHARMM will evaluate.

    uv run python scripts/validate_scaled_lj_prm.py --sidecar <hybrid_mm.json>

With no --sidecar a synthetic one is generated in-bounds, which still validates
the machinery end to end.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

KCAL = 1.0  # CGenFF epsilons are already kcal/mol

# dataviz categorical slots 1/2/3 (validated: CVD + normal-vision pass)
C_REF, C_DEP, C_BASE = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def lj_energy(pos_a, pos_b, sig_a, sig_b, eps_a, eps_b):
    """Total intermolecular LJ energy, Lorentz-Berthelot, conventional sigma."""
    d = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=-1)
    sig = 0.5 * (sig_a[:, None] + sig_b[None, :])
    eps = np.sqrt(np.abs(eps_a[:, None] * eps_b[None, :]))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x6 = (sig / d) ** 6
        e = 4.0 * eps * (x6 * x6 - x6)
    return float(np.nansum(e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, default=None)
    ap.add_argument("--n-configs", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/validation/scaled_lj_prm.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # apply_mm_lj_scales goes through jnp, which is float32 by default. Left
    # at float32 the two routes disagree by ~4e-4 relative purely from the
    # multiply, and r^-12 amplifies it further -- that is the harness losing
    # precision, not the prm rewrite being inexact.
    import jax
    jax.config.update("jax_enable_x64", True)

    from mmml.data.cgenff_dataset import (
        DEF_EXTRA_TOPPAR, DEF_PRM_PATH, DEF_RTF_PATH, load_reference,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        write_scaled_cgenff_prm,
    )
    from mmml.models.mm_lj_scales import apply_mm_lj_scales

    ref = load_reference(str(DEF_PRM_PATH), str(DEF_RTF_PATH))
    names = [""] * len(ref.nb_map)
    for name, idx in ref.nb_map.items():
        names[int(idx)] = str(name)

    rng = np.random.default_rng(args.seed)
    tmp = Path(tempfile.mkdtemp(prefix="mmml-lj-validate-"))

    if args.sidecar is not None:
        sidecar = args.sidecar
        payload = json.loads(sidecar.read_text())
        sig_scale = np.ones(len(names))
        eps_scale = np.ones(len(names))
        idx = {n: i for i, n in enumerate(names)}
        for n, s, e in zip(payload["cgenff_type_names"],
                           payload["mm_lj_sigma_scale"],
                           payload["mm_lj_epsilon_scale"]):
            if n in idx:
                sig_scale[idx[n]], eps_scale[idx[n]] = float(s), float(e)
        label = sidecar.name
    else:
        sig_scale = rng.uniform(0.95, 1.05, size=len(names))
        eps_scale = rng.uniform(0.25, 4.0, size=len(names))
        sig_scale[names.index("DEFAULT")] = 1.0
        eps_scale[names.index("DEFAULT")] = 1.0
        sidecar = tmp / "hybrid_mm.json"
        sidecar.write_text(json.dumps({
            "learn_mm_lj_scales": True,
            "cgenff_type_names": names,
            "mm_lj_sigma_scale": sig_scale.tolist(),
            "mm_lj_epsilon_scale": eps_scale.tolist(),
            "mm_lj_sigma_scale_bounds": [0.95, 1.05],
            "mm_lj_epsilon_scale_bounds": [0.25, 4.0],
        }))
        label = "synthetic (in-bounds)"

    out_dir = tmp / "scaled"
    stats = write_scaled_cgenff_prm(sidecar, out_dir, overwrite=True)
    n_scaled = sum(len(s.scaled) for s in stats.values())
    print(f"scaled {n_scaled} types across {len(stats)} parameter files")

    ref2 = load_reference(
        str(out_dir / DEF_PRM_PATH.name), str(DEF_RTF_PATH),
        extra_toppar=tuple(out_dir / Path(p).name for p in DEF_EXTRA_TOPPAR),
    )

    # route A: JAX helper on the base tables
    sigA, epsA = apply_mm_lj_scales(ref.sigmas, ref.epsilons, sig_scale, eps_scale)
    sigA, epsA = np.asarray(sigA), np.asarray(epsA)
    # route B: straight out of the scaled prm
    sigB = np.array([ref2.sigmas[ref2.nb_map[n]] for n in names])
    epsB = np.array([ref2.epsilons[ref2.nb_map[n]] for n in names])

    # Real types only (drop the zero-LJ sentinels).
    live = [i for i, n in enumerate(names)
            if n != "DEFAULT" and ref.sigmas[i] > 0 and ref.epsilons[i] > 0]

    e_ref, e_dep, e_base, seps = [], [], [], []
    for _ in range(args.n_configs):
        na, nb = rng.integers(2, 7), rng.integers(2, 7)
        ta = rng.choice(live, size=na)
        tb = rng.choice(live, size=nb)
        sep = rng.uniform(3.0, 9.0)
        # Compact monomers plus a real separation: without a floor on the
        # closest contact, r^-12 produces ~1e15 kcal/mol and the absolute
        # residual stops meaning anything.
        pa = rng.normal(0, 0.7, size=(na, 3))
        pb = rng.normal(0, 0.7, size=(nb, 3)) + np.array([sep, 0, 0])
        dmin = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min()
        if dmin < 2.5:
            pb = pb + np.array([2.5 - dmin, 0.0, 0.0])
        e_ref.append(lj_energy(pa, pb, sigA[ta], sigA[tb], epsA[ta], epsA[tb]))
        e_dep.append(lj_energy(pa, pb, sigB[ta], sigB[tb], epsB[ta], epsB[tb]))
        e_base.append(lj_energy(pa, pb, ref.sigmas[ta], ref.sigmas[tb],
                                ref.epsilons[ta], ref.epsilons[tb]))
        seps.append(sep)

    e_ref = np.array(e_ref); e_dep = np.array(e_dep); e_base = np.array(e_base)
    resid = e_dep - e_ref
    scale = np.maximum(np.abs(e_ref), 1e-12)
    max_abs = float(np.max(np.abs(resid)))
    max_rel = float(np.max(np.abs(resid) / scale))
    effect = float(np.max(np.abs(e_ref - e_base)))

    print(f"configs                 : {len(e_ref)}")
    print(f"max |E_deployed-E_ref|  : {max_abs:.3e} kcal/mol")
    print(f"max relative deviation  : {max_rel:.3e}")
    print(f"max |scaled - unscaled| : {effect:.3f} kcal/mol  (effect size)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    ax = axes[0]
    lo, hi = float(min(e_ref.min(), e_dep.min())), float(max(e_ref.max(), e_dep.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=1.0,
            ls="--", zorder=1, label="y = x")
    ax.scatter(e_ref, e_dep, s=26, color=C_REF, alpha=0.75, linewidths=0, zorder=2)
    ax.set_xlabel("route A — JAX apply_mm_lj_scales  (kcal/mol)", fontsize=9, color=INK2)
    ax.set_ylabel("route B — parsed from scaled prm  (kcal/mol)", fontsize=9, color=INK2)
    ax.set_title("Deployed LJ matches the trained LJ", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

    ax = axes[1]
    ax.axhline(0, color=INK2, lw=1.0, ls="--", zorder=1)
    ax.scatter(np.abs(e_ref), np.abs(resid), s=26, color=C_DEP,
               alpha=0.8, linewidths=0, zorder=2)
    ax.set_yscale("log"); ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("|E| (kcal/mol)", fontsize=9, color=INK2)
    ax.set_ylabel("|route B − route A| (kcal/mol)", fontsize=9, color=INK2)
    ax.set_title(f"Residual — max {max_abs:.1e} kcal/mol", fontsize=11,
                 color=INK, loc="left")
    ax.annotate("float64 round-off", xy=(0.03, 0.90), xycoords="axes fraction",
                fontsize=9, color=C_DEP)

    ax = axes[2]
    # A pair potential, not a scatter of random compositions: panel 3 has to be
    # readable, and E-vs-separation over random type draws is just noise.
    # Pick the two live types whose epsilon scale moved most.
    moved = sorted(live, key=lambda i: -abs(eps_scale[i] - 1.0))[:2]
    r = np.linspace(2.6, 8.0, 400)
    styles = [("-", C_REF), ("-", C_DEP)]
    for k, t in enumerate(moved):
        s_b, e_b = ref.sigmas[t], ref.epsilons[t]
        s_a, e_a = sigA[t], epsA[t]
        base = 4 * e_b * ((s_b / r) ** 12 - (s_b / r) ** 6)
        trained = 4 * e_a * ((s_a / r) ** 12 - (s_a / r) ** 6)
        ls, col = styles[k]
        ax.plot(r, base, lw=1.6, color=col, ls="--", alpha=0.55, zorder=2)
        ax.plot(r, trained, lw=2.0, color=col, ls=ls, zorder=3)
        j = int(np.argmin(trained))
        ax.annotate(f"{names[t]}  (x{eps_scale[t]:.2f} eps)",
                    xy=(r[j], trained[j]), xytext=(6, -4 + 12 * k),
                    textcoords="offset points", fontsize=9, color=col,
                    weight="bold")
    ax.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax.set_ylim(min(-1.2, 1.3 * min(
        float((4 * epsA[t] * ((sigA[t] / r) ** 12 - (sigA[t] / r) ** 6)).min())
        for t in moved)), 1.0)
    ax.set_xlabel("pair separation (Å)", fontsize=9, color=INK2)
    ax.set_ylabel("LJ pair energy (kcal/mol)", fontsize=9, color=INK2)
    ax.set_title(f"…and it is not a no-op — max Δ {effect:.1f} kcal/mol",
                 fontsize=11, color=INK, loc="left")
    ax.plot([], [], ls="--", color=INK2, lw=1.6, alpha=0.6, label="unscaled CGenFF")
    ax.plot([], [], ls="-", color=INK2, lw=2.0, label="trained scales")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    fig.suptitle(
        f"periodic_external can now consume trained LJ scales  —  {label}  "
        f"({n_scaled} types, {len(e_ref)} configs)",
        fontsize=12, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")

    # 1e-7 relative: comfortably below anything physically meaningful, and
    # ~600x tighter than the 6e-5 that the original 6-decimal prm write
    # introduced. Not tighter, because summing r^-12 terms in float64 leaves
    # ~1e-9 relative round-off that no amount of care in the writer removes.
    ok = max_rel < 1e-7 and effect > 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
