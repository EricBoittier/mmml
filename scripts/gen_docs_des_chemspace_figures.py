#!/usr/bin/env python3
"""Generate the figures and LJ tables for docs/des-so3lr-dimers.md.

Input is the JSON written by ``scripts/scan_des_chemical_space.py`` — a real
streaming pass over the SO3LR-format DES dimer set that ran the production
CGenFF assignment (:mod:`mmml.data.cgenff_dataset`) on a strided sample. Nothing
here is illustrative; every count comes from that scan.

Outputs -> docs/images/des-so3lr-dimers/:

* ``chemical_space.png``  — what is in the dataset, and which parts are typeable
* ``lj_coverage.png``     — which CGenFF LJ (sigma, epsilon) entries are reachable
* ``lj_types.md``         — markdown table of the reachable LJ parameters
* ``resi_coverage.md``    — markdown table of the covered CGenFF residues
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmml.data.cgenff_dataset import load_reference
from mmml.utils.plotting.styles import apply_plot_style

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images" / "des-so3lr-dimers"

# Okabe-Ito slots from the house ICML palette. Validated all-pairs (worst CVD
# dE 11.0, worst normal-vision dE 18.7) so the covered/not distinction survives
# colour-vision deficiency; every panel also carries a legend or direct labels.
C_OK = "#0072B2"     # typeable by the hybrid ML/MM CGenFF path
C_NO = "#D55E00"     # dropped
C_NEUTRAL = "#6E6E6E"


def _fmt(formula: str) -> str:
    """Subscript digits so C2H6O reads as a formula on the axis."""
    subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return formula.translate(subs)


def figure_chemical_space(scan: dict, out: Path) -> None:
    cg = scan["cgenff"]
    mono_ok = dict(cg["monomer_ok"])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    (ax_mono, ax_heat), (ax_elem, ax_size) = axes

    # (a) Most common monomers, coloured by whether hybrid ML/MM can type them.
    # Dots, not bars: water outruns the next monomer 14x, so the axis has to be
    # logarithmic -- and a bar on a log axis no longer encodes magnitude by length.
    top = scan["monomers"][:26][::-1]
    names = [t[0] for t in top]
    vals = np.array([t[1] for t in top], dtype=float)
    cols = [C_OK if mono_ok.get(n, 0) > 0 else C_NO for n in names]
    y = np.arange(len(names))
    ax_mono.hlines(y, vals.min() * 0.75, vals, color="0.85", lw=1.0, zorder=1)
    ax_mono.scatter(vals, y, s=64, c=cols, zorder=3,
                    edgecolors="white", linewidths=0.8)
    ax_mono.set_yticks(y, [_fmt(n) for n in names], fontsize=8.5)
    ax_mono.set_xscale("log")
    ax_mono.set_xlim(vals.min() * 0.72, vals.max() * 1.7)
    ax_mono.set_ylim(-0.9, len(names) - 0.1)
    ax_mono.set_xlabel("occurrences as a monomer (log)")
    ax_mono.set_title("(a) Most common monomers", loc="left", fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=C_OK),
               plt.Line2D([], [], marker="o", ls="", ms=8, color=C_NO)]
    ax_mono.legend(handles, ["CGenFF-typeable", "no template / topology"],
                   loc="lower right", fontsize=8.5, framealpha=0.95)
    ax_mono.grid(axis="x", alpha=0.25)
    ax_mono.set_axisbelow(True)

    # (b) Which monomer pairs actually co-occur.
    n_heat = 20
    heat_names = [m for m, _ in scan["monomers"][:n_heat]]
    pos = {m: i for i, m in enumerate(heat_names)}
    grid = np.zeros((n_heat, n_heat))
    for pair, c in scan["pairs"]:
        a, _, b = pair.partition(" + ")
        if a in pos and b in pos:
            grid[pos[a], pos[b]] += c
            if a != b:
                grid[pos[b], pos[a]] += c
    masked = np.ma.masked_where(grid == 0, grid)
    # Single hue, light -> dark: magnitude, not identity. Grey = pair never sampled.
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#E8E8E6")
    im = ax_heat.imshow(masked, cmap=cmap,
                        norm=matplotlib.colors.LogNorm(vmin=max(grid[grid > 0].min(), 1),
                                                       vmax=grid.max()))
    ax_heat.set_xticks(range(n_heat), [_fmt(m) for m in heat_names],
                       rotation=90, fontsize=7.5)
    ax_heat.set_yticks(range(n_heat), [_fmt(m) for m in heat_names], fontsize=7.5)
    ax_heat.set_title(f"(b) Pair co-occurrence, top {n_heat}", loc="left",
                      fontweight="bold")
    ax_heat.annotate("grey = pair never sampled", xy=(0, 1.015),
                     xycoords="axes fraction", fontsize=8, color="0.35")
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cb.set_label("frames (log)", fontsize=9)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # (c) Element inventory (frames containing the element).
    els = scan["elements"][::-1]
    ev = np.array([c for _, c in els], dtype=float)
    ey = np.arange(len(els))
    ax_elem.hlines(ey, ev.min() * 0.75, ev, color="0.85", lw=1.0, zorder=1)
    ax_elem.scatter(ev, ey, s=58, color=C_NEUTRAL, zorder=3,
                    edgecolors="white", linewidths=0.8)
    ax_elem.set_yticks(ey, [e for e, _ in els], fontsize=8.5)
    ax_elem.set_xscale("log")
    ax_elem.set_xlim(ev.min() * 0.72, ev.max() * 1.7)
    ax_elem.set_ylim(-0.9, len(els) - 0.1)
    ax_elem.set_xlabel("frames containing the element (log)")
    ax_elem.set_title("(c) Element inventory", loc="left", fontweight="bold")
    ax_elem.grid(axis="x", alpha=0.25)
    ax_elem.set_axisbelow(True)

    # (d) Frame sizes -- the padding width any NPZ conversion must use.
    sizes = np.array([k for k, _ in scan["natoms_hist"]])
    counts = np.array([v for _, v in scan["natoms_hist"]], dtype=float)
    ax_size.bar(sizes, counts, color=C_NEUTRAL, width=0.82)
    ax_size.set_xlabel("atoms per dimer frame")
    ax_size.set_ylabel("frames")
    ax_size.set_title("(d) Frame size distribution", loc="left", fontweight="bold")
    ax_size.grid(axis="y", alpha=0.25)
    ax_size.set_axisbelow(True)
    ax_size.annotate(f"max {sizes.max()} atoms", xy=(sizes.max(), counts[-1]),
                     xytext=(-8, 28), textcoords="offset points", ha="right",
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))

    frac = 100 * cg["n_typed"] / max(cg["frames_attempted"], 1)
    fig.suptitle(
        f"SO3LR / DES dimers — chemical space  ({scan['frames_scanned']:,} frames, "
        f"{len(scan['monomers']):,} monomers, {len(scan['pairs']):,} pairs; "
        f"{frac:.0f}% hybrid-MM typeable)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def lj_coverage_table(scan: dict) -> tuple[list[dict], Counter, dict]:
    """Expand covered RESI templates into the CGenFF LJ types they reach."""
    ref = load_reference()
    idx_to_name = {v: k for k, v in ref.nb_map.items()}

    resi_frames: Counter = Counter()
    for pair, n in scan["cgenff"]["resi_pairs"]:
        for res in pair.split(" + "):
            resi_frames[res] += n

    type_frames: Counter = Counter()
    type_resis = defaultdict(set)
    resi_types: dict[str, list[str]] = {}
    for resi, n in resi_frames.items():
        tmpl = ref.residues[resi]
        types = sorted({idx_to_name[int(i)] for i in tmpl["type_indices"]})
        resi_types[resi] = types
        for t in types:
            type_frames[t] += n
            type_resis[t].add(resi)

    rows = []
    for t, n in type_frames.most_common():
        i = ref.nb_map[t]
        rows.append({
            "type": t,
            "sigma": float(ref.sigmas[i]),
            "epsilon": float(ref.epsilons[i]),
            "frames": int(n),
            "resis": sorted(type_resis[t]),
        })
    return rows, resi_frames, {"ref": ref, "resi_types": resi_types,
                               "idx_to_name": idx_to_name}


def figure_lj_coverage(scan: dict, rows: list[dict], extra: dict, out: Path) -> None:
    ref = extra["ref"]
    idx_to_name = extra["idx_to_name"]
    covered = {r["type"]: r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax_sc, ax_bar = axes

    # (a) The LJ parameter plane: which (sigma, epsilon) entries get a gradient.
    all_types = [idx_to_name[i] for i in range(len(ref.sigmas))]
    un_s, un_e, cv_s, cv_e, cv_n = [], [], [], [], []
    for i, t in enumerate(all_types):
        s, e = float(ref.sigmas[i]), float(ref.epsilons[i])
        if e <= 0:  # LPH lone pair: zero LJ by design, not plottable on a log axis
            continue
        if t in covered:
            cv_s.append(s); cv_e.append(e); cv_n.append(covered[t]["frames"])
        else:
            un_s.append(s); un_e.append(e)
    ax_sc.scatter(un_s, un_e, s=26, color=C_NEUTRAL, alpha=0.45,
                  label=f"unreachable ({len(un_s)})", zorder=2)
    sizes = 24 + 300 * (np.array(cv_n) / max(cv_n))
    ax_sc.scatter(cv_s, cv_e, s=sizes, color=C_OK, alpha=0.85,
                  edgecolors="white", linewidths=1.2,
                  label=f"reachable from DES ({len(cv_s)})", zorder=3)
    ax_sc.set_yscale("log")
    ax_sc.set_xlabel(r"$\sigma$  ($\mathrm{\AA}$)")
    ax_sc.set_ylabel(r"$\epsilon$  (kcal mol$^{-1}$, log)")
    ax_sc.set_title("(a) CGenFF LJ parameter plane\n"
                    "marker area $\\propto$ frames touching the type",
                    loc="left", fontweight="bold")
    ax_sc.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax_sc.grid(alpha=0.25)
    ax_sc.set_axisbelow(True)
    # Direct-label the handful of types the fit sees most -- identity must not
    # rest on colour alone. Distinct CGenFF types can share a (sigma, epsilon)
    # cell exactly (HT and HGP1 are both 0.4000 / 0.04600), so merge labels by
    # position rather than stacking two strings on one dot.
    at_point: dict[tuple[float, float], list[str]] = defaultdict(list)
    for r in rows[:8]:
        if r["epsilon"] > 0:
            at_point[(round(r["sigma"], 4), round(r["epsilon"], 5))].append(r["type"])
    for (s, e), names in at_point.items():
        ax_sc.annotate("/".join(names), (s, e), textcoords="offset points",
                       xytext=(9, 6), fontsize=8.5, color="0.15", fontweight="bold")

    # (b) Which types the loss actually leans on.
    top = rows[:24][::-1]
    y = np.arange(len(top))
    ax_bar.barh(y, [r["frames"] for r in top], color=C_OK, height=0.74)
    ax_bar.set_yticks(y, [r["type"] for r in top], fontsize=8.5)
    ax_bar.set_xlabel("frames touching the type (sampled)")
    ax_bar.set_title("(b) Most-exercised LJ types", loc="left", fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.set_axisbelow(True)
    for yi, r in zip(y, top):
        ax_bar.annotate(f"  σ {r['sigma']:.2f} / ε {r['epsilon']:.3f}",
                        (r["frames"], yi), va="center", fontsize=7.2, color="0.3")
    ax_bar.set_xlim(0, max(r["frames"] for r in top) * 1.42)

    fig.suptitle(
        f"Hybrid ML/MM LJ coverage — {len(rows)} of {len(ref.sigmas)} CGenFF types "
        f"reachable from the DES dimer set",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def write_tables(scan: dict, rows: list[dict], resi_frames: Counter,
                 extra: dict, outdir: Path) -> None:
    resi_types = extra["resi_types"]
    total = sum(r["frames"] for r in rows)

    lines = ["| CGenFF type | σ (Å) | ε (kcal/mol) | sampled frames | residues |",
             "|---|---:|---:|---:|---|"]
    for r in rows:
        resis = ", ".join(r["resis"][:6]) + (" …" if len(r["resis"]) > 6 else "")
        lines.append(f"| `{r['type']}` | {r['sigma']:.4f} | {r['epsilon']:.5f} "
                     f"| {r['frames']:,} | {resis} |")
    lines.append("")
    lines.append(f"*{len(rows)} types; {total:,} type-frame incidences in the "
                 f"CGenFF sample.*")
    (outdir / "lj_types.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {outdir / 'lj_types.md'}")

    lines = ["| RESI | sampled frames | atoms | LJ types |", "|---|---:|---:|---|"]
    for resi, n in resi_frames.most_common():
        types = resi_types[resi]
        lines.append(f"| `{resi}` | {n:,} | {len(extra['ref'].residues[resi]['atoms'])} "
                     f"| {', '.join(f'`{t}`' for t in types)} |")
    (outdir / "resi_coverage.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {outdir / 'resi_coverage.md'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_json", type=Path,
                    help="JSON from scripts/scan_des_chemical_space.py")
    ap.add_argument("--outdir", type=Path, default=OUT)
    a = ap.parse_args(argv)

    apply_plot_style("icml")
    scan = json.loads(a.scan_json.read_text())
    a.outdir.mkdir(parents=True, exist_ok=True)

    figure_chemical_space(scan, a.outdir / "chemical_space.png")
    rows, resi_frames, extra = lj_coverage_table(scan)
    figure_lj_coverage(scan, rows, extra, a.outdir / "lj_coverage.png")
    write_tables(scan, rows, resi_frames, extra, a.outdir)

    cg = scan["cgenff"]
    print(f"\nsummary: {scan['frames_scanned']:,} frames, "
          f"{len(scan['monomers']):,} monomers, {len(scan['pairs']):,} pairs")
    print(f"         {cg['n_typed']}/{cg['frames_attempted']} typeable "
          f"({100 * cg['n_typed'] / cg['frames_attempted']:.1f}%), "
          f"{len(cg['resi_pairs'])} RESI pairs, {len(resi_frames)} residues, "
          f"{len(rows)} LJ types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
