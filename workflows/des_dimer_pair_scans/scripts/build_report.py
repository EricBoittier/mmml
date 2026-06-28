#!/usr/bin/env python3
"""Build HTML report with Matplotlib figures for all dimer pairs."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_pair_scan import plot_pair_scan_npz, plot_pending_pair  # noqa: E402
from scan_lib import iter_pairs, load_config, output_dir, workflow_root  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=workflow_root() / "config.yaml")
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=workflow_root() / "results" / "figures",
        help="Directory for per-pair PNG figures",
    )
    p.add_argument(
        "--output-html",
        type=Path,
        default=workflow_root() / "results" / "report.html",
        help="Combined HTML report path",
    )
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def _figure_name(pair_tag: str) -> str:
    return f"{pair_tag}.png"


def build_report(
    *,
    cfg_path: Path,
    figures_dir: Path,
    output_html: Path,
    dpi: int = 120,
) -> dict[str, int]:
    cfg = load_config(cfg_path)
    figures_dir = figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_plotted = 0
    n_pending = 0
    cards: list[str] = []

    for pair in iter_pairs(cfg):
        png = figures_dir / _figure_name(pair.tag)
        npz = output_dir(cfg, pair) / "scan_2d.npz"
        rel_img = png.relative_to(output_html.parent.resolve())

        stats: dict[str, str] = {}
        if npz.is_file():
            import numpy as np

            data = np.load(npz, allow_pickle=False)
            for key, label in (
                ("charmm_ENER_kcal", "CHARMM min"),
                ("xtb_energy_kcal", "xTB min"),
                ("orca_mp2_energy_kcal", "ORCA MP2 min"),
            ):
                if key in data:
                    arr = np.asarray(data[key], dtype=float)
                    if arr.size and not np.all(np.isnan(arr)):
                        stats[label] = f"{float(np.nanmin(arr)):.2f} kcal/mol"

        if plot_pair_scan_npz(npz, png, title=pair.label, dpi=dpi):
            n_plotted += 1
            badge = '<span class="badge ok">scanned</span>'
        else:
            plot_pending_pair(png, pair_tag=pair.tag, label=pair.label, dpi=dpi)
            n_pending += 1
            badge = '<span class="badge pending">pending</span>'

        stat_lines = "".join(
            f'<li><strong>{html.escape(k)}:</strong> {html.escape(v)}</li>'
            for k, v in stats.items()
        )
        cards.append(
            f"""
<section class="pair-card" id="{html.escape(pair.tag)}">
  <header>
    <h2>{html.escape(pair.label)}</h2>
    <p class="meta">{html.escape(pair.composition)} · {html.escape(pair.tag)} {badge}</p>
  </header>
  <figure>
    <a href="{html.escape(str(rel_img))}"><img src="{html.escape(str(rel_img))}" alt="{html.escape(pair.label)} scan" loading="lazy"/></a>
  </figure>
  <ul class="stats">{stat_lines or '<li>No energy data yet</li>'}</ul>
</section>
"""
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = n_plotted + n_pending
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>DES dimer pair 2D scans</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #fafafa; color: #222; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .subtitle {{ color: #555; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1.25rem; }}
    .pair-card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    .pair-card h2 {{ font-size: 1.05rem; margin: 0 0 0.25rem; }}
    .meta {{ font-size: 0.85rem; color: #666; margin: 0 0 0.75rem; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
    .stats {{ font-size: 0.8rem; margin: 0.75rem 0 0; padding-left: 1.1rem; }}
    .badge {{ font-size: 0.7rem; padding: 0.1rem 0.45rem; border-radius: 4px; margin-left: 0.35rem; }}
    .badge.ok {{ background: #e6f4ea; color: #137333; }}
    .badge.pending {{ background: #fef7e0; color: #b06000; }}
    nav.toc {{ columns: 3; font-size: 0.85rem; margin-bottom: 1.5rem; }}
    nav.toc a {{ text-decoration: none; color: #1967d2; }}
  </style>
</head>
<body>
  <h1>DES dimer pair 2D scans</h1>
  <p class="subtitle">Generated {ts} · {n_plotted}/{total} pairs with scan data · reference checkpoint: {html.escape(str(cfg.get('reference_checkpoint', '')))}</p>
  <nav class="toc">
    <ul>
      {''.join(f'<li><a href="#{html.escape(p.tag)}">{html.escape(p.label)}</a></li>' for p in iter_pairs(cfg))}
    </ul>
  </nav>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(doc, encoding="utf-8")

    manifest = {
        "timestamp": ts,
        "total_pairs": total,
        "plotted": n_plotted,
        "pending": n_pending,
        "figures_dir": str(figures_dir),
        "report_html": str(output_html),
    }
    (output_html.parent / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"plotted": n_plotted, "pending": n_pending, "total": total}


def main() -> int:
    args = _parse_args()
    stats = build_report(
        cfg_path=args.config,
        figures_dir=args.figures_dir,
        output_html=args.output_html,
        dpi=args.dpi,
    )
    print(
        f"Report: {args.output_html} "
        f"({stats['plotted']}/{stats['total']} pairs with scan NPZ, "
        f"{stats['pending']} pending placeholders)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
