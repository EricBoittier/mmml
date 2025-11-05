#!/usr/bin/env python3
"""
Generate reports from CV results CSV.

Takes a CSV file with CV results and generates summary tables and plots.

Usage:
    python generate_reports.py --data results.csv --report-dir reports
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from reporting import (
    build_summary_tables,
    style_table_html,
    to_latex_table,
    save_html_table,
    save_latex_table,
    plot_metric_by_model,
    plot_heatmap_by_group,
)


def main():
    parser = argparse.ArgumentParser(description="Generate reports from CV results")
    parser.add_argument("--data", type=str, required=True, help="Path to CV results CSV")
    parser.add_argument("--report-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--metric", type=str, default="val_rmse_mean", help="Metric to report")
    args = parser.parse_args()
    
    print(f"Loading results from {args.data}...")
    results = pd.read_csv(args.data)
    print(f"  Loaded {len(results)} rows")
    
    required_cols = {"model", "scheme", "level", "atom_index", args.metric}
    missing = required_cols - set(results.columns)
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
    
    # Create report directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating reports in {report_dir}...")
    
    metric = args.metric
    
    # Build summary tables
    summary_by_model, best_per_group, labeled = build_summary_tables(results, metric=metric)
    
    print(f"\nSummary by model ({metric}):")
    print(summary_by_model.to_string(index=False))
    
    # HTML tables
    html_summary = style_table_html(
        summary_by_model.rename(columns={"mean_metric": f"{metric}_mean", "std_metric": f"{metric}_std"}),
        metric=f"{metric}_mean",
        cmap="viridis",
        caption=f"Model summary ({metric})"
    )
    save_html_table(html_summary, str(report_dir / "summary_by_model.html"))
    print(f"  ✓ Saved {report_dir / 'summary_by_model.html'}")
    
    html_best = style_table_html(
        best_per_group[["model", "scheme", "level", "atom_index", metric, "n_folds", "n_samples"]],
        metric=metric,
        cmap="magma",
        caption=f"Best model per group ({metric})"
    )
    save_html_table(html_best, str(report_dir / "best_per_group.html"))
    print(f"  ✓ Saved {report_dir / 'best_per_group.html'}")
    
    # LaTeX tables
    latex_summary = to_latex_table(
        summary_by_model.rename(columns={"mean_metric": f"{metric}_mean", "std_metric": f"{metric}_std"}),
        metric=f"{metric}_mean",
        caption=f"Model summary ({metric})",
        label="tab:model_summary"
    )
    save_latex_table(latex_summary, str(report_dir / "summary_by_model.tex"))
    print(f"  ✓ Saved {report_dir / 'summary_by_model.tex'}")
    
    latex_best = to_latex_table(
        best_per_group[["model", "scheme", "level", "atom_index", metric, "n_folds", "n_samples"]],
        metric=metric,
        caption=f"Best model per group ({metric})",
        label="tab:best_per_group"
    )
    save_latex_table(latex_best, str(report_dir / "best_per_group.tex"))
    print(f"  ✓ Saved {report_dir / 'best_per_group.tex'}")
    
    # Plots
    plot_metric_by_model(
        results,
        metric=metric,
        cmap="tab10",
        figsize=(8, 5),
        savepath=str(report_dir / "metric_by_model.png")
    )
    print(f"  ✓ Saved {report_dir / 'metric_by_model.png'}")
    
    plot_heatmap_by_group(
        results,
        metric=metric,
        cmap="magma",
        figsize=(11, 6),
        savepath=str(report_dir / "heatmap_by_group.png")
    )
    print(f"  ✓ Saved {report_dir / 'heatmap_by_group.png'}")
    
    print(f"\n✓ All reports generated in {report_dir}/")
    print(f"\nFiles created:")
    for f in sorted(report_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

