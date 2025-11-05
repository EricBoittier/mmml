# ========================= reporting.py =========================
# Nice tables (Pandas + LaTeX) and plots (matplotlib) for your CV results
# ========================================================================
# model, scheme, level, atom_index, val_rmse_mean, val_mae_mean, val_r2_mean, train_* , n_samples, n_folds.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# -------------------------------
# Table utilities (Pandas + LaTeX)
# -------------------------------
def _fmt_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standard numeric formatting for metrics."""
    fmt = {
        "val_rmse_mean": "{:.4f}",
        "val_rmse_std": "{:.4f}",
        "val_mae_mean": "{:.4f}",
        "val_r2_mean": "{:.3f}",
        "train_rmse_mean": "{:.4f}",
        "train_mae_mean": "{:.4f}",
        "train_r2_mean": "{:.3f}",
    }
    for c, f in fmt.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_group_label(row):
    return f"{row['scheme']}-{row['level']}-idx{int(row['atom_index'])}"

def build_summary_tables(results_df: pd.DataFrame, metric: str = "val_rmse_mean"):
    """
    Returns:
      summary_by_model: mean±std of metric per model across groups
      best_per_group: rows for the best model per (scheme, level, atom_index)
      long_with_labels: results with an added 'group' string label
    """
    df = results_df.copy()
    df = _fmt_cols(df)
    df["group"] = df.apply(make_group_label, axis=1)

    # Summary per model
    g = df.groupby("model", as_index=False)
    summary = g.agg(
        mean_metric=(metric, "mean"),
        std_metric=(metric, "std"),
        groups=("group", "nunique"),
        rows=("model", "count"),
    ).sort_values("mean_metric")

    # Best per group
    best_idx = df.groupby("group")[metric].idxmin()
    best_per_group = df.loc[best_idx].sort_values(["scheme", "level", "atom_index"])

    return summary, best_per_group, df

def style_table_html(df: pd.DataFrame, metric: str = "val_rmse_mean",
                     cmap: str = "viridis", caption: str | None = None):
    """
    Returns a pandas Styler with formatting, bar/gradient on the chosen metric.
    """
    df = df.copy()
    df = _fmt_cols(df)

    fmt = {
        "val_rmse_mean": "{:.4f}",
        "val_rmse_std": "{:.4f}",
        "val_mae_mean": "{:.4f}",
        "val_r2_mean": "{:.3f}",
        "train_rmse_mean": "{:.4f}",
        "train_mae_mean": "{:.4f}",
        "train_r2_mean": "{:.3f}",
    }
    styler = (
        df.style
        .format(fmt)
        .background_gradient(subset=[metric], cmap=cmap)
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "600"), ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
            {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "110%"), ("font-weight", "600")]}
        ])
        .hide(axis="index")
    )
    if caption:
        styler = styler.set_caption(caption)
    return styler

def to_latex_table(df: pd.DataFrame, metric: str = "val_rmse_mean",
                   float_format="%.4f", column_format=None, caption=None, label=None):
    """
    Returns a LaTeX table string using pandas.to_latex with some sensible defaults.
    """
    df = df.copy()
    df = _fmt_cols(df)
    # Reorder common columns if available
    preferred = ["model", "scheme", "level", "atom_index", "val_rmse_mean", "val_rmse_std",
                 "val_mae_mean", "val_r2_mean", "n_folds", "n_samples"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    latex = df.to_latex(
        index=False,
        float_format=float_format.__mod__ if isinstance(float_format, str) else float_format,
        column_format=column_format,
        caption=caption,
        label=label,
        escape=True,  # keep safe
        longtable=False,
        bold_rows=False
    )
    return latex

def save_html_table(styler, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    styler.to_html(path)

def save_latex_table(latex_str: str, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

# -------------------------------
# Plotting utilities (matplotlib)
# -------------------------------
def plot_metric_by_model(results_df: pd.DataFrame, metric: str = "val_rmse_mean",
                         cmap: str = "tab10", figsize=(8,5), savepath: str | None = None):
    """
    Bar plot: mean ± std of metric per model.
    """
    df = results_df.copy()
    df = _fmt_cols(df)
    summary, _, _ = build_summary_tables(df, metric=metric)

    models = summary["model"].tolist()
    means = summary["mean_metric"].values
    stds  = summary["std_metric"].fillna(0.0).values

    colors = cm.get_cmap(cmap)(np.linspace(0, 1, len(models)))

    plt.figure(figsize=figsize)
    bars = plt.bar(models, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    plt.ylabel(metric)
    plt.title(f"{metric} by model (mean ± std across groups)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    return bars

def plot_heatmap_by_group(results_df: pd.DataFrame, metric: str = "val_rmse_mean",
                          cmap: str = "magma", figsize=(10, 6), savepath: str | None = None):
    """
    Heatmap of metric for (rows=models, cols=groups).
    """
    df = results_df.copy()
    df = _fmt_cols(df)
    df["group"] = df.apply(make_group_label, axis=1)

    pivot = df.pivot_table(index="model", columns="group", values=metric, aggfunc="mean")
    plt.figure(figsize=figsize)
    im = plt.imshow(pivot.values, aspect="auto", cmap=cmap)
    plt.colorbar(im, label=metric)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.title(f"{metric} heatmap (model × group)")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    return pivot

def plot_parity_by_model(results_df: pd.DataFrame, y_true: np.ndarray, y_pred: dict,
                         cmap: str = "viridis", figsize=(6,6), savepath: str | None = None):
    """
    Generic parity plot: provide y_true and a dict {model_name: y_pred_array} for a *single* fold/split.
    This is optional scaffolding if you want to capture per-fold predictions externally.
    """
    plt.figure(figsize=figsize)
    xs = np.array(y_true)
    vmin, vmax = np.min(xs), np.max(xs)
    grid = np.linspace(vmin, vmax, 100)
    plt.plot(grid, grid, linestyle="--", linewidth=1, color="black", alpha=0.7, label="y=x")

    cmx = cm.get_cmap(cmap)
    names = list(y_pred.keys())
    colors = cmx(np.linspace(0, 1, len(names)))

    for color, name in zip(colors, names):
        plt.scatter(xs, np.array(y_pred[name]), s=20, alpha=0.8, label=name, color=color, edgecolors="black", linewidths=0.3)

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.legend(frameon=False, fontsize=9)
    plt.title("Parity plot")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
# ======================= end reporting.py =======================


# if __name__ == "__main__":

#     from reporting import (
#         build_summary_tables, style_table_html, to_latex_table,
#         save_html_table, save_latex_table,
#         plot_metric_by_model, plot_heatmap_by_group
#     )

#     metric = "val_rmse_mean"  # change to "val_mae_mean" or "val_r2_mean" as you like

#     # --------- Tables ---------
#     summary_by_model, best_per_group, out_labeled = build_summary_tables(out, metric=metric)

#     # HTML tables
#     html_summary = style_table_html(summary_by_model.rename(columns={
#         "mean_metric": f"{metric}_mean", "std_metric": f"{metric}_std"
#     }), metric=f"{metric}_mean", cmap="viridis", caption=f"Model summary ({metric})")

#     save_html_table(html_summary, "reports/summary_by_model.html")

#     html_best = style_table_html(
#         best_per_group[["model", "scheme", "level", "atom_index", metric, "n_folds", "n_samples"]],
#         metric=metric, cmap="magma", caption=f"Best model per group ({metric})"
#     )
#     save_html_table(html_best, "reports/best_per_group.html")

#     # LaTeX tables
#     latex_summary = to_latex_table(
#         summary_by_model.rename(columns={"mean_metric": f"{metric}_mean", "std_metric": f"{metric}_std"}),
#         metric=f"{metric}_mean", caption=f"Model summary ({metric})", label="tab:model_summary"
#     )
#     save_latex_table(latex_summary, "reports/summary_by_model.tex")

#     latex_best = to_latex_table(
#         best_per_group[["model", "scheme", "level", "atom_index", metric, "n_folds", "n_samples"]],
#         metric=metric, caption=f"Best model per group ({metric})", label="tab:best_per_group"
#     )
#     save_latex_table(latex_best, "reports/best_per_group.tex")

#     # --------- Plots ---------
#     plot_metric_by_model(out, metric=metric, cmap="tab10", figsize=(8,5), savepath="reports/metric_by_model.png")
#     plot_heatmap_by_group(out, metric=metric, cmap="magma", figsize=(11,6), savepath="reports/heatmap_by_group.png")
