"""Hypothesis tests on oriented-volume sweep CSVs (multiple seeds / runs).

Interprets ``pts`` as **pseudotensors** (``include_pseudotensors``) and ``max_l`` as
``max_degree``. Uses `valid_mae_last10_mean` by default (change with ``--metric``).

Tests are **exploratory**: several samples share the same random split of data per seed file,
so p-values for *pooled* tests can be optimistic. Where possible we use **pairing** on
``(seed, max_degree)`` for pseudotensor comparisons.

Examples::

  python test_hypotheses_oriented_volume_sweeps.py oriented_volume_sweep_seed*.csv
  python test_hypotheses_oriented_volume_sweeps.py --glob 'oriented_volume_sweep_seed*.csv' --tests rank

If no CSV paths are given, loads ``oriented_volume_sweep_seed*.csv`` next to this script.

**Rank-based tests** (use ``--tests rank`` for only these): Wilcoxon signed-rank and sign tests
on paired contrasts; Kruskal–Wallis; Friedman; Mann–Whitney U. Parametric add-ons: *t*-tests,
Gaussian linear interaction *F* (H7 only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from summarize_oriented_volume_sweeps import _collect_paths


def _coerce_bool(s: pd.Series) -> pd.Series:
  return s.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower() in ("1", "true", "yes"))


def _load_frames(paths: list[Path]) -> pd.DataFrame:
  frames = [pd.read_csv(p) for p in paths]
  df = pd.concat(frames, ignore_index=True)
  need = {"seed", "max_degree", "include_pseudotensors", "valid_mae_last10_mean", "valid_mae_final"}
  miss = need - set(df.columns)
  if miss:
    raise SystemExit(f"Missing columns {sorted(miss)}")
  df["include_pseudotensors"] = _coerce_bool(df["include_pseudotensors"])
  df["odd_max_degree"] = (df["max_degree"].astype(int) % 2).astype(bool)
  return df


def _paired_pt_diffs(df: pd.DataFrame, metric: str) -> pd.Series:
  """Per (seed, max_degree), y(PT=True) - y(PT=False); drops incomplete pairs."""
  wide = df.pivot_table(
      index=["seed", "max_degree"],
      columns="include_pseudotensors",
      values=metric,
      aggfunc="first",
  )
  if False not in wide.columns or True not in wide.columns:
    return pd.Series(dtype=float)
  d = wide[True].sub(wide[False])
  return d.dropna()


def _cohens_d_one_sample(x: np.ndarray) -> float:
  x = np.asarray(x, dtype=float)
  x = x[np.isfinite(x)]
  if len(x) < 2:
    return float("nan")
  return float(np.mean(x) / np.std(x, ddof=1))


def _print_test(title: str, statistic: float, pvalue: float, extra: str = "") -> None:
  sig = "reject H0" if pvalue < 0.05 else "fail to reject H0"
  print(f"\n{title}")
  print(f"  statistic = {statistic:.6g}, p-value = {pvalue:.6g}  ({sig} at α=0.05){extra}")


def _sign_test_two_sided(d: np.ndarray) -> tuple[float, float]:
  """Exact two-sided sign test on paired differences; zeros discarded. Returns (n_plus, p)."""
  d = np.asarray(d, dtype=float).ravel()
  d = d[np.isfinite(d)]
  d_nz = d[d != 0]
  n = len(d_nz)
  if n == 0:
    return float("nan"), float("nan")
  pos = int(np.sum(d_nz > 0))
  res = stats.binomtest(pos, n, p=0.5, alternative="two-sided")
  return float(pos), float(res.pvalue)


def _wilcoxon_safe(x: np.ndarray, **kwargs) -> tuple[float, float]:
  x = np.asarray(x, dtype=float).ravel()
  x = x[np.isfinite(x)]
  if len(x) < 3:
    return float("nan"), float("nan")
  try:
    w = stats.wilcoxon(x, **kwargs)
    return float(w.statistic), float(w.pvalue)
  except ValueError:
    return float("nan"), float("nan")


def _wilcoxon_paired_safe(x: np.ndarray, y: np.ndarray, **kwargs) -> tuple[float, float]:
  x = np.asarray(x, dtype=float).ravel()
  y = np.asarray(y, dtype=float).ravel()
  m = np.isfinite(x) & np.isfinite(y)
  x, y = x[m], y[m]
  if len(x) < 3:
    return float("nan"), float("nan")
  try:
    w = stats.wilcoxon(x, y, **kwargs)
    return float(w.statistic), float(w.pvalue)
  except ValueError:
    return float("nan"), float("nan")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("csv_files", nargs="*", type=str)
  parser.add_argument("--glob", dest="glob_pattern", default=None)
  parser.add_argument(
      "--metric",
      choices=("valid_mae_last10_mean", "valid_mae_final"),
      default="valid_mae_last10_mean",
  )
  parser.add_argument("--alpha", type=float, default=0.05, help="Only affects printed label text.")
  parser.add_argument(
      "--tests",
      choices=("both", "parametric", "rank"),
      default="both",
      help="``rank``: only rank / nonparametric tests. ``parametric``: *t*-tests and H7 *F*-test. "
      "``both``: print parametric and rank blocks where applicable.",
  )
  args = parser.parse_args()
  do_param = args.tests in ("both", "parametric")
  do_rank = args.tests in ("both", "rank")
  paths = _collect_paths(args.csv_files, args.glob_pattern)
  if not paths:
    raise SystemExit("No CSV files found.")

  df = _load_frames(paths)
  y = df[args.metric]

  print(f"Rows: {len(df)} from {len(paths)} file(s). Metric: {args.metric}")
  print(f"Test mode: {args.tests}")
  print("H0 phrases below are informal; read each line.")

  # --- H: Pseudotensors have no effect (paired on seed × max_degree)
  d_pt = _paired_pt_diffs(df, args.metric)
  if len(d_pt) >= 3:
    dv = d_pt.values
    if do_param:
      t, p = stats.ttest_1samp(dv, 0.0, alternative="two-sided")
      _print_test(
          "H1 [parametric]: Pseudotensors have no effect on MAE (paired Δ = PT−noPT; one-sample t)",
          float(t),
          float(p),
          f" | Cohen d (on Δ) = {_cohens_d_one_sample(dv):.4g}",
      )
    if do_rank:
      w, pw = _wilcoxon_safe(dv, alternative="two-sided")
      _print_test(
          "H1 [rank]: same null (Wilcoxon signed-rank on paired Δ)",
          w,
          pw,
      )
      n_plus, psign = _sign_test_two_sided(dv)
      _print_test(
          "H1 [rank]: same null (sign test on paired Δ; zeros omitted; stat = count of positive Δ)",
          n_plus,
          psign,
      )
  else:
    print("\nH1 (pseudotensors): not enough complete (seed, max_degree) pairs for a paired test.")

  # --- H: max_degree has no effect (omnibus across 5 levels, pooled — non-independent)
  groups_l = [g[args.metric].values for _, g in df.groupby("max_degree", sort=True)]
  groups_l = [g for g in groups_l if len(g) > 0]
  if len(groups_l) >= 2 and do_rank:
    kw = stats.kruskal(*groups_l)
    _print_test(
        "H2 [rank]: max_degree has no effect (Kruskal–Wallis across max_degree levels; "
        "pooled over PT & seeds — optimistic p)",
        float(kw.statistic),
        float(kw.pvalue),
    )
    # Friedman: block = seed, treatments = max_degree, one PT level at a time
    for pt_val, label in ((False, "no PT"), (True, "PT")):
      sub = df[df["include_pseudotensors"] == pt_val]
      wide = sub.pivot_table(index="seed", columns="max_degree", values=args.metric, aggfunc="first")
      wide = wide.dropna(axis=0, how="any")
      if wide.shape[0] >= 3 and wide.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
        _print_test(
            f"H2b [rank]: max_degree has no effect (Friedman, seeds as blocks, {label} only)",
            float(fr.statistic),
            float(fr.pvalue),
        )

  # --- H: odd vs even max_degree (marginal, pooled)
  odd_y = df.loc[df["odd_max_degree"], args.metric]
  even_y = df.loc[~df["odd_max_degree"], args.metric]
  if len(odd_y) > 2 and len(even_y) > 2 and do_rank:
    u = stats.mannwhitneyu(odd_y, even_y, alternative="two-sided")
    _print_test(
        "H3 [rank]: Odd vs even max_degree makes no difference (Mann–Whitney U on metric; "
        "pooled — non-independent)",
        float(u.statistic),
        float(u.pvalue),
    )

  # --- H: PT has no effect restricted to odd L / even L
  for name, mask in (
      ("odd max_degree", df["odd_max_degree"]),
      ("even max_degree", ~df["odd_max_degree"]),
  ):
    sub = df[mask]
    d = _paired_pt_diffs(sub, args.metric)
    if len(d) >= 3:
      dv = d.values
      if do_param:
        t, p = stats.ttest_1samp(dv, 0.0, alternative="two-sided")
        _print_test(
            f"H4 [parametric]: Pseudotensors have no effect when restricting to {name} (paired Δ; t)",
            float(t),
            float(p),
        )
      if do_rank:
        w, pw = _wilcoxon_safe(dv, alternative="two-sided")
        _print_test(
            f"H4 [rank]: same null for {name} (Wilcoxon on paired Δ)",
            w,
            pw,
        )
        n_plus, ps = _sign_test_two_sided(dv)
        _print_test(
            f"H4 [rank]: same null for {name} (sign test on paired Δ; stat = # positive Δ)",
            n_plus,
            ps,
        )
    else:
      print(f"\nH4 ({name}): insufficient paired rows.")

  # --- H: PT effect does not depend on parity of max_degree
  # Per seed: mean Δ over odd L vs mean Δ over even L; then paired across seeds.
  wide_full = df.pivot_table(
      index=["seed", "max_degree"],
      columns="include_pseudotensors",
      values=args.metric,
      aggfunc="first",
  )
  if False in wide_full.columns and True in wide_full.columns:
    wide_full = wide_full.dropna()
    wide_full["diff"] = wide_full[True] - wide_full[False]
    wide_full = wide_full.reset_index()
    def _mean_diff_odd(g: pd.DataFrame) -> float:
      return float(g.loc[g["max_degree"].astype(int) % 2 == 1, "diff"].mean())

    def _mean_diff_even(g: pd.DataFrame) -> float:
      return float(g.loc[g["max_degree"].astype(int) % 2 == 0, "diff"].mean())

    odd_diffs = wide_full.groupby("seed", sort=True, group_keys=False).apply(_mean_diff_odd)
    even_diffs = wide_full.groupby("seed", sort=True, group_keys=False).apply(_mean_diff_even)
    common = odd_diffs.index.intersection(even_diffs.index)
    o = odd_diffs.loc[common].astype(float)
    e = even_diffs.loc[common].astype(float)
    pair = pd.DataFrame({"odd_L_mean_delta": o, "even_L_mean_delta": e}).dropna()
    if len(pair) >= 3:
      ocol = pair["odd_L_mean_delta"].values
      ecol = pair["even_L_mean_delta"].values
      if do_param:
        t, p = stats.ttest_rel(ocol, ecol)
        _print_test(
            "H5 [parametric]: PT effect does not differ for odd vs even max_degree "
            "(per-seed mean paired Δ odd vs even L; paired t)",
            float(t),
            float(p),
        )
      if do_rank:
        w, pw = _wilcoxon_paired_safe(ocol, ecol, alternative="two-sided")
        _print_test(
            "H5 [rank]: same null (Wilcoxon signed-rank on paired per-seed contrasts)",
            w,
            pw,
        )
        diff = ocol - ecol
        n_plus, ps = _sign_test_two_sided(diff)
        _print_test(
            "H5 [rank]: same null (sign test on per-seed odd_L minus even_L mean PT deltas)",
            n_plus,
            ps,
        )
    else:
      print("\nH5 (PT×parity interaction via per-seed contrasts): not enough seeds with both parities.")

  # --- H: all (max_degree × PT) cells have the same median (omnibus)
  cells = df.groupby(["max_degree", "include_pseudotensors"], sort=True)
  grp = [g[args.metric].values for _, g in cells]
  grp = [g for g in grp if len(g) > 0]
  if len(grp) >= 2 and do_rank:
    kw = stats.kruskal(*grp)
    _print_test(
        "H6 [rank]: No difference among any (max_degree × pseudotensors) combination "
        "(Kruskal–Wallis across all cells — pooled runs)",
        float(kw.statistic),
        float(kw.pvalue),
    )

  # --- OLS-style interaction: y ~ PT + odd_L + PT*odd_L (Type I sequential F, numpy)
  T = df["include_pseudotensors"].astype(float).values
  P = df["odd_max_degree"].astype(float).values
  yv = y.astype(float).values
  n = len(yv)
  X_full = np.column_stack([np.ones(n), T, P, T * P])
  X_no_i = np.column_stack([np.ones(n), T, P])
  # SSE from lstsq
  def sse(X, yvec):
    beta, _, rank, _ = np.linalg.lstsq(X, yvec, rcond=None)
    if rank < X.shape[1]:
      return None, None
    pred = X @ beta
    return float(np.sum((yvec - pred) ** 2)), X.shape[1]

  s_full, p_full = sse(X_full, yv)
  s_no_int, p_no_int = sse(X_no_i, yv)
  if do_param and s_full is not None and s_no_int is not None and s_full > 0:
    df_int = p_full - p_no_int
    df_e = n - p_full
    if df_int > 0 and df_e > 0:
      ms_int = (s_no_int - s_full) / df_int
      ms_e = s_full / df_e
      F_int = ms_int / ms_e if ms_e > 0 else float("nan")
      p_int = 1 - stats.f.cdf(F_int, df_int, df_e)
      _print_test(
          "H7 [parametric]: no PT × odd_max_degree interaction on MAE "
          "(sequential F for interaction in y ~ 1 + PT + odd + PT*odd; Gaussian residuals)",
          float(F_int),
          float(p_int),
      )
  elif not do_param:
    print(
        "\nH7: skipped in rank-only mode (no exact rank analogue printed; consider aligned rank transform ANOVA)."
    )

  print(
      f"\n---\nMultiple tests were run; control FDR or use Bonferroni if you need a family-wise α "
      f"(nominal α={args.alpha} only guides the reject/fail labels above).\n"
  )


if __name__ == "__main__":
  main()
