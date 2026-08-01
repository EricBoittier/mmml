#!/usr/bin/env python3
"""Paired byte-compression and descriptor-coverage benchmark for PES designs."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np

from mmml.cli.misc.pes_design import (
    _coverage,
    _embed,
    bayesian_select,
    descriptors,
    physical_mask,
    physical_weights,
)


def _serialized_sizes(data: dict, indices: np.ndarray) -> tuple[int, int, int]:
    """Return uncompressed-NPZ, gzip(NPZ), and compressed-NPZ sizes.

    Sorting removes ordering as a confound. Both arms contain exactly the same
    fields and number of structures; selector-only metadata is excluded.
    """
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    n = len(data["R"])
    subset = {
        key: (arr[indices] if arr.ndim and arr.shape[0] == n else arr)
        for key, value in data.items()
        for arr in [np.asarray(value)]
    }
    raw = io.BytesIO()
    np.savez(raw, **subset)
    raw_bytes = raw.getvalue()
    zipped = io.BytesIO()
    np.savez_compressed(zipped, **subset)
    return len(raw_bytes), len(gzip.compress(raw_bytes, compresslevel=9)), len(zipped.getvalue())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-select", type=int, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--rdf-bins", type=int, default=24)
    p.add_argument("--type-hash-bins", type=int, default=16)
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    raw = np.load(a.input, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    keep, _ = physical_mask(data, 0.75, None, None)
    candidates = np.flatnonzero(keep)
    X, _, _ = descriptors(data, candidates, "pair-rdf", a.cutoff,
                           a.rdf_bins, a.type_hash_bins)
    Z, pca = _embed(X, a.pca_components, a.seed)
    weights = physical_weights(data, candidates, [300.0, 600.0, 1200.0])
    rows = []
    for repeat in range(a.repeats):
        seed = a.seed + repeat
        selected, _ = bayesian_select(Z, a.n_select, weights, 1.0, 1.0, seed)
        rng = np.random.default_rng(seed + 104729)
        # Match chemical composition exactly. Otherwise a selector that keeps
        # more padded monomers appears artificially gzip-friendly simply because
        # their unused atom slots are zeros.
        groups_all = np.asarray(data.get("res_name", np.repeat("all", len(data["R"])))).astype(str)
        candidate_groups = groups_all[candidates]
        random_parts = []
        for group, count in zip(*np.unique(candidate_groups[selected], return_counts=True)):
            eligible = np.flatnonzero(candidate_groups == group)
            pw = weights[eligible]; pw = pw / pw.sum()
            random_parts.append(rng.choice(eligible, int(count), replace=False, p=pw))
        random = np.concatenate(random_parts)
        for method, local in (("bayes_dopt", selected), ("random", random)):
            source = candidates[local]
            raw_n, gzip_n, npzc_n = _serialized_sizes(data, source)
            mean, p95, _ = _coverage(Z, local)
            rows.append({
                "repeat": repeat, "seed": seed, "method": method,
                "raw_npz_bytes": raw_n, "gzip_npz_bytes": gzip_n,
                "compressed_npz_bytes": npzc_n,
                "gzip_ratio": raw_n / gzip_n,
                "coverage_mean": mean, "coverage_p95": p95,
            })
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "paired_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    import csv
    with (a.out_dir / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "input": str(a.input.resolve()), "n_candidates": int(len(candidates)),
        "n_select": a.n_select, "repeats": a.repeats,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
    }
    for method in ("bayes_dopt", "random"):
        rr = [r for r in rows if r["method"] == method]
        for metric in ("gzip_npz_bytes", "compressed_npz_bytes", "coverage_mean", "coverage_p95"):
            values = np.array([r[metric] for r in rr], float)
            summary[f"{method}_{metric}_mean"] = float(values.mean())
            summary[f"{method}_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    bg = summary["bayes_dopt_gzip_npz_bytes_mean"]
    rg = summary["random_gzip_npz_bytes_mean"]
    summary["bayes_gzip_size_change_vs_random_percent"] = 100 * (bg - rg) / rg
    bc = summary["bayes_dopt_coverage_mean_mean"]
    rc = summary["random_coverage_mean_mean"]
    summary["bayes_coverage_improvement_percent"] = 100 * (rc - bc) / rc
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mmml.utils.plotting.styles import apply_plot_style
    style = apply_plot_style("icml")
    bayes_color = style.colors["train"]
    random_color = style.colors["valid"]
    connector_color = style.colors["muted"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for repeat in range(a.repeats):
        pair = {r["method"]: r for r in rows if r["repeat"] == repeat}
        axes[0].plot([0, 1], [pair["random"]["gzip_npz_bytes"] / 1024,
                             pair["bayes_dopt"]["gzip_npz_bytes"] / 1024],
                     color=connector_color, alpha=.45, marker=None)
        axes[1].plot([0, 1], [pair["random"]["coverage_mean"],
                             pair["bayes_dopt"]["coverage_mean"]],
                     color=connector_color, alpha=.45, marker=None)
        axes[0].scatter(0, pair["random"]["gzip_npz_bytes"] / 1024,
                        color=random_color, marker="s")
        axes[0].scatter(1, pair["bayes_dopt"]["gzip_npz_bytes"] / 1024,
                        color=bayes_color, marker="o")
        axes[1].scatter(0, pair["random"]["coverage_mean"],
                        color=random_color, marker="s")
        axes[1].scatter(1, pair["bayes_dopt"]["coverage_mean"],
                        color=bayes_color, marker="o")
    for ax, ylabel, title in (
        (axes[0], "gzip size (KiB)", "Literal byte compressibility"),
        (axes[1], "mean coverage distance", "Descriptor coverage (lower is better)"),
    ):
        ax.set_xticks([0, 1], ["Random", "Bayes/D-opt"]); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.savefig(a.out_dir / "paired_compression_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
