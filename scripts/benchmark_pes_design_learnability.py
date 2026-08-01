#!/usr/bin/env python3
"""Compare PES-design and random subsets with simple energy learning curves.

This is deliberately model-light. If a descriptor-selected set is genuinely
easier to learn, ridge-linear and quadratic models should reach lower error on
the same held-out structures at equal label count.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mmml.cli.misc.pes_design import bayesian_select, descriptors, physical_mask


def _load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _targets(data: dict) -> np.ndarray:
    E = np.asarray(data["E"], dtype=float).reshape(len(data["R"]), -1)[:, 0]
    if not np.isfinite(E).all():
        raise ValueError("energy target contains NaN/Inf; wait for complete collection")
    return E


def _groups(data: dict) -> np.ndarray:
    return np.asarray(data.get("res_name", np.repeat("all", len(data["R"])))).astype(str)


def _composition_matched_random(groups: np.ndarray, selected: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    parts = []
    for group, count in zip(*np.unique(groups[selected], return_counts=True)):
        eligible = np.flatnonzero(groups == group)
        parts.append(rng.choice(eligible, int(count), replace=False))
    return np.concatenate(parts)


def _features_with_composition(Z: np.ndarray, groups: np.ndarray,
                               known_groups: list[str]) -> np.ndarray:
    onehot = np.zeros((len(groups), len(known_groups)))
    by_name = {name: i for i, name in enumerate(known_groups)}
    for row, group in enumerate(groups):
        if group in by_name:
            onehot[row, by_name[group]] = 1.0
    return np.concatenate([Z, onehot], axis=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sizes", default="50,100,200,500,1000")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--rdf-bins", type=int, default=24)
    p.add_argument("--type-hash-bins", type=int, default=16)
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    train, test = _load(a.train), _load(a.test)
    keep_tr, _ = physical_mask(train, 0.75, None, None)
    keep_te, _ = physical_mask(test, 0.75, None, None)
    itr, ite = np.flatnonzero(keep_tr), np.flatnonzero(keep_te)
    Xtr, _, _ = descriptors(train, itr, "pair-rdf", 6.0, a.rdf_bins, a.type_hash_bins)
    Xte, _, _ = descriptors(test, ite, "pair-rdf", 6.0, a.rdf_bins, a.type_hash_bins)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    nc = min(a.pca_components, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    pca = PCA(n_components=nc, svd_solver="randomized", random_state=a.seed).fit(Xtr_s)
    Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
    gtr, gte = _groups(train)[itr], _groups(test)[ite]
    known = sorted(set(gtr))
    Ztr = _features_with_composition(Ztr, gtr, known)
    Zte = _features_with_composition(Zte, gte, known)
    ytr, yte = _targets(train)[itr], _targets(test)[ite]
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    if max(sizes) > len(itr):
        raise ValueError(f"largest size {max(sizes)} exceeds {len(itr)} physical train frames")
    alphas = np.logspace(-8, 4, 25)
    rows = []
    for repeat in range(a.repeats):
        seed = a.seed + repeat
        rng = np.random.default_rng(seed + 104729)
        for size in sizes:
            selected, _ = bayesian_select(Ztr[:, :nc], size, np.ones(len(Ztr)), 1.0, 1.0, seed)
            random = _composition_matched_random(gtr, selected, rng)
            for design, idx in (("bayes_dopt", selected), ("random", random)):
                for model_name, degree in (("linear", 1), ("quadratic", 2)):
                    if degree == 2:
                        poly = PolynomialFeatures(2, include_bias=False).fit(Ztr[idx])
                        A, B = poly.transform(Ztr[idx]), poly.transform(Zte)
                    else:
                        A, B = Ztr[idx], Zte
                    model = RidgeCV(alphas=alphas).fit(A, ytr[idx])
                    pred = model.predict(B)
                    rows.append({
                        "repeat": repeat, "seed": seed, "n_train": size,
                        "design": design, "model": model_name,
                        "mae": float(mean_absolute_error(yte, pred)),
                        "rmse": float(root_mean_squared_error(yte, pred)),
                        "alpha": float(model.alpha_),
                    })
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "train": str(a.train.resolve()), "test": str(a.test.resolve()),
        "n_train_pool": int(len(itr)), "n_test": int(len(ite)),
        "repeats": a.repeats, "sizes": sizes,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "results": [],
    }
    for size in sizes:
        for design in ("bayes_dopt", "random"):
            for model in ("linear", "quadratic"):
                rr = [r for r in rows if r["n_train"] == size and r["design"] == design and r["model"] == model]
                summary["results"].append({
                    "n_train": size, "design": design, "model": model,
                    "mae_mean": float(np.mean([r["mae"] for r in rr])),
                    "mae_std": float(np.std([r["mae"] for r in rr], ddof=1)),
                    "rmse_mean": float(np.mean([r["rmse"] for r in rr])),
                })
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for ax, model in zip(axes, ("linear", "quadratic")):
        for design in ("bayes_dopt", "random"):
            means, stds = [], []
            for size in sizes:
                vals = [r["mae"] for r in rows if r["n_train"] == size and r["design"] == design and r["model"] == model]
                means.append(np.mean(vals)); stds.append(np.std(vals, ddof=1))
            ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=design)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(model); ax.set_xlabel("RI-MP2 labels")
        ax.set_ylabel("held-out energy MAE (eV)"); ax.legend()
    fig.suptitle("Does descriptor design improve simple-model learnability?")
    fig.tight_layout(); fig.savefig(a.out_dir / "learning_curves.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
