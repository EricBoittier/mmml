"""Bayesian, physically filtered design of compact PES labeling sets.

The command selects a diverse subset from a large geometry-only or cheaply
labeled NPZ pool.  Its default descriptor is intentionally inexpensive:
element-pair RDF channels plus hashed force-field-type pair spectra.  Optional
SOAP can be added when DScribe is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

K_B_EV_K = 8.617333262e-5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml pes-design",
        description=(
            "Select a physical, Bayesian D-optimal, descriptor-diverse PES subset "
            "and validate it against equally sized random sampling."
        ),
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--n-select", type=int, required=True)
    p.add_argument("--descriptor", choices=("pair-rdf", "soap", "combined"), default="pair-rdf")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--rdf-bins", type=int, default=48)
    p.add_argument("--type-hash-bins", type=int, default=64)
    p.add_argument("--pca-components", type=int, default=32)
    p.add_argument("--prior-precision", type=float, default=1.0)
    p.add_argument("--uncertainty-power", type=float, default=1.0)
    p.add_argument("--temperatures", default="300,600,1200", help="Boltzmann-mixture temperatures in K")
    p.add_argument("--min-distance", type=float, default=0.75)
    p.add_argument("--max-force", type=float, default=None, help="Reject frames above max |F| (dataset units)")
    p.add_argument("--max-relative-energy", type=float, default=None, help="Reject frames above group minimum + this value")
    p.add_argument("--seed", type=int, default=0)
    return p


def _frame_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(data["R"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.int32)
    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"R must have shape (frames, atoms, 3), got {R.shape}")
    if Z.ndim == 1:
        Z = np.broadcast_to(Z, R.shape[:2])
    N = np.asarray(data.get("N", (Z > 0).sum(axis=1)), dtype=np.int32).reshape(-1)
    if len(R) != len(Z) or len(R) != len(N):
        raise ValueError("R/Z/N frame counts differ")
    return R, Z, N


def _pair_data(r: np.ndarray, z: np.ndarray, n: int):
    r, z = r[:n], z[:n]
    ii, jj = np.triu_indices(n, 1)
    d = np.linalg.norm(r[ii] - r[jj], axis=1)
    za, zb = np.minimum(z[ii], z[jj]), np.maximum(z[ii], z[jj])
    return d, za, zb, ii, jj


def physical_mask(data: dict, min_distance: float, max_force: float | None,
                  max_relative_energy: float | None) -> tuple[np.ndarray, np.ndarray]:
    R, Z, N = _frame_arrays(data)
    keep = np.isfinite(R).all(axis=(1, 2))
    min_contacts = np.full(len(R), np.inf)
    for i in range(len(R)):
        d, *_ = _pair_data(R[i], Z[i], int(N[i]))
        if len(d):
            min_contacts[i] = d.min()
    keep &= min_contacts >= min_distance
    if max_force is not None and "F" in data:
        F = np.asarray(data["F"], dtype=np.float64)
        keep &= np.nanmax(np.abs(F), axis=(1, 2)) <= max_force
    if max_relative_energy is not None and "E" in data:
        E = np.asarray(data["E"], dtype=np.float64).reshape(len(R), -1)[:, 0]
        groups = np.asarray(data.get("res_name", np.repeat("all", len(R)))).astype(str)
        rel = np.zeros(len(R))
        for group in np.unique(groups):
            m = groups == group
            rel[m] = E[m] - np.nanmin(E[m])
        keep &= np.isfinite(rel) & (rel <= max_relative_energy)
    return keep, min_contacts


def _pair_rdf(data: dict, indices: np.ndarray, cutoff: float, n_bins: int,
              type_hash_bins: int) -> tuple[np.ndarray, list[str]]:
    R, Z, N = _frame_arrays(data)
    species = sorted(int(x) for x in np.unique(Z[Z > 0]))
    pairs = [(a, b) for ia, a in enumerate(species) for b in species[ia:]]
    pair_to_channel = {p: i for i, p in enumerate(pairs)}
    edges = np.linspace(0.0, cutoff, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = max(edges[1] - edges[0], 1e-6)
    n_type = type_hash_bins if "cgenff_type_idx" in data else 0
    X = np.zeros((len(indices), len(pairs) * n_bins + n_type * n_bins), np.float64)
    type_idx = np.asarray(data.get("cgenff_type_idx", []))
    for row, frame in enumerate(indices):
        d, za, zb, ii, jj = _pair_data(R[frame], Z[frame], int(N[frame]))
        within = d < cutoff
        for dist, a, b in zip(d[within], za[within], zb[within]):
            ch = pair_to_channel[(int(a), int(b))]
            # Smooth radial basis is less grid-sensitive than hard histogram counts.
            X[row, ch * n_bins:(ch + 1) * n_bins] += np.exp(-0.5 * ((centers - dist) / width) ** 2)
        if n_type:
            t = type_idx[frame, :int(N[frame])]
            for dist, a, b in zip(d[within], t[ii[within]], t[jj[within]]):
                if a < 0 or b < 0:
                    continue
                a, b = sorted((int(a), int(b)))
                ch = ((a * 1009 + b * 9176) % n_type)
                off = len(pairs) * n_bins + ch * n_bins
                X[row, off:off + n_bins] += np.exp(-0.5 * ((centers - dist) / width) ** 2)
        # Normalize by the number of real pairs; size should not dominate shape.
        X[row] /= max(len(d), 1)
    labels = [f"Z{a}-Z{b}@{r:.2f}" for a, b in pairs for r in centers]
    labels += [f"typehash{h}@{r:.2f}" for h in range(n_type) for r in centers]
    return X, labels


def _soap(data: dict, indices: np.ndarray, cutoff: float) -> tuple[np.ndarray, list[str]]:
    try:
        from ase import Atoms
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise RuntimeError("--descriptor soap/combined requires DScribe (install mmml[quantum])") from exc
    R, Z, N = _frame_arrays(data)
    species = sorted(int(x) for x in np.unique(Z[Z > 0]))
    soap = SOAP(species=species, periodic=False, r_cut=cutoff, n_max=4, l_max=3,
                average="inner", sparse=False)
    atoms = [Atoms(numbers=Z[i, :N[i]], positions=R[i, :N[i]]) for i in indices]
    X = np.asarray(soap.create(atoms, n_jobs=1), dtype=np.float64)
    return X, [f"soap_{i}" for i in range(X.shape[1])]


def descriptors(data: dict, indices: np.ndarray, kind: str, cutoff: float,
                rdf_bins: int, type_hash_bins: int) -> tuple[np.ndarray, list[str], int]:
    rdf, rdf_labels = _pair_rdf(data, indices, cutoff, rdf_bins, type_hash_bins)
    if kind == "pair-rdf":
        return rdf, rdf_labels, rdf.shape[1]
    soap, soap_labels = _soap(data, indices, cutoff)
    if kind == "soap":
        return soap, soap_labels, 0
    return np.concatenate([rdf, soap], axis=1), rdf_labels + soap_labels, rdf.shape[1]


def physical_weights(data: dict, indices: np.ndarray, temperatures: list[float]) -> np.ndarray:
    if "E" not in data:
        return np.ones(len(indices))
    E = np.asarray(data["E"], dtype=np.float64).reshape(len(data["R"]), -1)[:, 0]
    if not np.isfinite(E[indices]).any():
        return np.ones(len(indices))
    groups = np.asarray(data.get("res_name", np.repeat("all", len(E)))).astype(str)
    rel = np.zeros(len(indices))
    for group in np.unique(groups[indices]):
        loc = np.where(groups[indices] == group)[0]
        values = E[indices[loc]]
        finite = np.isfinite(values)
        if finite.any():
            rel[loc[finite]] = values[finite] - np.min(values[finite])
            rel[loc[~finite]] = np.max(rel[loc[finite]], initial=0.0)
    parts = [np.exp(-np.clip(rel / (K_B_EV_K * t), 0.0, 50.0)) for t in temperatures]
    w = np.mean(parts, axis=0)
    # Keep deliberate high-energy coverage possible without letting it dominate.
    return 0.05 + 0.95 * w / max(float(w.max()), 1e-12)


def _embed(X: np.ndarray, n_components: int, seed: int):
    from sklearn.decomposition import PCA
    mean, scale = X.mean(0), X.std(0)
    scale[scale < 1e-10] = 1.0
    Xs = (X - mean) / scale
    n_comp = max(2, min(n_components, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(Xs), pca


def bayesian_select(Z: np.ndarray, n_select: int, physical_w: np.ndarray,
                    prior_precision: float, uncertainty_power: float,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Approximate Bayesian D-optimal batch via leverage-weighted clustering."""
    from sklearn.cluster import MiniBatchKMeans

    n, d = Z.shape
    A = Z.T @ (physical_w[:, None] * Z) / max(n, 1)
    A.flat[::d + 1] += prior_precision
    Ainv = np.linalg.pinv(A, hermitian=True)
    variance = np.einsum("ij,jk,ik->i", Z, Ainv, Z)
    score = physical_w * np.maximum(variance, 1e-12) ** uncertainty_power
    # Clustering supplies coverage/compressibility; leverage sample weights make
    # rare posterior directions receive proportionally more design centres.
    km = MiniBatchKMeans(
        n_clusters=n_select, init="k-means++", n_init=1,
        batch_size=min(max(1024, 4 * n_select), n), random_state=seed,
        max_iter=100, reassignment_ratio=0.005,
    ).fit(Z, sample_weight=score)
    chosen: list[int] = []
    used: set[int] = set()
    for center in km.cluster_centers_:
        dist = np.sum((Z - center) ** 2, axis=1)
        for idx in np.argsort(dist):
            i = int(idx)
            if i not in used:
                used.add(i); chosen.append(i); break
    if len(chosen) < n_select:
        for i in np.argsort(score)[::-1]:
            if int(i) not in used:
                used.add(int(i)); chosen.append(int(i))
            if len(chosen) == n_select:
                break
    return np.asarray(chosen, dtype=np.int64), score


def _coverage(Z: np.ndarray, selected: np.ndarray) -> tuple[float, float, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Z[selected])
    d = nn.kneighbors(Z, return_distance=True)[0][:, 0]
    return float(d.mean()), float(np.percentile(d, 95)), d


def _logdet(Z: np.ndarray, selected: np.ndarray, prior: float) -> float:
    d = Z.shape[1]
    A = Z[selected].T @ Z[selected]
    A.flat[::d + 1] += prior
    sign, value = np.linalg.slogdet(A)
    return float(value if sign > 0 else -np.inf)


def _save_subset(data: dict, source_indices: np.ndarray, score: np.ndarray,
                 output: Path) -> None:
    n = len(data["R"])
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr[source_indices] if arr.ndim and arr.shape[0] == n else arr
    out["pes_design_source_index"] = source_indices
    out["pes_design_score"] = score
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)


def _plots(report_dir: Path, Z: np.ndarray, Xrdf: np.ndarray, selected: np.ndarray,
           random: np.ndarray, d_sel: np.ndarray, d_random: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[:, 0], Z[:, 1], s=5, c="0.8", label="physical candidates")
    ax.scatter(Z[random, 0], Z[random, 1], s=10, alpha=.55, label="random")
    ax.scatter(Z[selected, 0], Z[selected, 1], s=12, alpha=.8, label="Bayes/D-opt")
    ax.set(xlabel="descriptor PC1", ylabel="descriptor PC2", title="Selected PES coverage")
    ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "descriptor_coverage.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for d, label in ((d_sel, "Bayes/D-opt"), (d_random, "random")):
        x = np.sort(d); y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=label)
    ax.set(xlabel="nearest selected-point distance (PCA space)", ylabel="candidate CDF",
           title="Feature-space coverage (left is better)")
    ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "coverage_cdf.png", dpi=180); plt.close(fig)

    if Xrdf.shape[1]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Xrdf.mean(0), color="black", lw=1.5, label="candidate mean")
        ax.plot(Xrdf[selected].mean(0), label="Bayes/D-opt mean")
        ax.plot(Xrdf[random].mean(0), label="random mean", alpha=.8)
        ax.set(xlabel="pair-spectrum feature", ylabel="mean normalized intensity",
               title="Pair-distance/RDF spectrum reproduction")
        ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "rdf_spectrum.png", dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    raw = np.load(args.input, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    keep, min_contacts = physical_mask(data, args.min_distance, args.max_force,
                                       args.max_relative_energy)
    candidates = np.flatnonzero(keep)
    if args.n_select <= 0 or args.n_select > len(candidates):
        raise ValueError(f"--n-select must be in 1..{len(candidates)} after filtering")
    X, labels, n_rdf = descriptors(data, candidates, args.descriptor, args.cutoff,
                                    args.rdf_bins, args.type_hash_bins)
    Z, pca = _embed(X, args.pca_components, args.seed)
    temps = [float(x) for x in args.temperatures.split(",") if x.strip()]
    if not temps or any(t <= 0 for t in temps):
        raise ValueError("--temperatures must contain positive comma-separated values")
    weights = physical_weights(data, candidates, temps)
    selected, score = bayesian_select(Z, args.n_select, weights,
                                      args.prior_precision, args.uncertainty_power, args.seed)
    rng = np.random.default_rng(args.seed + 104729)
    random = rng.choice(len(candidates), args.n_select, replace=False, p=weights / weights.sum())
    mean_s, p95_s, d_s = _coverage(Z, selected)
    mean_r, p95_r, d_r = _coverage(Z, random)
    report = {
        "input": str(args.input.resolve()), "output": str(args.output.resolve()),
        "n_input": int(len(data["R"])), "n_physical": int(len(candidates)),
        "n_rejected": int((~keep).sum()), "n_selected": int(args.n_select),
        "descriptor": args.descriptor, "descriptor_dimensions": int(X.shape[1]),
        "pca_dimensions": int(Z.shape[1]),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "coverage_mean_selected": mean_s, "coverage_mean_random": mean_r,
        "coverage_p95_selected": p95_s, "coverage_p95_random": p95_r,
        "coverage_improvement_percent": float(100 * (mean_r - mean_s) / max(mean_r, 1e-12)),
        "d_opt_logdet_selected": _logdet(Z, selected, args.prior_precision),
        "d_opt_logdet_random": _logdet(Z, random, args.prior_precision),
        "minimum_contact_selected": [float(x) for x in np.percentile(min_contacts[candidates[selected]], [0, 5, 50, 95, 100])],
        "temperatures_K": temps,
    }
    source_indices = candidates[selected]
    _save_subset(data, source_indices, score[selected], args.output)
    report_dir = args.report_dir or args.output.with_suffix("").with_name(args.output.stem + "_report")
    _plots(report_dir, Z, X[:, :n_rdf], selected, random, d_s, d_r)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (report_dir / "descriptor_labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2))
    print(f"Saved selected NPZ: {args.output}")
    print(f"Validation report: {args.report_dir or args.output.with_suffix('').with_name(args.output.stem + '_report')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
