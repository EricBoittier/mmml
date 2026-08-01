#!/usr/bin/env python3
"""Plot a sampled DES chemical-space map from averaged SOAP descriptors.

The left panel is a UMAP projection of cosine distances in SOAP space.  The
right panel retains only the minimum spanning tree of the SOAP k-nearest-
neighbour graph: a TMAP-style view that exposes branches and isolated chemical
families without implying that 2-D distances are quantitative.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from mmml.data.cgenff_dataset import (
    assign_frame_cgenff,
    find_covalent_components,
    format_composition,
    load_reference,
)


def _pair_label(z: np.ndarray, r: np.ndarray) -> str:
    comps = find_covalent_components(z, r)
    if len(comps) != 2:
        return f"{len(comps)} components"
    return " + ".join(sorted(format_composition(z[c]) for c in comps))


def _sample(path: Path, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as fh:
        names = sorted(k for k in fh if k != "metadata")
        chosen = np.sort(rng.choice(len(names), size=min(n_samples, len(names)), replace=False))
        frames = []
        for index in chosen:
            g = fh[names[index]]
            z = np.asarray(g["atomic_numbers"][()], dtype=np.int32).reshape(-1)
            r = np.asarray(g["positions"][()], dtype=np.float64).reshape(-1, 3)
            frames.append((names[index], z, r))
    return frames


def _mst_edges(x: np.ndarray, neighbors: int = 12):
    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, len(x)), metric="cosine").fit(x)
    distances, indices = nn.kneighbors(x)
    rows = np.repeat(np.arange(len(x)), indices.shape[1] - 1)
    cols = indices[:, 1:].ravel()
    vals = distances[:, 1:].ravel()
    graph = coo_matrix((vals, (rows, cols)), shape=(len(x), len(x)))
    graph = graph.maximum(graph.T)
    tree = minimum_spanning_tree(graph).tocoo()
    return zip(tree.row.tolist(), tree.col.tolist(), strict=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--csv", type=Path)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-jobs", type=int, default=1)
    args = p.parse_args(argv)

    frames = _sample(args.h5.expanduser(), args.samples, args.seed)
    species = sorted({int(v) for _, z, _ in frames for v in z})
    atoms = [Atoms(numbers=z, positions=r) for _, z, r in frames]
    soap = SOAP(species=species, periodic=False, r_cut=5.0, n_max=4, l_max=3,
                average="inner", sparse=False)
    descriptors = soap.create(atoms, n_jobs=args.n_jobs)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)

    # PCA removes numerical noise and makes the neighbour search tractable; it
    # retains substantially more dimensions than the final 2-D display.
    n_pc = min(50, len(frames) - 1, descriptors.shape[1])
    reduced = PCA(n_components=n_pc, random_state=args.seed).fit_transform(descriptors)
    embedding = UMAP(n_neighbors=25, min_dist=0.08, metric="cosine",
                     random_state=args.seed).fit_transform(reduced)

    ref = load_reference()
    records = []
    status = []
    for (name, z, r), (x, y) in zip(frames, embedding, strict=True):
        assignment, reason = assign_frame_cgenff(z, r, ref, compute_mm=False)
        label = "typeable" if assignment is not None else reason.split("(")[0].strip()
        status.append(label)
        records.append((name, _pair_label(z, r), label, float(x), float(y)))

    order = ["typeable"] + sorted(set(status) - {"typeable"})
    palette = ["#2878b5", "#d94841", "#e69f00", "#7b6fd0", "#888888"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.1), constrained_layout=True)
    for ax, title in zip(axes, ("SOAP / UMAP", "SOAP neighbour tree"), strict=True):
        if ax is axes[1]:
            for i, j in _mst_edges(reduced):
                ax.plot(embedding[[i, j], 0], embedding[[i, j], 1], color="#b8b8b8",
                        lw=0.3, alpha=0.45, zorder=0)
        for k, label in enumerate(order):
            keep = np.asarray(status) == label
            ax.scatter(embedding[keep, 0], embedding[keep, 1], s=7, alpha=0.65,
                       linewidths=0, color=palette[min(k, len(palette) - 1)], label=label)
        ax.set_title(title)
        ax.set_xlabel("embedding 1")
        ax.set_ylabel("embedding 2")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(frameon=False, markerscale=2, fontsize=8, loc="best")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.csv or args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("h5_group", "formula_pair", "cgenff_status", "soap_x", "soap_y"))
        writer.writerows(records)
    print(f"wrote {args.output} and {csv_path} ({len(frames)} sampled frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
