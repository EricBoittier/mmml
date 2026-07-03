"""Capture, compare, save, and visualize CHARMM vs MMML inter-monomer neighbor lists."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

MmNlBackendName = Literal["auto", "vesin", "cell_list", "jax_md"]


@dataclass(frozen=True)
class InterMonomerPair:
    i: int
    j: int
    distance_A: float
    monomer_i: int
    monomer_j: int


@dataclass
class NeighborListSnapshot:
    source: str
    cutoff_A: float
    pairs: list[InterMonomerPair]
    meta: dict[str, Any] = field(default_factory=dict)


def uniform_monomer_offsets(n_monomers: int, atoms_per_monomer: int) -> np.ndarray:
    n = int(n_monomers)
    apm = int(atoms_per_monomer)
    if n <= 0 or apm <= 0:
        raise ValueError(f"invalid monomer layout n={n_monomers} apm={atoms_per_monomer}")
    return np.arange(0, n * apm + 1, apm, dtype=np.int32)


def monomer_id_from_offsets(monomer_offsets: Sequence[int], n_atoms: int) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.nl_reference import monomer_id_from_offsets as _mid

    return _mid(monomer_offsets, n_atoms)


def cubic_cell_matrix(box_side: float) -> np.ndarray:
    side = float(box_side)
    return np.diag([side, side, side]).astype(np.float64)


def _pair_records(
    pairs: Iterable[tuple[int, int]],
    *,
    positions: np.ndarray,
    cell: np.ndarray | None,
    monomer_id: np.ndarray,
) -> list[InterMonomerPair]:
    from mmml.interfaces.pycharmmInterface.nl_reference import cell_matrix_3x3, mic_distance

    R = np.asarray(positions, dtype=np.float64)
    cell_mat = cell_matrix_3x3(cell) if cell is not None else None
    mid = np.asarray(monomer_id, dtype=np.int32)
    out: list[InterMonomerPair] = []
    for ai, aj in pairs:
        i, j = int(ai), int(aj)
        if i > j:
            i, j = j, i
        if cell_mat is None:
            dist = float(np.linalg.norm(R[j] - R[i]))
        else:
            dist = mic_distance(R, i, j, cell_mat)
        out.append(
            InterMonomerPair(
                i=i,
                j=j,
                distance_A=dist,
                monomer_i=int(mid[i]),
                monomer_j=int(mid[j]),
            )
        )
    out.sort(key=lambda p: p.distance_A)
    return out


def capture_charmm_inter_monomer_pairs(
    *,
    cutoff_A: float,
    monomer_offsets: Sequence[int],
    positions: np.ndarray | None = None,
    work_dir: Path | None = None,
) -> NeighborListSnapshot:
    """Capture inter-monomer pairs from CHARMM ``COOR DMAT`` (live PyCHARMM session)."""
    import pycharmm.nbonds as nbonds

    from mmml.interfaces.pycharmmInterface.import_pycharmm import capture_neighbour_list

    nbonds.update_bnbnd()
    cleanup: tempfile.TemporaryDirectory[str] | None = None
    cwd_before = Path.cwd()
    try:
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            cleanup = tempfile.TemporaryDirectory(prefix="mmml_nl_dmat_")
            work_dir = Path(cleanup.name)
        import os

        os.chdir(work_dir)
        nl_info = capture_neighbour_list()
    finally:
        import os

        os.chdir(cwd_before)
        if cleanup is not None:
            cleanup.cleanup()

    if positions is None:
        import pandas as pd

        from mmml.interfaces.pycharmmInterface.import_pycharmm import coor

        positions = coor.get_positions().to_numpy(dtype=np.float64)
    else:
        positions = np.asarray(positions, dtype=np.float64)

    mid = monomer_id_from_offsets(monomer_offsets, positions.shape[0])
    resid = nl_info["atom_number_resid_dict"]
    cutoff = float(cutoff_A)
    raw_pairs: set[tuple[int, int]] = set()
    for (a, b), dist in nl_info["pair_distance_dict"].items():
        if float(dist) >= cutoff:
            continue
        i, j = int(a), int(b)
        if int(mid[i]) == int(mid[j]):
            continue
        if i > j:
            i, j = j, i
        raw_pairs.add((i, j))

    pairs = _pair_records(raw_pairs, positions=positions, cell=None, monomer_id=mid)
    return NeighborListSnapshot(
        source="charmm_dmat",
        cutoff_A=cutoff,
        pairs=pairs,
        meta={
            "n_atoms": int(positions.shape[0]),
            "n_pairs_total_dmat": len(nl_info["pair_distance_dict"]),
            "n_inter_monomer_pairs": len(pairs),
            "resid_per_atom": {int(k): int(v) for k, v in resid.items()},
        },
    )


def capture_charmm_jnb_inter_monomer_pairs(
    *,
    cutoff_A: float,
    monomer_offsets: Sequence[int],
    positions: np.ndarray | None = None,
) -> NeighborListSnapshot | None:
    """Capture inter-monomer pairs from CHARMM primary ``JNB/INBLO`` (not DMAT)."""
    import pycharmm.nbonds as nbonds

    from mmml.interfaces.pycharmmInterface.nl_reference import (
        apply_mm_pair_filters,
        filter_pairs_under_cutoff,
        inter_monomer_pair_set,
        walk_charmm_primary_jnb_pair_set,
    )

    nbonds.update_bnbnd()
    exported = nbonds.export_primary_pairs()
    if exported is None:
        return None
    pair_i, pair_j = exported
    raw = walk_charmm_primary_jnb_pair_set(pair_i, pair_j)

    if positions is None:
        import pandas as pd

        from mmml.interfaces.pycharmmInterface.import_pycharmm import coor

        positions = coor.get_positions().to_numpy(dtype=np.float64)
    else:
        positions = np.asarray(positions, dtype=np.float64)

    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, positions.shape[0])
    inter = inter_monomer_pair_set(raw, monomer_id=mid)
    cutoff = float(cutoff_A)
    filtered = apply_mm_pair_filters(inter, monomer_id=mid, positions=positions, cell=None)
    mic_filtered = {
        pair
        for pair in filtered
        if float(np.linalg.norm(positions[pair[1]] - positions[pair[0]])) < cutoff
    }
    pairs = _pair_records(mic_filtered, positions=positions, cell=None, monomer_id=mid)
    return NeighborListSnapshot(
        source="charmm_jnb",
        cutoff_A=cutoff,
        pairs=pairs,
        meta={
            "n_primary_pairs_total": len(raw),
            "n_inter_monomer_pairs": len(pairs),
        },
    )


def capture_mlpot_mlmm_inter_monomer_pairs(
    *,
    cutoff_A: float,
    monomer_offsets: Sequence[int],
    natom: int,
    positions: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    mm_r_min: float | None = None,
) -> NeighborListSnapshot | None:
    """Capture inter-monomer ML–MM pairs from Fortran ``idxu/idxv`` (callback path)."""
    from pycharmm.energy_mlpot import export_mlpot_mlmm_pairs

    from mmml.interfaces.pycharmmInterface.nl_reference import (
        apply_mm_pair_filters,
        canonical_half_pair,
        filter_pairs_under_cutoff,
        inter_monomer_pair_set,
    )

    exported = export_mlpot_mlmm_pairs()
    if exported is None:
        return None
    pair_u, pair_v = exported
    primary: set[tuple[int, int]] = set()
    for u, v in zip(pair_u, pair_v, strict=False):
        if int(u) >= int(natom) or int(v) >= int(natom):
            continue
        primary.add(canonical_half_pair(u, v))

    if positions is None:
        import pandas as pd

        from mmml.interfaces.pycharmmInterface.import_pycharmm import coor

        positions = coor.get_positions().to_numpy(dtype=np.float64)
    else:
        positions = np.asarray(positions, dtype=np.float64)

    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, positions.shape[0])
    inter = inter_monomer_pair_set(primary, monomer_id=mid)
    cutoff = float(cutoff_A)
    filtered = apply_mm_pair_filters(
        inter,
        monomer_id=mid,
        positions=positions,
        cell=cell,
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
    )
    if cell is not None:
        mic_filtered = filter_pairs_under_cutoff(filtered, positions, cell, cutoff)
    else:
        mic_filtered = {
            pair
            for pair in filtered
            if float(np.linalg.norm(positions[pair[1]] - positions[pair[0]])) < cutoff
        }
    pairs = _pair_records(mic_filtered, positions=positions, cell=cell, monomer_id=mid)
    return NeighborListSnapshot(
        source="mlpot_mlmm",
        cutoff_A=cutoff,
        pairs=pairs,
        meta={
            "n_mlmm_pairs_exported": len(pair_u),
            "n_primary_inter_monomer_pairs": len(pairs),
        },
    )


def compare_snapshots_aligned(
    left: NeighborListSnapshot,
    right: NeighborListSnapshot,
    *,
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    monomer_offsets: Sequence[int],
    mm_r_min: float | None = None,
) -> dict[str, Any]:
    """Pair diff plus semantic tags (cutoff skew, ``mm_r_min``, true mismatch)."""
    from mmml.interfaces.pycharmmInterface.nl_reference import (
        canonical_half_pair,
        classify_inter_monomer_diff,
    )

    base = compare_snapshots(left, right)
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, np.asarray(positions).shape[0])
    left_set = {
        canonical_half_pair(p.i, p.j) for p in left.pairs
    }
    right_set = {
        canonical_half_pair(p.i, p.j) for p in right.pairs
    }
    tags = classify_inter_monomer_diff(
        only_left=left_set - right_set,
        only_right=right_set - left_set,
        positions=np.asarray(positions, dtype=np.float64),
        cell=cell,
        monomer_id=mid,
        left_cutoff_A=float(left.cutoff_A),
        right_cutoff_A=float(right.cutoff_A),
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
    )
    base["semantic_tags"] = tags
    base["common_cutoff_A"] = min(float(left.cutoff_A), float(right.cutoff_A))
    return base


def capture_mmml_inter_monomer_pairs(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff_A: float,
    monomer_offsets: Sequence[int],
    backend: MmNlBackendName = "vesin",
    mm_r_min: float | None = None,
) -> NeighborListSnapshot:
    """Capture MMML switched-MM inter-monomer pairs (Vesin / cell-list / jax-md)."""
    from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend
    from mmml.interfaces.pycharmmInterface.nl_reference import (
        apply_mm_pair_filters,
        extract_valid_pairs,
        filter_pairs_under_cutoff,
    )

    R = np.asarray(positions, dtype=np.float64)
    cell_arr = np.asarray(cell, dtype=np.float64)
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, R.shape[0])
    cutoff = float(cutoff_A)
    backend_name = str(backend).strip().lower()
    if backend_name == "auto":
        backend_name = "vesin"

    atoms_per = [int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)]
    pi, pj, mask, n_valid, capacity, used = build_mm_pairs_with_backend(
        backend_name,  # type: ignore[arg-type]
        R,
        cell_arr,
        cutoff=cutoff,
        monomer_offsets=offsets,
        atoms_per_monomer_list=atoms_per,
        mm_r_min=mm_r_min,
        total_atoms=R.shape[0],
    )
    raw = extract_valid_pairs(pi, pj, mask)
    filtered = apply_mm_pair_filters(
        raw,
        monomer_id=mid,
        positions=R,
        cell=cell_arr,
        mm_r_min=mm_r_min,
        monomer_offsets=offsets,
    )
    mic_filtered = filter_pairs_under_cutoff(filtered, R, cell_arr, cutoff)
    pairs = _pair_records(mic_filtered, positions=R, cell=cell_arr, monomer_id=mid)
    return NeighborListSnapshot(
        source=f"mmml_{used}",
        cutoff_A=cutoff,
        pairs=pairs,
        meta={
            "backend_requested": backend_name,
            "backend_used": used,
            "n_valid_pairs": int(n_valid),
            "capacity": int(capacity),
            "mm_r_min": mm_r_min,
            "n_atoms": int(R.shape[0]),
        },
    )


def compare_snapshots(
    left: NeighborListSnapshot,
    right: NeighborListSnapshot,
) -> dict[str, Any]:
    """Symmetric pair-set diff keyed by ``(i, j)``."""
    left_map = {(p.i, p.j): p.distance_A for p in left.pairs}
    right_map = {(p.i, p.j): p.distance_A for p in right.pairs}
    only_left = sorted(set(left_map) - set(right_map))
    only_right = sorted(set(right_map) - set(left_map))
    shared = sorted(set(left_map) & set(right_map))
    dist_delta = [
        {
            "i": i,
            "j": j,
            f"{left.source}_A": left_map[(i, j)],
            f"{right.source}_A": right_map[(i, j)],
            "delta_A": float(right_map[(i, j)] - left_map[(i, j)]),
        }
        for i, j in shared
        if abs(right_map[(i, j)] - left_map[(i, j)]) > 1.0e-4
    ]
    dist_delta.sort(key=lambda row: abs(float(row["delta_A"])), reverse=True)
    return {
        "left": left.source,
        "right": right.source,
        "n_left": len(left_map),
        "n_right": len(right_map),
        "n_shared": len(shared),
        "n_only_left": len(only_left),
        "n_only_right": len(only_right),
        "only_left": [{"i": i, "j": j, "distance_A": left_map[(i, j)]} for i, j in only_left[:200]],
        "only_right": [{"i": i, "j": j, "distance_A": right_map[(i, j)]} for i, j in only_right[:200]],
        "distance_deltas": dist_delta[:200],
        "worst_left": _worst_pairs(left.pairs, 10),
        "worst_right": _worst_pairs(right.pairs, 10),
    }


def _worst_pairs(pairs: Sequence[InterMonomerPair], n: int) -> list[dict[str, Any]]:
    rows = sorted(pairs, key=lambda p: p.distance_A)[: int(n)]
    return [asdict(p) for p in rows]


def snapshot_to_jsonable(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    monomer_offsets: Sequence[int],
    charmm: NeighborListSnapshot | None = None,
    mmml: NeighborListSnapshot | None = None,
    comparison: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "positions_shape": list(np.asarray(positions).shape),
        "cell_A": np.asarray(cell, dtype=float).tolist(),
        "monomer_offsets": [int(x) for x in monomer_offsets],
        "meta": dict(extra_meta or {}),
    }
    if charmm is not None:
        payload["charmm"] = {
            "source": charmm.source,
            "cutoff_A": charmm.cutoff_A,
            "meta": charmm.meta,
            "pairs": [asdict(p) for p in charmm.pairs],
        }
    if mmml is not None:
        payload["mmml"] = {
            "source": mmml.source,
            "cutoff_A": mmml.cutoff_A,
            "meta": mmml.meta,
            "pairs": [asdict(p) for p in mmml.pairs],
        }
    if comparison is not None:
        payload["comparison"] = comparison
    return payload


def write_pairs_csv(path: Path, pairs: Sequence[InterMonomerPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["i", "j", "distance_A", "monomer_i", "monomer_j"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(asdict(pair))


def save_neighbor_list_artifacts(
    out_dir: Path,
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    monomer_offsets: Sequence[int],
    charmm: NeighborListSnapshot | None = None,
    mmml: NeighborListSnapshot | None = None,
    extra_meta: dict[str, Any] | None = None,
    top_pairs: int = 30,
) -> dict[str, Path]:
    """Write JSON, CSV pair tables, positions NPY, and a matplotlib PNG."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    comparison = compare_snapshots(charmm, mmml) if charmm is not None and mmml is not None else None
    payload = snapshot_to_jsonable(
        positions=positions,
        cell=cell,
        monomer_offsets=monomer_offsets,
        charmm=charmm,
        mmml=mmml,
        comparison=comparison,
        extra_meta=extra_meta,
    )
    json_path = out / "neighbor_lists.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(out / "positions.npy", np.asarray(positions, dtype=np.float64))
    paths: dict[str, Path] = {"json": json_path, "positions": out / "positions.npy"}
    if charmm is not None:
        csv_path = out / "charmm_pairs.csv"
        write_pairs_csv(csv_path, charmm.pairs)
        paths["charmm_csv"] = csv_path
    if mmml is not None:
        csv_path = out / "mmml_pairs.csv"
        write_pairs_csv(csv_path, mmml.pairs)
        paths["mmml_csv"] = csv_path
    if comparison is not None:
        cmp_path = out / "comparison.json"
        cmp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        paths["comparison"] = cmp_path
    plot_path = out / "neighbor_lists.png"
    plot_neighbor_lists(
        positions=positions,
        cell=cell,
        monomer_offsets=monomer_offsets,
        charmm=charmm,
        mmml=mmml,
        out_path=plot_path,
        top_pairs=top_pairs,
    )
    paths["plot"] = plot_path
    return paths


def plot_neighbor_lists(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    monomer_offsets: Sequence[int],
    charmm: NeighborListSnapshot | None = None,
    mmml: NeighborListSnapshot | None = None,
    out_path: Path,
    top_pairs: int = 30,
) -> None:
    """3D matplotlib view: atoms + closest inter-monomer pair segments."""
    import matplotlib.pyplot as plt

    R = np.asarray(positions, dtype=np.float64)
    offsets = np.asarray(monomer_offsets, dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, R.shape[0])
    n_panels = int(charmm is not None) + int(mmml is not None)
    if n_panels == 0:
        raise ValueError("plot_neighbor_lists requires at least one snapshot")

    fig = plt.figure(figsize=(6.5 * n_panels, 6.0))
    panel_idx = 1

    def _draw_panel(
        snap: NeighborListSnapshot,
        *,
        title: str,
        color: str,
    ) -> None:
        nonlocal panel_idx
        ax = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
        panel_idx += 1
        cmap = plt.get_cmap("tab20")
        for mi in range(int(mid.max()) + 1):
            mask = mid == mi
            pts = R[mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=8,
                color=cmap(mi % 20),
                alpha=0.65,
                linewidths=0,
            )
        worst = sorted(snap.pairs, key=lambda p: p.distance_A)[: int(top_pairs)]
        for pair in worst:
            a = R[pair.i]
            b = R[pair.j]
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color=color,
                linewidth=1.2,
                alpha=0.85,
            )
        if worst:
            p0 = worst[0]
            ax.set_title(
                f"{title}\nclosest {p0.distance_A:.3f} Å "
                f"(mon {p0.monomer_i}/{p0.monomer_j}, atoms {p0.i}/{p0.j})"
            )
        else:
            ax.set_title(f"{title}\n(no pairs under cutoff)")
        side = float(np.diag(np.asarray(cell, dtype=float).reshape(3, 3))[0])
        ax.set_xlim(0, side)
        ax.set_ylim(0, side)
        ax.set_zlim(0, side)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")

    if charmm is not None:
        _draw_panel(charmm, title=f"CHARMM ({charmm.cutoff_A:.1f} Å)", color="#d62728")
    if mmml is not None:
        _draw_panel(mmml, title=f"MMML ({mmml.cutoff_A:.1f} Å)", color="#1f77b4")

    fig.tight_layout()
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def setup_charmm_from_psf_crd(
    *,
    psf_path: Path,
    crd_path: Path,
    box_side: float,
    charmm_cutoff_A: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load PSF+CRD into PyCHARMM, apply PBC nbonds, return positions and effective cutoff."""
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import read_psf_card_file
    from mmml.interfaces.pycharmmInterface.import_pycharmm import read_cgenff_toppar
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import apply_crd_file_to_charmm
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import apply_pbc_nbonds, prepare_charmm_pbc
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        charmm_relaxed_bomlev,
        get_charmm_positions_array,
    )

    read_cgenff_toppar()
    with charmm_relaxed_bomlev():
        read_psf_card_file(psf_path)
        apply_crd_file_to_charmm(crd_path)
    prepare_charmm_pbc(float(box_side))
    cuts = apply_pbc_nbonds(
        nbxmod=5,
        cutnb=float(charmm_cutoff_A),
        cubic_box_side_A=float(box_side),
    )
    positions = np.asarray(get_charmm_positions_array(), dtype=np.float64)
    return positions, cubic_cell_matrix(box_side), float(cuts.cutnb)


def find_artifact_geometry(artifact_dir: Path) -> tuple[Path, Path]:
    """Return ``(psf, crd)`` from a campaign leg output directory."""
    root = Path(artifact_dir).expanduser().resolve()
    psf_candidates = sorted(root.rglob("*.psf"), key=lambda p: p.stat().st_mtime, reverse=True)
    crd_candidates = sorted(root.rglob("*.crd"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not psf_candidates or not crd_candidates:
        raise FileNotFoundError(
            f"no PSF/CRD under {root} (need model.psf or mini_full_mlpot_*.psf + matching CRD)"
        )
    return psf_candidates[0], crd_candidates[0]
