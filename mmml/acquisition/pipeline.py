"""One-shot acquisition pipeline stages (Snakemake and CLI share these).

Later active-learning rounds recompute representations from a fine-tuned
checkpoint by pointing ``student`` at that checkpoint and incrementing
``round``.  This module only implements round 0 (one-shot) but keeps
artifacts keyed by round so that is a config change, not a rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from mmml.acquisition.labels import (
    LabelCache,
    build_backend,
    labels_to_npz,
)
from mmml.acquisition.linear_student import (
    LinearStudent,
    init_linear_student,
    init_linear_teacher,
)
from mmml.acquisition.md_eval import outcome_to_dict, run_nve
from mmml.acquisition.metrics import evaluate_predictions
from mmml.acquisition.pca import assert_no_leakage, project_jacobian_rows
from mmml.acquisition.report import selection_overlap, size_dependence, write_report
from mmml.acquisition.representations import (
    activation_energy_alignment,
    extract_activations,
    extract_energy_jacobians,
    extract_force_information_blocks,
    extract_loss_gradients,
    fit_embedding_pca,
    fit_jacobian_basis,
    linear_readout_pooled_equals_energy_grad,
)
from mmml.acquisition.selection import (
    farthest_point_sampling,
    greedy_doptimal,
    largest_norm_indices,
    stratified_random,
)
from mmml.acquisition.splits import (
    SPLIT_CANDIDATE,
    SPLIT_SEED,
    SPLIT_TEST,
    SPLIT_VALID,
    PoolManifest,
    StructureRecord,
    build_pool,
    subset_arrays,
)
from mmml.acquisition.synthetic import default_seed_groups, make_smoke_pool
from mmml.acquisition.train import TrainSettings, finetune_linear, predict_dataset

METHODS = (
    "stratified_random",
    "activation_fps",
    "activation_largest_norm",
    "output_grad_fps",
    "output_grad_largest_norm",
    "force_doptimal",
    "loss_grad_fps",
    "loss_grad_largest_norm",
    "unmodified_student",
)

UNRESOLVED_DEFAULTS = [
    "Production reference backend (PySCF/ORCA/Molpro/…) is unset; smoke uses mock_morse.",
    "Compute budget (QC walltime, GPU hours, number of unique labels) is unset.",
    "Deployment MD conditions (ensemble, box, thermostat, production length) are unset.",
    "Student and teacher checkpoint paths for the scientific campaign are unset.",
    "Candidate pool source (which trajectories / thermodynamic states) is unset.",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} is not a mapping")
    return cfg


def workdir(cfg: Mapping[str, Any], override: str | Path | None = None) -> Path:
    if override:
        root = Path(override)
    else:
        root = Path(cfg.get("output_root") or "artifacts/label_acquisition/default")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_ser) + "\n")


def _ser(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def stage_prepare_pool(cfg: Mapping[str, Any], out: Path) -> Path:
    src = cfg.get("candidate") or {}
    kind = str(src.get("source") or "synthetic_smoke")
    if kind == "synthetic_smoke":
        data = make_smoke_pool(
            n_traj_per_stratum=int(src.get("n_traj_per_stratum", 2)),
            n_frames=int(src.get("n_frames", 6)),
            seed=int(src.get("seed", 0)),
        )
        seed_groups = default_seed_groups(data, n_groups=int(src.get("n_seed_groups", 2)))
    elif kind in ("npz", "file"):
        path = Path(src["path"])
        data = _load_npz(path)
        seed_groups = list(src.get("seed_groups") or [])
    else:
        raise ValueError(
            f"candidate.source={kind!r} is unresolved. Use synthetic_smoke for "
            "tests or point path at an unlabeled NPZ."
        )
    splits_cfg = cfg.get("splits") or {}
    manifest = build_pool(
        data,
        valid_fraction=float(splits_cfg.get("valid_fraction", 0.2)),
        test_fraction=float(splits_cfg.get("test_fraction", 0.2)),
        seed=int(splits_cfg.get("seed", 0)),
        seed_groups=seed_groups,
    )
    pool_dir = out / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(pool_dir / "structures.npz", data)
    _write_manifest(pool_dir / "manifest.json", manifest)
    for split in (SPLIT_CANDIDATE, SPLIT_VALID, SPLIT_TEST, SPLIT_SEED):
        recs = manifest.by_split(split)
        _save_npz(pool_dir / f"{split}.npz", subset_arrays(data, recs) if recs else _empty_like(data))
        (pool_dir / f"{split}_ids.json").write_text(
            json.dumps([r.structure_id for r in recs], indent=2) + "\n"
        )
    return pool_dir / "manifest.json"


def _object_arr(items: list) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _empty_like(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    return {
        "R": np.zeros((0,) + R.shape[1:], dtype=R.dtype),
        "Z": np.zeros((0,) + Z.shape[1:], dtype=Z.dtype),
        "N": np.zeros((0,), dtype=np.int32),
        "E": np.zeros((0,), dtype=np.float64),
        "F": np.zeros((0,) + np.asarray(data.get("F", np.zeros((1,) + R.shape[1:]))).shape[1:], dtype=np.float64),
    }


def _write_manifest(path: Path, manifest: PoolManifest) -> None:
    payload = {
        "n_input": manifest.n_input,
        "n_unique": manifest.n_unique,
        "n_duplicates_removed": manifest.n_duplicates_removed,
        "duplicate_of": manifest.duplicate_of,
        "records": [
            {
                "index": r.index,
                "structure_id": r.structure_id,
                "geometry_fingerprint": r.geometry_fingerprint,
                "composition": r.composition,
                "stratum": r.stratum,
                "group": r.group,
                "n_atoms": r.n_atoms,
                "split": r.split,
                "temperature": r.temperature,
                "pressure": r.pressure,
                "phase": r.phase,
                "frame_index": r.frame_index,
                "source": r.source,
                "atomic_numbers": r.atomic_numbers.tolist(),
                "positions": r.positions.tolist(),
                "cell": None if r.cell is None else np.asarray(r.cell).tolist(),
            }
            for r in manifest.records
        ],
    }
    _json_dump(path, payload)


def load_manifest(path: Path) -> PoolManifest:
    payload = json.loads(Path(path).read_text())
    records = []
    for row in payload["records"]:
        records.append(
            StructureRecord(
                index=int(row["index"]),
                structure_id=row["structure_id"],
                geometry_fingerprint=row["geometry_fingerprint"],
                composition=row["composition"],
                stratum=row["stratum"],
                group=row["group"],
                n_atoms=int(row["n_atoms"]),
                atomic_numbers=np.asarray(row["atomic_numbers"], dtype=np.int32),
                positions=np.asarray(row["positions"], dtype=np.float64),
                cell=None if row.get("cell") is None else np.asarray(row["cell"]),
                temperature=row.get("temperature"),
                pressure=row.get("pressure"),
                phase=row.get("phase"),
                frame_index=row.get("frame_index"),
                source=row.get("source"),
                split=row["split"],
            )
        )
    return PoolManifest(
        records=records,
        duplicate_of=payload.get("duplicate_of") or {},
        n_input=int(payload.get("n_input", len(records))),
        n_unique=int(payload.get("n_unique", len(records))),
        n_duplicates_removed=int(payload.get("n_duplicates_removed", 0)),
    )


def _models(cfg: Mapping[str, Any]) -> tuple[LinearStudent, LinearStudent]:
    sc = cfg.get("student") or {}
    kind = str(sc.get("kind") or "linear_smoke")
    if kind not in ("linear_smoke", "linear", "mock"):
        raise ValueError(
            f"student.kind={kind!r} is unresolved for this workflow revision; "
            "use linear_smoke (tests) or implement a PhysNet adapter checkpoint."
        )
    student = init_linear_student(
        max_z=int(sc.get("max_z", 8)),
        k_dist=int(sc.get("k_dist", 4)),
        seed=int(sc.get("seed", 0)),
        scale=float(sc.get("scale", 0.05)),
        name="student",
    )
    tc = cfg.get("teacher") or {}
    teacher = init_linear_teacher(
        student,
        seed=int(tc.get("seed", 1)),
        scale=float(tc.get("scale", 0.08)),
    )
    return student, teacher


def stage_fingerprint_models(cfg: Mapping[str, Any], out: Path) -> Path:
    student, teacher = _models(cfg)
    d = out / "models"
    d.mkdir(parents=True, exist_ok=True)
    _json_dump(d / "student.json", student.fingerprint())
    _json_dump(d / "teacher.json", teacher.fingerprint())
    note = {
        "teacher_role": (
            "Cheap surrogate for loss-gradient acquisition. Not ground truth. "
            "Shared errors of student and teacher are invisible to this method."
        ),
        "seed_coverage_prior": (
            "Default D-optimal seed coverage is the existing student-training "
            "structures (split=seed). Foundation-labeled training coverage is a "
            "design prior, not calibrated uncertainty vs the expensive reference."
        ),
    }
    _json_dump(d / "notes.json", note)
    return d / "student.json"


def _candidate_and_seed(out: Path) -> tuple[PoolManifest, list[StructureRecord], list[StructureRecord]]:
    manifest = load_manifest(out / "pool" / "manifest.json")
    return manifest, manifest.by_split(SPLIT_CANDIDATE), manifest.by_split(SPLIT_SEED)


def stage_extract(cfg: Mapping[str, Any], out: Path) -> Path:
    student, teacher = _models(cfg)
    _manifest, candidates, seeds = _candidate_and_seed(out)
    held_out = [r.structure_id for r in _manifest.records if r.split in (SPLIT_VALID, SPLIT_TEST)]
    loss_cfg = cfg.get("loss") or {}
    a_e = float(loss_cfg.get("energy_weight", 1.0))
    a_f = float(loss_cfg.get("forces_weight", 52.91))
    rep_dir = out / "representations"
    rep_dir.mkdir(parents=True, exist_ok=True)

    act, act_ids, act_stats, _atom = extract_activations(student, candidates)
    ejac, ejac_ids, ejac_stats = extract_energy_jacobians(student, candidates)
    blocks, blk_ids, blk_stats = extract_force_information_blocks(
        student, candidates, energy_weight=a_e, forces_weight=a_f
    )
    loss_g, loss_ids, loss_stats = extract_loss_gradients(
        student, teacher, candidates, energy_weight=a_e, forces_weight=a_f
    )
    # Seed coverage blocks for D-optimal (acquisition-accessible only).
    seed_blocks = []
    if seeds:
        seed_blocks, _, _ = extract_force_information_blocks(
            student, seeds, energy_weight=a_e, forces_weight=a_f
        )

    _save_npz(rep_dir / "activations.npz", {"X": act, "structure_id": np.asarray(act_ids, dtype=object)})
    _save_npz(rep_dir / "energy_jacobian.npz", {"X": ejac, "structure_id": np.asarray(ejac_ids, dtype=object)})
    _save_npz(rep_dir / "loss_gradient.npz", {"X": loss_g, "structure_id": np.asarray(loss_ids, dtype=object)})
    np.savez_compressed(
        rep_dir / "force_blocks.npz",
        blocks=_object_arr(blocks),
        structure_id=np.asarray(blk_ids, dtype=object),
        seed_blocks=_object_arr(seed_blocks),
    )
    align = linear_readout_pooled_equals_energy_grad(student, candidates)
    pooled_species, _, _, _ = extract_activations(student, candidates)
    species_align = activation_energy_alignment(pooled_species, ejac)
    stats = {
        "activations": act_stats.__dict__,
        "energy_jacobian": ejac_stats.__dict__,
        "force_jacobian": blk_stats.__dict__,
        "loss_gradient": loss_stats.__dict__,
        "held_out_ids_excluded": held_out,
        "linear_readout_sum_pool_vs_energy_grad": align,
        "species_pooled_activations_vs_energy_grad": species_align,
        "architecture_note": (
            "On LinearStudent, sum-pooled invariant features equal ∇_w E. "
            "Species-aware pooled activations are a *different* vector (per-element "
            "channels + max pool) and are not equivalent to energy Jacobians. "
            "PhysNet uses e3x.Dense(1) then nn.Dense(1) plus optional energy_bias, "
            "ZBL and electrostatics, so activation-vs-Jacobian equivalence must be "
            "measured on that architecture rather than assumed."
        ),
    }
    _json_dump(rep_dir / "stats.json", stats)
    return rep_dir / "stats.json"


def stage_fit_pca(cfg: Mapping[str, Any], out: Path) -> Path:
    pca_cfg = cfg.get("pca") or {}
    n_comp = pca_cfg.get("n_components")
    n_comp = int(n_comp) if n_comp is not None else None
    var_thr = pca_cfg.get("variance_threshold")
    var_thr = float(var_thr) if var_thr is not None else None
    row_norm = bool(pca_cfg.get("row_normalize", False))
    jac_comp = pca_cfg.get("jacobian_components")
    jac_comp = int(jac_comp) if jac_comp is not None else n_comp

    manifest = load_manifest(out / "pool" / "manifest.json")
    held = [r.structure_id for r in manifest.records if r.split in (SPLIT_VALID, SPLIT_TEST)]
    rep = out / "representations"
    pca_dir = out / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    fits = {}
    for name in ("activations", "energy_jacobian", "loss_gradient"):
        data = _load_npz(rep / f"{name}.npz")
        ids = [str(x) for x in data["structure_id"].tolist()]
        fit, Z = fit_embedding_pca(
            data["X"], ids, n_components=n_comp, variance_threshold=var_thr, row_normalize=row_norm
        )
        assert_no_leakage(fit, held)
        _save_npz(pca_dir / f"{name}_embedding.npz", {"Z": Z, "structure_id": data["structure_id"]})
        _json_dump(pca_dir / f"{name}_pca.json", fit.to_dict())
        fits[name] = fit.to_dict()
        _pca_scatter(Z, ids, manifest, pca_dir / f"{name}_pca.png")

    fb = np.load(rep / "force_blocks.npz", allow_pickle=True)
    blocks = list(fb["blocks"])
    ids = [str(x) for x in fb["structure_id"].tolist()]
    jfit, proj = fit_jacobian_basis(blocks, ids, n_components=jac_comp, variance_threshold=var_thr)
    assert_no_leakage(jfit, held)
    np.savez_compressed(
        pca_dir / "force_blocks_projected.npz",
        blocks=_object_arr(proj),
        structure_id=fb["structure_id"],
    )
    seed_blocks = list(fb["seed_blocks"]) if "seed_blocks" in fb.files else []
    seed_proj = [
        project_jacobian_rows(np.asarray(b, dtype=np.float64), jfit)
        for b in seed_blocks
        if np.asarray(b).size
    ]
    np.savez_compressed(
        pca_dir / "force_seed_projected.npz",
        blocks=_object_arr(seed_proj),
    )
    _json_dump(pca_dir / "force_jacobian_basis.json", jfit.to_dict())
    fits["force_jacobian_basis"] = jfit.to_dict()
    _json_dump(pca_dir / "summary.json", fits)
    return pca_dir / "summary.json"


def _pca_scatter(Z: np.ndarray, ids: list[str], manifest: PoolManifest, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_id = {r.structure_id: r for r in manifest.records}
    comps = [by_id[i].composition if i in by_id else "?" for i in ids]
    uniq = sorted(set(comps))
    fig, ax = plt.subplots(figsize=(5, 4))
    if Z.shape[1] >= 2:
        for c in uniq:
            m = np.array([x == c for x in comps])
            ax.scatter(Z[m, 0], Z[m, 1], s=18, label=c, alpha=0.8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        ax.scatter(Z[:, 0], np.zeros(len(Z)), s=18)
        ax.set_xlabel("PC1")
    ax.legend(fontsize=8)
    ax.set_title("PCA embedding (candidates only)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _load_embedding(out: Path, name: str) -> tuple[np.ndarray, list[str]]:
    data = _load_npz(out / "pca" / f"{name}_embedding.npz")
    ids = [str(x) for x in data["structure_id"].tolist()]
    return np.asarray(data["Z"], dtype=np.float64), ids


def stage_select(cfg: Mapping[str, Any], out: Path) -> Path:
    sel_cfg = cfg.get("selection") or {}
    budgets = [int(b) for b in (sel_cfg.get("budgets") or [8])]
    seeds = [int(s) for s in (sel_cfg.get("seeds") or [0])]
    methods = list(sel_cfg.get("methods") or METHODS)
    lam = float(sel_cfg.get("doptimal_lambda", 1e-3))
    manifest = load_manifest(out / "pool" / "manifest.json")
    candidates = manifest.by_split(SPLIT_CANDIDATE)
    cand_ids = [r.structure_id for r in candidates]
    strata = [r.stratum for r in candidates]

    act_Z, _ = _load_embedding(out, "activations")
    ejac_Z, _ = _load_embedding(out, "energy_jacobian")
    loss_Z, _ = _load_embedding(out, "loss_gradient")
    act_raw = _load_npz(out / "representations" / "activations.npz")["X"]
    ejac_raw = _load_npz(out / "representations" / "energy_jacobian.npz")["X"]
    loss_raw = _load_npz(out / "representations" / "loss_gradient.npz")["X"]
    proj = np.load(out / "pca" / "force_blocks_projected.npz", allow_pickle=True)
    blocks = list(proj["blocks"])
    seed_file = out / "pca" / "force_seed_projected.npz"
    seed_blocks = []
    if seed_file.is_file():
        sf = np.load(seed_file, allow_pickle=True)
        if "blocks" in sf.files and np.asarray(sf["blocks"]).size:
            seed_blocks = list(sf["blocks"])

    sel_root = out / "selections"
    sel_root.mkdir(parents=True, exist_ok=True)
    written = []
    for method in methods:
        for budget in budgets:
            for seed in seeds:
                if method == "unmodified_student":
                    picked: list[str] = []
                else:
                    idx = _select_indices(
                        method,
                        budget=budget,
                        seed=seed,
                        strata=strata,
                        act_Z=act_Z,
                        ejac_Z=ejac_Z,
                        loss_Z=loss_Z,
                        act_raw=act_raw,
                        ejac_raw=ejac_raw,
                        loss_raw=loss_raw,
                        blocks=blocks,
                        seed_blocks=seed_blocks,
                        lam=lam,
                    )
                    picked = [cand_ids[i] for i in idx]
                d = sel_root / method / f"budget{budget}" / f"seed{seed}"
                d.mkdir(parents=True, exist_ok=True)
                payload = {
                    "method": method,
                    "budget": budget,
                    "nominal_budget": 0 if method == "unmodified_student" else budget,
                    "seed": seed,
                    "structure_ids": picked,
                    "n_selected": len(picked),
                    "immutable": True,
                    "round": int(cfg.get("round", 0)),
                    "notes": _method_notes(method),
                }
                _json_dump(d / "manifest.json", payload)
                written.append(str(d / "manifest.json"))
    _json_dump(sel_root / "index.json", {"manifests": written})
    # Freeze: labeling may begin.
    _json_dump(out / "selections_finalized.json", {"ok": True, "n_manifests": len(written)})
    return out / "selections_finalized.json"


def _method_notes(method: str) -> list[str]:
    notes = {
        "stratified_random": ["Random within composition × thermodynamic-condition strata."],
        "activation_fps": [
            "Species-aware pooled invariant features → PCA → Euclidean FPS.",
            "Not UMAP/t-SNE/random projection.",
        ],
        "activation_largest_norm": ["Ablation: largest L2 of pooled activations (pre-PCA)."],
        "output_grad_fps": ["Energy Jacobian w.r.t. final readout params → PCA → FPS."],
        "output_grad_largest_norm": ["Ablation: largest L2 of energy Jacobians."],
        "force_doptimal": [
            "Greedy regularized D-optimal on weighted [∇E; ∇F] blocks.",
            "Seed coverage = student-training split (design prior, not calibrated uncertainty).",
        ],
        "loss_grad_fps": [
            "Student–teacher loss gradient w.r.t. readout → PCA → FPS.",
            "Not the original classification BADGE algorithm.",
        ],
        "loss_grad_largest_norm": ["Ablation: largest L2 of loss gradients."],
        "unmodified_student": ["Baseline: no extra expensive labels, original checkpoint."],
    }
    return notes.get(method, [])


def _select_indices(
    method: str,
    *,
    budget: int,
    seed: int,
    strata: list[str],
    act_Z,
    ejac_Z,
    loss_Z,
    act_raw,
    ejac_raw,
    loss_raw,
    blocks,
    seed_blocks,
    lam: float,
) -> np.ndarray:
    if method == "stratified_random":
        return stratified_random(strata, budget, seed=seed)
    if method == "activation_fps":
        return farthest_point_sampling(act_Z, budget, seed=seed)
    if method == "activation_largest_norm":
        return largest_norm_indices(np.asarray(act_raw, dtype=np.float64), budget)
    if method == "output_grad_fps":
        return farthest_point_sampling(ejac_Z, budget, seed=seed)
    if method == "output_grad_largest_norm":
        return largest_norm_indices(np.asarray(ejac_raw, dtype=np.float64), budget)
    if method == "loss_grad_fps":
        return farthest_point_sampling(loss_Z, budget, seed=seed)
    if method == "loss_grad_largest_norm":
        return largest_norm_indices(np.asarray(loss_raw, dtype=np.float64), budget)
    if method == "force_doptimal":
        idx, _gains = greedy_doptimal(blocks, budget, lam=lam, seed_blocks=seed_blocks)
        return idx
    raise ValueError(f"unknown method {method}")


def _require_finalized(out: Path) -> None:
    marker = out / "selections_finalized.json"
    if not marker.is_file():
        raise RuntimeError("refusing to label: selections are not finalized")


def stage_label(cfg: Mapping[str, Any], out: Path) -> Path:
    _require_finalized(out)
    ref_cfg = cfg.get("reference") or {"backend": "mock_morse"}
    backend = build_backend(ref_cfg)
    cache = LabelCache(out / "labels" / "cache")
    manifest = load_manifest(out / "pool" / "manifest.json")
    by_id = {r.structure_id: r for r in manifest.records}

    # Evaluation set: held-out test (+ valid if configured), labeled independently.
    eval_splits = list((cfg.get("evaluation") or {}).get("splits") or [SPLIT_TEST])
    eval_recs = [r for r in manifest.records if r.split in eval_splits]
    eval_budget = (cfg.get("evaluation") or {}).get("max_structures")
    if eval_budget is not None:
        eval_recs = eval_recs[: int(eval_budget)]

    labeled_eval = [cache.compute_or_cached(r, backend) for r in eval_recs]
    eval_npz = labels_to_npz(eval_recs, labeled_eval)
    _save_npz(out / "labels" / "eval" / "labeled.npz", eval_npz)
    _json_dump(
        out / "labels" / "eval" / "accounting.json",
        {
            "n_requested": len(eval_recs),
            "n_success": int(sum(x.success for x in labeled_eval)),
            "n_failed": int(sum(not x.success for x in labeled_eval)),
            "cache_hits_so_far": cache.hits,
            "cache_misses_so_far": cache.misses,
            "separate_from_method_budgets": True,
        },
    )

    sel_index = json.loads((out / "selections" / "index.json").read_text())
    unique_ids: set[str] = set()
    nominal = 0
    for path in sel_index["manifests"]:
        man = json.loads(Path(path).read_text())
        method = man["method"]
        budget = man["budget"]
        seed = man["seed"]
        ids = list(man["structure_ids"])
        nominal += int(man.get("nominal_budget", len(ids)))
        recs = [by_id[i] for i in ids if i in by_id]
        unique_ids.update(i for i in ids if i in by_id)
        results = [cache.compute_or_cached(r, backend) for r in recs]
        npz = labels_to_npz(recs, results) if recs else _empty_like(eval_npz)
        dest = out / "labels" / "selected" / method / f"budget{budget}" / f"seed{seed}"
        _save_npz(dest / "labeled.npz", npz)
        _json_dump(
            dest / "accounting.json",
            {
                "nominal_budget": man.get("nominal_budget", len(ids)),
                "n_requested": len(recs),
                "n_success": int(sum(x.success for x in results)),
                "n_failed": int(sum(not x.success for x in results)),
                "failed_ids": [r.structure_id for r, x in zip(recs, results) if not x.success],
            },
        )
    _json_dump(
        out / "labels" / "cache_summary.json",
        {
            "unique_calculations": cache.hits + cache.misses,  # stored entries requested
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "failures": cache.failures,
            "unique_structure_ids_labeled_for_training": len(unique_ids),
            "sum_nominal_budgets": nominal,
            "eval_labels": len(eval_recs),
            "method_fingerprint": backend.fingerprint(),
        },
    )
    # Recompute unique calcs from disk (hits+misses counts requests, not unique).
    n_unique = 0
    cache_root = out / "labels" / "cache"
    if cache_root.is_dir():
        n_unique = sum(1 for p in cache_root.glob("*/*/labels.npz"))
    summary = json.loads((out / "labels" / "cache_summary.json").read_text())
    summary["unique_calculations"] = n_unique
    _json_dump(out / "labels" / "cache_summary.json", summary)
    return out / "labels" / "cache_summary.json"


def stage_train_eval(cfg: Mapping[str, Any], out: Path) -> Path:
    student0, _teacher = _models(cfg)
    tcfg = cfg.get("training") or {}
    modes = list(tcfg.get("modes") or ["readout", "full"])
    settings = TrainSettings(
        energy_weight=float((cfg.get("loss") or {}).get("energy_weight", tcfg.get("energy_weight", 1.0))),
        forces_weight=float((cfg.get("loss") or {}).get("forces_weight", tcfg.get("forces_weight", 52.91))),
        learning_rate=float(tcfg.get("learning_rate", 0.05)),
        n_steps=int(tcfg.get("n_steps", 30)),
        seed=int(tcfg.get("seed", 0)),
        batch_size=int(tcfg.get("batch_size", 4)),
    )
    md_cfg = cfg.get("md") or {}
    eval_npz = _load_npz(out / "labels" / "eval" / "labeled.npz")
    manifest = load_manifest(out / "pool" / "manifest.json")
    by_id = {r.structure_id: r for r in manifest.records}

    sel_index = json.loads((out / "selections" / "index.json").read_text())
    rows = []
    id_sets: dict[str, list[str]] = {}
    for path in sel_index["manifests"]:
        man = json.loads(Path(path).read_text())
        method, budget, seed = man["method"], man["budget"], man["seed"]
        key = f"{method}/budget{budget}/seed{seed}"
        id_sets[key] = list(man["structure_ids"])
        labeled_path = out / "labels" / "selected" / method / f"budget{budget}" / f"seed{seed}" / "labeled.npz"
        labeled = _load_npz(labeled_path) if labeled_path.is_file() else None
        if labeled is not None and labeled["E"].shape[0]:
            keep = np.asarray(labeled.get("label_success", np.ones(len(labeled["E"]), dtype=bool)))
            if keep.any():
                labeled = {
                    k: (v[keep] if np.asarray(v).shape[:1] == keep.shape else v)
                    for k, v in labeled.items()
                }
            else:
                labeled = None
        tune_modes = ["unmodified"] if method == "unmodified_student" else modes
        for mode in tune_modes:
            use_unmodified = (
                method == "unmodified_student"
                or labeled is None
                or labeled["E"].shape[0] == 0
                or mode == "unmodified"
            )
            if use_unmodified:
                trained, train_meta = finetune_linear(
                    student0, eval_npz, settings, mode="unmodified"
                )
            else:
                trained, train_meta = finetune_linear(student0, labeled, settings, mode=mode)
            pred = predict_dataset(trained, eval_npz)
            metrics = evaluate_predictions(
                pred,
                eval_npz,
                compositions=[
                    by_id[sid].composition if sid in by_id else "?"
                    for sid in eval_npz.get("structure_id", [])
                ] if "structure_id" in eval_npz else None,
            )
            # Matched MD: same first eval structure, same seed.
            md_stats = []
            n_md = min(int(md_cfg.get("n_structures", 2)), int(eval_npz["R"].shape[0]))
            for i in range(n_md):
                ni = int(eval_npz["N"][i])
                outc = run_nve(
                    trained.energy_forces,
                    eval_npz["R"][i, :ni],
                    eval_npz["Z"][i, :ni],
                    dt=float(md_cfg.get("dt", 0.5)),
                    n_steps=int(md_cfg.get("n_steps", 30)),
                    seed=int(md_cfg.get("seed", 0)) + i,
                    temperature=float(md_cfg.get("temperature", 300.0)),
                )
                md_stats.append(outcome_to_dict(outc))
            dest = out / "eval" / method / f"budget{budget}" / f"seed{seed}" / mode
            dest.mkdir(parents=True, exist_ok=True)
            _json_dump(dest / "metrics.json", metrics)
            _json_dump(dest / "train.json", train_meta)
            _json_dump(dest / "md.json", md_stats)
            _save_npz(dest / "predictions.npz", pred)
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "seed": seed,
                    "tune_mode": mode,
                    "energy_mae": metrics.get("energy_mae"),
                    "force_rmse": metrics.get("force_rmse"),
                    "md_failures": int(sum(1 for m in md_stats if m["failed"])),
                    "nve_drift": float(np.nanmean([m["energy_drift"] for m in md_stats])),
                    "n_train_labels": 0 if labeled is None else int(np.asarray(labeled["E"]).shape[0]),
                }
            )
    # overlap vs stratified_random of matching budget/seed
    for row in rows:
        rand_key = f"stratified_random/budget{row['budget']}/seed{row['seed']}"
        this_key = f"{row['method']}/budget{row['budget']}/seed{row['seed']}"
        ov = selection_overlap(
            {
                "this": id_sets.get(this_key, []),
                "random": id_sets.get(rand_key, []),
            }
        )
        row["jaccard_vs_random"] = ov.get("this∩random")
        row.update(size_dependence(id_sets.get(this_key, []), {r.structure_id: r.n_atoms for r in manifest.records}))
    _json_dump(out / "eval" / "rows.json", rows)
    return out / "eval" / "rows.json"


def stage_report(cfg: Mapping[str, Any], out: Path) -> Path:
    rows = json.loads((out / "eval" / "rows.json").read_text()) if (out / "eval" / "rows.json").is_file() else []
    cache_summary = {}
    cs = out / "labels" / "cache_summary.json"
    if cs.is_file():
        cache_summary = json.loads(cs.read_text())
    stats = {}
    sp = out / "representations" / "stats.json"
    if sp.is_file():
        stats = json.loads(sp.read_text())
    pca_sum = {}
    pp = out / "pca" / "summary.json"
    if pp.is_file():
        pca_sum = json.loads(pp.read_text())
    unresolved = list(cfg.get("unresolved") or UNRESOLVED_DEFAULTS)
    if cfg.get("reference", {}).get("backend") in (None, "UNRESOLVED"):
        if UNRESOLVED_DEFAULTS[0] not in unresolved:
            unresolved.insert(0, UNRESOLVED_DEFAULTS[0])
    payload = {
        "unresolved": unresolved,
        "label_accounting": {
            "unique_calculations": cache_summary.get("unique_calculations"),
            "sum_nominal_budgets": cache_summary.get("sum_nominal_budgets"),
            "eval_labels": cache_summary.get("eval_labels"),
            "failures": cache_summary.get("failures"),
            "cache_hits": cache_summary.get("cache_hits"),
            "cache_misses": cache_summary.get("cache_misses"),
        },
        "activation_energy_alignment": {
            **(stats.get("linear_readout_sum_pool_vs_energy_grad") or {}),
            "architecture_note": stats.get("architecture_note"),
            "species_pooled": stats.get("species_pooled_activations_vs_energy_grad"),
        },
        "pca": {
            name: {"retained_variance": fit.get("retained_variance"), "n_components": fit.get("n_components")}
            for name, fit in pca_sum.items()
        },
        "results": rows,
        "acquisition_cost": stats,
        "notes": [
            "One-shot acquisition (round 0). Re-run extract after fine-tuning for later rounds.",
            "Mock results in this tree are not scientific results." if (cfg.get("reference") or {}).get("backend") in ("mock_morse", "mock", "smoke") else "",
            "Avoid declaring a winner from a single seed or aggregate force RMSE alone.",
        ],
    }
    payload["notes"] = [n for n in payload["notes"] if n]
    write_report(payload, out / "report" / "report.md", out / "report" / "summary.json")
    return out / "report" / "report.md"


def run_all(cfg: Mapping[str, Any], out: Path) -> Path:
    stage_prepare_pool(cfg, out)
    stage_fingerprint_models(cfg, out)
    stage_extract(cfg, out)
    stage_fit_pca(cfg, out)
    stage_select(cfg, out)
    stage_label(cfg, out)
    stage_train_eval(cfg, out)
    return stage_report(cfg, out)
