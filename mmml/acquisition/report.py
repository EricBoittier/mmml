"""Aggregate learning curves, overlap, diversity, cost, and stability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def selection_overlap(manifests: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Pairwise Jaccard overlap of selected structure-ID sets."""
    names = sorted(manifests)
    out: dict[str, float] = {}
    for i, a in enumerate(names):
        sa = set(manifests[a])
        for b in names[i + 1 :]:
            sb = set(manifests[b])
            union = sa | sb
            out[f"{a}∩{b}"] = float(len(sa & sb) / len(union)) if union else 0.0
    return out


def size_dependence(
    selected_ids: Sequence[str],
    id_to_natoms: Mapping[str, int],
) -> dict[str, float]:
    ns = [id_to_natoms[i] for i in selected_ids if i in id_to_natoms]
    if not ns:
        return {"mean_n_atoms": float("nan"), "std_n_atoms": float("nan")}
    arr = np.asarray(ns, dtype=np.float64)
    return {"mean_n_atoms": float(arr.mean()), "std_n_atoms": float(arr.std())}


def write_report(
    payload: Mapping[str, Any],
    output_md: Path,
    output_json: Path,
) -> None:
    output_json = Path(output_json)
    output_md = Path(output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    lines = _render_markdown(payload)
    output_md.write_text("\n".join(lines) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _render_markdown(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Structure-selection acquisition report",
        "",
        "The foundation / teacher potential is a cheap surrogate, not ground truth.",
        "Expensive reference labels were unavailable to acquisition methods until",
        "selection manifests were written.  Evaluation uses a fixed held-out set",
        "whose labeling cost is accounted separately from each method's nominal budget.",
        "",
        "## Unresolved production choices",
        "",
    ]
    for item in payload.get("unresolved", []) or ["(none recorded)"]:
        lines.append(f"- {item}")
    lines += ["", "## Label accounting", ""]
    acc = payload.get("label_accounting") or {}
    lines.append(
        f"- Unique expensive calculations executed: **{acc.get('unique_calculations', 'n/a')}**"
    )
    lines.append(
        f"- Sum of nominal method budgets (training labels): **{acc.get('sum_nominal_budgets', 'n/a')}**"
    )
    lines.append(
        f"- Evaluation-set labels (separate): **{acc.get('eval_labels', 'n/a')}**"
    )
    lines.append(
        f"- Failures: **{acc.get('failures', 'n/a')}**"
    )
    lines += ["", "## Linear-readout equivalence (activations vs energy Jacobians)", ""]
    eq = payload.get("activation_energy_alignment") or {}
    lines.append(
        f"- Mean cosine (sum-pooled φ vs ∇_w E on the linear student): "
        f"{eq.get('mean_cosine', 'n/a')}"
    )
    lines.append(
        f"- Architecture note: {eq.get('architecture_note', '')}"
    )
    lines += ["", "## Methods (do not declare a winner from one seed or force RMSE alone)", ""]
    results = payload.get("results") or {}
    if results:
        lines += [
            "| method | budget | seed | tune | E MAE | F RMSE | MD fail | NVE drift | overlap-vs-random |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in results:
            lines.append(
                "| {method} | {budget} | {seed} | {tune_mode} | {energy_mae} | {force_rmse} | "
                "{md_fail} | {nve_drift} | {jaccard_vs_random} |".format(
                    method=row.get("method"),
                    budget=row.get("budget"),
                    seed=row.get("seed"),
                    tune_mode=row.get("tune_mode"),
                    energy_mae=_fmt(row.get("energy_mae")),
                    force_rmse=_fmt(row.get("force_rmse")),
                    md_fail=row.get("md_failures"),
                    nve_drift=_fmt(row.get("nve_drift")),
                    jaccard_vs_random=_fmt(row.get("jaccard_vs_random")),
                )
            )
    lines += ["", "## Baselines", ""]
    lines.append("- `unmodified_student`: the frozen starting checkpoint (no extra labels).")
    lines.append("- `stratified_random`: random within composition × condition strata.")
    lines.append(
        "- `*_largest_norm`: diagnostic ablation — select representations with largest L2."
    )
    lines += ["", "## Acquisition cost", ""]
    cost = payload.get("acquisition_cost") or {}
    for k, v in cost.items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Seed coverage prior",
        "",
        "Default information-gain seed coverage is the existing student-training",
        "set.  That is a design prior from foundation-labeled data, **not**",
        "evidence of calibrated uncertainty against the expensive reference method.",
        "",
    ]
    notes = payload.get("notes") or []
    if notes:
        lines += ["## Notes", ""]
        for n in notes:
            lines.append(f"- {n}")
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.4g}"
