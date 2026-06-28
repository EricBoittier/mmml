"""Compare hybrid GRMS across long-range Coulomb solvers on the same geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
    CLEANUP_SUBDIR,
    PREP_LADDER_SUBDIR,
    SNAPSHOTS_JSON,
)


@dataclass(frozen=True)
class LrSolverGrmsRow:
    lr_solver: str
    jax_pme_method: str = ""
    scafacos_method: str = ""
    mm_nonbond_mode: str = ""
    run_dir: str = ""
    status: str = ""
    hybrid_grms_kcalmol_A: float | None = None

    @property
    def label(self) -> str:
        parts = [self.lr_solver]
        if self.jax_pme_method:
            parts.append(self.jax_pme_method)
        if self.scafacos_method:
            parts.append(self.scafacos_method)
        return "_".join(parts)


@dataclass
class GrmsValidationResult:
    ok: bool
    messages: list[str] = field(default_factory=list)
    rows: list[LrSolverGrmsRow] = field(default_factory=list)


def _journal_steps(journal_path: Path) -> list[dict[str, Any]]:
    if not journal_path.is_file():
        return []
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    steps = payload.get("steps")
    return steps if isinstance(steps, list) else []


def _best_grms_from_steps(steps: Iterable[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        grms = step.get("hybrid_grms_kcalmol_A")
        if grms is None:
            grms = step.get("grms_kcalmol_A")
        if grms is None:
            continue
        try:
            val = float(grms)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            values.append(val)
    if not values:
        return None
    return float(min(values))


def read_hybrid_grms_from_output_dir(output_dir: Path | str) -> float | None:
    """Best recorded hybrid GRMS from an ``md-system`` output directory."""
    root = Path(output_dir).expanduser().resolve()
    candidates: list[float] = []

    snapshots_path = root / SNAPSHOTS_JSON
    if snapshots_path.is_file():
        try:
            payload = json.loads(snapshots_path.read_text(encoding="utf-8"))
            for snap in payload.get("snapshots") or []:
                if not isinstance(snap, dict):
                    continue
                grms = snap.get("grms_kcalmol_A")
                if grms is not None and math.isfinite(float(grms)):
                    candidates.append(float(grms))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    for sub in (PREP_LADDER_SUBDIR, CLEANUP_SUBDIR):
        journal = root / sub / "journal.json"
        best = _best_grms_from_steps(_journal_steps(journal))
        if best is not None:
            candidates.append(best)

    if not candidates:
        return None
    return float(min(candidates))


def parse_solver_comparison_tsv(path: Path | str) -> list[LrSolverGrmsRow]:
    """Parse ``solver_comparison.tsv`` from ``run_dcm_long_range_workflow.sh``."""
    tsv = Path(path).expanduser().resolve()
    rows: list[LrSolverGrmsRow] = []
    with tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            run_dir = str(raw.get("run_dir") or "").strip()
            grms_raw = (raw.get("hybrid_grms_kcalmol_A") or "").strip()
            grms: float | None
            if grms_raw:
                try:
                    grms = float(grms_raw)
                except ValueError:
                    grms = None
            else:
                grms = read_hybrid_grms_from_output_dir(run_dir) if run_dir else None
            rows.append(
                LrSolverGrmsRow(
                    lr_solver=str(raw.get("lr_solver") or "").strip(),
                    jax_pme_method=str(raw.get("jax_pme_method") or "").strip(),
                    scafacos_method=str(raw.get("scafacos_method") or "").strip(),
                    mm_nonbond_mode=str(raw.get("mm_nonbond_mode") or "").strip(),
                    run_dir=run_dir,
                    status=str(raw.get("status") or "").strip(),
                    hybrid_grms_kcalmol_A=grms,
                )
            )
    return rows


def validate_lr_solver_hybrid_grms(
    rows: Sequence[LrSolverGrmsRow],
    *,
    jax_pme_rtol: float = 0.10,
) -> GrmsValidationResult:
    """Check hybrid GRMS agreement across solvers on the same certified box."""
    messages: list[str] = []
    ok = True

    ok_rows = [r for r in rows if r.status == "ok"]
    if not ok_rows:
        return GrmsValidationResult(
            ok=False,
            messages=["no successful solver runs to compare"],
            rows=list(rows),
        )

    finite = [
        r
        for r in ok_rows
        if r.hybrid_grms_kcalmol_A is not None and math.isfinite(r.hybrid_grms_kcalmol_A)
    ]
    if len(finite) < len(ok_rows):
        ok = False
        missing = len(ok_rows) - len(finite)
        messages.append(f"{missing} successful run(s) missing finite hybrid GRMS")

    jax_pme_rows = [r for r in finite if r.lr_solver == "jax_pme"]
    if len(jax_pme_rows) >= 2:
        ref = next(
            (r for r in jax_pme_rows if r.jax_pme_method == "ewald"),
            jax_pme_rows[0],
        )
        ref_grms = float(ref.hybrid_grms_kcalmol_A)  # type: ignore[arg-type]
        for row in jax_pme_rows:
            if row is ref:
                continue
            grms = float(row.hybrid_grms_kcalmol_A)  # type: ignore[arg-type]
            rtol = abs(grms - ref_grms) / max(ref_grms, 1.0e-6)
            if rtol > float(jax_pme_rtol):
                ok = False
                messages.append(
                    f"jax_pme {row.jax_pme_method or '?'} GRMS {grms:.4f} vs "
                    f"{ref.jax_pme_method or 'reference'} {ref_grms:.4f} "
                    f"(rtol {rtol:.3f} > {jax_pme_rtol:g})"
                )
        if ok:
            methods = ", ".join(
                f"{r.jax_pme_method or 'default'}={float(r.hybrid_grms_kcalmol_A):.4f}"
                for r in jax_pme_rows
            )
            messages.append(f"jax_pme methods agree within rtol {jax_pme_rtol:g}: {methods}")

    mic_rows = [r for r in finite if r.lr_solver == "mic"]
    if mic_rows and jax_pme_rows:
        mic_grms = float(mic_rows[0].hybrid_grms_kcalmol_A)  # type: ignore[arg-type]
        pme_grms = float(jax_pme_rows[0].hybrid_grms_kcalmol_A)  # type: ignore[arg-type]
        messages.append(
            f"MIC GRMS={mic_grms:.4f} kcal/mol/Å; jax_pme reference={pme_grms:.4f} "
            "(Coulomb path differs; no strict match required)"
        )

    for row in finite:
        messages.append(
            f"{row.label}: hybrid GRMS={float(row.hybrid_grms_kcalmol_A):.4f} kcal/mol/Å"
        )

    return GrmsValidationResult(ok=ok, messages=messages, rows=list(rows))


def probe_hybrid_grms_at_certified_box(
    *,
    psf: Path | str,
    crd: Path | str,
    checkpoint: Path | str,
    box_size: float,
    lr_configs: Sequence[tuple[str, str | None]],
    composition: str = "DCM:60",
    mm_switch_on: float = 9.0,
    mm_switch_width: float = 1.5,
    ml_switch_width: float = 1.0,
    verbose: bool = False,
) -> list[LrSolverGrmsRow]:
    """Measure hybrid GRMS for each ``lr_solver`` at fixed certified-box coordinates."""
    import numpy as np

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        mlpot_hybrid_grms_from_calculator,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        _register_mlpot_context,
        setup_charmm_environment,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        load_cluster_from_artifacts,
        reconcile_n_monomers_with_psf,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    setup_default_nbonds()
    base_args = argparse.Namespace(
        quiet=not verbose,
        from_psf=str(Path(psf).expanduser().resolve()),
        from_crd=str(Path(crd).expanduser().resolve()),
        skip_cluster_build=True,
        setup="pbc_npt",
        composition=composition,
        box_size=float(box_size),
        include_mm=True,
        mm_nonbond_mode="jax_mic",
        lr_solver="mic",
        jax_pme_method="ewald",
        jax_pme_sr_cutoff=6.0,
        mlpot_mm_internal_scale=0.0,
        mm_switch_on=float(mm_switch_on),
        mm_switch_width=float(mm_switch_width),
        ml_switch_width=float(ml_switch_width),
        ml_batch_size=64,
        ml_gpu_count=1,
        charmm_pre_minimize=False,
        bonded_mm_mini=False,
    )
    z, r, n_mol, _tag = load_cluster_from_artifacts(base_args)
    n_mol, _atoms_per = reconcile_n_monomers_with_psf(base_args, z, n_mol)
    setup_charmm_environment(use_pbc=True, cubic_box_side_A=float(box_size))
    sync_charmm_positions(r)
    baseline = np.asarray(get_charmm_positions_array(), dtype=np.float64).copy()
    ckpt = Path(checkpoint).expanduser().resolve()
    n_atoms = int(len(z))

    rows: list[LrSolverGrmsRow] = []
    ctx = None
    try:
        for lr_solver, jax_pme_method in lr_configs:
            sync_charmm_positions(baseline)
            if ctx is not None:
                ctx.unset()
                ctx = None
            probe_args = argparse.Namespace(**vars(base_args))
            probe_args.lr_solver = str(lr_solver)
            if jax_pme_method:
                probe_args.jax_pme_method = str(jax_pme_method)
            ctx, _pyCModel = _register_mlpot_context(
                z,
                baseline,
                ckpt,
                n_atoms,
                n_mol,
                atoms_per_monomer=getattr(probe_args, "_cluster_atoms_per_list", None),
                ml_batch_size=getattr(probe_args, "ml_batch_size", None),
                ml_gpu_count=getattr(probe_args, "ml_gpu_count", None),
                cubic_box_side_A=float(box_size),
                mlpot_use_pbc=True,
                verbose=verbose,
                args=probe_args,
                defer_jax_warmup=False,
            )
            grms = mlpot_hybrid_grms_from_calculator(ctx)
            rows.append(
                LrSolverGrmsRow(
                    lr_solver=str(lr_solver),
                    jax_pme_method=str(jax_pme_method or ""),
                    hybrid_grms_kcalmol_A=float(grms) if grms is not None else None,
                    status="ok" if grms is not None and math.isfinite(float(grms)) else "fail",
                )
            )
    finally:
        if ctx is not None:
            ctx.unset()
    return rows
