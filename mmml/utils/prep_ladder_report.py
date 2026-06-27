"""Rich dashboards for liquid-prep ladder, pre-SD recovery, and MLpot SD chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mmml.data.units import (
    format_energy_kcal_ev,
    format_fmax_ev_kcal_a,
    format_grms_kcal_ev_a,
)
from mmml.utils.rich_report import emit_dashboard, emit_tagged, rich_enabled


_DIAG_NOTES = {
    "ok": "hybrid/CHARMM GRMS consistent",
    "desync_suspected": "possible hybrid/CHARMM desync — light resync recommended",
    "geometry_stress": (
        "ML geometry stress (CHARMM bonded/MM GRMS low; ELEC/VDW blocked on ML atoms)"
    ),
    "both_high": "high hybrid and CHARMM GRMS",
    "unknown": "non-finite GRMS",
}


def try_user_energy_kcal(mlpot_ctx: Any | None = None) -> float | None:
    """Best-effort CHARMM USER term (kcal/mol); None when unavailable."""
    if mlpot_ctx is None:
        return None
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm.energy as energy

        user = float(energy.get_term_by_name("USER"))
        if not np.isfinite(user):
            return None
        return user
    except Exception:
        return None


def _fmt_grms(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return format_grms_kcal_ev_a(float(value))


def _fmt_user(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return format_energy_kcal_ev(float(value))


def _fmt_fmax(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return format_fmax_ev_kcal_a(float(value))


def _fmt_box(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.3f} Å"


@dataclass
class PrepMetrics:
    hybrid_grms: float | None = None
    charmm_grms: float | None = None
    user_kcal: float | None = None
    fmax_ev_a: float | None = None
    box_L_A: float | None = None
    diag_kind: str | None = None

    @classmethod
    def from_mlpot(
        cls,
        mlpot_ctx: Any | None,
        *,
        hybrid_grms: float | None = None,
        charmm_grms: float | None = None,
        diag_kind: str | None = None,
    ) -> PrepMetrics:
        box_f: float | None = None
        if mlpot_ctx is not None:
            box = getattr(mlpot_ctx, "cubic_box_side_A", None)
            if box is None:
                box = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)
            if box is not None:
                try:
                    box_f = float(box)
                except (TypeError, ValueError):
                    box_f = None
        return cls(
            hybrid_grms=hybrid_grms,
            charmm_grms=charmm_grms,
            user_kcal=try_user_energy_kcal(mlpot_ctx),
            box_L_A=box_f,
            diag_kind=diag_kind,
        )


def metrics_mapping(metrics: PrepMetrics, *, note: str = "") -> dict[str, str]:
    """Dashboard row mapping for one prep/ladder checkpoint."""
    mapping: dict[str, str] = {
        "Hybrid GRMS": _fmt_grms(metrics.hybrid_grms),
    }
    if metrics.charmm_grms is not None and np.isfinite(metrics.charmm_grms):
        mapping["CHARMM GRMS"] = _fmt_grms(metrics.charmm_grms)
    mapping["USER energy"] = _fmt_user(metrics.user_kcal)
    if metrics.fmax_ev_a is not None and np.isfinite(metrics.fmax_ev_a):
        mapping["Max |F|"] = _fmt_fmax(metrics.fmax_ev_a)
    if metrics.box_L_A is not None and np.isfinite(metrics.box_L_A):
        mapping["Box L"] = _fmt_box(metrics.box_L_A)
    if metrics.diag_kind:
        mapping["Diag"] = _DIAG_NOTES.get(metrics.diag_kind, metrics.diag_kind)
    if note:
        mapping["Note"] = note
    return mapping


def emit_prep_checkpoint(
    title: str,
    metrics: PrepMetrics,
    *,
    note: str = "",
    quiet: bool = False,
    border_style: str = "cyan",
) -> None:
    """Single Rich dashboard for a prep/ladder/SD checkpoint."""
    if quiet or not title:
        return
    emit_dashboard(
        title,
        [("State", metrics_mapping(metrics, note=note))],
        border_style=border_style,
        quiet=quiet,
    )


def emit_prep_phase(
    phase: str,
    step: str,
    *,
    metrics: PrepMetrics | None = None,
    note: str = "",
    quiet: bool = False,
) -> None:
    """Announce a workflow phase (e.g. Pre-SD packing → monomer_repack)."""
    if quiet:
        return
    title = f"{phase} → {step}" if step else phase
    if metrics is None:
        emit_tagged("prep", title, tag_style="bold green", quiet=quiet)
        return
    emit_prep_checkpoint(title, metrics, note=note, quiet=quiet, border_style="green")


def emit_hybrid_grms_diag(
    context: str,
    *,
    hybrid: float,
    charmm: float | None = None,
    kind: str = "ok",
    ratio: float | None = None,
    user_kcal: float | None = None,
    mlpot_ctx: Any | None = None,
    quiet: bool = False,
) -> None:
    """Replace plain hybrid/CHARMM GRMS one-liners with a dashboard."""
    if quiet or not context:
        return
    note = _DIAG_NOTES.get(kind, kind)
    if kind == "desync_suspected" and ratio is not None and np.isfinite(ratio):
        note = f"{note} (ratio={float(ratio):.1f})"
    if user_kcal is None:
        user_kcal = try_user_energy_kcal(mlpot_ctx)
    metrics = PrepMetrics(
        hybrid_grms=float(hybrid),
        charmm_grms=charmm,
        user_kcal=user_kcal,
        diag_kind=kind,
    )
    if rich_enabled(quiet=quiet):
        emit_prep_checkpoint(context, metrics, note=note, quiet=quiet)
        return
    if kind == "ok":
        print(
            f"{context}: hybrid GRMS={float(hybrid):.4f} kcal/mol/Å (calculator)",
            flush=True,
        )
        return
    charmm_txt = f"{float(charmm):.4f}" if charmm is not None else "?"
    print(
        f"{context}: hybrid GRMS={float(hybrid):.4f} kcal/mol/Å (calculator); "
        f"CHARMM GRMS={charmm_txt} ({note})",
        flush=True,
    )


def emit_ase_jax_verify(
    context: str,
    *,
    ase_grms: float,
    jax_grms: float,
    ratio: float,
    consistent: bool,
    quiet: bool = False,
) -> None:
    if quiet or not context:
        return
    note = "consistent" if consistent else f"possible desync (ASE/JAX GRMS ratio={ratio:.2f})"
    metrics = PrepMetrics(hybrid_grms=jax_grms, charmm_grms=ase_grms, diag_kind="ok" if consistent else "desync_suspected")
    mapping = metrics_mapping(metrics, note=note)
    mapping["ASE hybrid GRMS"] = _fmt_grms(ase_grms)
    mapping["JAX hybrid GRMS"] = _fmt_grms(jax_grms)
    mapping.pop("Hybrid GRMS", None)
    emit_dashboard(context, [("ASE/JAX verify", mapping)], border_style="yellow", quiet=quiet)


@dataclass
class PrepLadderJournal:
    """Track density-prep ladder rounds and emit a summary table."""

    title: str = "Density prep ladder"
    max_grms: float = 0.0
    quiet: bool = False
    initial_grms: float | None = None
    current_grms: float | None = None
    max_rounds: int = 1
    current_round: int = 0
    steps: list[dict[str, str]] = field(default_factory=list)

    def begin(
        self,
        *,
        initial_grms: float,
        max_grms: float,
        max_rounds: int,
    ) -> None:
        self.initial_grms = float(initial_grms)
        self.current_grms = float(initial_grms)
        self.max_grms = float(max_grms)
        self.max_rounds = int(max_rounds)
        self.steps.clear()
        if self.quiet:
            return
        metrics = PrepMetrics(hybrid_grms=self.initial_grms)
        emit_prep_checkpoint(
            f"{self.title}: start",
            metrics,
            note=f"target GRMS ≤ {self.max_grms:.4f}; up to {self.max_rounds} round(s)",
            quiet=self.quiet,
            border_style="magenta",
        )

    def begin_round(self, round_idx: int, grms: float) -> None:
        self.current_round = int(round_idx)
        self.current_grms = float(grms)
        if self.quiet:
            return
        emit_tagged(
            "ladder",
            f"round {round_idx + 1}/{self.max_rounds} "
            f"(GRMS {_fmt_grms(grms)}, limit {_fmt_grms(self.max_grms)})",
            tag_style="bold magenta",
            quiet=self.quiet,
        )

    def record_step(
        self,
        step_label: str,
        metrics: PrepMetrics,
        *,
        status: str = "ok",
        note: str = "",
    ) -> None:
        self.current_grms = metrics.hybrid_grms
        self.steps.append(
            {
                "step": step_label,
                "status": status,
                "hybrid_grms": _fmt_grms(metrics.hybrid_grms),
                "user": _fmt_user(metrics.user_kcal),
            }
        )
        if self.quiet:
            return
        border = "green" if status == "ok" else "yellow"
        emit_prep_checkpoint(
            f"{self.title}: {step_label}",
            metrics,
            note=note or status,
            quiet=self.quiet,
            border_style=border,
        )

    def skip_step(self, step_label: str, reason: str) -> None:
        self.steps.append(
            {
                "step": step_label,
                "status": "skip",
                "hybrid_grms": _fmt_grms(self.current_grms),
                "user": "—",
            }
        )
        if self.quiet:
            return
        emit_tagged(
            "ladder",
            f"skip {step_label} ({reason})",
            tag_style="dim yellow",
            quiet=self.quiet,
        )

    def finish(self, final_grms: float, *, reason: str) -> None:
        self.current_grms = float(final_grms)
        if self.quiet:
            return
        ok = final_grms <= self.max_grms
        sections: list[tuple[str, Mapping[str, Any]]] = [
            (
                "Outcome",
                {
                    "Initial GRMS": _fmt_grms(self.initial_grms),
                    "Final GRMS": _fmt_grms(final_grms),
                    "Limit": _fmt_grms(self.max_grms),
                    "Rounds": self.current_round,
                    "Steps": len(self.steps),
                    "Status": "converged" if ok else reason,
                },
            ),
        ]
        if self.steps:
            trail = ", ".join(
                f"{row['step']} [{row['status']}]" for row in self.steps[-8:]
            )
            if len(self.steps) > 8:
                trail = f"…{len(self.steps) - 8} more, " + trail
            sections.append(("Trail", {"recent": trail}))
        emit_dashboard(
            f"{self.title}: done",
            sections,
            border_style="green" if ok else "red",
            quiet=self.quiet,
        )


def emit_sd_pass_header(
    method: str,
    pass_label: str,
    *,
    remaining: int,
    n_chunks: int,
    chunk_cap: int,
    min_chunk: int,
    metrics: PrepMetrics | None = None,
    quiet: bool = False,
) -> None:
    if quiet:
        return
    note = (
        f"{remaining} steps in up to {n_chunks} chunks "
        f"(≤{chunk_cap}/chunk, adaptive down to {min_chunk}, inbfrq=0 + UPDATE between chunks)"
    )
    title = f"{method} {pass_label}"
    if metrics is None:
        emit_tagged("SD", f"{title}: {note}", tag_style="bold blue", quiet=quiet)
        return
    mapping = metrics_mapping(metrics, note=note)
    emit_dashboard(title, [("SD pass", mapping)], border_style="blue", quiet=quiet)


def emit_sd_chunk_progress(
    method: str,
    pass_label: str,
    *,
    chunk_index: int,
    n_chunks: int,
    nstep: int,
    metrics: PrepMetrics,
    quiet: bool = False,
) -> None:
    if quiet:
        return
    title = f"{method} {pass_label}: chunk {chunk_index}/{n_chunks}"
    note = f"nstep={nstep}"
    if rich_enabled(quiet=quiet):
        emit_prep_checkpoint(title, metrics, note=note, quiet=quiet, border_style="blue")
        return
    hybrid = metrics.hybrid_grms
    hybrid_txt = f"{float(hybrid):.4f}" if hybrid is not None and np.isfinite(hybrid) else "?"
    print(
        f"{title} nstep={nstep}; hybrid GRMS={hybrid_txt} kcal/mol/Å",
        flush=True,
    )


def emit_sd_event(
    event: str,
    pass_label: str,
    *,
    metrics: PrepMetrics | None = None,
    detail: str = "",
    quiet: bool = False,
) -> None:
    if quiet:
        return
    title = f"MLpot SD {event} ({pass_label})"
    if metrics is None:
        emit_tagged("SD", f"{title}: {detail}", tag_style="bold yellow", quiet=quiet)
        return
    emit_prep_checkpoint(title, metrics, note=detail, quiet=quiet, border_style="yellow")
