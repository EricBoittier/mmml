"""Tier 3 DOMDEC + MLpot readiness survey (production blocked on open risks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_info import (
    DomdecApiSurvey,
    survey_domdec_api,
)


@dataclass
class Tier3DomdecReport:
    """Readiness report for DOMDEC coexistence with MLpot (Tier 3).

    ``ok`` means the survey itself completed. ``blocked`` means Tier 3 is not
    production-ready (DOMDEC+MLpot/JAX stability and related open questions).
    Strict mode intentionally maps a blocked survey to ``ok=False``.
    """

    ok: bool
    tier: str = "tier3_domdec_mlpot"
    blocked: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    survey: DomdecApiSurvey | None = None
    spike_doc: str = "tests/functionality/mlpot/SPATIAL_MPI_DOMDEC.md"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "tier": self.tier,
            "blocked": self.blocked,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "spike_doc": self.spike_doc,
        }
        if self.survey is not None:
            payload["survey"] = {
                "charmm_fortran_domdec": self.survey.charmm_fortran_domdec,
                "pycharmm_domdec_script": self.survey.pycharmm_domdec_script,
                "pycharmm_local_atom_api": self.survey.pycharmm_local_atom_api,
                "pycharmm_ghost_atom_api": self.survey.pycharmm_ghost_atom_api,
                "mmml_disable_domdec_for_mlpot": self.survey.mmml_disable_domdec_for_mlpot,
                "recommended_phase2_grid": self.survey.recommended_phase2_grid,
                "halo_width_formula": self.survey.halo_width_formula,
                "open_questions": list(self.survey.open_questions),
            }
        return payload


def validate_tier3_domdec_env(*, strict: bool = False) -> Tier3DomdecReport:
    """Survey DOMDEC + MLpot blockers.

    Non-strict mode is informational: the check exits successfully once the
    survey runs, while ``blocked=True`` continues to mark production Tier 3 as
    unavailable until DOMDEC+MLpot engineering risks are cleared.
    """
    survey = survey_domdec_api()
    report = Tier3DomdecReport(ok=False, blocked=True, survey=survey)

    if not survey.pycharmm_local_atom_api or not survey.pycharmm_ghost_atom_api:
        report.errors.append(
            "PyCHARMM does not expose per-rank local/ghost atom maps; Tier 3 DOMDEC+MLpot is blocked"
        )
    else:
        report.errors.append(
            "Tier 3 DOMDEC+MLpot production is blocked pending DOMDEC/JAX stability "
            "and global energy reduction (see survey open_questions)"
        )

    if survey.mmml_disable_domdec_for_mlpot:
        report.warnings.append(
            "MMML can send ``domdec off`` via MMML_FORCE_DOMDEC_OFF for MLpot stability; use Tier 2 spatial MPI instead"
        )

    if survey.pycharmm_domdec_script:
        report.warnings.append(
            "Only ``domdec off`` is exposed via pycharmm.lingo; no domain metadata API"
        )

    report.warnings.append(
        f"Manual spike procedure: {report.spike_doc}"
    )

    if strict:
        report.ok = False
    else:
        # Informational pass: survey completed; production Tier 3 still blocked.
        report.ok = True

    return report


def render_tier3_report(report: Tier3DomdecReport) -> str:
    check_state = "survey completed" if report.ok else "survey failed or strict blocker"
    lines = [
        "MMML Tier 3 DOMDEC + MLpot survey",
        "================================",
        f"Production status: {'BLOCKED' if report.blocked else 'OK'}",
        f"Check status: {check_state} (ok={report.ok})",
        f"Spike doc: {report.spike_doc}",
    ]
    if report.survey is not None:
        s = report.survey
        lines.extend(
            [
                "",
                "PyCHARMM / CHARMM capability:",
                f"  Fortran DOMDEC symbols: {s.charmm_fortran_domdec}",
                f"  domdec script control: {s.pycharmm_domdec_script}",
                f"  per-rank local atom API: {s.pycharmm_local_atom_api}",
                f"  ghost atom API: {s.pycharmm_ghost_atom_api}",
                f"  MMML domdec-off command available for MLpot guard: {s.mmml_disable_domdec_for_mlpot}",
                f"  Phase 2 fallback grid: {s.recommended_phase2_grid}",
                f"  Halo width: {s.halo_width_formula}",
            ]
        )
        if s.open_questions:
            lines.extend(["", "Open questions:"] + [f"  - {q}" for q in s.open_questions])
    if report.warnings:
        lines.extend(["", "Warnings:"] + [f"  - {w}" for w in report.warnings])
    if report.errors:
        lines.extend(["", "Blockers:"] + [f"  - {e}" for e in report.errors])
    lines.extend(
        [
            "",
            "Until Tier 3 DOMDEC+MLpot is production-ready, use Tier 2:",
            "  MMML_MLPOT_SPATIAL_MPI=1 --ml-spatial-mpi --ml-gpu-count 1",
        ]
    )
    return "\n".join(lines)
