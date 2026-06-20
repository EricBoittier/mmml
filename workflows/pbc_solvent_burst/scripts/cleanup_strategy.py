"""Map ``cleanup_strategy`` workflow YAML to md-system campaign job flags.

The strategy describes the hybrid recovery ladder used when geometry or handoff
quality breaks during a burst campaign:

1. **charmm_mm** — CGENFF pretreat (heat/equi/prod) before MLpot; CHARMM SD/ABNR
   during overlap rescue on PyCHARMM legs.
2. **mlpot** — bonded-MM mini + MLpot overlap rescue during PyCHARMM dynamics.
3. **jaxmd_pbc** — handoff quality gate (PBC FIRE) and CHARMM overlap rescue on
   JAX-MD bursts.

When ``cleanup_strategy`` is absent, legacy flat keys in ``config.yaml`` are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CleanupStrategy:
    name: str
    charmm_mm: dict[str, Any]
    mlpot: dict[str, Any]
    jaxmd_pbc: dict[str, Any]


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    block = raw.get(key)
    return dict(block) if isinstance(block, dict) else {}


def resolve_cleanup_strategy(cfg: dict[str, Any]) -> CleanupStrategy:
    """Resolve strategy from ``cleanup_strategy`` block or legacy flat keys."""
    raw = cfg.get("cleanup_strategy")
    if isinstance(raw, dict):
        return _from_cleanup_block(raw, cfg)
    return _from_legacy_flat_keys(cfg)


def _from_cleanup_block(raw: dict[str, Any], cfg: dict[str, Any]) -> CleanupStrategy:
    name = str(raw.get("name", "pbc_burst_default"))
    charmm_mm = _section(raw, "charmm_mm")
    mlpot = _section(raw, "mlpot")
    jaxmd_pbc = _section(raw, "jaxmd_pbc")
    # Timing keys may live at workflow root; inherit for pretreat ps.
    if "ps_heat" not in charmm_mm:
        charmm_mm.setdefault("ps_heat", cfg.get("charmm_mm_pretreat_ps_heat", cfg.get("ps_heat", 30.0)))
    if "ps_equi" not in charmm_mm:
        charmm_mm.setdefault("ps_equi", cfg.get("charmm_mm_pretreat_ps_equi", 0.0))
    if "ps_prod" not in charmm_mm:
        charmm_mm.setdefault("ps_prod", cfg.get("charmm_mm_pretreat_ps_prod", 0.0))
    return CleanupStrategy(name=name, charmm_mm=charmm_mm, mlpot=mlpot, jaxmd_pbc=jaxmd_pbc)


def _from_legacy_flat_keys(cfg: dict[str, Any]) -> CleanupStrategy:
    pretreat_on = bool(cfg.get("charmm_mm_pretreat", False))
    charmm_mm: dict[str, Any] = {
        "pretreat_on_pycharmm": pretreat_on,
        "ps_heat": cfg.get("charmm_mm_pretreat_ps_heat", cfg.get("ps_heat", 30.0)),
        "ps_equi": cfg.get("charmm_mm_pretreat_ps_equi", 0.0),
        "ps_prod": cfg.get("charmm_mm_pretreat_ps_prod", 0.0),
        "overlap_rescue_sd_steps": cfg.get("dynamics_overlap_charmm_sd_steps", 200),
        "overlap_rescue_abnr_steps": cfg.get("dynamics_overlap_charmm_abnr_steps", 400),
    }
    mlpot: dict[str, Any] = {
        "dynamics_overlap_action": cfg.get("dynamics_overlap_action", "rescue"),
        "dynamics_overlap_min_distance": cfg.get("dynamics_overlap_min_distance", 1.5),
        "dynamics_intra_min_distance": cfg.get("dynamics_intra_min_distance", 0.5),
        "dynamics_overlap_check_interval": cfg.get("dynamics_overlap_check_interval", 500),
        "bonded_mm_mini": cfg.get("bonded_mm_mini", True),
        "bonded_mm_mini_after": cfg.get("bonded_mm_mini_after", "mini,heat"),
        "bonded_mm_mini_steps": cfg.get("bonded_mm_mini_steps", 100),
        "charmm_pre_minimize": cfg.get("charmm_pre_minimize", True),
        "charmm_sd_steps": cfg.get("charmm_sd_steps", 200),
        "charmm_abnr_steps": cfg.get("charmm_abnr_steps", 400),
        "mini_nstep": cfg.get("mini_nstep", 150),
        "dcd_nsavc": cfg.get("dcd_nsavc", 500),
        "dyn_nprint": cfg.get("dyn_nprint", 500),
        "no_echeck_heat": bool(cfg.get("no_echeck_heat", False)),
    }
    jaxmd_pbc: dict[str, Any] = {
        "handoff_quality_gate": cfg.get("handoff_quality_gate", True),
        "handoff_quality_fmax_eVA": cfg.get("handoff_quality_fmax_eVA", 1.0),
        "handoff_quality_action": cfg.get("handoff_quality_action", "minimize"),
        "jaxmd_minimize_steps": cfg.get("jaxmd_minimize_steps", 200),
        "jaxmd_pbc_minimize_steps": cfg.get("jaxmd_pbc_minimize_steps", 200),
        "dynamics_overlap_action": cfg.get("jaxmd_dynamics_overlap_action", "rescue"),
        "overlap_rescue_sd_steps": cfg.get("dynamics_overlap_charmm_sd_steps", 200),
        "overlap_rescue_abnr_steps": cfg.get("dynamics_overlap_charmm_abnr_steps", 400),
        "steps_per_recording": cfg.get("steps_per_recording", 800),
        "jax_md_update_interval": cfg.get("jax_md_update_interval", 1),
    }
    return CleanupStrategy(name="legacy_flat", charmm_mm=charmm_mm, mlpot=mlpot, jaxmd_pbc=jaxmd_pbc)


def pretreat_job_flags(strategy: CleanupStrategy) -> dict[str, Any]:
    mm = strategy.charmm_mm
    if not bool(mm.get("pretreat_on_pycharmm", False)):
        return {}
    return {
        "charmm_mm_pretreat": True,
        "charmm_mm_pretreat_ps_heat": float(mm.get("ps_heat", 30.0)),
        "charmm_mm_pretreat_ps_equi": float(mm.get("ps_equi", 0.0)),
        "charmm_mm_pretreat_ps_prod": float(mm.get("ps_prod", 0.0)),
    }


def pycharmm_job_flags(strategy: CleanupStrategy) -> dict[str, Any]:
    mm = strategy.charmm_mm
    ml = strategy.mlpot
    return {
        "dynamics_overlap_action": str(ml.get("dynamics_overlap_action", "rescue")),
        "dynamics_overlap_min_distance": float(ml.get("dynamics_overlap_min_distance", 1.5)),
        "dynamics_intra_min_distance": float(ml.get("dynamics_intra_min_distance", 0.5)),
        "dynamics_overlap_check_interval": int(ml.get("dynamics_overlap_check_interval", 500)),
        "dynamics_overlap_charmm_sd_steps": int(
            mm.get("overlap_rescue_sd_steps", ml.get("dynamics_overlap_charmm_sd_steps", 200))
        ),
        "dynamics_overlap_charmm_abnr_steps": int(
            mm.get("overlap_rescue_abnr_steps", ml.get("dynamics_overlap_charmm_abnr_steps", 400))
        ),
        "bonded_mm_mini": bool(ml.get("bonded_mm_mini", True)),
        "bonded_mm_mini_after": str(ml.get("bonded_mm_mini_after", "mini,heat")),
        "bonded_mm_mini_steps": int(ml.get("bonded_mm_mini_steps", 100)),
        "charmm_pre_minimize": bool(ml.get("charmm_pre_minimize", True)),
        "charmm_sd_steps": int(ml.get("charmm_sd_steps", 200)),
        "charmm_abnr_steps": int(ml.get("charmm_abnr_steps", 400)),
        "mini_nstep": int(ml.get("mini_nstep", 150)),
        "dcd_nsavc": int(ml.get("dcd_nsavc", 500)),
        "dyn_nprint": int(ml.get("dyn_nprint", 500)),
        "no_echeck_heat": bool(ml.get("no_echeck_heat", False)),
    }


def jaxmd_job_flags(strategy: CleanupStrategy) -> dict[str, Any]:
    jd = strategy.jaxmd_pbc
    sd = int(jd.get("overlap_rescue_sd_steps", 200))
    abnr = int(jd.get("overlap_rescue_abnr_steps", 400))
    return {
        "handoff_quality_gate": bool(jd.get("handoff_quality_gate", True)),
        "handoff_quality_fmax_eVA": float(jd.get("handoff_quality_fmax_eVA", 1.0)),
        "handoff_quality_action": str(jd.get("handoff_quality_action", "minimize")),
        "jaxmd_minimize_steps": int(jd.get("jaxmd_minimize_steps", 200)),
        "jaxmd_pbc_minimize_steps": int(jd.get("jaxmd_pbc_minimize_steps", 200)),
        "dynamics_overlap_action": str(jd.get("dynamics_overlap_action", "rescue")),
        "dynamics_overlap_charmm_sd_steps": sd,
        "dynamics_overlap_charmm_abnr_steps": abnr,
        "extra_args": [
            "--steps-per-recording",
            str(int(jd.get("steps_per_recording", 800))),
            "--jax-md-update-interval",
            str(int(jd.get("jax_md_update_interval", 1))),
        ],
    }
