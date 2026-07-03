"""Named minimization / prep setups for DCM density comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SetupVariant:
    id: str
    description: str
    config_overrides: dict[str, Any]
    job_overrides: dict[str, Any]
    use_cleanup_strategy: bool = False
    cleanup_strategy: dict[str, Any] | None = None


_BURST_HYBRID_CLEANUP: dict[str, Any] = {
    "name": "pbc_hybrid_default",
    "charmm_mm": {
        "pretreat_on_pycharmm": True,
        "ps_heat": 1.0,
        "ps_equi": 1.0,
        "ps_prod": 1.0,
        "overlap_rescue_sd_steps": 200,
        "overlap_rescue_abnr_steps": 400,
    },
    "mlpot": {
        "no_echeck_heat": True,
        "dynamics_overlap_action": "rescue",
        "dynamics_overlap_min_distance": 1.5,
        "dynamics_intra_min_distance": 0.5,
        "dynamics_overlap_check_interval": 250,
        "dynamics_overlap_memory_handoff": True,
        "bonded_mm_mini": True,
        "bonded_recovery_backend": "jax",
        "bonded_mm_mini_after": "mini",
        "bonded_mm_mini_steps": 1000,
        "charmm_pre_minimize": True,
        "charmm_sd_steps": 1000,
        "charmm_abnr_steps": 1000,
        "mini_nstep": 500,
        "dcd_nsavc": 100,
        "dyn_nprint": 100,
        "save_run_state": True,
        "overlap_run_state_every_chunks": 4,
    },
    "jaxmd_pbc": {},
}

_RESILIENT_CLEANUP: dict[str, Any] = {
    "name": "pbc_liquid_density_default",
    "charmm_mm": {
        "pretreat_on_pycharmm": True,
        "ps_heat": 2.0,
        "ps_equi": 2.0,
        "ps_prod": 1.0,
        "overlap_rescue_sd_steps": 400,
        "overlap_rescue_abnr_steps": 800,
    },
    "mlpot": {
        "cleanup": True,
        "no_echeck_heat": True,
        "dynamics_overlap_action": "rescue",
        "dynamics_overlap_min_distance": 1.5,
        "dynamics_intra_min_distance": 0.5,
        "dynamics_overlap_check_interval": 250,
        "dynamics_overlap_memory_handoff": True,
        "bonded_mm_mini": False,
        "bonded_mm_mini_after": "mini",
        "bonded_mm_mini_steps": 1000,
        "charmm_pre_minimize": True,
        "charmm_sd_steps": 1000,
        "charmm_abnr_steps": 1000,
        "mini_nstep": 500,
        "dcd_nsavc": 100,
        "dyn_nprint": 100,
        "save_run_state": True,
        "overlap_run_state_every_chunks": 4,
    },
    "jaxmd_pbc": {},
}

_SETUPS: dict[str, SetupVariant] = {
    "minimal": SetupVariant(
        id="minimal",
        description="Packmol placement → MLpot CHARMM SD only (no prep ladder or rescue)",
        config_overrides={
            "liquid_prep": False,
            "density_prep_ladder": False,
            "calculator_pre_minimize": False,
        },
        job_overrides={
            "cleanup": False,
            "calculator_pre_minimize": False,
            "liquid_prep": False,
            "density_prep_ladder": False,
            "charmm_mm_pretreat": False,
            "charmm_pre_minimize": False,
            "bonded_mm_mini": False,
            "dynamics_overlap_action": "abort",
            "mini_nstep": 150,
            "charmm_sd_steps": 200,
            "charmm_abnr_steps": 400,
        },
    ),
    "calculator_pre_sd": SetupVariant(
        id="calculator_pre_sd",
        description="ASE hybrid FIRE/BFGS calculator pre-minimize before CHARMM SD",
        config_overrides={
            "liquid_prep": False,
            "density_prep_ladder": False,
            "calculator_pre_minimize": True,
        },
        job_overrides={
            "calculator_pre_minimize": True,
            "pre_min_steps": 200,
            "pre_min_fmax": 0.75,
            "fire_min_steps": 200,
            "fire_min_fmax": 0.5,
            "cleanup": True,
            "dynamics_overlap_action": "rescue",
            "dynamics_overlap_charmm_sd_steps": 400,
            "dynamics_overlap_charmm_abnr_steps": 800,
            "mini_nstep": 300,
            "charmm_sd_steps": 500,
            "charmm_abnr_steps": 800,
        },
    ),
    "liquid_prep_dense": SetupVariant(
        id="liquid_prep_dense",
        description="liquid_prep + density_prep_ladder (no CHARMM MM pretreat)",
        config_overrides={
            "liquid_prep": True,
            "density_prep_ladder": True,
            "density_prep_ladder_max_rounds": 5,
            "density_prep_lattice_abnr_steps": 200,
            "mini_lattice_abnr_steps": 200,
            "mc_density_equalize": True,
            "mc_density_steps": 80,
            "min_intermonomer_atom_distance": 1.0,
            "max_grms_before_dyn": 50.0,
            "calculator_pre_minimize": False,
        },
        job_overrides={
            "liquid_prep": True,
            "density_prep_ladder": True,
            "cleanup": True,
            "dynamics_overlap_action": "rescue",
            "bonded_mm_mini": True,
            "bonded_mm_mini_after": "mini",
            "bonded_mm_mini_steps": 500,
            "mini_nstep": 500,
            "charmm_sd_steps": 500,
            "charmm_abnr_steps": 800,
        },
    ),
    "burst_hybrid": SetupVariant(
        id="burst_hybrid",
        description="pbc_solvent_burst cleanup ladder (pretreat + overlap rescue)",
        config_overrides={
            "liquid_prep": False,
            "density_prep_ladder": False,
            "calculator_pre_minimize": False,
        },
        job_overrides={},
        use_cleanup_strategy=True,
        cleanup_strategy=_BURST_HYBRID_CLEANUP,
    ),
    "resilient": SetupVariant(
        id="resilient",
        description="liquid_prep_dense + calculator pre-SD + resilient cleanup ladder",
        config_overrides={
            "liquid_prep": True,
            "density_prep_ladder": True,
            "density_prep_ladder_max_rounds": 5,
            "density_prep_lattice_abnr_steps": 300,
            "mini_lattice_abnr_steps": 300,
            "mc_density_equalize": True,
            "mc_density_steps": 128,
            "min_intermonomer_atom_distance": 1.0,
            "max_grms_before_dyn": 50.0,
            "mini_box_equil_ps": 5.0,
            "calculator_pre_minimize": True,
            "bonded_mm_mini": False,
        },
        job_overrides={
            "liquid_prep": True,
            "density_prep_ladder": True,
            "calculator_pre_minimize": True,
            "bonded_mm_mini": False,
            "pre_min_steps": 200,
            "pre_min_fmax": 0.75,
            "fire_min_steps": 200,
            "fire_min_fmax": 0.5,
        },
        use_cleanup_strategy=True,
        cleanup_strategy=_RESILIENT_CLEANUP,
    ),
}


def all_setup_variants() -> dict[str, SetupVariant]:
    return dict(_SETUPS)


def resolve_setup_variant(setup_id: str) -> SetupVariant:
    key = str(setup_id).strip()
    if key not in _SETUPS:
        known = ", ".join(sorted(_SETUPS))
        raise KeyError(f"Unknown setup {setup_id!r} (known: {known})")
    return _SETUPS[key]


def merge_setup_into_config(cfg: dict[str, Any], setup_id: str) -> dict[str, Any]:
    """Return cfg copy with setup variant defaults applied."""
    variant = resolve_setup_variant(setup_id)
    # Variant defaults first; workflow config.yaml wins on conflict.
    merged = {**variant.config_overrides, **cfg}
    if variant.cleanup_strategy is not None:
        merged["cleanup_strategy"] = variant.cleanup_strategy
    return merged
