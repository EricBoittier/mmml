"""CHARMM MLpot registration, minimization, and MD workflow helpers.

Validated against the scripts in ``tests/functionality/mlpot/`` (ASE / callback / ENER).

Exports are resolved lazily so importing a lightweight submodule (e.g.
``mlpot.box_sizing`` / ``mlpot.overlap_guard`` for CLI ``--help``) does not pull
dynamics/setup/pandas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CharmmMmMinimizeConfig",
    "CharmmTrajectoryFiles",
    "FlatBottomSphereConfig",
    "MinimizeWithMlpotConfig",
    "apply_flat_bottom_workflow",
    "assign_boltzmann_velocities",
    "boltzmann_velocity_kwargs",
    "center_cluster_at_origin",
    "clear_mmfp_restraints",
    "MlpotContext",
    "PartialMlMmConfig",
    "apply_charmm_mm_block",
    "apply_charmm_verbosity",
    "apply_mlpot_energy_block",
    "build_cpt_equilibration_dynamics",
    "build_cpt_production_dynamics",
    "build_heat_dynamics",
    "build_nve_dynamics",
    "charmm_energy_terms",
    "compute_cpt_piston_masses",
    "parse_cubic_box_side_from_charmm_restart",
    "final_npt_segment_restart",
    "npt_restart_chain",
    "load_minimized_coordinates",
    "get_charmm_positions_array",
    "resolve_export_positions",
    "load_physnet_mlpot_bundle",
    "minimize_charmm_mm_only",
    "minimize_with_mlpot",
    "sync_charmm_positions",
    "open_minimize_dcd",
    "register_mlpot",
    "register_mlpot_partial_mm",
    "save_cluster_topology_for_vmd",
    "write_charmm_psf",
    "production_restart_chain",
    "run_dynamics",
    "run_dynamics_with_io",
    "save_minimization_results",
    "select_all_atoms",
    "select_by_resid",
    "select_by_resids",
    "select_by_seg_id",
    "disable_charmm_domdec",
    "prepare_charmm_vacuum",
    "refresh_nbonds_after_mlpot",
    "setup_default_nbonds",
    "setup_flat_bottom_sphere_mmfp",
    "write_minimized_coordinates",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "CharmmMmMinimizeConfig": (".dynamics", "CharmmMmMinimizeConfig"),
    "CharmmTrajectoryFiles": (".dynamics", "CharmmTrajectoryFiles"),
    "MinimizeWithMlpotConfig": (".dynamics", "MinimizeWithMlpotConfig"),
    "minimize_charmm_mm_only": (".dynamics", "minimize_charmm_mm_only"),
    "assign_boltzmann_velocities": (".dynamics", "assign_boltzmann_velocities"),
    "boltzmann_velocity_kwargs": (".dynamics", "boltzmann_velocity_kwargs"),
    "build_cpt_production_dynamics": (".dynamics", "build_cpt_production_dynamics"),
    "build_cpt_equilibration_dynamics": (".dynamics", "build_cpt_equilibration_dynamics"),
    "build_heat_dynamics": (".dynamics", "build_heat_dynamics"),
    "build_nve_dynamics": (".dynamics", "build_nve_dynamics"),
    "compute_cpt_piston_masses": (".dynamics", "compute_cpt_piston_masses"),
    "final_npt_segment_restart": (".dynamics", "final_npt_segment_restart"),
    "npt_restart_chain": (".dynamics", "npt_restart_chain"),
    "load_minimized_coordinates": (".dynamics", "load_minimized_coordinates"),
    "charmm_energy_terms": (".dynamics", "charmm_energy_terms"),
    "minimize_with_mlpot": (".dynamics", "minimize_with_mlpot"),
    "open_minimize_dcd": (".dynamics", "open_minimize_dcd"),
    "production_restart_chain": (".dynamics", "production_restart_chain"),
    "run_dynamics": (".dynamics", "run_dynamics"),
    "run_dynamics_with_io": (".dynamics", "run_dynamics_with_io"),
    "save_minimization_results": (".dynamics", "save_minimization_results"),
    "write_minimized_coordinates": (".dynamics", "write_minimized_coordinates"),
    "parse_cubic_box_side_from_charmm_restart": (
        ".pbc_env",
        "parse_cubic_box_side_from_charmm_restart",
    ),
    "FlatBottomSphereConfig": (".restraints", "FlatBottomSphereConfig"),
    "apply_flat_bottom_workflow": (".restraints", "apply_flat_bottom_workflow"),
    "center_cluster_at_origin": (".restraints", "center_cluster_at_origin"),
    "clear_mmfp_restraints": (".restraints", "clear_mmfp_restraints"),
    "setup_flat_bottom_sphere_mmfp": (".restraints", "setup_flat_bottom_sphere_mmfp"),
    "apply_charmm_mm_block": (".block_terms", "apply_charmm_mm_block"),
    "apply_mlpot_energy_block": (".block_terms", "apply_mlpot_energy_block"),
    "PartialMlMmConfig": (".partial_mm", "PartialMlMmConfig"),
    "register_mlpot_partial_mm": (".partial_mm", "register_mlpot_partial_mm"),
    "MlpotContext": (".setup", "MlpotContext"),
    "apply_charmm_verbosity": (".setup", "apply_charmm_verbosity"),
    "get_charmm_positions_array": (".setup", "get_charmm_positions_array"),
    "resolve_export_positions": (".setup", "resolve_export_positions"),
    "load_physnet_mlpot_bundle": (".setup", "load_physnet_mlpot_bundle"),
    "register_mlpot": (".setup", "register_mlpot"),
    "save_cluster_topology_for_vmd": (".setup", "save_cluster_topology_for_vmd"),
    "select_all_atoms": (".setup", "select_all_atoms"),
    "sync_charmm_positions": (".setup", "sync_charmm_positions"),
    "write_charmm_psf": (".setup", "write_charmm_psf"),
    "select_by_resid": (".setup", "select_by_resid"),
    "select_by_resids": (".setup", "select_by_resids"),
    "select_by_seg_id": (".setup", "select_by_seg_id"),
    "disable_charmm_domdec": (".setup", "disable_charmm_domdec"),
    "prepare_charmm_vacuum": (".setup", "prepare_charmm_vacuum"),
    "refresh_nbonds_after_mlpot": (".setup", "refresh_nbonds_after_mlpot"),
    "setup_default_nbonds": (".setup", "setup_default_nbonds"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(__all__))
