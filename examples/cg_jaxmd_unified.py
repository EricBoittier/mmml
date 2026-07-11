#!/usr/bin/env python
"""Thin front-end for cg_jaxmd-style peptide-water ML/MM simulations.

Reads a cg_jaxmd JSON config (see ``examples/cg_jaxmd.example.json``) and runs
its fire -> nvt -> nve phases through the unified ``mmml.md`` pipeline:

    JSON config --runconfig_from_cg_config--> RunConfig --assemble_and_run-->
    PeptideWaterSystemBuilder -> HybridEnergy -> JaxmdDriver

This is the validated replacement for the *simulation loop* of
``examples/cg_jaxmd.py`` (build + energy + dynamics), not a full feature-parity
port. Deliberately NOT yet supported (raise clearly rather than silently
diverge from the legacy script):

- ``constrain_phi_psi`` — needs phi/psi dihedral-index derivation from the
  peptide sequence/PSF, not yet wired into this front-end.
- CHARMM-side H-X bond / overlap repair, DCD/ASE trajectory export, the
  detailed force/energy diagnostics the original script prints.

See ``docs/md-cg-unification-design.md`` (§0, §9, §11) and
``docs/md-cg-unification-handoff.md`` for the surrounding architecture and
what remains.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

# jax_enable_x64 MUST be set before any jax array is created (matches the
# original examples/cg_jaxmd.py). Energy terms and the driver assume float64.
import jax

jax.config.update("jax_enable_x64", True)

from mmml.md.assemble import assemble_and_run, build_system  # noqa: E402
from mmml.md.energy.registry import EnergyContext  # noqa: E402
from mmml.md.results import Trajectory  # noqa: E402
from mmml.md.system import MolecularSystem  # noqa: E402

_UNSUPPORTED_TOGGLES = ("constrain_phi_psi",)


def _load_model(checkpoint_path: Path) -> tuple[Any, Any]:
    """Load a physnet/spooky model + params from a portable JSON checkpoint."""
    from mmml.interfaces.calculators.simple_inference import (
        create_calculator_from_checkpoint,
    )

    calc = create_calculator_from_checkpoint(str(checkpoint_path))
    model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
    params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
    if model is None or params is None:
        raise ValueError(f"could not extract model/params from checkpoint {checkpoint_path}")
    return model, params


def term_kwargs_from_cg_config(cfg: Mapping[str, Any], system: MolecularSystem) -> dict[str, dict]:
    """Build per-term constructor kwargs that need the *built* system.

    ``terms_from_cg_config`` (in ``mmml.md.lowering``) only selects term
    *names* from the config, since it is pure and runs before the system
    exists. Terms whose constructor needs concrete atom indices (``ml_pep_water``,
    ``vdw_core``, ``smd``) are wired here, after ``build_system``.
    """
    kwargs: dict[str, dict] = {}
    core_indices = system.monomer_indices[0] if system.monomer_indices else None

    if cfg.get("peptide_water_ml", False):
        if core_indices is None or not system.water_indices:
            raise ValueError("peptide_water_ml requires a built peptide + water system")
        term_kw = {"core_indices": core_indices, "group_indices": system.water_indices}
        cutoff = cfg.get("peptide_water_ml_cutoff_A")
        if cutoff is not None:
            term_kw["interaction_cutoff_A"] = float(cutoff)
        kwargs["ml_pep_water"] = term_kw

        if cfg.get("peptide_water_ml_core_vdw", False):
            if system.ff_params is None:
                raise ValueError("peptide_water_ml_core_vdw requires FFParams (CHARMM-built system)")
            kwargs["vdw_core"] = {
                "n_core_atoms": len(core_indices),
                "group_indices": system.water_indices,
                "lj_epsilon": system.ff_params.epsilon,
                "lj_rmin_half": system.ff_params.rmin_half,
                "cutoff_A": float(cfg.get("peptide_water_ml_core_cutoff_A", 3.0)),
                "switch_width_A": float(cfg.get("peptide_water_ml_core_switch_width_A", 1.0)),
            }

    if cfg.get("smd_enable", False):
        kwargs["smd"] = {
            "atom_i": int(cfg["smd_atom_i"]),
            "atom_j": int(cfg["smd_atom_j"]),
            "k_ev_per_A2": float(cfg.get("smd_k_ev_a2", cfg.get("smd_k", 1.0))),
            "target": cfg.get("smd_d"),
        }

    return kwargs


def check_cg_config_supported(cfg: Mapping[str, Any]) -> None:
    """Raise clearly (before any CHARMM build) for toggles not yet wired here."""
    for toggle in _UNSUPPORTED_TOGGLES:
        if cfg.get(toggle, False):
            raise NotImplementedError(
                f"cg_jaxmd_unified.py does not yet support {toggle!r}; "
                "see the module docstring."
            )


def run_cg_config(cfg: Mapping[str, Any], phases: tuple[str, ...] = ("fire", "nvt", "nve")) -> dict[str, Trajectory]:
    """Run the requested phases in sequence, carrying positions forward.

    Velocities are NOT carried between phases (each phase re-initializes its
    own integrator state) — a simplification vs. the legacy script's continuous
    fire->nvt->nve handoff; acceptable for minimization->equilibration->
    production staging, but not a substitute for a true restart/handoff.
    """
    check_cg_config_supported(cfg)

    from mmml.md.lowering import runconfig_from_cg_config

    first_config = runconfig_from_cg_config(cfg, phase=phases[0])
    if first_config.checkpoint is None:
        raise ValueError("cg config must set 'checkpoint' (or 'peptide_checkpoint')")
    model, params = _load_model(first_config.checkpoint)
    ctx = EnergyContext(model=model, params=params)

    system = build_system(first_config.system)
    trajectories: dict[str, Trajectory] = {}

    for phase in phases:
        run_config = runconfig_from_cg_config(cfg, phase=phase)
        if run_config.ensemble.n_steps <= 0:
            continue  # phase not requested in this config
        term_kwargs = term_kwargs_from_cg_config(cfg, system)
        traj = assemble_and_run(run_config, system=system, ctx=ctx, term_kwargs=term_kwargs)
        trajectories[phase] = traj
        final_R = traj.metadata["positions"][-1]
        system = dataclasses.replace(system, R=final_R)

    return trajectories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="cg_jaxmd-style JSON config")
    parser.add_argument(
        "--phases", default="fire,nvt,nve",
        help="comma-separated subset/order of {fire,nvt,nve} to run",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    phases = tuple(p.strip() for p in args.phases.split(",") if p.strip())

    trajectories = run_cg_config(cfg, phases=phases)
    for phase, traj in trajectories.items():
        energies = traj.metadata["energies"]
        print(
            f"[{phase}] {traj.n_frames} frames, "
            f"E0={energies[0]:.4f} eV, Efinal={energies[-1]:.4f} eV"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
