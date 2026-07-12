#!/usr/bin/env python3
"""Run one (setting, seed) of the mixed-system / calculator sweep.

A "setting" is a named entry in config.yaml combining: a system (plain MM
water box, or the mixed ML-peptide + ML/MM-water system), a calculator
(checkpoint + electrostatics_damping_sigma + disable_physnet_point_coulomb),
and mm_nonbonded cutoffs. Every setting runs 10000-step NVE with
``JaxmdDriver``'s default ``record_every=100`` (100 recorded energy samples),
so the resulting drift reflects real trajectory behavior, not a 2-3 point
endpoint delta.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-config", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow_config = yaml.safe_load(args.workflow_config.read_text(encoding="utf-8"))
    spec = workflow_config["settings"][args.setting]
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(spec.get("n_steps", workflow_config["default_n_steps"]))
    dt_fs = float(spec.get("dt_fs", workflow_config["default_dt_fs"]))
    temperature_K = float(spec.get("temperature_K", workflow_config["default_temperature_K"]))
    checkpoint_key = spec["checkpoint"]
    checkpoint_rel = workflow_config["checkpoints"][checkpoint_key]
    calculator_kwargs = dict(spec.get("calculator", {}))
    mm_nonbonded_kwargs = dict(spec.get("mm_nonbonded", {}))

    run_config_record = {
        "setting": args.setting,
        "seed": args.seed,
        "system": spec["system"],
        "checkpoint": checkpoint_key,
        "calculator_kwargs": calculator_kwargs,
        "mm_nonbonded_kwargs": mm_nonbonded_kwargs,
        "n_steps": n_steps,
        "dt_fs": dt_fs,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config_record, indent=2) + "\n", encoding="utf-8"
    )

    os.chdir(output_dir)

    import jax

    jax.config.update("jax_enable_x64", True)

    import numpy as np

    from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams
    from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.md.assemble import assemble_and_run, build_system
    from mmml.md.config import EnsembleSpec, RunConfig
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.system import SystemSpec

    status: dict[str, object] = {
        "setting": args.setting,
        "seed": args.seed,
        "system": spec["system"],
        "checkpoint": checkpoint_key,
        "calculator_kwargs": calculator_kwargs,
        "mm_nonbonded_kwargs": mm_nonbonded_kwargs,
        "n_steps": n_steps,
        "completed": False,
        "error": None,
    }

    started = time.monotonic()
    try:
        if not ensure_pycharmm_loaded():
            raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

        checkpoint_path = repo_root / checkpoint_rel
        calc = create_calculator_from_checkpoint(str(checkpoint_path), **calculator_kwargs)
        model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
        params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
        ctx = EnergyContext(model=model, params=params, options=mm_nonbonded_kwargs)

        if spec["system"] == "water_box":
            sys_spec = SystemSpec(
                builder="packmol",
                composition=spec["composition"],
                box_size=float(spec["box_size"]),
                seed=args.seed,
            )
            system = build_packmol_system_with_ffparams(sys_spec)
            terms: tuple[str, ...] = ("ml_intra", "mm_nonbonded")
            term_kwargs: dict[str, dict] = {}
        elif spec["system"] == "peptide_water":
            sys_spec = SystemSpec(
                builder="peptide_water",
                n_molecules=int(spec["n_waters"]),
                box_size=float(spec["box_size"]),
                seed=args.seed,
            )
            system = build_system(sys_spec)
            core_indices = system.monomer_indices[0]
            term_kwargs = {
                "ml_pep_water": {
                    "core_indices": core_indices,
                    "group_indices": system.water_indices,
                }
            }
            cutoff = spec.get("peptide_water_ml_cutoff_A")
            if cutoff is not None:
                term_kwargs["ml_pep_water"]["interaction_cutoff_A"] = float(cutoff)
            terms = ("ml_intra", "ml_pep_water", "mm_nonbonded")
            if spec.get("peptide_water_ml_core_vdw", False):
                term_kwargs["vdw_core"] = {
                    "n_core_atoms": len(core_indices),
                    "group_indices": system.water_indices,
                    "lj_epsilon": system.ff_params.epsilon,
                    "lj_rmin_half": system.ff_params.rmin_half,
                    "cutoff_A": float(spec.get("peptide_water_ml_core_cutoff_A", 3.0)),
                    "switch_width_A": float(spec.get("peptide_water_ml_core_switch_width_A", 1.0)),
                }
                terms = terms + ("vdw_core",)
        else:
            raise ValueError(f"unknown system {spec['system']!r}")

        status["atom_count"] = system.n_atoms

        ensemble = EnsembleSpec(
            ensemble="nve",
            space="pbc",
            temperature_K=temperature_K,
            dt_fs=dt_fs,
            n_steps=n_steps,
            params={"float64": True, "seed": args.seed},
        )
        run_config = RunConfig(
            system=sys_spec,
            terms=terms,
            ensemble=ensemble,
            backend="jaxmd",
            sampler="md",
            seed=args.seed,
            output_dir=output_dir,
        )

        traj = assemble_and_run(run_config, system=system, ctx=ctx, term_kwargs=term_kwargs)

        energies = np.asarray(traj.metadata["energies"], dtype=float)
        finite = bool(np.all(np.isfinite(energies)))
        status.update(
            {
                "completed": finite,
                "n_frames": int(traj.n_frames),
                "energy_initial_ev": float(energies[0]),
                "energy_final_ev": float(energies[-1]),
                "energy_mean_ev": float(energies.mean()),
                "energy_std_ev": float(energies.std()),
                "energy_drift_ev": float(energies[-1] - energies[0]),
                "energy_max_abs_deviation_ev": float(np.max(np.abs(energies - energies.mean()))),
            }
        )
        if not finite:
            status["error"] = "non-finite energy encountered"
    except Exception as exc:  # noqa: BLE001 - want every failure captured in status.json
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["elapsed_seconds"] = round(time.monotonic() - started, 3)

    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 0 if status["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
