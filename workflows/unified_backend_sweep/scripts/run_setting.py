#!/usr/bin/env python3
"""Run one (backend, seed) setting of the unified mmml.md pipeline.

A "backend" here is a driver/sampler + ensemble combination reachable through
``mmml.md.assemble.assemble_and_run``: FIRE minimization, NVE, NVT, and NPT via
the ``JaxmdDriver``, plus Metropolis MC via the ``RigidBodySampler``. Every
backend builds the *same* small TIP3-water box (via the packmol composition
builder, same path as ``mmml.cli.run.md_system_unified``) and scores it with
the same ``ml_intra`` + ``mm_nonbonded`` terms, so the sweep is a like-for-like
smoke test across everything the unified stack currently supports.
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
    parser.add_argument("--backend", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow_config = yaml.safe_load(args.workflow_config.read_text(encoding="utf-8"))
    backend_cfg = workflow_config["backends"][args.backend]
    system_cfg = workflow_config["system"]
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_record = {
        "backend": args.backend,
        "description": backend_cfg["description"],
        "sampler": backend_cfg.get("sampler", "md"),
        "ensemble": backend_cfg.get("ensemble", "nve"),
        "n_steps": backend_cfg["n_steps"],
        "seed": args.seed,
        **system_cfg,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config_record, indent=2) + "\n", encoding="utf-8"
    )

    # cwd matters: packmol/CHARMM scratch files land relative to it, and
    # workers may run from a shared filesystem, so isolate each setting.
    os.chdir(output_dir)

    import jax

    jax.config.update("jax_enable_x64", True)

    import numpy as np

    from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams
    from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.md.assemble import assemble_and_run
    from mmml.md.config import EnsembleSpec, RunConfig
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.system import SystemSpec

    status: dict[str, object] = {
        "backend": args.backend,
        "description": backend_cfg["description"],
        "seed": args.seed,
        "dt_fs": system_cfg["dt_fs"],
        "n_steps": backend_cfg["n_steps"],
        "completed": False,
        "error": None,
    }

    started = time.monotonic()
    try:
        if not ensure_pycharmm_loaded():
            raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

        checkpoint_path = repo_root / workflow_config["checkpoint"]
        calc = create_calculator_from_checkpoint(str(checkpoint_path))
        model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
        params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
        ctx = EnergyContext(model=model, params=params)

        spec = SystemSpec(
            builder="packmol",
            composition=system_cfg["composition"],
            box_size=float(system_cfg["box_size"]),
            seed=args.seed,
        )
        system = build_packmol_system_with_ffparams(spec)
        status["atom_count"] = system.n_atoms

        sampler = backend_cfg.get("sampler", "md")
        ensemble = EnsembleSpec(
            ensemble=backend_cfg.get("ensemble", "nve"),
            space="pbc",
            temperature_K=float(system_cfg.get("temperature", 300.0)),
            pressure_bar=float(system_cfg.get("pressure", 1.0)),
            dt_fs=float(system_cfg["dt_fs"]),
            n_steps=int(backend_cfg["n_steps"]),
            params={"float64": True, "seed": args.seed},
        )
        run_config = RunConfig(
            system=spec,
            terms=("ml_intra", "mm_nonbonded"),
            ensemble=ensemble,
            backend="jaxmd",
            sampler=sampler,
            seed=args.seed,
        )

        # Let assemble_and_run dispatch on config.sampler/config.backend itself:
        # RigidBodySampler.run(system, energy, config: RunConfig) has a different
        # signature than Driver.run(system, energy, ensemble, *, on_overlap), so a
        # custom sampler/driver must not be passed here for either path.
        traj = assemble_and_run(run_config, system=system, ctx=ctx)

        energies = np.asarray(traj.metadata["energies"], dtype=float)
        finite = bool(np.all(np.isfinite(energies)))
        status.update(
            {
                "completed": finite,
                "n_frames": int(traj.n_frames),
                "energy_initial_ev": float(energies[0]),
                "energy_final_ev": float(energies[-1]),
                "energy_drift_ev": float(energies[-1] - energies[0]),
                "energy_max_abs_ev": float(np.max(np.abs(energies))),
            }
        )
        if sampler == "rigid":
            status["acceptance_ratio"] = traj.metadata.get("acceptance_ratio")
        if not finite:
            status["error"] = "non-finite energy encountered"
    except Exception as exc:  # noqa: BLE001 - want every failure captured in status.json
        status["error"] = f"{type(exc).__name__}: {exc}"
    status["elapsed_seconds"] = round(time.monotonic() - started, 3)

    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 0 if status["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
