#!/usr/bin/env python3
"""Validate the unified stack beyond the sweep's endpoint-only 'drift'.

The sweep's status.json ``energy_drift_ev`` is just E[last recorded frame] -
E[first recorded frame], sampled every ``record_every`` (default 100) steps --
fine as a smoke test, but it says nothing about whether energy is actually
conserved between those samples, and nothing about whether the forces driving
the dynamics are even correct. This script, run in the exact same environment
(same checkpoint, same packmol-built system, same ml_intra + mm_nonbonded
energy terms, same neighbor_fn) as one sweep setting, does two things instead:

1. Runs NVE with ``record_every=1`` so every step's energy is visible, and
   reports the energy trace (not just first/last) -- the real conservation
   signature of a correct force field is fluctuation around a constant, not
   monotonic drift.
2. Finite-difference validates the analytic (autodiff) force against a
   central-difference numerical gradient of the same jax energy_fn, at the
   same positions/neighbor lists/dynamic_kwargs the driver actually uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=500, help="NVE steps, recorded every step")
    parser.add_argument("--fd-h", type=float, default=1e-4, help="finite-difference step (Angstrom)")
    parser.add_argument("--output-dir", type=Path, default=Path("results-validate/nve_fd"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    workflow_config = yaml.safe_load(args.workflow_config.read_text(encoding="utf-8"))
    system_cfg = workflow_config["system"]
    checkpoint_path = repo_root / workflow_config["checkpoint"]

    import os

    os.chdir(output_dir)

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams
    from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.md.assemble import _auto_neighbor_fn, build_hybrid_energy
    from mmml.md.config import EnsembleSpec, RunConfig
    from mmml.md.drivers import JaxmdDriver
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.system import SystemSpec

    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

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
    terms = ("ml_intra", "mm_nonbonded")
    energy = build_hybrid_energy(system, terms, ctx)

    run_config = RunConfig(
        system=spec, terms=terms, ensemble=EnsembleSpec(ensemble="nve", n_steps=0),
        backend="jaxmd", sampler="md", seed=args.seed,
    )
    neighbor_fn = _auto_neighbor_fn(system, energy, run_config)

    # --- 1. Finite-difference force validation, at the *initial* positions and
    # the *same* dynamic_kwargs (neighbor lists) the driver would use there. ---
    energy_fn = energy.as_jax_energy_fn()
    R0 = np.asarray(system.R, dtype=np.float64)
    box = None if system.box is None else np.asarray(system.box, dtype=np.float64)
    dyn0 = neighbor_fn(R0, box) if neighbor_fn is not None else {}
    dyn0 = {k: jnp.asarray(v) for k, v in dict(dyn0).items()}

    def e_of(R):
        return energy_fn(jnp.asarray(R), **dyn0)

    analytic_force = -np.asarray(jax.grad(e_of)(jnp.asarray(R0)))

    h = args.fd_h
    numeric_force = np.zeros_like(R0)
    n_atoms = R0.shape[0]
    for i in range(n_atoms):
        for d in range(3):
            Rp = R0.copy()
            Rp[i, d] += h
            Rm = R0.copy()
            Rm[i, d] -= h
            e_plus = float(e_of(Rp))
            e_minus = float(e_of(Rm))
            numeric_force[i, d] = -(e_plus - e_minus) / (2.0 * h)

    abs_err = np.abs(analytic_force - numeric_force)
    denom = np.maximum(np.abs(analytic_force), 1e-8)
    rel_err = abs_err / denom
    fd_report = {
        "fd_h_angstrom": h,
        "n_atoms": n_atoms,
        "max_abs_error_ev_per_A": float(abs_err.max()),
        "rms_abs_error_ev_per_A": float(np.sqrt(np.mean(abs_err**2))),
        "max_rel_error": float(rel_err.max()),
        "mean_rel_error": float(rel_err.mean()),
        "max_force_component_ev_per_A": float(np.abs(analytic_force).max()),
    }
    print("Finite-difference force check:", json.dumps(fd_report, indent=2))

    # --- 2. Dense-recorded NVE trajectory: real conservation signature. ---
    driver = JaxmdDriver(
        neighbor_fn=neighbor_fn,
        record_every=1,
        output_path=output_dir / "trajectory_dense.npz",
    )
    ensemble = EnsembleSpec(
        ensemble="nve",
        space="pbc",
        temperature_K=float(system_cfg.get("temperature", 300.0)),
        dt_fs=float(system_cfg["dt_fs"]),
        n_steps=args.n_steps,
        params={"float64": True, "seed": args.seed},
    )
    traj = driver.run(system, energy, ensemble)
    energies = np.asarray(traj.metadata["energies"], dtype=float)

    nve_report = {
        "n_steps": args.n_steps,
        "n_frames": int(traj.n_frames),
        "energy_initial_ev": float(energies[0]),
        "energy_final_ev": float(energies[-1]),
        "energy_mean_ev": float(energies.mean()),
        "energy_std_ev": float(energies.std()),
        "endpoint_drift_ev": float(energies[-1] - energies[0]),
        "max_abs_deviation_from_mean_ev": float(np.max(np.abs(energies - energies.mean()))),
        "drift_per_step_ev": float((energies[-1] - energies[0]) / max(args.n_steps, 1)),
    }
    print("Dense-recorded NVE energy trace:", json.dumps(nve_report, indent=2))

    (output_dir / "report.json").write_text(
        json.dumps({"finite_difference": fd_report, "nve_energy_trace": nve_report}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    np.savetxt(output_dir / "energy_trace.csv", energies, header="energy_ev", comments="")


if __name__ == "__main__":
    main()
