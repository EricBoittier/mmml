#!/usr/bin/env python3
"""Driver for ``peptide_gas.smoke``: alanine and trialanine in gas phase.

Computes the five acceptance checks declared for this task in ``campaign.yaml``:

    build_audit, finite_energy_force, fd_force_pass,
    minimized_structure, short_nve_nvt_pass

CHARMM runs in a **child process** (``--_worker``). PyCHARMM can take down the
interpreter with SIGSEGV, and a driver that died with it would leave no receipt
at all -- the campaign would read that as "never ran" rather than "crashed".
Isolating the child turns a crash into a measured, reported result.

As in the liquids driver: every number here is measured from something this run
produced, and a check that cannot be evaluated is a failed check, never a
passing one.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_lib as lib  # noqa: E402

# Gas-phase peptides, as ACE-capped CGenFF/protein sequences.
PEPTIDES: dict[str, list[str]] = {
    "alanine": ["ALA"],
    "trialanine": ["ALA", "ALA", "ALA"],
}

# Post-minimization RMS gradient. Above this the structure is not a minimum and
# dynamics will be sampling a strained geometry rather than the PES basin.
MAX_GRMS_KCAL_MOL_A = 1.0

# Analytic vs finite-difference force agreement. 1e-4 A is too small: the energy
# difference it produces is at the edge of CHARMM's reported precision, so the
# central difference is dominated by round-off. 1e-3 A gives ~1e-4 agreement.
FD_MAX_ABS_ERROR_KCAL_MOL_A = 1.0e-2
FD_STEP_A = 1.0e-3

# NVE energy drift budget and NVT temperature tolerance.
MAX_NVE_DRIFT_KCAL_MOL = 5.0
NVT_TEMPERATURE_TOLERANCE_K = 100.0


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _exit_detail(returncode: int) -> str:
    if returncode < 0:
        return f"killed by signal {-returncode} ({signal.Signals(-returncode).name})"
    if returncode > 128:
        sig = returncode - 128
        try:
            return f"exit {returncode} (signal {sig}: {signal.Signals(sig).name})"
        except ValueError:
            return f"exit {returncode}"
    return f"exit {returncode}"


# ---------------------------------------------------------------------------
# Child: everything that touches CHARMM
# ---------------------------------------------------------------------------


def _worker(system: str, workdir: Path, temperature: float, steps: int) -> int:
    """Build, minimize, probe forces and run short NVE/NVT. Emits JSON on stdout."""
    import numpy as np

    from mmml.interfaces.pycharmmInterface.peptide_builder import build_peptide_in_charmm

    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"system": system}

    sequence = PEPTIDES[system]
    build = build_peptide_in_charmm(
        sequence, minimize=True, mini_steps=2000, workdir=workdir
    )

    out["n_atoms"] = int(build.n_atoms)
    out["sequence"] = list(build.sequence)
    out["psf_path"] = str(build.psf_path) if build.psf_path else None
    out["pdb_path"] = str(build.pdb_path) if build.pdb_path else None
    positions = np.asarray(build.positions, dtype=float)
    out["positions_finite"] = bool(np.isfinite(positions).all())

    import pandas as pd

    import pycharmm.coor as coor
    import pycharmm.energy as energy_mod

    def _set_positions(p: "np.ndarray") -> None:
        coor.set_positions(pd.DataFrame(p, columns=["x", "y", "z"]))

    def _energy_and_forces() -> tuple[float, "np.ndarray"]:
        energy_mod.show()
        e = float(energy_mod.get_total())
        # coor.get_forces() returns the *gradient* dE/dx, not the force. Verified
        # against a central finite difference: every component came back at
        # exactly ratio -1.000 with magnitudes agreeing to 4 decimals. Negate it.
        #
        # (Use coor.get_forces(), NOT import_pycharmm.get_forces_pycharmm(),
        # which hands back the coordinate array unchanged.)
        forces = -np.asarray(coor.get_forces(), dtype=float)
        return e, forces

    # The builder's own minimization leaves GRMS around 1.3 kcal/mol/A, which is
    # not a minimum. Converge it properly with ABNR before probing forces and
    # launching dynamics, or the run samples a strained geometry.
    import pycharmm.minimize as minimize_mod

    minimize_mod.run_abnr(nstep=2000, tolenr=1.0e-6, tolgrd=1.0e-4)
    positions = np.asarray(coor.get_positions(), dtype=float)
    out["positions_finite"] = bool(np.isfinite(positions).all())

    e0, f0 = _energy_and_forces()
    out["energy_kcal_mol"] = e0
    out["max_force_kcal_mol_A"] = float(np.abs(f0).max()) if f0.size else None
    out["energy_finite"] = bool(math.isfinite(e0))
    out["forces_finite"] = bool(np.isfinite(f0).all())
    out["grms_kcal_mol_A"] = (
        float(np.sqrt((f0**2).sum() / len(f0))) if len(f0) else float("nan")
    )

    # ---- finite-difference forces on a few atoms ---------------------------
    # F = -dE/dx. Probing every atom would be slow and adds nothing: a wiring or
    # sign error shows up on the first few.
    n_probe = min(3, positions.shape[0])
    fd_errors: list[float] = []
    for i in range(n_probe):
        for axis in range(3):
            plus = positions.copy()
            plus[i, axis] += FD_STEP_A
            _set_positions(plus)
            e_plus, _ = _energy_and_forces()

            minus = positions.copy()
            minus[i, axis] -= FD_STEP_A
            _set_positions(minus)
            e_minus, _ = _energy_and_forces()

            fd_force = -(e_plus - e_minus) / (2.0 * FD_STEP_A)
            fd_errors.append(abs(fd_force - float(f0[i, axis])))
    _set_positions(positions)

    out["fd_atoms_probed"] = n_probe
    out["fd_max_abs_error_kcal_mol_A"] = max(fd_errors) if fd_errors else None

    # ---- short NVE and NVT -------------------------------------------------
    import os

    import pycharmm
    import pycharmm.settings as settings_mod

    def _parse_dyna_temperatures(text: str) -> list[float]:
        """Temperatures from CHARMM's ``DYNA>`` trace.

        Read from the trace rather than dynamics.get_velos(), which returns an
        empty array in this pycharmm build (hence the earlier nan). On a
        ``DYNA>`` line TEMPerature is the last column.
        """
        temps: list[float] = []
        for line in text.splitlines():
            if not line.startswith("DYNA>"):
                continue
            parts = line.split()
            try:
                temps.append(float(parts[-1]))
            except (IndexError, ValueError):
                continue
        return temps

    def _run(ensemble: str) -> dict[str, Any]:
        # Same construction the rest of the repo uses (mlpot/dynamics.py):
        # pycharmm.DynamicsScript, not a bare dynamics.run().
        kw: dict[str, Any] = {
            "leap": True,
            "verlet": True,
            "start": True,
            "nstep": steps,
            "timest": 0.0005,
            "firstt": temperature,
            "finalt": temperature,
            "nprint": max(1, steps // 10),
            "inbfrq": 50,
            "ihbfrq": 0,
            "nsavc": 0,
            "echeck": -1.0,
        }
        if ensemble == "nvt":
            kw.update({"hoover": True, "reft": temperature, "tmass": 250.0})

        # The peptide builder lowers CHARMM's print level, which suppresses the
        # DYNA trace entirely -- without raising it there is nothing to parse and
        # the temperature silently reads back as nan. Raise it for the run only.
        #
        # CHARMM writes the trace to fd 1 from Fortran, so redirect_stdout cannot
        # catch it; point fd 1 at a file and parse the trace back out.
        trace_path = workdir / f"dyna_{ensemble}.log"
        previous_verbosity = settings_mod.set_verbosity(5)
        sys.stdout.flush()
        saved_fd = os.dup(1)
        trace_fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(trace_fd, 1)
            pycharmm.DynamicsScript(**kw).run()
        finally:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            os.close(trace_fd)
            settings_mod.set_verbosity(previous_verbosity)

        temps = _parse_dyna_temperatures(
            trace_path.read_text(errors="replace")
        )
        e, _ = _energy_and_forces()
        return {
            "final_energy_kcal_mol": e,
            "final_temperature_K": temps[-1] if temps else float("nan"),
            "mean_temperature_K": (sum(temps) / len(temps)) if temps else float("nan"),
            "n_temperature_samples": len(temps),
        }

    nve = _run("nve")
    out["nve"] = nve
    out["nve_drift_kcal_mol"] = abs(nve["final_energy_kcal_mol"] - e0)

    nvt = _run("nvt")
    out["nvt"] = nvt

    print("<<<JSON>>>" + json.dumps(out))
    return 0


# ---------------------------------------------------------------------------
# Parent: spawn the child, measure the checks
# ---------------------------------------------------------------------------


def run_system(
    system: str, *, out_dir: Path, temperature: float, steps: int
) -> dict[str, Any]:
    sys_dir = out_dir / system
    sys_dir.mkdir(parents=True, exist_ok=True)

    # The worker loads CHARMM directly, so an MPI-linked libcharmm needs an MPI
    # launcher here; a bare `python` child aborts in MPI_Init.
    cmd = [
        *lib.charmm_mpi_prefix(),
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        system,
        "--output-dir",
        str(sys_dir),
        "--temperature",
        str(temperature),
        "--steps",
        str(steps),
    ]
    proc = subprocess.run(cmd, cwd=lib.REPO, capture_output=True, text=True)
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (sys_dir / "worker.log").write_text(log, encoding="utf-8")

    metrics: dict[str, Any] = {"system": system, "worker_returncode": proc.returncode}
    checks: list[dict[str, Any]] = []

    payload: dict[str, Any] | None = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("<<<JSON>>>"):
            payload = json.loads(line[len("<<<JSON>>>") :])

    if payload is None:
        reason = _exit_detail(proc.returncode)
        metrics["failure_reason"] = reason
        for name in (
            "build_audit",
            "finite_energy_force",
            "fd_force_pass",
            "minimized_structure",
            "short_nve_nvt_pass",
        ):
            checks.append(_check(name, False, f"CHARMM worker produced no result: {reason}"))
        return {"metrics": metrics, "checks": checks}

    metrics.update(payload)

    # ---- build_audit -------------------------------------------------------
    n_atoms = payload.get("n_atoms") or 0
    seq_ok = payload.get("sequence") == PEPTIDES[system]
    ok = n_atoms > 0 and seq_ok and bool(payload.get("positions_finite"))
    checks.append(
        _check(
            "build_audit",
            ok,
            f"built {n_atoms} atoms, sequence={payload.get('sequence')}, "
            f"coordinates finite={payload.get('positions_finite')}",
        )
    )

    # ---- finite_energy_force ----------------------------------------------
    e = payload.get("energy_kcal_mol")
    fmax = payload.get("max_force_kcal_mol_A")
    ok = bool(payload.get("energy_finite")) and bool(payload.get("forces_finite"))
    checks.append(
        _check(
            "finite_energy_force",
            ok,
            f"E = {e} kcal/mol, max|F| = {fmax} kcal/mol/A (both finite: {ok})",
        )
    )

    # ---- fd_force_pass -----------------------------------------------------
    fd_err = payload.get("fd_max_abs_error_kcal_mol_A")
    ok = fd_err is not None and math.isfinite(fd_err) and fd_err <= FD_MAX_ABS_ERROR_KCAL_MOL_A
    checks.append(
        _check(
            "fd_force_pass",
            ok,
            f"max |analytic - finite-difference| = {fd_err} kcal/mol/A over "
            f"{payload.get('fd_atoms_probed')} atoms "
            f"(tolerance {FD_MAX_ABS_ERROR_KCAL_MOL_A})"
            if fd_err is not None
            else "no finite-difference comparison produced",
        )
    )

    # ---- minimized_structure ----------------------------------------------
    grms = payload.get("grms_kcal_mol_A")
    ok = grms is not None and math.isfinite(grms) and grms <= MAX_GRMS_KCAL_MOL_A
    checks.append(
        _check(
            "minimized_structure",
            ok,
            f"post-minimization GRMS = {grms} kcal/mol/A "
            f"(threshold {MAX_GRMS_KCAL_MOL_A})",
        )
    )

    # ---- short_nve_nvt_pass ------------------------------------------------
    drift = payload.get("nve_drift_kcal_mol")
    nvt = payload.get("nvt") or {}
    # Mean over the trace, not the final instantaneous value: a short run's last
    # sample fluctuates by tens of K even when the thermostat is working.
    nvt_t = nvt.get("mean_temperature_K")
    drift_ok = drift is not None and math.isfinite(drift) and drift <= MAX_NVE_DRIFT_KCAL_MOL
    temp_ok = (
        nvt_t is not None
        and math.isfinite(nvt_t)
        and abs(nvt_t - temperature) <= NVT_TEMPERATURE_TOLERANCE_K
    )
    checks.append(
        _check(
            "short_nve_nvt_pass",
            drift_ok and temp_ok,
            f"NVE |dE| = {drift} kcal/mol (budget {MAX_NVE_DRIFT_KCAL_MOL}); "
            f"NVT mean T = {nvt_t} K over {nvt.get('n_temperature_samples')} samples "
            f"vs setpoint {temperature:.0f} K "
            f"(tolerance {NVT_TEMPERATURE_TOLERANCE_K:.0f} K)",
        )
    )

    return {"metrics": metrics, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(PEPTIDES))
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--_worker", dest="worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return _worker(args.worker, args.output_dir, args.temperature, args.steps)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    per_system: dict[str, Any] = {}
    for system in args.systems:
        print(f"=== {system} ===", flush=True)
        result = run_system(
            system, out_dir=out, temperature=args.temperature, steps=args.steps
        )
        per_system[system] = result
        for c in result["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    names = [
        "build_audit",
        "finite_energy_force",
        "fd_force_pass",
        "minimized_structure",
        "short_nve_nvt_pass",
    ]
    proof_checks = []
    summary: dict[str, bool] = {}
    for name in names:
        failing = [
            s
            for s, r in per_system.items()
            if not next((c["passed"] for c in r["checks"] if c["name"] == name), False)
        ]
        summary[name] = not failing
        proof_checks.append(
            _check(
                name,
                not failing,
                "all peptides pass" if not failing else f"failing: {', '.join(failing)}",
            )
        )

    lib.write_json(out / "metrics.json", {s: r["metrics"] for s, r in per_system.items()})
    lib.write_json(
        out / "proof.json",
        {
            "passed": all(summary.values()),
            "checks": proof_checks,
            "asserted_by": "peptide_gas_smoke.py (measured from generated artifacts)",
            "systems": {s: r["checks"] for s, r in per_system.items()},
            "sources": ["metrics.json", "*/worker.log"],
        },
    )

    ok = all(summary.values())
    print(f"\npeptide_gas.smoke: {'PASS' if ok else 'FAIL'}")
    for name, passed in summary.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
