#!/usr/bin/env python3
"""Driver for ``pure_liquids.smoke``: BENZ, TIP3, DCM, ACO.

Computes the five acceptance checks declared for this task in ``campaign.yaml``:

    packmol_audit, finite_energy_force, positive_temperature,
    no_overlap, short_nvt_pass

Every metric written here is measured from an artifact this driver produced. If
a stage does not run, its checks are reported false with the real reason (exit
signal, missing artifact) -- never defaulted to a plausible-looking number. A
check that cannot be evaluated is a failed check, not a passing one.

The build/certify stage is ``mmml liquid-box``, which packs the box with Packmol
and pretreats it under CHARMM MM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_lib as lib  # noqa: E402

SYSTEMS = ("BENZ", "TIP3", "DCM", "ACO")

# Heavy-atom + hydrogen counts per CGenFF monomer, used to audit what Packmol
# actually placed against what we asked for.
ATOMS_PER_MONOMER: dict[str, int] = {
    "BENZ": 12,
    "TIP3": 3,
    "DCM": 5,
    "ACO": 10,
}

# Closest approach we tolerate between atoms of different molecules. Below this
# the box is not physical and CHARMM will blow up rather than sample.
MIN_INTERMOLECULAR_DISTANCE_A = 1.5

# A short NVT is "stable" if the mean temperature lands near the setpoint.
NVT_TEMPERATURE_TOLERANCE_K = 100.0


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _read_pdb_coords(path: Path) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Coordinates and residue names from a PDB, without pulling in a parser."""
    coords: list[tuple[float, float, float]] = []
    resnames: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append((x, y, z))
        resnames.append(line[17:21].strip())
    return coords, resnames


def _min_intermolecular_distance(
    coords: list[tuple[float, float, float]], atoms_per_molecule: int
) -> float:
    """Smallest distance between atoms belonging to different molecules.

    Intramolecular pairs are excluded: a C-H bond at 1.09 A is not an overlap.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return float("nan")

    if len(coords) < 2 or atoms_per_molecule < 1:
        return float("nan")

    xyz = np.asarray(coords, dtype=float)
    molecule = np.arange(len(xyz)) // atoms_per_molecule

    best = math.inf
    for i in range(len(xyz)):
        other = molecule != molecule[i]
        if not other.any():
            continue
        d = np.linalg.norm(xyz[other] - xyz[i], axis=1).min()
        best = min(best, float(d))
    return best if best is not math.inf else float("nan")


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


def _find_packed_pdb(out_dir: Path) -> Path | None:
    for candidate in sorted(out_dir.rglob("*.pdb")):
        if "packmol" in str(candidate).lower():
            return candidate
    pdbs = sorted(out_dir.rglob("*.pdb"))
    return pdbs[0] if pdbs else None


def _parse_temperatures(log: str) -> list[float]:
    """Instantaneous temperatures from the CHARMM dynamics trace."""
    temps: list[float] = []
    for match in re.finditer(r"^\s*DYNA\b.*?TEMP\w*\s*[:=]?\s*(-?\d+\.\d+)", log, re.M):
        temps.append(float(match.group(1)))
    if temps:
        return temps
    # CHARMM's fixed-column DYNA PROP line: TEMPerature is the 3rd field.
    for line in log.splitlines():
        if line.startswith("DYNA>"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    temps.append(float(parts[3]))
                except ValueError:
                    continue
    return temps


def run_system(
    system: str,
    *,
    out_dir: Path,
    n_molecules: int,
    box_size: float,
    temperature: float,
    python: str,
) -> dict[str, Any]:
    """Build and certify one liquid; return measured metrics and checks."""
    sys_dir = out_dir / system
    sys_dir.mkdir(parents=True, exist_ok=True)
    build_dir = sys_dir / "box"

    cmd = [
        python,
        "-m",
        "mmml.cli.__main__",
        "liquid-box",
        "--composition",
        f"{system}:{n_molecules}",
        "--output-dir",
        str(build_dir),
        "--box-size",
        str(box_size),
        "--temperature",
        str(temperature),
    ]

    proc = subprocess.run(
        cmd, cwd=lib.REPO, capture_output=True, text=True, env=os.environ.copy()
    )
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (sys_dir / "liquid_box.log").write_text(log, encoding="utf-8")

    metrics: dict[str, Any] = {
        "system": system,
        "requested_molecules": n_molecules,
        "box_size_A": box_size,
        "target_temperature_K": temperature,
        "liquid_box_returncode": proc.returncode,
    }
    checks: list[dict[str, Any]] = []

    # `mmml liquid-box` can print "liquid-box failed:" and still exit 0, so the
    # return code alone is not trusted -- the log is inspected too.
    failure_line = ""
    for line in log.splitlines():
        if "liquid-box failed" in line.lower():
            failure_line = line.strip()
            break
    metrics["liquid_box_failure_line"] = failure_line or None

    build_ok = proc.returncode == 0 and not failure_line
    if not build_ok:
        reason = failure_line or _exit_detail(proc.returncode)
        metrics["build_failure_reason"] = reason

    # ---- packmol_audit -----------------------------------------------------
    packed = _find_packed_pdb(build_dir) if build_dir.is_dir() else None
    apm = ATOMS_PER_MONOMER.get(system)
    if packed is None:
        checks.append(
            _check(
                "packmol_audit",
                False,
                f"no packed structure produced ({metrics.get('build_failure_reason', 'unknown')})",
            )
        )
        coords: list[tuple[float, float, float]] = []
    else:
        coords, _ = _read_pdb_coords(packed)
        expected = n_molecules * apm if apm else None
        metrics["packed_structure"] = str(packed.relative_to(lib.REPO))
        metrics["packed_atoms"] = len(coords)
        metrics["expected_atoms"] = expected
        ok = expected is not None and len(coords) == expected
        checks.append(
            _check(
                "packmol_audit",
                ok,
                f"packed {len(coords)} atoms, expected {expected} "
                f"({n_molecules} x {apm} for {system})",
            )
        )

    # ---- no_overlap --------------------------------------------------------
    if coords and apm:
        min_d = _min_intermolecular_distance(coords, apm)
        metrics["min_intermolecular_distance_A"] = (
            round(min_d, 4) if math.isfinite(min_d) else None
        )
        ok = math.isfinite(min_d) and min_d >= MIN_INTERMOLECULAR_DISTANCE_A
        checks.append(
            _check(
                "no_overlap",
                ok,
                f"closest intermolecular contact {min_d:.3f} A "
                f"(threshold {MIN_INTERMOLECULAR_DISTANCE_A} A)"
                if math.isfinite(min_d)
                else "could not measure contacts",
            )
        )
    else:
        checks.append(_check("no_overlap", False, "no coordinates to measure"))

    # ---- finite_energy_force ----------------------------------------------
    # Energies come from the CHARMM pretreat stage inside liquid-box. If that
    # stage never ran, we have no energy -- which is a failed check, not a pass.
    energy = None
    for match in re.finditer(r"ENER ENR:\s*(-?\d+\.\d+)", log):
        energy = float(match.group(1))
    if energy is None:
        for match in re.finditer(r"^\s*ENER>\s*\S+\s+(-?\d+\.\d+)", log, re.M):
            energy = float(match.group(1))
    metrics["final_potential_energy_kcal_mol"] = energy
    checks.append(
        _check(
            "finite_energy_force",
            energy is not None and math.isfinite(energy),
            f"CHARMM potential energy {energy} kcal/mol"
            if energy is not None
            else f"no energy evaluated ({metrics.get('build_failure_reason', 'stage did not run')})",
        )
    )

    # ---- positive_temperature / short_nvt_pass -----------------------------
    temps = _parse_temperatures(log)
    metrics["n_temperature_samples"] = len(temps)
    if temps:
        mean_t = sum(temps) / len(temps)
        metrics["mean_temperature_K"] = round(mean_t, 3)
        metrics["final_temperature_K"] = round(temps[-1], 3)
        checks.append(
            _check(
                "positive_temperature",
                mean_t > 0.0,
                f"mean T = {mean_t:.1f} K over {len(temps)} samples",
            )
        )
        drift = abs(mean_t - temperature)
        checks.append(
            _check(
                "short_nvt_pass",
                build_ok and drift <= NVT_TEMPERATURE_TOLERANCE_K,
                f"mean T = {mean_t:.1f} K vs setpoint {temperature:.0f} K "
                f"(|drift| {drift:.1f} K, tolerance {NVT_TEMPERATURE_TOLERANCE_K:.0f} K), "
                f"build_ok={build_ok}",
            )
        )
    else:
        reason = metrics.get("build_failure_reason", "no dynamics trace produced")
        checks.append(_check("positive_temperature", False, f"no temperature samples: {reason}"))
        checks.append(_check("short_nvt_pass", False, f"NVT did not run: {reason}"))

    return {"metrics": metrics, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--n-molecules", type=int, default=20)
    parser.add_argument("--box-size", type=float, default=22.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    per_system: dict[str, Any] = {}
    all_checks: dict[str, bool] = {}

    for system in args.systems:
        print(f"=== {system} ===", flush=True)
        result = run_system(
            system,
            out_dir=out,
            n_molecules=args.n_molecules,
            box_size=args.box_size,
            temperature=args.temperature,
            python=python,
        )
        per_system[system] = result
        for check in result["checks"]:
            print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")

    # A check passes for the task only if it passed for every system: the task is
    # "BENZ, TIP3, DCM and ACO all prepare and sample stably", not "at least one did".
    check_names = [
        "packmol_audit",
        "finite_energy_force",
        "positive_temperature",
        "no_overlap",
        "short_nvt_pass",
    ]
    proof_checks = []
    for name in check_names:
        failing = [
            s
            for s, r in per_system.items()
            if not next((c["passed"] for c in r["checks"] if c["name"] == name), False)
        ]
        passed = not failing
        all_checks[name] = passed
        proof_checks.append(
            _check(
                name,
                passed,
                "all systems pass" if passed else f"failing systems: {', '.join(failing)}",
            )
        )

    lib.write_json(out / "metrics.json", {s: r["metrics"] for s, r in per_system.items()})
    lib.write_json(
        out / "proof.json",
        {
            "passed": all(all_checks.values()),
            "checks": proof_checks,
            "asserted_by": "pure_liquids_smoke.py (measured from generated artifacts)",
            "systems": {s: r["checks"] for s, r in per_system.items()},
            "sources": ["metrics.json", "*/liquid_box.log", "*/box/"],
        },
    )

    ok = all(all_checks.values())
    print(f"\npure_liquids.smoke: {'PASS' if ok else 'FAIL'}")
    for name, passed in all_checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
