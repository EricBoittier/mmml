"""Run mode-check at every cutoff-region COM station (vacuum dimers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import write as ase_write

from .config import HybridModeCheckSetup, ModeCheckConfig
from .cutoff_ladder import (
    CutoffStation,
    cutoff_region_stations,
    region_boundaries,
)
from .hybrid import (
    assert_resolved_vacuum_geometry,
    build_psf_and_attach_hybrid,
    com_separations_along_chain,
    min_intermolecular_distance_A,
    reposition_monomers_along_x,
)
from .run import run_mode_check


def _station_summary(result, meta: dict[str, Any], station: CutoffStation) -> dict[str, Any]:
    return {
        "label": station.label,
        "region": station.region,
        "description": station.description,
        "com_A_requested": station.com_A,
        "com_separations_A": meta.get("com_separations_A"),
        "min_intermolecular_distance_A": meta.get("min_intermolecular_distance_A"),
        "energy_eV": result.energy_eV,
        "max_force_eVA": result.max_force_eVA,
        "fd": result.fd,
        "bond_nu_cm_from_E": {
            k: v.get("nu_cm_from_E") for k, v in result.bond_scans.items()
        },
        "vib_max_cm": (result.vibrations or {}).get("max_cm"),
        "kick_fft_peak_cm": (result.kick or {}).get("fft_peak_cm"),
        "errors": result.errors,
        "notes": list(result.notes),
    }


def run_cutoff_sweep(
    setup: HybridModeCheckSetup,
    *,
    output_dir: Path,
    config: ModeCheckConfig | None = None,
    stations: list[CutoffStation] | None = None,
    min_intermolecular_distance_threshold_A: float = 1.2,
) -> dict[str, Any]:
    """Build hybrid once, then evaluate mode-check at each cutoff COM station.

    Stations are visited far→near so the initial PSF attach uses a clash-safe
    geometry. Per-station minimize is discouraged (it drifts COM); prefer
    ``fd,bond-scan`` for the sweep.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config or ModeCheckConfig(
        checks=("minimize", "fd", "bond-scan"),
        minimize_freeze_monomer_coms=True,
    )

    if stations is None:
        stations = cutoff_region_stations(
            ml_switch_width=float(setup.ml_switch_width),
            mm_switch_on=float(setup.mm_switch_on),
            mm_switch_width=float(setup.mm_switch_width),
        )
    # Attach at the largest COM first.
    stations_sorted = sorted(stations, key=lambda s: float(s.com_A), reverse=True)
    setup_far = HybridModeCheckSetup(
        composition=setup.composition,
        checkpoint=setup.checkpoint,
        do_mm=setup.do_mm,
        do_ml=setup.do_ml,
        do_ml_dimer=setup.do_ml_dimer,
        ml_switch_width=setup.ml_switch_width,
        mm_switch_on=setup.mm_switch_on,
        mm_switch_width=setup.mm_switch_width,
        mm_charge_mode=setup.mm_charge_mode,
        lr_solver=setup.lr_solver,
        monomer_separation_A=float(stations_sorted[0].com_A),
        xyz=setup.xyz,
        max_pairs=setup.max_pairs,
    )
    atoms, apm, base_meta = build_psf_and_attach_hybrid(
        setup_far,
        write_psf_to=out / "cluster.psf",
    )
    ase_write(str(out / "cluster_attach.xyz"), atoms)
    # Freeze-COM minimize needs the monomer layout on the config.
    object.__setattr__(cfg, "atoms_per_monomer", tuple(int(n) for n in apm))
    if "minimize" in cfg.checks and not cfg.minimize_freeze_monomer_coms:
        # Cutoff stations are meaningless if FIRE can collapse COM.
        object.__setattr__(cfg, "minimize_freeze_monomer_coms", True)

    # Restore rigid monomer templates from the attach geometry (pre-minimize).
    template_pos = np.asarray(atoms.get_positions(), dtype=float).copy()

    rows: list[dict[str, Any]] = []
    any_errors = False
    for station in sorted(stations, key=lambda s: float(s.com_A)):
        station_dir = out / f"r_{station.label}_{station.com_A:.3f}"
        station_dir.mkdir(parents=True, exist_ok=True)
        atoms.set_positions(template_pos)
        reposition_monomers_along_x(atoms, apm, separation_A=float(station.com_A))
        try:
            assert_resolved_vacuum_geometry(
                atoms.get_positions(),
                apm,
                min_intermolecular_distance_threshold_A=float(
                    min_intermolecular_distance_threshold_A
                ),
            )
        except RuntimeError as exc:
            any_errors = True
            rows.append(
                {
                    "label": station.label,
                    "region": station.region,
                    "description": station.description,
                    "com_A_requested": station.com_A,
                    "skipped": True,
                    "skip_reason": str(exc),
                    "min_intermolecular_distance_A": min_intermolecular_distance_A(
                        atoms.get_positions(), apm
                    ),
                    "com_separations_A": com_separations_along_chain(
                        atoms.get_positions(), apm
                    ),
                    "errors": {"geometry": str(exc)},
                }
            )
            continue

        ase_write(str(station_dir / "cluster_initial.xyz"), atoms)
        station_meta = {
            **base_meta,
            "monomer_separation_A": float(station.com_A),
            "com_separations_A": com_separations_along_chain(
                atoms.get_positions(), apm
            ),
            "min_intermolecular_distance_A": min_intermolecular_distance_A(
                atoms.get_positions(), apm
            ),
            "cutoff_station": station.to_dict(),
        }
        result = run_mode_check(
            atoms,
            cfg,
            output_dir=station_dir,
            setup_meta=station_meta,
        )
        ase_write(str(station_dir / "cluster_final.xyz"), atoms)
        row = _station_summary(result, station_meta, station)
        row["skipped"] = False
        row["summary"] = str(station_dir / "mode_check_summary.json")
        if result.errors:
            any_errors = True
        rows.append(row)

    # Restore order by increasing COM for the JSON table.
    rows.sort(key=lambda r: float(r.get("com_A_requested", 0.0)))
    payload = {
        "schema": "mode_check_cutoff_sweep/1.0",
        "boundaries": region_boundaries(
            ml_switch_width=float(setup.ml_switch_width),
            mm_switch_on=float(setup.mm_switch_on),
            mm_switch_width=float(setup.mm_switch_width),
        ),
        "stations": [s.to_dict() for s in sorted(stations, key=lambda s: s.com_A)],
        "checks": list(cfg.checks),
        "minimize_freeze_monomer_coms": bool(cfg.minimize_freeze_monomer_coms),
        "base_meta": {
            k: base_meta[k]
            for k in (
                "composition",
                "checkpoint",
                "mm_charge_mode",
                "do_mm_effective",
                "do_ml_dimer",
            )
            if k in base_meta
        },
        "results": rows,
        "ok": not any_errors,
    }
    summary_path = out / "cutoff_sweep_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary"] = str(summary_path)
    return payload
