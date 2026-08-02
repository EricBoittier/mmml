"""Calculator-agnostic mode-check orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from .bonds import infer_xh_bond_pairs
from .config import ModeCheckConfig, ModeCheckPaths
from .forces import bond_stretch_scan, force_fd_check
from .kick import kick_bond_fft
from .result import ModeCheckResult
from .vibrations import run_ase_vibrations


def run_mode_check(
    atoms: Atoms,
    config: ModeCheckConfig | None = None,
    *,
    output_dir: Path | None = None,
    setup_meta: dict[str, Any] | None = None,
    bond_pairs: list[tuple[int, int]] | None = None,
) -> ModeCheckResult:
    """Run selected local-mode / force diagnostics on ``atoms`` (calc attached).

    This path is calculator-agnostic: any ASE calculator that provides energy and
    forces works. Hybrid PSF attachment lives in :mod:`mmml.mode_check.hybrid`.
    """
    if atoms.calc is None:
        raise ValueError("atoms.calc must be set before run_mode_check")
    cfg = config or ModeCheckConfig()
    paths = ModeCheckPaths(output_dir) if output_dir is not None else None
    if paths is not None:
        paths.output_dir.mkdir(parents=True, exist_ok=True)

    result = ModeCheckResult(
        config=cfg.to_dict(),
        setup=dict(setup_meta or {}),
    )
    checks = set(cfg.checks)

    if "minimize" in checks:
        try:
            log = None
            if paths is not None:
                log = str(paths.output_dir / "fire.log")
            prior_constraints = list(atoms.constraints) if atoms.constraints else []
            if cfg.minimize_freeze_monomer_coms:
                apm = cfg.atoms_per_monomer
                if apm is None:
                    raise ValueError(
                        "minimize_freeze_monomer_coms requires "
                        "ModeCheckConfig.atoms_per_monomer"
                    )
                from .constraints import FixMonomerCOMs

                atoms.set_constraint(
                    [*prior_constraints, FixMonomerCOMs(atoms, list(apm))]
                )
            try:
                FIRE(atoms, logfile=log).run(
                    fmax=float(cfg.minimize_fmax),
                    steps=int(cfg.minimize_steps),
                )
            finally:
                atoms.set_constraint(prior_constraints)
        except Exception as exc:  # pragma: no cover - optimizer failures are env-dependent
            result.errors["minimize"] = f"{type(exc).__name__}: {exc}"

    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        result.energy_eV = energy
        result.max_force_eVA = float(np.max(np.linalg.norm(forces, axis=1)))
    except Exception as exc:
        result.errors["energy"] = f"{type(exc).__name__}: {exc}"
        return result

    apm = cfg.atoms_per_monomer
    if bond_pairs is None:
        bond_pairs = infer_xh_bond_pairs(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            atoms_per_monomer=list(apm) if apm is not None else None,
        )
    result.bond_pairs = [[int(i), int(j)] for i, j in bond_pairs]
    result.r_bonds_A = [
        float(np.linalg.norm(atoms.positions[j] - atoms.positions[i]))
        for i, j in bond_pairs
    ]

    if "fd" in checks:
        try:
            result.fd = force_fd_check(atoms, int(cfg.fd_atoms), float(cfg.fd_dx_A))
        except Exception as exc:
            result.errors["fd"] = f"{type(exc).__name__}: {exc}"

    if "bond-scan" in checks:
        if not bond_pairs:
            result.notes.append("bond-scan skipped: no X–H pairs inferred")
        deltas = np.asarray(cfg.bond_deltas, dtype=float)
        for ip, (i, j) in enumerate(bond_pairs):
            tag = f"XH{ip}"
            try:
                scan = bond_stretch_scan(
                    atoms,
                    int(i),
                    int(j),
                    deltas=deltas,
                    fit_abs_delta_max=float(cfg.bond_fit_abs_delta_max),
                )
                result.bond_scans[tag] = scan
                if paths is not None:
                    rows = np.array(
                        [
                            [
                                r["delta_A"],
                                r["r_A"],
                                r["E_eV"],
                                r["F_stretch_eV_A"],
                            ]
                            for r in scan["rows"]
                        ],
                        dtype=float,
                    )
                    np.savetxt(
                        paths.output_dir / f"{tag}_scan.txt",
                        rows,
                        header="delta_A r_A E_eV F_stretch_eV_A",
                    )
            except Exception as exc:
                result.errors[f"bond-scan:{tag}"] = f"{type(exc).__name__}: {exc}"

    if "vibrations" in checks:
        try:
            vib_dir = paths.vib_dir if paths is not None else None
            result.vibrations = run_ase_vibrations(
                atoms,
                output_dir=vib_dir,
                delta=float(cfg.vib_delta_A),
                nfree=int(cfg.vib_nfree),
            )
        except Exception as exc:
            result.errors["vibrations"] = f"{type(exc).__name__}: {exc}"

    if "kick" in checks:
        if not bond_pairs:
            result.notes.append("kick skipped: no X–H pairs inferred")
        else:
            i, j = bond_pairs[0]
            try:
                r_path = paths.kick_r_npy if paths is not None else None
                result.kick = kick_bond_fft(
                    atoms,
                    int(i),
                    int(j),
                    kick_delta_A=float(cfg.kick_delta_A),
                    timestep_fs=float(cfg.kick_timestep_fs),
                    n_steps=int(cfg.kick_steps),
                    output_r_path=r_path,
                )
            except Exception as exc:
                result.errors["kick"] = f"{type(exc).__name__}: {exc}"

    if paths is not None:
        result.write(paths.summary_json, overwrite=True)
    return result
