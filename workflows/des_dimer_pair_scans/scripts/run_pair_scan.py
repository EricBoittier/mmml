#!/usr/bin/env python3
"""Run one DES dimer-pair 2D scan with CHARMM, xTB, and ORCA MP2 energies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scan_lib import (  # noqa: E402
    load_config,
    output_dir,
    pair_from_tag,
    scan_grids,
    workflow_root,
)

EV_PER_KCAL = 1.0 / 23.0605
HARTREE_TO_KCAL = 627.5094740631


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", required=True, help="Pair tag, e.g. aco__meoh")
    p.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="NPZ output path (default: artifacts/.../<pair>/scan_2d.npz)",
    )
    return p.parse_args(argv)


def _build_cluster(composition: str, spacing: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_from_composition
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import parse_composition

    return _build_cluster_from_composition(
        composition=parse_composition(composition),
        spacing=spacing,
    )[:3]


def _charmm_energy_kcal(positions: np.ndarray) -> dict[str, float]:
    import pycharmm
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_energy_row
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    sync_charmm_positions(positions)
    pycharmm.lingo.charmm_script("ENER")
    terms = charmm_energy_row()
    return {
        "charmm_ENER_kcal": float(terms.get("ENER", np.nan)),
        "charmm_VDW_kcal": float(terms.get("VDW", np.nan)),
        "charmm_ELEC_kcal": float(terms.get("ELEC", np.nan)),
        "charmm_GRMS_kcal_A": float(terms.get("GRMS", np.nan)),
    }


def _ase_atoms(positions: np.ndarray, z: np.ndarray):
    from ase import Atoms

    return Atoms(numbers=np.asarray(z, dtype=int), positions=np.asarray(positions, dtype=float))


def _xtb_energy_ev(atoms, cfg: dict[str, Any]) -> float:
    from mmml.interfaces.qc_backends.xtb import XTBBackend

    xtb_cfg = cfg.get("xtb") or {}
    backend = XTBBackend(
        method=str(xtb_cfg.get("method", "GFN2-xTB")),
        charge=0,
        multiplicity=1,
    )
    result = backend.evaluate_batch([atoms], properties=frozenset({"energy"}))
    return float(result["E"][0])


def _orca_mp2_energy_hartree(atoms, cfg: dict[str, Any], workdir: Path) -> float:
    from mmml.interfaces.qc_backends.orca_qm import OrcaQMBackend

    orca_cfg = cfg.get("orca_mp2") or {}
    backend = OrcaQMBackend(
        method=str(orca_cfg.get("method", "MP2")),
        basis=str(orca_cfg.get("basis", "def2-SVP")),
        charge=0,
        multiplicity=1,
        pal=int(orca_cfg.get("pal", 4)),
    )
    energy, _grad = backend._run_single(atoms, workdir)
    return float(energy)


def _backend_enabled(cfg: dict[str, Any], name: str) -> bool:
    backends = cfg.get("backends") or {}
    return bool(backends.get(name, False))


def _backend_available(name: str) -> bool:
    if name == "charmm":
        try:
            import pycharmm  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "xtb":
        try:
            import tblite  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "orca_mp2":
        exe = os.environ.get("ORCA", "orca")
        return shutil.which(exe) is not None
    return False


def run_pair_scan(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    pair = pair_from_tag(cfg, args.pair)
    scan = cfg.get("scan") or {}
    spacing = float(scan.get("spacing", 5.0))
    angle_deg = float(scan.get("angle_deg", 60.0))

    out_path = args.output
    if out_path is None:
        out_path = output_dir(cfg, pair) / "scan_2d.npz"
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d1_grid, d2_grid = scan_grids(cfg)
    n1, n2 = len(d1_grid), len(d2_grid)

    store: dict[str, Any] = {
        "pair_tag": pair.tag,
        "composition": pair.composition,
        "species_a": pair.species_a.tag,
        "species_b": pair.species_b.tag,
        "label": pair.label,
        "d01_grid": d1_grid,
        "d02_grid": d2_grid,
        "angle_02_deg": angle_deg,
        "reference_checkpoint": str(cfg.get("reference_checkpoint", "")),
    }

    use_charmm = _backend_enabled(cfg, "charmm") and _backend_available("charmm")
    use_xtb = _backend_enabled(cfg, "xtb") and _backend_available("xtb")
    use_orca = _backend_enabled(cfg, "orca_mp2") and _backend_available("orca_mp2")

    if use_charmm:
        for key in ("charmm_ENER_kcal", "charmm_VDW_kcal", "charmm_ELEC_kcal", "charmm_GRMS_kcal_A"):
            store[key] = np.full((n1, n2), np.nan, dtype=np.float64)
    if use_xtb:
        store["xtb_energy_ev"] = np.full((n1, n2), np.nan, dtype=np.float64)
        store["xtb_energy_kcal"] = np.full((n1, n2), np.nan, dtype=np.float64)
    if use_orca:
        store["orca_mp2_energy_hartree"] = np.full((n1, n2), np.nan, dtype=np.float64)
        store["orca_mp2_energy_kcal"] = np.full((n1, n2), np.nan, dtype=np.float64)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    pycharmm_quiet()
    z, ref_pos, atoms_per = _build_cluster(pair.composition, spacing=spacing)
    setup_default_nbonds()
    z = np.asarray(z, dtype=int)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)
    atoms_per = [int(x) for x in atoms_per]

    from mmml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
        distance_report,
        place_trimer,
    )

    orca_root = Path(tempfile.mkdtemp(prefix=f"orca_{pair.tag}_"))

    def eval_at(positions: np.ndarray) -> dict[str, float]:
        rec: dict[str, float] = {}
        rec.update(distance_report(positions, atoms_per))
        atoms = _ase_atoms(positions, z)
        if use_charmm:
            rec.update(_charmm_energy_kcal(positions))
        if use_xtb:
            try:
                e_ev = _xtb_energy_ev(atoms, cfg)
                rec["xtb_energy_ev"] = e_ev
                rec["xtb_energy_kcal"] = e_ev * EV_PER_KCAL
            except Exception as exc:
                print(f"xtb failed: {exc}", file=sys.stderr)
        if use_orca:
            i = int(eval_at.grid_i)  # type: ignore[attr-defined]
            j = int(eval_at.grid_j)  # type: ignore[attr-defined]
            workdir = orca_root / f"{i:03d}_{j:03d}"
            workdir.mkdir(parents=True, exist_ok=True)
            try:
                e_ha = _orca_mp2_energy_hartree(atoms, cfg, workdir)
                rec["orca_mp2_energy_hartree"] = e_ha
                rec["orca_mp2_energy_kcal"] = e_ha * HARTREE_TO_KCAL
            except Exception as exc:
                print(f"orca failed at {i},{j}: {exc}", file=sys.stderr)
        return rec

    metric_keys: tuple[str, ...] = ()
    if use_charmm:
        metric_keys += ("charmm_ENER_kcal", "charmm_VDW_kcal", "charmm_ELEC_kcal")
    if use_xtb:
        metric_keys += ("xtb_energy_ev", "xtb_energy_kcal")
    if use_orca:
        metric_keys += ("orca_mp2_energy_hartree", "orca_mp2_energy_kcal")

    # Attach grid indices for ORCA scratch dirs.
    angle = np.deg2rad(angle_deg)
    for i, d01 in enumerate(d1_grid):
        for j, d02 in enumerate(d2_grid):
            pos = place_trimer(ref_pos, atoms_per, float(d01), float(d02), angle)
            eval_at.grid_i = i  # type: ignore[attr-defined]
            eval_at.grid_j = j  # type: ignore[attr-defined]
            rec = eval_at(pos)
            for key in metric_keys:
                if key in store:
                    store[key][i, j] = float(rec.get(key, np.nan))

    meta = {
        "backends": {
            "charmm": use_charmm,
            "xtb": use_xtb,
            "orca_mp2": use_orca,
        },
        "atoms_per_monomer": atoms_per,
        "n_atoms": int(len(z)),
    }
    store["meta_json"] = np.array(json.dumps(meta))

    np.savez_compressed(out_path, **store)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "pair_tag": pair.tag,
                "composition": pair.composition,
                "label": pair.label,
                "output": str(out_path),
                **meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_path.parent / "done.txt").write_text("ok\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    args = _parse_args()
    run_pair_scan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
