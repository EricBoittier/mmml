"""Tri-alanine water embedding: checkpoint export, ML validation, hybrid ASE calculator."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import (
    DEFAULT_ML_SEG_ID,
    TRAINING_N_ATOMS_AAA,
    build_embedding_box,
    register_embedding_mlpot,
)


@dataclass(slots=True)
class EmbeddingValidationResult:
    """PhysNet monomer potential vs ``valid.npz`` (pre-hybrid gate)."""

    checkpoint_json: Path
    valid_npz: Path
    metrics_path: Path
    energy_mae_kcal_mol: float
    force_mae_kcal_mol_A: float
    metrics: dict[str, Any]


@dataclass(slots=True)
class EmbeddingHybridSession:
    """Live partial-MLpot session for TRIA + TIP3 (caller must ``close()`` when done)."""

    output_dir: Path
    checkpoint_json: Path
    mlpot_ctx: Any
    atoms: Any
    calculator: Any
    box_meta: dict[str, Any]

    def charmm_total_energy_kcalmol(self) -> float:
        import pycharmm.energy as energy

        energy.show()
        return float(energy.get_total())

    def close(self) -> None:
        unset = getattr(self.mlpot_ctx, "unset", None)
        if callable(unset):
            unset()


def _repo_root() -> Path:
    # .../mmml/interfaces/pycharmmInterface/mlpot/embedding_hybrid.py -> repo root
    return Path(__file__).resolve().parents[4]


def export_embedding_checkpoint(
    epoch_dir: Path | str,
    output_json: Path | str,
    *,
    params_key: str = "params",
) -> Path:
    """Export Orbax ``epoch-*`` checkpoint to portable JSON."""
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.utils.model_checkpoint import orbax_to_json

    _, resolved = resolve_checkpoint_paths(Path(epoch_dir))
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    return orbax_to_json(
        orbax_checkpoint_dir=resolved,
        output_path=out,
        params_key=params_key,
    )


def validate_embedding_monomer_potential(
    checkpoint_json: Path | str,
    valid_npz: Path | str,
    out_dir: Path | str,
    *,
    natoms: int = TRAINING_N_ATOMS_AAA,
    batch_size: int = 32,
    repo_root: Path | str | None = None,
) -> EmbeddingValidationResult:
    """Validate PhysNet monomer E/F on ``valid.npz`` before wiring hybrid MLpot.

    This is the correct pre-flight for ``md-embedding`` (``n_monomers=1``): the
    aaa.ama NPZ is a single peptide, not a DES-style dimer cluster. Cluster
    dimer scans belong to ``md-system`` / ``validate_mlpot_sparse_dimers.py``.
    """
    ckpt = Path(checkpoint_json)
    valid = Path(valid_npz)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root is not None else _repo_root()
    cmd = [
        sys.executable,
        "-m",
        "mmml.cli.__main__",
        "physnet-evaluate",
        "--checkpoint",
        str(Path(checkpoint_json).resolve()),
        "--data",
        str(Path(valid_npz).resolve()),
        "-o",
        str(out.resolve()),
        "--natoms",
        str(int(natoms)),
        "--batch-size",
        str(int(batch_size)),
        "--no-save-npz",
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(root) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    subprocess.run(cmd, check=True, cwd=root, env=env)
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "energy" in metrics and isinstance(metrics["energy"], dict):
        e_mae = float(metrics["energy"].get("mae_kcal_mol", np.nan))
        f_mae = float(metrics["forces"].get("mae_kcal_mol", np.nan))
    else:
        e_mae = float(metrics.get("energy_mae_kcal_mol", np.nan))
        f_mae = float(metrics.get("force_mae_kcal_mol_A", np.nan))
    return EmbeddingValidationResult(
        checkpoint_json=ckpt,
        valid_npz=valid,
        metrics_path=metrics_path,
        energy_mae_kcal_mol=e_mae,
        force_mae_kcal_mol_A=f_mae,
        metrics=metrics,
    )


def prepare_trialanine_hybrid_session(
    output_dir: Path | str,
    checkpoint_json: Path | str,
    *,
    ml_seg_id: str = DEFAULT_ML_SEG_ID,
    ml_charge: float = 1.0,
    ml_fq: bool = True,
    build_if_missing: bool = True,
    n_waters: int = 10,
    box_side_A: float = 28.0,
    seed: int = 11,
) -> EmbeddingHybridSession:
    """Load TRIA+TIP3 box, register partial MLpot, return ASE hybrid calculator."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    ensure_pycharmm_loaded()
    import pycharmm.coor as coor
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import read_psf_card_file
    from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
        _hybrid_mlpot_ase_calculator_class,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        prepare_charmm_for_trialanine_box_psf,
    )
    from mmml.utils.charmm_ase import atoms_from_psf_box

    out = Path(output_dir)
    box_json = out / "box.json"
    if build_if_missing and not box_json.is_file():
        build_embedding_box(
            out,
            n_waters=n_waters,
            box_side_A=box_side_A,
            seed=seed,
        )
    if not box_json.is_file():
        raise FileNotFoundError(f"Missing {box_json}; run md-embedding build or set build_if_missing=True")

    meta = json.loads(box_json.read_text(encoding="utf-8"))
    side = float(meta["box_side_A"])
    psf_path = out / str(meta.get("psf", "model.psf"))
    crd_path = out / str(meta.get("crd", "model.crd"))

    prepare_charmm_for_trialanine_box_psf()
    read_psf_card_file(psf_path)
    read.coor_card(str(crd_path))
    prepare_charmm_pbc(side)
    apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=side)

    import pycharmm.psf as psf

    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_by_seg_id

    n_pept = len(tuple(select_by_seg_id(ml_seg_id).get_atom_indexes()))
    if n_pept != TRAINING_N_ATOMS_AAA:
        import warnings

        warnings.warn(
            f"PEPT segment has {n_pept} atoms but training NPZ uses "
            f"{TRAINING_N_ATOMS_AAA}; MLpot E/F may fail until PSF matches "
            "aaa.ama topology (see docs/examples/aaa-ama-workflow.md).",
            stacklevel=2,
        )

    ctx = register_embedding_mlpot(
        checkpoint_json,
        ml_seg_id=ml_seg_id,
        ml_charge=ml_charge,
        ml_fq=ml_fq,
    )
    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    atoms = atoms_from_psf_box(psf_path, positions, box_side_A=side, pbc=True)
    calc_cls = _hybrid_mlpot_ase_calculator_class()
    calculator = calc_cls(ctx)
    atoms.calc = calculator
    return EmbeddingHybridSession(
        output_dir=out,
        checkpoint_json=Path(checkpoint_json),
        mlpot_ctx=ctx,
        atoms=atoms,
        calculator=calculator,
        box_meta=meta,
    )


__all__ = [
    "EmbeddingHybridSession",
    "EmbeddingValidationResult",
    "export_embedding_checkpoint",
    "prepare_trialanine_hybrid_session",
    "validate_embedding_monomer_potential",
]
