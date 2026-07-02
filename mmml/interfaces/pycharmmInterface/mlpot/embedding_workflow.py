"""Solvated-peptide MD embedding workflow (partial MLpot, separate from cluster md-system)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_ML_SEG_ID = "PEPT"
DEFAULT_TRAIN_TAG = "aaa_smoke"
TRAINING_N_ATOMS_AAA = 34


@dataclass(frozen=True, slots=True)
class TrainPhaseResult:
    output_dir: Path
    train_npz: Path
    valid_npz: Path
    train_config: Path
    manifest_path: Path
    checkpoint_json: Path | None
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BuildPhaseResult:
    output_dir: Path
    psf_path: Path
    crd_path: Path
    box_json_path: Path
    n_peptide_atoms: int
    n_waters: int
    box_side_A: float
    bonded_report: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class RunPhaseResult:
    output_dir: Path
    ml_seg_id: str
    n_ml_atoms: int
    n_total_atoms: int
    charmm_total_energy_kcalmol: float | None
    minimized: bool


def split_npz_dataset(
    npz_path: Path | str,
    out_dir: Path | str,
    *,
    train_fraction: float = 0.9,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Shuffle-split an NPZ into train/valid files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(Path(npz_path), allow_pickle=True)
    n = int(len(data["E"]))
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = max(1, int(train_fraction * n))
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:]
    if valid_idx.size == 0:
        valid_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    def _subset(idxs: np.ndarray) -> dict[str, np.ndarray]:
        return {k: np.asarray(data[k])[idxs] for k in data.files}

    train_path = out / "train.npz"
    valid_path = out / "valid.npz"
    np.savez(train_path, **_subset(train_idx))
    np.savez(valid_path, **_subset(valid_idx))
    return train_path, valid_path


def split_via_fix_and_split(
    npz_path: Path | str,
    out_dir: Path | str,
    *,
    train_fraction: float = 0.9,
    seed: int = 0,
) -> tuple[Path, Path, Path]:
    """Split NPZ with ``mmml fix-and-split`` (units manifest + reproducible indices)."""
    from mmml.cli.misc.fix_and_split import fix_and_split_data

    out = Path(out_dir)
    split_root = out / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    valid_fraction = max(0.0, 1.0 - float(train_fraction))
    ok = fix_and_split_data(
        efd_file=Path(npz_path),
        grid_file=None,
        output_dir=split_root,
        train_frac=float(train_fraction),
        valid_frac=valid_fraction,
        test_frac=0.0,
        seed=int(seed),
        skip_validation=True,
        preserve_units=True,
        coords_in="angstrom",
        coords_out="same",
        energy_in="ev",
        energy_out="same",
        force_in="ev_angstrom",
        force_out="same",
        verbose=False,
    )
    if ok is False:
        raise RuntimeError(f"fix-and-split failed for {npz_path}")

    train_src = split_root / "energies_forces_dipoles_train.npz"
    valid_src = split_root / "energies_forces_dipoles_valid.npz"
    if not train_src.is_file() or not valid_src.is_file():
        raise FileNotFoundError(f"fix-and-split did not write train/valid NPZ under {split_root}")

    train_path = out / "train.npz"
    valid_path = out / "valid.npz"
    shutil.copy2(train_src, train_path)
    shutil.copy2(valid_src, valid_path)
    return train_path, valid_path, split_root


def prepare_train_valid_npz(
    npz_path: Path | str,
    out_dir: Path | str,
    *,
    train_fraction: float = 0.9,
    seed: int = 0,
    use_fix_and_split: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Return ``(train.npz, valid.npz, splits_dir_or_none)``."""
    if use_fix_and_split:
        train_p, valid_p, split_root = split_via_fix_and_split(
            npz_path,
            out_dir,
            train_fraction=train_fraction,
            seed=seed,
        )
        return train_p, valid_p, split_root
    train_p, valid_p = split_npz_dataset(
        npz_path, out_dir, train_fraction=train_fraction, seed=seed
    )
    return train_p, valid_p, None


def plot_npz_peptide_frame(
    data: dict[str, np.ndarray],
    out_path: Path | str,
    *,
    frame: int = 0,
    title: str | None = None,
) -> Path:
    """Orthographic ASE figure (bonds) for one peptide NPZ frame."""
    from mmml.data.external.aaa_ama import atoms_from_npz_frame, inspect_dataset_aaa
    from mmml.utils.ase_structure_plot import (
        SCALE_PEPTIDE_ML,
        save_structure_figure,
        use_matplotlib_agg,
    )

    use_matplotlib_agg()
    report = inspect_dataset_aaa(data)
    atoms = atoms_from_npz_frame(data, frame=frame)
    label = title or f"{report.molecule_label} (frame {frame})"
    return save_structure_figure(
        atoms,
        out_path,
        title=label,
        rotation="25x,15y,0z",
        scale=SCALE_PEPTIDE_ML,
    )


def plot_embedding_box_structures(
    psf_path: Path | str,
    positions: np.ndarray,
    *,
    box_side_A: float,
    n_peptide_atoms: int,
    out_dir: Path | str,
) -> list[Path]:
    """Write full-box and peptide-zoom PNGs (ASE bonds, docs style)."""
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        peptide_atoms_from_trialanine_box,
    )
    from mmml.utils.ase_structure_plot import (
        SCALE_TRIALANINE_BOX,
        SCALE_TRIALANINE_PEPTIDE,
        save_structure_figure,
        use_matplotlib_agg,
    )
    from mmml.utils.charmm_ase import atoms_from_psf_box

    use_matplotlib_agg()
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    full = atoms_from_psf_box(
        psf_path,
        positions,
        box_side_A=box_side_A,
        pbc=True,
    )
    peptide = peptide_atoms_from_trialanine_box(
        full,
        n_peptide_atoms=n_peptide_atoms,
    )
    paths = [
        save_structure_figure(
            full,
            fig_dir / "embedding_box.png",
            title=f"md-embedding box ({len(full)} atoms, L={box_side_A:.1f} Å)",
            rotation="55x,25y,0z",
            scale=SCALE_TRIALANINE_BOX,
        ),
        save_structure_figure(
            peptide,
            fig_dir / "embedding_peptide.png",
            title=f"PEPT segment ({len(peptide)} atoms; CGENFF TRIA)",
            rotation="25x,15y,0z",
            scale=SCALE_TRIALANINE_PEPTIDE,
        ),
    ]
    return paths


def default_train_config_dict(
    output_dir: Path | str,
    *,
    tag: str = DEFAULT_TRAIN_TAG,
    num_atoms: int = TRAINING_N_ATOMS_AAA,
) -> dict[str, Any]:
    """Small PhysNet smoke hyperparameters for aaa.ama peptide."""
    root = Path(output_dir)
    return {
        "data": str(root / "train.npz"),
        "valid_data": str(root / "valid.npz"),
        "ckpt_dir": str(root / "checkpoints"),
        "tag": tag,
        "num_atoms": int(num_atoms),
        "max_atomic_number": 8,
        "features": 32,
        "num_basis_functions": 32,
        "num_iterations": 2,
        "n_res": 1,
        "cutoff": 5.0,
        "batch_size": 16,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "energy_weight": 1.0,
        "forces_weight": 50.0,
        "objective": "valid_loss",
        "seed": 42,
    }


def write_train_config(
    output_dir: Path | str,
    path: Path | str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg = default_train_config_dict(output_dir)
    if overrides:
        cfg.update(overrides)
    out = Path(path) if path is not None else Path(output_dir) / "train_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _latest_orbax_epoch(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    epochs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("epoch-")),
        key=lambda p: int(p.name.split("-", 1)[1]) if "-" in p.name else 0,
    )
    return epochs[-1] if epochs else None


def _latest_train_run_dir(ckpt_dir: Path, tag: str) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    runs = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith(f"{tag}-")),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None


def run_train_phase(
    output_dir: Path | str,
    *,
    npz_path: Path | str | None = None,
    download: bool = True,
    train_fraction: float = 0.9,
    seed: int = 0,
    skip_train: bool = False,
    skip_export_json: bool = False,
    tag: str = DEFAULT_TRAIN_TAG,
    config_overrides: dict[str, Any] | None = None,
    use_fix_and_split: bool = True,
    write_plots: bool = True,
) -> TrainPhaseResult:
    """Download/split NPZ, optional PhysNet train, export JSON checkpoint manifest."""
    from mmml.data.external.aaa_ama import (
        AAA_DATASET_URL,
        download_dataset_aaa,
        inspect_dataset_aaa,
        load_dataset_aaa,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if npz_path is None:
        npz_path = out / "dataset_aaa.npz"
    npz_path = Path(npz_path)
    if download or not npz_path.is_file():
        download_dataset_aaa(npz_path)

    report = inspect_dataset_aaa(load_dataset_aaa(npz_path))
    train_npz, valid_npz, split_root = prepare_train_valid_npz(
        npz_path,
        out,
        train_fraction=train_fraction,
        seed=seed,
        use_fix_and_split=use_fix_and_split,
    )
    figure_paths: list[str] = []
    if write_plots:
        try:
            fig = plot_npz_peptide_frame(
                load_dataset_aaa(npz_path),
                out / "figures" / "peptide_frame0.png",
            )
            figure_paths.append(str(fig.resolve()))
        except Exception:
            pass
    train_config = write_train_config(out, overrides=config_overrides)

    checkpoint_json: Path | None = None
    if not skip_train:
        cmd = [
            sys.executable,
            "-m",
            "mmml.cli.__main__",
            "physnet-train",
            "--config",
            str(train_config),
        ]
        subprocess.run(cmd, check=True, cwd=out)
        if not skip_export_json:
            run_dir = _latest_train_run_dir(out / "checkpoints", tag)
            epoch_dir = _latest_orbax_epoch(run_dir) if run_dir is not None else None
            if epoch_dir is not None:
                checkpoint_json = out / f"{tag}_params.json"
                export_cmd = [
                    sys.executable,
                    "-m",
                    "mmml.cli.__main__",
                    "orbax-to-json",
                    str(epoch_dir),
                    "-o",
                    str(checkpoint_json),
                ]
                subprocess.run(export_cmd, check=True)

    manifest = {
        "phase": "train",
        "dataset_url": AAA_DATASET_URL,
        "npz_path": str(npz_path.resolve()),
        "train_npz": str(train_npz.resolve()),
        "valid_npz": str(valid_npz.resolve()),
        "split_dir": str(split_root.resolve()) if split_root is not None else None,
        "use_fix_and_split": use_fix_and_split,
        "figures": figure_paths,
        "train_config": str(train_config.resolve()),
        "checkpoint_json": str(checkpoint_json.resolve()) if checkpoint_json else None,
        "tag": tag,
        "dataset_report": report.to_json_dict(),
        "skip_train": skip_train,
    }
    manifest_path = out / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return TrainPhaseResult(
        output_dir=out,
        train_npz=train_npz,
        valid_npz=valid_npz,
        train_config=train_config,
        manifest_path=manifest_path,
        checkpoint_json=checkpoint_json,
        report=manifest,
    )


def build_embedding_box(
    output_dir: Path | str,
    *,
    n_waters: int = 10,
    box_side_A: float = 28.0,
    seed: int = 11,
    skip_reset_block: bool = True,
    charmm_mm_minimize: bool = True,
    charmm_sd_steps: int = 200,
    write_plots: bool = True,
) -> BuildPhaseResult:
    """Build CGENFF TRIA + TIP3 box; MM-only minimize; write PSF/CRD/box.json."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        TRIA_RESI_NAME,
        build_trialanine_water_box_in_charmm,
        n_peptide_atoms_in_trialanine_box,
    )

    ensure_pycharmm_loaded()
    import pycharmm.coor as coor
    import pycharmm.minimize as minimize
    import pycharmm.write as write

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        run_charmm_bonded_ener_force,
        setup_bonded_only_charmm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    box = build_trialanine_water_box_in_charmm(
        n_waters=n_waters,
        box_side_A=box_side_A,
        seed=seed,
        workdir=out,
        skip_reset_block=skip_reset_block,
    )
    n_peptide = n_peptide_atoms_in_trialanine_box(box.psf_path)

    if charmm_mm_minimize:
        apply_charmm_mm_block()
        minimize.run_sd(nstep=int(charmm_sd_steps), tolenr=1e-4, tolgrd=1e-3)

    psf_path = out / "model.psf"
    crd_path = out / "model.crd"
    prev = os.getcwd()
    try:
        os.chdir(out)
        write.psf_card(psf_path.name)
        write.coor_card(crd_path.name)
    finally:
        os.chdir(prev)

    bonded_report: dict[str, float] | None = None
    try:
        setup_bonded_only_charmm()
        run_charmm_bonded_ener_force(silent=True)
        bonded = charmm_bonded_energy_components_kcalmol()
        bonded_report = {k: float(v) for k, v in bonded.items()}
        (out / "bonded_report.json").write_text(
            json.dumps(bonded_report, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        bonded_report = None

    box_meta = {
        "workflow": "md-embedding",
        "peptide_resi": TRIA_RESI_NAME,
        "ml_seg_id": DEFAULT_ML_SEG_ID,
        "solvent_seg_id": "SOLV",
        "n_peptide_atoms": int(n_peptide),
        "n_waters": int(box.n_waters),
        "n_total_atoms": int(box.positions.shape[0]),
        "box_side_A": float(box.box_side_A),
        "training_n_atoms": TRAINING_N_ATOMS_AAA,
        "topology_note": (
            "Bundled CGENFF TRIA build has 42 peptide atoms; aaa.ama NPZ uses 34. "
            "Align PSF with training topology before comparing to NPZ E/F."
        ),
        "psf": str(psf_path.name),
        "crd": str(crd_path.name),
        "bonded_report": bonded_report,
    }
    box_json_path = out / "box.json"
    box_json_path.write_text(json.dumps(box_meta, indent=2) + "\n", encoding="utf-8")

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    np.save(out / "positions.npy", positions)

    figure_paths: list[str] = []
    if write_plots:
        try:
            for fig in plot_embedding_box_structures(
                psf_path,
                positions,
                box_side_A=float(box.box_side_A),
                n_peptide_atoms=int(n_peptide),
                out_dir=out,
            ):
                figure_paths.append(str(fig.resolve()))
            box_meta["figures"] = figure_paths
            box_json_path.write_text(json.dumps(box_meta, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    return BuildPhaseResult(
        output_dir=out,
        psf_path=psf_path,
        crd_path=crd_path,
        box_json_path=box_json_path,
        n_peptide_atoms=int(n_peptide),
        n_waters=int(box.n_waters),
        box_side_A=float(box.box_side_A),
        bonded_report=bonded_report,
    )


def register_embedding_mlpot(
    checkpoint: Path | str,
    *,
    ml_seg_id: str = DEFAULT_ML_SEG_ID,
    ml_charge: float = 1.0,
    ml_fq: bool = True,
    mlmm_ctonnb: float | None = None,
    mlmm_ctofnb: float | None = None,
) -> Any:
    """Register partial MLpot on ``ml_seg_id`` using single-monomer PhysNet (``n_monomers=1``)."""
    from ase import Atoms

    from mmml.interfaces.pycharmmInterface.mlpot.partial_mm import (
        PartialMlMmConfig,
        register_mlpot_partial_mm,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        load_physnet_mlpot_bundle,
        select_by_seg_id,
    )

    sel = select_by_seg_id(ml_seg_id)
    ml_indices = tuple(int(i) for i in sel.get_atom_indexes())
    if not ml_indices:
        raise ValueError(f"ML segment {ml_seg_id!r} has no atoms")

    import pycharmm.coor as coor
    import pycharmm.psf as psf

    positions = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    atypes = psf.get_atype()
    # Map CHARMM atom types to Z for ASE (fallback: C for unknown protein types)
    _atype_z = {
        "H": 1,
        "HC": 1,
        "HA": 1,
        "HP": 1,
        "HN": 1,
        "CT": 6,
        "CA": 6,
        "C": 6,
        "CB": 6,
        "N": 7,
        "NH1": 7,
        "NH2": 7,
        "NH3": 7,
        "O": 8,
        "OT": 8,
        "OH1": 8,
        "OG2D1": 8,
        "NG2S1": 7,
        "CG311": 6,
        "CG2O1": 6,
        "CG331": 6,
        "HGA1": 1,
        "HGA2": 1,
        "HGA3": 1,
        "HGP1": 1,
        "OG2D1": 8,
    }
    z = np.array(
        [_atype_z.get(str(atypes[i]).strip().upper(), 6) for i in ml_indices],
        dtype=int,
    )
    r = positions[np.asarray(ml_indices, dtype=int)]
    atoms = Atoms(numbers=z, positions=r)

    _, _, pyCModel = load_physnet_mlpot_bundle(
        Path(checkpoint),
        len(ml_indices),
        atoms,
        n_monomers=1,
    )
    config = PartialMlMmConfig(
        ml_seg_id=ml_seg_id,
        ml_charge=float(ml_charge),
        ml_fq=bool(ml_fq),
        mlmm_ctonnb=mlmm_ctonnb,
        mlmm_ctofnb=mlmm_ctofnb,
        use_mlmm_pair_lists=False,
    )
    return register_mlpot_partial_mm(pyCModel, z.tolist(), config)


def run_embedding_phase(
    output_dir: Path | str,
    checkpoint: Path | str,
    *,
    ml_seg_id: str = DEFAULT_ML_SEG_ID,
    ml_charge: float = 1.0,
    ml_fq: bool = True,
    mini_nstep: int = 0,
    box_side_A: float | None = None,
    mlmm_cutoff: float | None = None,
    mlmm_cuton: float | None = None,
) -> RunPhaseResult:
    """Load built box, register partial MLpot, optional MLpot SD minimize."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    ensure_pycharmm_loaded()
    import pycharmm.energy as energy
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import read_psf_card_file
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        prepare_charmm_pbc,
    )

    out = Path(output_dir)
    box_json = out / "box.json"
    if not box_json.is_file():
        raise FileNotFoundError(f"Missing {box_json}; run md-embedding build first")
    meta = json.loads(box_json.read_text(encoding="utf-8"))
    side = float(box_side_A if box_side_A is not None else meta["box_side_A"])
    psf_path = out / str(meta.get("psf", "model.psf"))
    crd_path = out / str(meta.get("crd", "model.crd"))

    read_psf_card_file(psf_path)
    read.coor_card(str(crd_path))
    prepare_charmm_pbc(side)
    apply_pbc_nbonds(nbxmod=5, cubic_box_side_A=side)

    ctx = register_embedding_mlpot(
        checkpoint,
        ml_seg_id=ml_seg_id,
        ml_charge=ml_charge,
        ml_fq=ml_fq,
        mlmm_ctonnb=mlmm_cuton,
        mlmm_ctofnb=mlmm_cutoff,
    )
    minimized = False
    total: float | None = None
    try:
        energy.show()
        total = float(energy.get_total())
        if mini_nstep > 0:
            from mmml.interfaces.pycharmmInterface.mlpot import (
                MinimizeWithMlpotConfig,
                minimize_with_mlpot,
            )

            minimize_with_mlpot(
                MinimizeWithMlpotConfig(
                    mlpot_ctx=ctx,
                    nstep=int(mini_nstep),
                    save=False,
                    calculator_pre_minimize=False,
                    verbose=False,
                )
            )
            minimized = True
            total = float(energy.get_total())
    finally:
        ctx.unset()

    n_ml = int(meta.get("n_peptide_atoms", 0))
    n_total = int(meta.get("n_total_atoms", 0))
    run_manifest = {
        "phase": "run",
        "checkpoint": str(Path(checkpoint).resolve()),
        "ml_seg_id": ml_seg_id,
        "mini_nstep": int(mini_nstep),
        "minimized": minimized,
        "charmm_total_energy_kcalmol": total,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return RunPhaseResult(
        output_dir=out,
        ml_seg_id=ml_seg_id,
        n_ml_atoms=n_ml,
        n_total_atoms=n_total,
        charmm_total_energy_kcalmol=total,
        minimized=minimized,
    )


def load_box_from_output(output_dir: Path | str) -> dict[str, Any]:
    path = Path(output_dir) / "box.json"
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BuildPhaseResult",
    "DEFAULT_ML_SEG_ID",
    "DEFAULT_TRAIN_TAG",
    "RunPhaseResult",
    "TRAINING_N_ATOMS_AAA",
    "TrainPhaseResult",
    "build_embedding_box",
    "default_train_config_dict",
    "load_box_from_output",
    "plot_embedding_box_structures",
    "plot_npz_peptide_frame",
    "prepare_train_valid_npz",
    "register_embedding_mlpot",
    "run_embedding_phase",
    "run_train_phase",
    "split_npz_dataset",
    "split_via_fix_and_split",
    "write_train_config",
]
