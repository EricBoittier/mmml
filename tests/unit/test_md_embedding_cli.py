"""Unit tests for ``mmml md-embedding`` (no CHARMM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmml.cli.run import md_embedding
from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import (
    TRAINING_N_ATOMS_AAA,
    default_train_config_dict,
    run_train_phase,
    split_npz_dataset,
    write_train_config,
)


def _minimal_aaa_npz(path: Path, n_frames: int = 20) -> None:
    n_atoms = 34
    z0 = np.array(
        [7, 1, 1, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 8, 1],
        dtype=int,
    )
    r = np.random.default_rng(0).normal(size=(n_frames, n_atoms, 3))
    e = np.linspace(-100.0, -90.0, n_frames)
    f = np.random.default_rng(1).normal(scale=0.1, size=(n_frames, n_atoms, 3))
    np.savez(
        path,
        N=np.full(n_frames, n_atoms, dtype=int),
        Z=np.tile(z0, (n_frames, 1)),
        R=r,
        E=e,
        F=f,
        Q=np.ones(n_frames),
        D=np.zeros((n_frames, 3)),
    )


def test_md_embedding_parser_subcommands():
    parser = md_embedding.build_parser()
    args = parser.parse_args(["train", "-o", "/tmp/out"])
    assert args.phase == "train"
    assert args.output_dir == Path("/tmp/out")
    args = parser.parse_args(["build", "-o", "artifacts/x", "--n-waters", "5"])
    assert args.n_waters == 5
    args = parser.parse_args(
        ["run", "-o", "artifacts/x", "--checkpoint", "ckpt.json", "--mini-nstep", "10"]
    )
    assert args.checkpoint == Path("ckpt.json")
    assert args.mini_nstep == 10


def test_split_npz_dataset_preserves_atom_count(tmp_path: Path):
    npz = tmp_path / "data.npz"
    _minimal_aaa_npz(npz, n_frames=10)
    train_p, valid_p = split_npz_dataset(npz, tmp_path, train_fraction=0.8, seed=0)
    train = np.load(train_p)
    valid = np.load(valid_p)
    assert train["Z"].shape[-1] == 34
    assert valid["Z"].shape[-1] == 34
    assert len(train["E"]) + len(valid["E"]) == 10
    assert len(train["E"]) >= 1
    assert len(valid["E"]) >= 1


def test_default_train_config_uses_34_atoms(tmp_path: Path):
    cfg = default_train_config_dict(tmp_path)
    assert cfg["num_atoms"] == TRAINING_N_ATOMS_AAA
    assert cfg["tag"] == "aaa_smoke"
    path = write_train_config(tmp_path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "num_atoms: 34" in text


def test_run_train_phase_skip_train_writes_manifest(tmp_path: Path, monkeypatch):
    npz = tmp_path / "dataset_aaa.npz"
    _minimal_aaa_npz(npz, n_frames=12)

    def _no_download(dest):
        return npz

    monkeypatch.setattr(
        "mmml.data.external.aaa_ama.download_dataset_aaa",
        _no_download,
    )
    result = run_train_phase(
        tmp_path / "out",
        npz_path=npz,
        download=False,
        skip_train=True,
        use_fix_and_split=False,
        write_plots=False,
    )
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "train"
    assert manifest["skip_train"] is True
    assert result.train_npz.is_file()
    assert result.valid_npz.is_file()
    assert result.train_config.is_file()


def test_main_train_dispatch(tmp_path: Path, monkeypatch):
    npz = tmp_path / "dataset_aaa.npz"
    _minimal_aaa_npz(npz, n_frames=8)

    monkeypatch.setattr(
        "mmml.data.external.aaa_ama.download_dataset_aaa",
        lambda dest: npz,
    )
    code = md_embedding.main(
        [
            "train",
            "-o",
            str(tmp_path / "artifacts"),
            "--npz",
            str(npz),
            "--no-download",
            "--skip-train",
            "--simple-split",
            "--no-plot",
        ]
    )
    assert code == 0
    assert (tmp_path / "artifacts" / "train_manifest.json").is_file()


def test_prepare_train_valid_npz_fix_and_split(tmp_path: Path, monkeypatch):
    npz = tmp_path / "dataset_aaa.npz"
    _minimal_aaa_npz(npz, n_frames=12)
    out = tmp_path / "out"
    train_p = out / "train.npz"
    valid_p = out / "valid.npz"
    split_root = out / "splits"

    def _fake_fix_and_split(**kwargs):
        split_root.mkdir(parents=True, exist_ok=True)
        data = np.load(npz, allow_pickle=True)
        n = len(data["E"])
        idx = np.arange(n)
        train_idx = idx[:9]
        valid_idx = idx[9:]
        subset = lambda idc: {k: np.asarray(data[k])[idc] for k in data.files}
        np.savez(split_root / "energies_forces_dipoles_train.npz", **subset(train_idx))
        np.savez(split_root / "energies_forces_dipoles_valid.npz", **subset(valid_idx))
        (split_root / "units_manifest.json").write_text("{}", encoding="utf-8")
        return True

    monkeypatch.setattr(
        "mmml.cli.misc.fix_and_split.fix_and_split_data",
        _fake_fix_and_split,
    )

    from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import (
        prepare_train_valid_npz,
    )

    t, v, sdir = prepare_train_valid_npz(npz, out, use_fix_and_split=True, seed=0)
    assert t == train_p
    assert v == valid_p
    assert sdir == split_root
    assert train_p.is_file()
    assert valid_p.is_file()
