"""Example YAML + lattice packing for the 32 Å PET-MAD ethanol PBC recipe."""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest
from ase.io import read as ase_read

from mmml.cli.run.md_config import load_yaml_config
from mmml.cli.run.md_system import build_pycharmm_command, parse_md_system_args
from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
    SOLVENT_BULK_PROPS,
    apply_box_auto_count_composition,
    n_molecules_for_target_density_in_fixed_box,
    total_mass_g_for_composition,
)
from mmml.utils.geometry_checks import tile_monomer_in_cubic_cell

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "pet_mad_etoh_pbc"
YAML = EXAMPLE / "yaml" / "pbc_nvt.yaml"
MONOMER = EXAMPLE / "etoh.xyz"
BOX_A = 32.0
ETOH_RHO = float(SOLVENT_BULK_PROPS["ETOH"]["rho_g_cm3"])


def test_example_files_exist() -> None:
    assert YAML.is_file()
    assert MONOMER.is_file()
    assert (EXAMPLE / "ase_pbc_md.py").is_file()
    assert (EXAMPLE / "run_smoke.sh").is_file()
    assert (EXAMPLE / "run_nve.sh").is_file()
    assert (EXAMPLE / "plot_interaction_pes.py").is_file()
    assert (EXAMPLE / "run_interaction_pes.sh").is_file()
    atoms = ase_read(str(MONOMER))
    assert len(atoms) == 9
    assert set(atoms.get_chemical_symbols()) == {"C", "H", "O"}


def test_etoh_32A_count_is_338() -> None:
    scaled = n_molecules_for_target_density_in_fixed_box(
        composition={"ETOH": 1},
        box_side_A=BOX_A,
        target_density_g_cm3=ETOH_RHO,
    )
    assert scaled == {"ETOH": 338}
    mass = total_mass_g_for_composition(scaled)
    vol_cm3 = (BOX_A * 1.0e-8) ** 3
    assert mass / vol_cm3 == pytest.approx(ETOH_RHO, rel=0.01)


def test_box_auto_count_etoh_composition() -> None:
    import argparse

    args = argparse.Namespace(
        box_auto="count",
        box_size=BOX_A,
        target_density_g_cm3=ETOH_RHO,
        bulk_density_fraction=None,
        composition="ETOH:1",
        box_auto_count_min_molecules=1,
        box_auto_count_max_molecules=None,
        quiet=True,
    )
    scaled = apply_box_auto_count_composition(args)
    assert scaled["ETOH"] == 338
    assert args.composition == "ETOH:338"
    assert args.n_molecules == 338


def test_tile_monomer_fills_32A_liquid_count() -> None:
    monomer = ase_read(str(MONOMER))
    n_mol = 338
    pos, offsets = tile_monomer_in_cubic_cell(
        monomer.get_positions(),
        n_mol,
        BOX_A,
        seed=42,
        random_rotations=True,
    )
    assert pos.shape == (n_mol * 9, 3)
    assert offsets.tolist() == list(range(0, n_mol * 9 + 1, 9))
    assert np.all(np.isfinite(pos))
    assert np.all(pos >= -1.0) and np.all(pos <= BOX_A + 1.0)
    coms = np.stack(
        [pos[int(offsets[i]) : int(offsets[i + 1])].mean(axis=0) for i in range(n_mol)]
    )
    assert coms.min() >= 0.0
    assert coms.max() <= BOX_A


def test_pbc_nvt_yaml_defaults() -> None:
    cfg = load_yaml_config(YAML)
    defaults = cfg["defaults"]
    assert defaults["composition"] == "ETOH:338"
    assert defaults["box_size"] == pytest.approx(32.0)
    assert defaults["target_density_g_cm3"] == pytest.approx(0.789)
    assert defaults["temperature"] == pytest.approx(300.0)
    assert defaults["dt_fs"] == pytest.approx(0.5)
    assert defaults["ml_potential_mode"] == "metatomic"
    assert defaults["metatomic_eval_mode"] == "whole_system"
    assert defaults["include_mm"] is False
    assert defaults["mlpot_pbc"] is True
    assert "nve_smoke" in cfg["runs"]
    assert "nve" in cfg["runs"]
    assert "nvt" in cfg["runs"]
    smoke = cfg["runs"]["nve_smoke"]
    assert smoke["setup"] == "pbc_nve"
    assert smoke["ps_nve"] == pytest.approx(0.0025)
    assert smoke["backend"] == "pycharmm"
    nve = cfg["runs"]["nve"]
    assert nve["setup"] == "pbc_nve"
    assert nve["ps_nve"] == pytest.approx(0.2)
    nvt = cfg["runs"]["nvt"]
    assert nvt["setup"] == "pbc_nvt"
    assert defaults["temperature"] == pytest.approx(300.0)


def test_pbc_nvt_yaml_forwards_metatomic_flags(tmp_path: Path) -> None:
    dummy = tmp_path / "pet-mad.pt"
    dummy.write_bytes(b"not-a-real-model")
    args = parse_md_system_args(
        [
            "--config",
            str(YAML),
            "--checkpoint",
            str(dummy),
        ]
    )
    assert args.composition == "ETOH:338"
    assert args.box_size == pytest.approx(32.0)
    assert args.dt_fs == pytest.approx(0.5)
    assert args.temperature == pytest.approx(300.0)
    assert args.ml_potential_mode == "metatomic"
    assert args.metatomic_eval_mode == "whole_system"
    assert args.include_mm is False
    assert args.mlpot_pbc is True
    cmd = build_pycharmm_command(args)
    assert "--ml-potential-mode" in cmd
    assert "metatomic" in cmd
    assert "--metatomic-eval-mode" in cmd
    assert "whole_system" in cmd
    assert "--no-include-mm" in cmd
    assert "--mlpot-pbc" in cmd
    assert cmd[cmd.index("--dt-fs") + 1] == "0.5"
    assert cmd[cmd.index("--temperature") + 1] == "300.0"
    assert "ETOH:338" in cmd
    assert cmd[cmd.index("--box-size") + 1] == "32.0"


def test_example_wrapper_reexports_conservation_stats() -> None:
    import importlib.util

    path = EXAMPLE / "ase_pbc_md.py"
    spec = importlib.util.spec_from_file_location("pet_mad_etoh_ase_pbc_md", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from mmml.md.metatomic_pbc import nve_conservation_stats as pkg_stats

    assert mod.nve_conservation_stats is pkg_stats


def test_metatomic_pbc_md_parser_etoh_defaults() -> None:
    from mmml.cli.misc.metatomic_pbc_md import DEFAULT_MONOMER_XYZ, DEFAULT_OUTPUT_DIR, build_parser
    from mmml.md.metatomic_pbc import n_molecules_for_residue_box

    args = build_parser().parse_args([])
    assert args.residue == "ETOH"
    assert args.box_size == pytest.approx(32.0)
    assert args.dt_fs == pytest.approx(0.5)
    assert args.temperature == pytest.approx(300.0)
    assert args.ensemble == "nvt"
    assert args.n_steps == 5
    assert args.checkpoint is None
    assert args.monomer_xyz is None
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert n_molecules_for_residue_box("ETOH", box_side_A=32.0) == 338
    nve = build_parser().parse_args(
        ["--ensemble", "nve", "--minimize-steps", "60", "--n-steps", "400"]
    )
    assert nve.ensemble == "nve"
    assert nve.minimize_steps == 60
    assert nve.n_steps == 400
    help_text = build_parser().format_help()
    assert DEFAULT_MONOMER_XYZ.as_posix() in help_text
    assert DEFAULT_OUTPUT_DIR.as_posix() in help_text
    assert str(REPO.resolve()) not in help_text


def test_default_etoh_monomer_xyz_is_repo_example() -> None:
    from mmml.md.metatomic_pbc import default_etoh_monomer_xyz

    path = default_etoh_monomer_xyz()
    assert path == MONOMER.resolve()
    assert path.is_file()
    assert path.name == "etoh.xyz"
    assert path.parent.name == "pet_mad_etoh_pbc"


def test_metatomic_pbc_md_missing_checkpoint(tmp_path: Path) -> None:
    from mmml.cli.misc.metatomic_pbc_md import main

    rc = main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--output-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(tmp_path / "report.json"),
        ]
    )
    assert rc == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["ok"] is False
    assert "missing.pt" in report["checkpoint"]


def test_nve_conservation_stats_zero_drift() -> None:
    from mmml.md.metatomic_pbc import nve_conservation_stats

    t = np.array([0.0, 0.1, 0.2, 0.3])
    e = np.array([-10.0, -10.0, -10.0, -10.0])
    stats = nve_conservation_stats(t, e, n_atoms=100)
    assert stats["etot_drift_eV"] == pytest.approx(0.0)
    assert stats["drift_eV_per_ps"] == pytest.approx(0.0, abs=1e-12)
    assert stats["etot_span_eV"] == pytest.approx(0.0)
    assert stats["time_span_ps"] == pytest.approx(0.3)


def test_nve_conservation_stats_linear_drift() -> None:
    from mmml.md.metatomic_pbc import nve_conservation_stats

    t = np.array([0.0, 0.5, 1.0])
    e = np.array([0.0, 1.0, 2.0])  # 2 eV/ps
    stats = nve_conservation_stats(t, e, n_atoms=200)
    assert stats["drift_eV_per_ps"] == pytest.approx(2.0)
    assert stats["etot_drift_eV"] == pytest.approx(2.0)
    assert stats["drift_meV_per_atom_ps"] == pytest.approx(10.0)
    assert stats["rel_drift_per_ps"] == pytest.approx(2.0)  # mean E = 1


def test_nve_conservation_stats_rejects_bad_traces() -> None:
    from mmml.md.metatomic_pbc import nve_conservation_stats

    with pytest.raises(ValueError, match="length >= 2"):
        nve_conservation_stats(np.array([0.0]), np.array([1.0]), n_atoms=1)
    with pytest.raises(ValueError, match="must increase"):
        nve_conservation_stats(np.array([1.0, 1.0]), np.array([0.0, 1.0]), n_atoms=1)
    with pytest.raises(ValueError, match="finite"):
        nve_conservation_stats(
            np.array([0.0, 1.0]), np.array([0.0, np.nan]), n_atoms=1
        )

