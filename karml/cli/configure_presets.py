"""Bundled ``mmml configure --preset`` definitions (cpu_tests / tutorial layouts)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_PKG = Path(__file__).resolve().parent
_TEMPLATES = _PKG / "templates" / "workflows"
_EXAMPLES = _PKG / "run"


@dataclass(frozen=True)
class ConfigurePreset:
    key: str
    title: str
    description: str
    kind: Literal["yaml-copy", "snakemake-template", "multi-yaml"]
    run_hint: str
    source_yaml: Path | None = None
    template_dir: Path | None = None
    extra_yamls: tuple[tuple[str, Path], ...] = ()


PRESETS: tuple[ConfigurePreset, ...] = (
    ConfigurePreset(
        key="cpu-spatial-mpi-mini",
        title="CPU/GPU: spatial MPI mini smoke",
        description=(
            "PyCHARMM pbc_npt mini-only (DCM:20, ~20 SD steps). "
            "Matches mmml/cli/run/md_system.spatial_mpi.example.yaml and "
            "tests/functionality/mlpot/07_md_system_spatial_mpi_mini.py."
        ),
        kind="yaml-copy",
        source_yaml=_EXAMPLES / "md_system.spatial_mpi.example.yaml",
        run_hint=(
            "MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 "
            "./scripts/mmml-charmm-mpirun.sh md-system --config md_system.spatial_mpi.yaml"
        ),
    ),
    ConfigurePreset(
        key="cpu-dense-liquid-prep",
        title="CPU/GPU: dense liquid prep campaign",
        description=(
            "PyCHARMM equil → JAX-MD prod handoff (DCM:206). "
            "Uses composable presets under mmml/cli/run/presets/."
        ),
        kind="yaml-copy",
        source_yaml=_EXAMPLES / "md_system.dense_liquid_prep.example.yaml",
        run_hint="mmml md-system --config md_campaign.dense_liquid.yaml --run-all",
    ),
    ConfigurePreset(
        key="cpu-md-benchmark",
        title="Snakemake: DCM:5 cross-backend benchmark",
        description=(
            "2 ps smoke matrix (ASE/JAX-MD/PyCHARMM). "
            "Mirrors workflows/dcm5_md_benchmark with Slurm profile."
        ),
        kind="snakemake-template",
        template_dir=_TEMPLATES / "md_benchmark",
        run_hint="cd md_benchmark_workflow && snakemake -n && snakemake -j2 --resources gpu=1 mpi=1",
    ),
    ConfigurePreset(
        key="cpu-heat-scaling-smoke",
        title="Snakemake: DCM heat scaling (smoke)",
        description=(
            "PyCHARMM heat-only scaling smoke (N=5,10 × dt=0.25). "
            "Full sweep: workflows/dcm_heat_scaling in repo."
        ),
        kind="snakemake-template",
        template_dir=_TEMPLATES / "heat_scaling_smoke",
        run_hint="cd heat_scaling_smoke && snakemake -n -j1 --resources gpu=1 charmm_slot=1",
    ),
    ConfigurePreset(
        key="cpu-nve-cutoff-sweep",
        title="Snakemake: NVE cutoff sweep (smoke)",
        description="Extended MM cutoff × NVE leg smoke. See workflows/dcm3_nve_cutoff_sweep.",
        kind="snakemake-template",
        template_dir=_TEMPLATES / "nve_cutoff_smoke",
        run_hint="cd nve_cutoff_smoke && snakemake -n",
    ),
    ConfigurePreset(
        key="qm-physnet-pipeline",
        title="QM → PhysNet training pipeline (YAML set)",
        description=(
            "Writes fix-and-split + pyscf-evaluate + physnet-train example configs "
            "for a standard MP2 → PhysNet workflow."
        ),
        kind="multi-yaml",
        run_hint=(
            "mmml fix-and-split --config qm_pipeline/fix_and_split.yaml && "
            "mmml pyscf-evaluate --config qm_pipeline/pyscf_evaluate.yaml && "
            "mmml physnet-train --config qm_pipeline/physnet_train.yaml"
        ),
    ),
)

PRESET_BY_KEY: dict[str, ConfigurePreset] = {p.key: p for p in PRESETS}


def list_presets_text() -> str:
    lines = ["Available configure presets:", ""]
    for p in PRESETS:
        lines.append(f"  {p.key}")
        lines.append(f"    {p.title}")
        lines.append(f"    {p.description}")
        lines.append("")
    lines.append("Apply:  mmml configure --preset <key> -o ./out")
    return "\n".join(lines).rstrip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _copy_tree(src: Path, dest: Path) -> list[Path]:
    import shutil

    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    shutil.copytree(src, dest)
    return [p for p in dest.rglob("*") if p.is_file()]


def apply_preset(preset: ConfigurePreset, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if preset.kind == "yaml-copy":
        assert preset.source_yaml is not None
        dest = out_dir / preset.source_yaml.name.replace(".example", "")
        dest.write_text(_read_text(preset.source_yaml), encoding="utf-8")
        written.append(dest)
        return written

    if preset.kind == "snakemake-template":
        assert preset.template_dir is not None
        dest = out_dir / f"{preset.template_dir.name}_workflow"
        written = _copy_tree(preset.template_dir, dest)
        return written

    if preset.kind == "multi-yaml":
        sub = out_dir / "qm_pipeline"
        sub.mkdir(parents=True, exist_ok=True)
        configs: dict[str, dict[str, Any]] = {
            "fix_and_split.yaml": {
                "efd": "data/raw_evaluated.npz",
                "output_dir": "splits",
                "train_frac": 0.8,
                "valid_frac": 0.1,
                "test_frac": 0.1,
                "dipole_in": "debye",
                "dipole_out": "e-angstrom",
            },
            "pyscf_evaluate.yaml": {
                "input": "sampled.npz",
                "output": "data/raw_evaluated.npz",
                "basis": "def2-SVP",
                "xc": "PBE0",
            },
            "physnet_train.yaml": {
                "data": "splits/energies_forces_dipoles_train.npz",
                "valid_data": "splits/energies_forces_dipoles_valid.npz",
                "ckpt_dir": "./ckpts/qm_pipeline",
                "tag": "qm_pipeline",
                "batch_size": 32,
                "num_epochs": 500,
            },
        }
        import yaml

        for name, body in configs.items():
            path = sub / name
            path.write_text(
                f"# Generated by mmml configure --preset qm-physnet-pipeline\n"
                + yaml.safe_dump(body, sort_keys=False),
                encoding="utf-8",
            )
            written.append(path)
        readme = sub / "README.md"
        readme.write_text(
            "# QM → PhysNet pipeline\n\n"
            "1. `mmml normal-mode-sample` or provide geometries → `sampled.npz`\n"
            "2. `mmml pyscf-evaluate -i sampled.npz -o data/raw_evaluated.npz`\n"
            "3. `mmml fix-and-split --efd data/raw_evaluated.npz -o splits`\n"
            "4. `mmml physnet-train --config qm_pipeline/physnet_train.yaml`\n",
            encoding="utf-8",
        )
        written.append(readme)
        return written

    raise ValueError(f"Unknown preset kind: {preset.kind}")
