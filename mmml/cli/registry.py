"""MMML CLI command registry: dispatch metadata, completion, deprecation audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandStatus = Literal["active", "deprecated", "legacy"]


@dataclass(frozen=True)
class CommandSpec:
    """One ``mmml <command>`` entry."""

    name: str
    module: str
    summary: str
    status: CommandStatus = "active"
    replacement: str | None = None
    note: str | None = None
    parser_module: str | None = None
    """Import path for ``build_parser`` when different from ``module``."""


# Keep in sync with ``mmml.cli.__main__`` dispatch and ``MMML_COMMANDS``.
COMMAND_REGISTRY: tuple[CommandSpec, ...] = (
    CommandSpec("make-res", "mmml.cli.make.make_res", "CGENFF residue → PDB/PSF/topology"),
    CommandSpec("make-box", "mmml.cli.make.make_box", "Pack molecules into a periodic box"),
    CommandSpec("build-crystal", "mmml.cli.misc.build_crystal", "Symmetry-aware crystals (PyXtal)"),
    CommandSpec(
        "run",
        "mmml.cli.run.run_sim",
        "MM/ML simulation (ASE + JAX-MD hybrid)",
        status="legacy",
        replacement="md-system",
        note="Prefer md-system for new MD; run kept for hybrid calculator demos.",
    ),
    CommandSpec("md-system", "mmml.cli.run.md_system", "Mixed-composition MD (ASE/JAX-MD/PyCHARMM)"),
    CommandSpec("liquid-box", "mmml.cli.run.liquid_box", "Build/certify periodic liquid boxes (MM only)"),
    CommandSpec("mpi-check", "mmml.cli.run.mpi_check", "Validate OpenMPI/CHARMM/mpi4py for MLpot"),
    CommandSpec("health-check", "mmml.cli.run.health_check", "Validate MMML/PyCHARMM/JAX interface health"),
    CommandSpec("warmup-mlpot-jax", "mmml.cli.run.warmup_mlpot_jax", "Serial JAX JIT warmup for MLpot"),
    CommandSpec("lambda-mbar", "mmml.cli.run.lambda_mbar", "MBAR post-processing for lambda TI"),
    CommandSpec(
        "run-pycharmm",
        "mmml.cli.run.run_pycharmm",
        "Pure CHARMM heating/equilibration",
        status="legacy",
        replacement="md-system --backend pycharmm (no ML checkpoint)",
        note="Pure MM CHARMM without MLpot; md-system covers ML workflows.",
    ),
    CommandSpec(
        "pycharmm-two-residue-sample",
        "mmml.cli.run.pycharmm_two_residue_sample",
        "Restrained sampling for two-residue CHARMM system",
    ),
    CommandSpec("xml2npz", "mmml.cli.misc.xml2npz", "Molpro XML → NPZ"),
    CommandSpec("validate", "mmml.cli.misc.validate_cli", "Validate NPZ against schema"),
    CommandSpec(
        "train",
        "mmml.cli.train.train",
        "Train DCMNet or legacy unified trainer",
        status="deprecated",
        replacement="physnet-train (PhysNet) or train-joint (PhysNet+DCMNet)",
    ),
    CommandSpec("train-joint", "mmml.cli.misc.train_joint", "Joint PhysNet+DCMNet training"),
    CommandSpec(
        "evaluate",
        "mmml.cli.misc.evaluate",
        "Legacy unified model evaluation",
        status="deprecated",
        replacement="physnet-evaluate or efield-evaluate",
    ),
    CommandSpec("downstream", "mmml.cli.misc.downstream", "Downstream analysis utilities"),
    CommandSpec("fix-and-split", "mmml.cli.misc.fix_and_split", "Unit fixes + train/valid/test splits"),
    CommandSpec("pyscf-dft", "mmml.cli.misc.pyscf_dft", "GPU DFT (energy, gradient, hessian, …)", parser_module="mmml.interfaces.pyscf4gpuInterface.calcs"),
    CommandSpec("pyscf-mp2", "mmml.cli.misc.pyscf_mp2", "GPU MP2"),
    CommandSpec("pyscf-evaluate", "mmml.cli.misc.pyscf_evaluate", "Batch E/F/D/ESP evaluation"),
    CommandSpec("pyscf-evaluate-mp2", "mmml.cli.misc.pyscf_evaluate_mp2", "Batch MP2 evaluation"),
    CommandSpec("verify-esp-alignment", "mmml.cli.misc.verify_esp_alignment", "Verify ESP grid alignment in NPZ"),
    CommandSpec("normal-mode-sample", "mmml.cli.misc.normal_mode_sample", "Sample along vibrational modes"),
    CommandSpec("physnet-train", "mmml.cli.make.make_training", "Train PhysNet message-passing model (E/F)"),
    CommandSpec("physnet-md", "mmml.cli.misc.physnet_md", "PhysNet MD sampling"),
    CommandSpec("physnet-evaluate", "mmml.cli.misc.physnet_evaluate", "Evaluate PhysNet checkpoint"),
    CommandSpec("compare-npz", "mmml.cli.misc.compare_npz", "Reference vs model NPZ plots"),
    CommandSpec("cross-check", "mmml.cli.misc.cross_check", "Supplementary QC cross-check"),
    CommandSpec("efield-train", "mmml.cli.misc.efield_train", "Train external electric-field PhysNet", parser_module="mmml.models.efield.training"),
    CommandSpec("efield-evaluate", "mmml.cli.misc.efield_evaluate", "Evaluate external electric-field PhysNet", parser_module="mmml.models.efield.evaluate"),
    CommandSpec("efield-md", "mmml.cli.misc.efield_md", "MD with external electric-field PhysNet"),
    CommandSpec(
        "ef-train",
        "mmml.cli.misc.ef_train",
        "Train EF equivariant model (deprecated)",
        status="deprecated",
        replacement="efield-train",
        parser_module="mmml.models.EF.training",
    ),
    CommandSpec(
        "ef-evaluate",
        "mmml.cli.misc.ef_evaluate",
        "Evaluate EF model (deprecated)",
        status="deprecated",
        replacement="efield-evaluate",
        parser_module="mmml.models.EF.evaluate",
    ),
    CommandSpec(
        "ef-md",
        "mmml.cli.misc.ef_md",
        "MD with trained EF model (deprecated)",
        status="deprecated",
        replacement="efield-md",
    ),
    CommandSpec("active-learning", "mmml.cli.misc.active_learning", "Sample structures for re-labeling"),
    CommandSpec("kernel-fit", "mmml.cli.misc.kernel_fit", "Kernel fitting utilities"),
    CommandSpec("interpolate-xyz", "mmml.cli.misc.interpolate_xyz", "Interpolate XYZ via Z-matrix → NPZ"),
    CommandSpec("unwrap-traj", "mmml.cli.misc.unwrap_traj", "Unwrap periodic trajectories"),
    CommandSpec(
        "sample-diverse-xyz",
        "mmml.generate.sample",
        "Pick diverse structures (SOAP) → NPZ",
        parser_module="mmml.generate.sample",
    ),
    CommandSpec("gui", "mmml.cli.gui", "Molecular viewer GUI"),
    CommandSpec(
        "extract-checkpoint-metrics",
        "mmml.cli.misc.extract_checkpoint_metrics",
        "Plot training metrics from Orbax checkpoints",
    ),
    CommandSpec("orbax-to-json", "mmml.cli.misc.orbax_to_json_cmd", "Export Orbax checkpoint to JSON"),
    CommandSpec("orca-server", "mmml.interfaces.orca_external.server", "Persistent JAX server for ORCA"),
    CommandSpec("orca-client", "mmml.interfaces.orca_external.client", "ORCA client → orca-server"),
    CommandSpec("orca-external", "mmml.interfaces.orca_external.runner", "Standalone ORCA external wrapper"),
    CommandSpec("configure", "mmml.cli.configure", "Interactive config / Snakemake wizard"),
    CommandSpec("commands", "mmml.cli.commands_help", "Browse subcommands (grouped)"),
    CommandSpec("examples", "mmml.cli.commands_help", "Copy-paste example invocations"),
    CommandSpec("completion", "mmml.cli.completion", "Shell tab-completion setup"),
)

MMML_COMMANDS: tuple[str, ...] = tuple(spec.name for spec in COMMAND_REGISTRY)

_DISPATCH_COMMANDS: tuple[str, ...] = tuple(
    name for name in MMML_COMMANDS if name != "completion"
)


def command_by_name(name: str) -> CommandSpec | None:
    for spec in COMMAND_REGISTRY:
        if spec.name == name:
            return spec
    return None


def format_audit_report() -> str:
    lines = [
        "MMML CLI audit — active, legacy, and deprecated commands",
        "",
        "Deprecated / legacy (prefer replacement):",
    ]
    for spec in COMMAND_REGISTRY:
        if spec.status == "active":
            continue
        rep = f" → use {spec.replacement}" if spec.replacement else ""
        lines.append(f"  {spec.name:<28} [{spec.status}]{rep}")
        if spec.note:
            lines.append(f"    {spec.note}")
    lines.extend(["", "Active commands with tab-completion when build_parser() exists:", ""])
    from mmml.cli.parser_utils import parser_available

    for spec in COMMAND_REGISTRY:
        if spec.status != "active":
            continue
        flag = "✓ flags" if parser_available(spec.name, import_module=False) else "  (top-level only)"
        lines.append(f"  {spec.name:<28} {flag}  {spec.summary}")
    lines.append("")
    lines.append("Install: pip install 'mmml[cli]' && eval \"$(register-python-argcomplete mmml)\"")
    return "\n".join(lines)
