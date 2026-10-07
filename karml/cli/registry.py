"""KARML CLI command registry: dispatch metadata, completion, deprecation audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandStatus = Literal["active", "deprecated", "legacy"]


@dataclass(frozen=True)
class CommandSpec:
    """One ``karml <command>`` entry."""

    name: str
    module: str
    summary: str
    status: CommandStatus = "active"
    replacement: str | None = None
    removal_date: str | None = None
    note: str | None = None
    parser_module: str | None = None
    """Import path for ``build_parser`` when different from ``module``.

    Must be argparse-only. Do not point this at JAX/PySCF training or
    evaluation modules: ``karml <cmd> --help`` and docs generation import it.
    """


# Keep in sync with ``karml.cli.__main__`` dispatch and ``KARML_COMMANDS``.
COMMAND_REGISTRY: tuple[CommandSpec, ...] = (
    CommandSpec("make-res", "karml.cli.make.make_res", "CGENFF residue → PDB/PSF/topology"),
    CommandSpec("make-box", "karml.cli.make.make_box", "Pack molecules into a periodic box"),
    CommandSpec("build-crystal", "karml.cli.misc.build_crystal", "Symmetry-aware crystals (PyXtal)"),
    CommandSpec(
        "run",
        "karml.cli.run.run_sim",
        "MM/ML simulation (ASE + JAX-MD hybrid)",
        status="legacy",
        replacement="md-system",
        removal_date="2026-09-01",
        note="Prefer md-system for new MD; run kept for hybrid calculator demos.",
    ),
    CommandSpec("md-system", "karml.cli.run.md_system", "Mixed-composition MD (ASE/JAX-MD/PyCHARMM)"),
    CommandSpec(
        "metatomic-pbc-md",
        "karml.cli.misc.metatomic_pbc_md",
        "CHARMM-free metatomic ASE MD in a cubic liquid box (NVT/NVE)",
    ),
    CommandSpec("liquid-box", "karml.cli.run.liquid_box", "Build/certify periodic liquid boxes (MM only)"),
    CommandSpec(
        "md-embedding",
        "karml.cli.run.md_embedding",
        "Solvated peptide partial MLpot (train/build/run)",
    ),
    CommandSpec("mpi-check", "karml.cli.run.mpi_check", "Validate OpenMPI/CHARMM/mpi4py for MLpot"),
    CommandSpec("mpi-launch", "karml.cli.run.mpi_launch", "Launch OpenMPI with an explicit JAX execution policy"),
    CommandSpec("doctor", "karml.cli.doctor", "Is this machine ready? (JAX, CHARMM, Packmol)"),
    CommandSpec("health-check", "karml.cli.run.health_check", "Validate KARML/PyCHARMM/JAX interface health"),
    CommandSpec("env", "karml.cli.env", "Find resolved/bundled checkpoints and CHARMM paths"),
    CommandSpec("warmup-mlpot-jax", "karml.cli.run.warmup_mlpot_jax", "Serial JAX JIT warmup for MLpot"),
    CommandSpec("lambda-mbar", "karml.cli.run.lambda_mbar", "MBAR post-processing for lambda TI"),
    CommandSpec(
        "run-pycharmm",
        "karml.cli.run.run_pycharmm",
        "Pure CHARMM heating/equilibration",
        status="legacy",
        replacement="md-system --backend pycharmm (no ML checkpoint)",
        removal_date="2026-09-01",
        note="Pure MM CHARMM without MLpot; md-system covers ML workflows.",
    ),
    CommandSpec(
        "pycharmm-two-residue-sample",
        "karml.cli.run.pycharmm_two_residue_sample",
        "Restrained sampling for two-residue CHARMM system",
    ),
    CommandSpec("xml2npz", "karml.cli.misc.xml2npz", "Molpro XML → NPZ"),
    CommandSpec(
        "npz2traj",
        "karml.cli.misc.convert_npz_traj",
        "NPZ → ASE trajectory (E/F/dipole/charges)",
    ),
    CommandSpec("validate", "karml.cli.misc.validate_cli", "Validate NPZ against schema"),
    CommandSpec("train-joint", "karml.cli.misc.train_joint", "Joint PhysNet+DCMNet training"),
    CommandSpec("downstream", "karml.cli.misc.downstream", "Downstream analysis utilities"),
    CommandSpec("fix-and-split", "karml.cli.misc.fix_and_split", "Unit fixes + train/valid/test splits"),
    CommandSpec(
        "prepare-mm-dataset",
        "karml.cli.misc.prepare_mm_dataset",
        "Assign CGenFF types/charges to a dimer NPZ (hybrid ML/MM)",
    ),
    CommandSpec("pyscf-dft", "karml.cli.misc.pyscf_dft", "GPU DFT (energy, gradient, hessian, …)"),
    CommandSpec("pyscf-mp2", "karml.cli.misc.pyscf_mp2", "GPU MP2"),
    CommandSpec("pyscf-evaluate", "karml.cli.misc.pyscf_evaluate", "Batch E/F/D/ESP evaluation"),
    CommandSpec("pyscf-evaluate-mp2", "karml.cli.misc.pyscf_evaluate_mp2", "Batch MP2 evaluation"),
    CommandSpec("verify-esp-alignment", "karml.cli.misc.verify_esp_alignment", "Verify ESP grid alignment in NPZ"),
    CommandSpec("normal-mode-sample", "karml.cli.misc.normal_mode_sample", "Sample along vibrational modes"),
    CommandSpec(
        "dimer-scan",
        "karml.cli.misc.dimer_scan",
        "Reproducible rigid 1D dimer energy/force scan",
    ),
    CommandSpec(
        "pet-interaction-pes",
        "karml.cli.misc.pet_interaction_pes",
        "PET-MAD interaction slices, surfaces, and trimer many-body leftover",
    ),
    CommandSpec(
        "ic-scan",
        "karml.cli.misc.ic_scan",
        "Bond/angle/dihedral scans (1D or N-D) for QM/ML",
    ),
    CommandSpec(
        "neb",
        "karml.cli.misc.neb",
        "Nudged elastic band (NEB) path sampling with PhysNet",
    ),
    CommandSpec(
        "umbrella-sample",
        "karml.cli.misc.umbrella_sample",
        "Batched distance umbrella NVT sampling (PhysNet/SpookyNet)",
    ),
    CommandSpec(
        "umbrella-mbar",
        "karml.cli.misc.umbrella_mbar",
        "MBAR post-processing for umbrella-sample runs",
    ),
    CommandSpec(
        "dmc",
        "karml.generate.dmc.dmc",
        "Diffusion Monte Carlo with PhysNetJax (batched walkers)",
    ),
    CommandSpec(
        "mode-check",
        "karml.cli.misc.mode_check",
        "Monomer/cluster FD, X–H stretch, vib, kick (+ PBC FD)",
    ),
    CommandSpec("physnet-train", "karml.cli.make.make_training", "Train PhysNet message-passing model (E/F)"),
    CommandSpec(
        "label-acquire",
        "karml.cli.misc.label_acquire",
        "Select structures for expensive labels (activation / Jacobian / teacher-gradient)",
    ),
    CommandSpec(
        "pet-box-dataset",
        "karml.cli.misc.pet_box_dataset",
        "Many-seed PET box dataset: random packing, FIRE intermediates, NVT (extxyz)",
    ),
    CommandSpec(
        "pet-physnet-distill",
        "karml.cli.misc.pet_physnet_distill",
        "PET-MAD teacher → PhysNet NPZ (acetone dataset + synthetic pool)",
    ),
    CommandSpec(
        "tune-mm-nonbonded",
        "karml.cli.misc.tune_mm_nonbonded",
        "Fit CGenFF LJ/charge scales of the ML/MM tail to teacher liquid frames",
    ),
    CommandSpec("physnet-md", "karml.cli.misc.physnet_md", "PhysNet MD sampling"),
    CommandSpec("physnet-evaluate", "karml.cli.misc.physnet_evaluate", "Evaluate PhysNet checkpoint"),
    CommandSpec("compare-npz", "karml.cli.misc.compare_npz", "Reference vs model NPZ plots"),
    CommandSpec(
        "compare-charmm-ml",
        "karml.cli.misc.compare_charmm_ml",
        "CHARMM PSF charges vs joint ML dipoles/ESP",
    ),
    CommandSpec("cross-check", "karml.cli.misc.cross_check", "Supplementary QC cross-check"),
    CommandSpec("efield-train", "karml.cli.misc.efield_train", "Train external electric-field PhysNet"),
    CommandSpec("efield-evaluate", "karml.cli.misc.efield_evaluate", "Evaluate external electric-field PhysNet"),
    CommandSpec("efield-md", "karml.cli.misc.efield_md", "MD with external electric-field PhysNet"),
    CommandSpec(
        "kernnn-train",
        "karml.cli.misc.kernnn_train",
        "Train KerNN kernel Softplus MLP (E/F)",
    ),
    CommandSpec(
        "kernnn-evaluate",
        "karml.cli.misc.kernnn_evaluate",
        "Evaluate KerNN checkpoint",
    ),
    CommandSpec("active-learning", "karml.cli.misc.active_learning", "Sample structures for re-labeling"),
    CommandSpec(
        "pes-design",
        "karml.cli.misc.pes_design",
        "Bayesian physical/diverse PES subset design + validation plots",
    ),
    CommandSpec("kernel-fit", "karml.cli.misc.kernel_fit", "Kernel fitting utilities"),
    CommandSpec("interpolate-xyz", "karml.cli.misc.interpolate_xyz", "Interpolate XYZ via Z-matrix → NPZ"),
    CommandSpec("unwrap-traj", "karml.cli.misc.unwrap_traj", "Unwrap periodic trajectories"),
    CommandSpec(
        "analyze-liquid",
        "karml.cli.misc.analyze_liquid",
        "Neat-liquid MD analysis (density, RDF, MSD, plots)",
    ),
    CommandSpec(
        "sample-diverse-xyz",
        "karml.generate.sample",
        "Pick diverse structures (SOAP) → NPZ",
    ),
    CommandSpec("gui", "karml.cli.gui", "Molecular viewer GUI"),
    CommandSpec(
        "extract-checkpoint-metrics",
        "karml.cli.misc.extract_checkpoint_metrics",
        "Plot training metrics from Orbax checkpoints",
    ),
    CommandSpec(
        "diagnose-lc-outliers",
        "karml.cli.misc.diagnose_learning_curve_outliers",
        "Inspect learning-curve sweeps for bad seeds and NPZ outliers",
    ),
    CommandSpec("orbax-to-json", "karml.cli.misc.orbax_to_json_cmd", "Export Orbax checkpoint to JSON"),
    CommandSpec("orca-server", "karml.interfaces.orca_external.server", "Persistent JAX server for ORCA"),
    CommandSpec("orca-client", "karml.interfaces.orca_external.client", "ORCA client → orca-server"),
    CommandSpec("orca-external", "karml.interfaces.orca_external.runner", "Standalone ORCA external wrapper"),
    CommandSpec("configure", "karml.cli.configure", "Interactive config / Snakemake wizard"),
    CommandSpec(
        "plot-restart-velocities",
        "karml.cli.plot.plot_restart_velocities",
        "Plot |v| distributions and outliers from CHARMM .res files",
    ),
    CommandSpec("commands", "karml.cli.commands_help", "Browse subcommands (grouped)"),
    CommandSpec("examples", "karml.cli.commands_help", "Copy-paste example invocations"),
    CommandSpec("completion", "karml.cli.completion", "Shell tab-completion setup"),
)

KARML_COMMANDS: tuple[str, ...] = tuple(spec.name for spec in COMMAND_REGISTRY)

_DISPATCH_COMMANDS: tuple[str, ...] = tuple(
    name for name in KARML_COMMANDS if name != "completion"
)


def command_by_name(name: str) -> CommandSpec | None:
    for spec in COMMAND_REGISTRY:
        if spec.name == name:
            return spec
    return None


def format_audit_report() -> str:
    lines = [
        "KARML CLI audit — active, legacy, and deprecated commands",
        "",
        "Deprecated / legacy (prefer replacement):",
    ]
    for spec in COMMAND_REGISTRY:
        if spec.status == "active":
            continue
        rep = f" → use {spec.replacement}" if spec.replacement else ""
        deadline = f"; removal {spec.removal_date}" if spec.removal_date else ""
        lines.append(f"  {spec.name:<28} [{spec.status}]{rep}{deadline}")
        if spec.note:
            lines.append(f"    {spec.note}")
    lines.extend(["", "Active commands with tab-completion when build_parser() exists:", ""])
    from karml.cli.parser_utils import parser_available

    for spec in COMMAND_REGISTRY:
        if spec.status != "active":
            continue
        flag = "✓ flags" if parser_available(spec.name, import_module=False) else "  (top-level only)"
        lines.append(f"  {spec.name:<28} {flag}  {spec.summary}")
    lines.append("")
    lines.append("Install: pip install 'karml[cli]' && eval \"$(register-python-argcomplete karml)\"")
    return "\n".join(lines)
