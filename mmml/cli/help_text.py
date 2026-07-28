"""Compact CLI help catalog for ``mmml`` (used by ``-h``, ``commands``, ``examples``)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandInfo:
    name: str
    summary: str


COMMAND_GROUPS: tuple[tuple[str, tuple[CommandInfo, ...]], ...] = (
    (
        "Structure & boxes",
        (
            CommandInfo("make-res", "CGENFF residue → PDB/PSF/topology"),
            CommandInfo("make-box", "Pack molecules into a periodic box"),
            CommandInfo("build-crystal", "Symmetry-aware crystals (PyXtal)"),
            CommandInfo("liquid-box", "Build/certify periodic liquid boxes (MM only)"),
        ),
    ),
    (
        "MD & campaigns",
        (
            CommandInfo("md-system", "Mixed-composition MD (ASE/JAX-MD/PyCHARMM MLpot)"),
            CommandInfo("run", "MM/ML simulation (ASE + JAX-MD hybrid)"),
            CommandInfo("run-pycharmm", "Pure CHARMM heating/equilibration"),
            CommandInfo("lambda-mbar", "MBAR post-processing for lambda TI"),
            CommandInfo("warmup-mlpot-jax", "Serial JAX JIT warmup for MLpot"),
            CommandInfo("mpi-check", "Validate OpenMPI/CHARMM/mpi4py for MLpot"),
            CommandInfo("mpi-launch", "Launch OpenMPI with an explicit JAX execution policy"),
            CommandInfo("health-check", "Validate MMML/PyCHARMM/JAX interface health"),
        ),
    ),
    (
        "QM & data",
        (
            CommandInfo("pyscf-dft", "GPU DFT (energy, gradient, hessian, …)"),
            CommandInfo("pyscf-mp2", "GPU MP2"),
            CommandInfo("pyscf-evaluate", "Batch E/F/D/ESP evaluation"),
            CommandInfo("fix-and-split", "Unit fixes + train/valid/test splits"),
            CommandInfo("prepare-mm-dataset", "Assign CGenFF types/charges to a dimer NPZ (hybrid ML/MM)"),
            CommandInfo("xml2npz", "Molpro XML → NPZ"),
            CommandInfo("npz2traj", "NPZ → ASE trajectory (E/F/dipole/charges)"),
            CommandInfo("validate", "Validate NPZ against schema"),
            CommandInfo("dimer-scan", "Rigid 1D dimer energy/force scan"),
            CommandInfo("ic-scan", "Bond/angle/dihedral scans (1D or N-D) for QM/ML"),
            CommandInfo(
                "mode-check",
                "Monomer/cluster FD, X–H stretch, vib, kick (+ PBC FD)",
            ),
        ),
    ),
    (
        "ML training & MD",
        (
            CommandInfo("physnet-train", "Train PhysNetJAX EF from NPZ"),
            CommandInfo("physnet-evaluate", "Evaluate PhysNet checkpoint"),
            CommandInfo("physnet-md", "PhysNet MD sampling"),
            CommandInfo("neb", "NEB reaction-path sampling with PhysNet"),
            CommandInfo("dmc", "Diffusion Monte Carlo (batched PhysNetJax walkers)"),
            CommandInfo("ef-train", "Train EF equivariant model"),
            CommandInfo("ef-evaluate", "Evaluate EF model"),
            CommandInfo("ef-md", "MD with trained EF model"),
            CommandInfo("kernnn-train", "Train KerNN kernel Softplus MLP"),
            CommandInfo("kernnn-evaluate", "Evaluate KerNN checkpoint"),
            CommandInfo("active-learning", "Sample structures for re-labeling"),
        ),
    ),
    (
        "Workflow helpers",
        (
            CommandInfo("configure", "Interactive wizard for YAML configs & Snakemake"),
            CommandInfo("env", "Find resolved/bundled checkpoints and CHARMM paths"),
            CommandInfo("commands", "Browse all subcommands (grouped)"),
            CommandInfo("examples", "Copy-paste example invocations"),
            CommandInfo("completion", "Shell tab-completion setup"),
            CommandInfo("gui", "Molecular viewer GUI"),
        ),
    ),
    (
        "Other",
        (
            CommandInfo("cross-check", "Supplementary QC cross-check"),
            CommandInfo("compare-npz", "Reference vs model NPZ plots"),
            CommandInfo("unwrap-traj", "Unwrap periodic trajectories"),
            CommandInfo("orca-server", "Persistent JAX server for ORCA"),
        ),
    ),
)

EXAMPLE_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Residues & boxes",
        (
            "mmml make-res --list-residues",
            "mmml make-res --res CYBZ",
            "mmml make-box --res CYBZ --n 50 --box-size 25.0",
            "mmml liquid-box --composition DCM:206 --target-density-g-cm3 1.326 -o boxes/dcm206",
            "mmml health-check --require-gpu",
        ),
    ),
    (
        "MD & campaigns",
        (
            "mmml configure",
            "mmml md-system --setup pbc_npt --composition MEOH:5,TIP3:5 --temperature 300",
            "mmml md-system --config campaign.yaml --run-all",
            "mmml warmup-mlpot-jax --checkpoint \"$MMML_CKPT\" --n-monomers 20",
        ),
    ),
    (
        "QM pipeline",
        (
            "mmml fix-and-split --efd data.npz --output-dir ./splits",
            "mmml npz2traj data.npz -o trajectory.traj",
            "mmml pyscf-evaluate -i traj.npz -o out.npz --EF --esp",
            "mmml physnet-train --config train.yaml",
            "mmml mode-check --composition TIP3:1 --checkpoint \"$MMML_CKPT\" --output-dir ./mode_tip3_1",
            "mmml mode-check --composition TIP3:2 --checkpoint \"$MMML_CKPT\" --output-dir ./mode_tip3_2 --checks minimize,fd,bond-scan,vibrations,kick",
            "mmml mode-check --pbc-fd --checkpoint \"$MMML_CKPT\" --output artifacts/fd_force_check.json",
            "mmml neb --config examples/m/yaml/neb.yaml --overwrite",
            "mmml neb --checkpoint examples/m/kl.json "
            "--initial examples/m/neb/reag_0_opt.xyz "
            "--final examples/m/neb/prod_0_opt.xyz "
            "--output-dir artifacts/nh3_ch3cl/neb --n-images 11 --fmax 0.05",
            "mmml dmc --natm 20 --nwalker 512 --stepsize 5e-4 --nstep 5000 --eqstep 1000 "
            "--alpha 1200.0 --checkpoint \"$MMML_CKPT\" "
            "--input mmml/generate/dmc/examples/acetone_dmc.extxyz",
        ),
    ),
)


def all_commands() -> list[str]:
    out: list[str] = []
    for _, cmds in COMMAND_GROUPS:
        out.extend(c.name for c in cmds)
    return out


def format_commands_help(*, width: int = 78) -> str:
    lines = [
        "MMML subcommands (grouped). For flag help: mmml <command> --help",
        "",
    ]
    for group, cmds in COMMAND_GROUPS:
        lines.append(f"{group}:")
        for cmd in cmds:
            pad = max(1, 22 - len(cmd.name))
            lines.append(f"  {cmd.name}{' ' * pad}{cmd.summary}")
        lines.append("")
    lines.append("Find checkpoints / env paths: mmml env")
    lines.append("Setup wizard:  mmml configure")
    lines.append("More examples: mmml examples")
    return "\n".join(lines).rstrip()


def format_examples_help() -> str:
    lines = ["Copy-paste examples:", ""]
    for title, examples in EXAMPLE_BLOCKS:
        lines.append(f"{title}:")
        for ex in examples:
            lines.append(f"  {ex}")
        lines.append("")
    lines.append("Per-command flags: mmml <command> --help")
    return "\n".join(lines).rstrip()


def format_top_level_help(prog: str = "mmml") -> str:
    """Short help for ``mmml -h`` (no giant epilog)."""
    from mmml.cli.registry import _DISPATCH_COMMANDS

    n_cmds = len(_DISPATCH_COMMANDS)
    lines = [
        f"usage: {prog} [-h] <command> ...",
        "",
        "MMML: Machine Learning for Molecular Modeling",
        "",
        f"Subcommands ({n_cmds} total). Common:",
        "  md-system      mixed-composition MD (YAML + campaigns)",
        "  physnet-train  train PhysNetJAX from NPZ",
        "  configure      interactive config / Snakemake wizard",
        "  env            find resolved/bundled checkpoints and CHARMM paths",
        "  liquid-box     build periodic liquid boxes",
        "",
        "Browse:   mmml commands",
        "Setup:    mmml configure",
        "Examples: mmml examples",
        "Flags:    mmml <command> --help",
        "",
        "Tab completion (bash/zsh/fish):",
        "  pip install 'mmml[cli]'",
        f"  eval \"$(register-python-argcomplete {prog})\"",
        "",
        f"options:",
        f"  -h, --help  show this help message and exit",
    ]
    return "\n".join(lines)


def validate_command(name: str, *, allowed: Iterable[str]) -> str | None:
    """Return error message if ``name`` is not a known command."""
    import difflib

    allowed_list = sorted(set(allowed))
    if name in allowed_list:
        return None
    close = difflib.get_close_matches(name, allowed_list, n=5, cutoff=0.45)
    if not close and len(name) >= 3:
        close = [c for c in allowed_list if c.startswith(name[:3])][:5]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return f"Unknown command {name!r}.{hint} Run 'mmml commands'."
