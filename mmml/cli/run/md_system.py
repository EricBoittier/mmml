#!/usr/bin/env python3
"""CLI wrapper for generalized mixed-composition MD setup scripts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
)
from mmml.interfaces.pycharmmInterface.ml_dtypes import add_ml_compute_dtype_args

_DEFAULT_OUTPUT_DIR_STEMS = frozenset({"pycharmm_mlpot", "lambda_ti"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run predefined MD setups (free-space NVE/NVT, periodic NVE/NVT, periodic NPT, "
            "lambda TI for arbitrary compositions) for arbitrary residue compositions. "
            "Runs mmml.cli.run.md_pbc_suite (ASE, JAX-MD, or CHARMM MLpot) and "
            "mmml.cli.run.lambda_dynamics (lambda_ti). "
            "MBAR: mmml lambda-mbar --run-dir <output-dir>."
        )
    )
    parser.add_argument(
        "--setup",
        choices=[
            "free_nve",
            "free_nvt",
            "pbc_nve",
            "pbc_nvt",
            "pbc_npt",
            "lambda_ti",
            "pycharmm_minimize",
            "pycharmm_full",
            "all",
        ],
        default="pbc_nve",
        help=(
            "Simulation setup preset. lambda_ti: alchemical TI with CHARMM+MMML minimization "
            "per λ window (--lambda-md-mode, --backend ase|jaxmd); mmml lambda-mbar afterward. "
            "pycharmm_minimize: CHARMM MLpot SD only (--backend pycharmm). "
            "pycharmm_full: mini → heat → NVE → equi → prod (--backend pycharmm). "
            "pbc_* with --backend pycharmm: same staged pipeline with CHARMM crystal/IMAGE. "
            "free_nve/free_nvt with --backend pycharmm: mini + NVE or mini + heat."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ase", "jaxmd", "pycharmm"],
        default="auto",
        help=(
            "MD engine: ase runs md_pbc_suite.ase; jaxmd runs md_pbc_suite.jaxmd; "
            "pycharmm runs CHARMM MLpot (vacuum, non-PBC: SD + NVE/NVT). "
            "auto uses ASE for vacuum (free_*) and fixed-volume PBC, JAX-MD for NPT. "
            "Use jaxmd with --setup free_nve or free_nvt for open-boundary JAX-MD."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for artifacts.")
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help=(
            "Logical job id for this run. Writes artifacts/md_system/jobs/<job-name>.json "
            "with the last command and options (default: basename of --output-dir when set)."
        ),
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("artifacts/md_system/jobs"),
        help="Directory for per-job last-run manifest JSON files (default: artifacts/md_system/jobs).",
    )
    parser.add_argument("--template-pdb", type=Path, default=None, help="Monomer template PDB path.")
    parser.add_argument("--n-molecules", type=int, default=10, help="Number of molecules for single-residue runs.")
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help=(
            "Residue composition: comma-separated RES:N entries, e.g. MEOH:5,TIP3:5. "
            "A bare RES (no ':N') implies a single copy (N=1); when this option is set, "
            "--n-molecules is not passed to the backend (use DCM:10 for ten DCM)."
        ),
    )
    parser.add_argument("--spacing", type=float, default=5.0, help="Target minimum random COM spacing in Angstrom.")
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Override periodic cubic box side length in Angstrom (default: auto from initial geometry).",
    )
    parser.add_argument("--ps", type=float, default=1.0, help="Simulation length in ps.")
    parser.add_argument("--dt-fs", type=float, default=0.25, help="Timestep in fs.")
    parser.add_argument(
        "--traj-chunk-frames",
        type=int,
        default=0,
        help="Split trajectory output into multi-file chunks with at most this many frames (0 = single file).",
    )
    parser.add_argument(
        "--traj-export-molecular-wrap",
        action="store_true",
        help="JAX-MD only: molecular COM wrap when writing HDF5/.traj (slower).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Target temperature in K (NVT/NPT).",
    )
    parser.add_argument(
        "--nvt-integrator",
        choices=["auto", "nhc", "langevin"],
        default="auto",
        help="Integrator for NVT in ASE route. auto=nhc for homogeneous, langevin for mixed composition.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Target pressure in atm (NPT).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for initial placement and velocities.")
    parser.add_argument(
        "--min-intermonomer-atom-distance",
        type=float,
        default=0.1,
        help="Abort if atoms from different monomers get closer than this distance in Angstrom (<=0 disables).",
    )
    parser.add_argument(
        "--packmol",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pack --composition with Packmol (default: inside cube from --box-size). "
            "Use --no-packmol for legacy grid placement."
        ),
    )
    parser.add_argument(
        "--packmol-placement",
        choices=("cube", "sphere"),
        default=None,
        help="Packmol constraint: cube (default) or sphere (--packmol-radius).",
    )
    parser.add_argument(
        "--packmol-sphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy alias for --packmol-placement sphere.",
    )
    parser.add_argument(
        "--packmol-radius",
        type=float,
        default=None,
        metavar="Å",
        help="Packmol sphere radius in Angstrom (required for --packmol-placement sphere).",
    )
    parser.add_argument(
        "--packmol-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Packmol constraint center in Angstrom (default: 0 0 0).",
    )
    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol distance tolerance (Å) for composition packing (default: 2.0).",
    )
    parser.add_argument(
        "--reuse-packmol-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pycharmm: reuse disk cache for Packmol cluster builds (default: on).",
    )
    parser.add_argument(
        "--rebuild-packmol",
        action="store_true",
        help="pycharmm: ignore Packmol cache and rebuild placement.",
    )
    parser.add_argument(
        "--packmol-cache-dir",
        type=Path,
        default=None,
        help="pycharmm: Packmol cache root (default: output-dir/.packmol_cache or MMML_PACKMOL_CACHE).",
    )
    parser.add_argument(
        "--save-run-state",
        action="store_true",
        help=(
            "pycharmm: after staged MD, save positions/velocities + metadata "
            "(Orbax if installed, else NPZ; PhysNet stays in --checkpoint)."
        ),
    )
    parser.add_argument(
        "--run-state-dir",
        type=Path,
        default=None,
        help="pycharmm: run-state output directory (default: output-dir/run_state).",
    )
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Harmonic flat-bottom on system COM: V=0 inside radius R, V=k(|d|-R)^2 outside. "
            "Independent of --packmol-radius. Vacuum: center at origin; PBC: MIC to box center."
        ),
    )
    parser.add_argument(
        "--flat-bottom-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Flat-bottom force constant when COM is outside --flat-bottom-radius (default: 1.0).",
    )
    parser.add_argument(
        "--flat-bottom-selection",
        type=str,
        default="all",
        help="pycharmm: CHARMM atom selection for MMFP wall (default: all).",
    )
    parser.add_argument(
        "--flat-bottom-mode",
        choices=["system", "monomer"],
        default="system",
        help=(
            "Flat-bottom anchor: system = one restraint on mass-weighted cluster COM; "
            "monomer = sum of harmonic restraints on each monomer COM (same R and k)."
        ),
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw args forwarded to the underlying script; put this option last.",
    )

    # --- pycharmm / MLpot (--backend pycharmm) --------------------------------
    parser.add_argument(
        "--fix-resids",
        type=str,
        default="",
        help=(
            "pycharmm: monomers held in SD pass 2 (comma-separated 1-based resids; "
            "default: none — use e.g. 1 or 1,3 to anchor monomers during minimize)"
        ),
    )
    parser.add_argument(
        "--constrain-resids",
        type=str,
        default="",
        help="pycharmm: freeze these resids during MD (comma-separated)",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="pycharmm: skip constrained SD pass 2",
    )
    parser.add_argument(
        "--mini-nstep",
        type=int,
        default=20,
        help="pycharmm: SD steps per minimization pass before dynamics",
    )
    parser.add_argument(
        "--no-pre-minimize",
        action="store_true",
        help="pycharmm: skip SD minimization before dynamics",
    )
    parser.add_argument(
        "--echeck",
        type=float,
        default=100.0,
        help="pycharmm: CHARMM ECHECK tolerance (kcal/mol); use --no-echeck to disable",
    )
    parser.add_argument(
        "--no-echeck",
        action="store_true",
        help="pycharmm: disable CHARMM ECHECK early stop",
    )
    parser.add_argument(
        "--allow-incomplete-dynamics",
        action="store_true",
        help=(
            "pycharmm: do not fail when dynamics stops early or the stage DCD is truncated"
        ),
    )
    parser.add_argument(
        "--nprint",
        type=int,
        default=50,
        help="pycharmm: print SD minimization energy every N steps (default: 50)",
    )
    parser.add_argument(
        "--dyn-nprint",
        type=int,
        default=500,
        help="pycharmm: print dynamics energy every N steps (default: 500)",
    )
    parser.add_argument(
        "--dyn-iprfrq",
        type=int,
        default=2000,
        help="pycharmm: detailed dynamics status every N steps (default: 2000)",
    )
    parser.add_argument(
        "--heat-ihtfrq",
        type=int,
        default=0,
        metavar="N",
        help=(
            "pycharmm: heating velocity rescale every N steps when --heat-thermostat "
            "scale (0 = match --dyn-nprint). Ignored for hoover."
        ),
    )
    parser.add_argument(
        "--heat-thermostat",
        choices=("scale", "hoover"),
        default="scale",
        help=(
            "pycharmm heat stage: scale=IHTFRQ velocity rescaling; hoover=CHARMM Hoover "
            "NVT (vacuum hoover reft/tmass, no CPT / no ML PBC required)."
        ),
    )
    parser.add_argument(
        "--heat-firstt",
        type=float,
        default=None,
        metavar="K",
        help=(
            "pycharmm: heat start temperature (FIRSTT). Default 0.2×--temperature; "
            "use 0 for cold start + IHTFRQ scaling."
        ),
    )
    parser.add_argument(
        "--heat-finalt",
        type=float,
        default=None,
        metavar="K",
        help=(
            "pycharmm: heat end temperature (FINALT). Default --temperature; "
            "DCM:9 stability often uses 240."
        ),
    )
    parser.add_argument(
        "--heat-hoover-tmass",
        type=int,
        default=None,
        metavar="M",
        help=(
            "pycharmm Hoover heat only: thermostat mass tmass (kcal·mol⁻¹·ps²). "
            "Default clamps PSF tmass to 400–1200. Lower = stronger T coupling."
        ),
    )
    parser.add_argument(
        "--nve-boltzmann-temp",
        type=float,
        default=None,
        metavar="K",
        help=(
            "pycharmm: Boltzmann velocity temperature before NVE after mini. "
            "Default 0.2×--temperature; use 50–100 K for a gentler start than 300 K."
        ),
    )
    parser.add_argument(
        "--heat-comp-damp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pycharmm: experimental COMP force copy before heat (default: off).",
    )
    parser.add_argument(
        "--heat-comp-hydrogen-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pycharmm: with --heat-comp-damp, select high-|F| hydrogens only "
            "(default). --no-heat-comp-hydrogen-only = all atom types."
        ),
    )
    parser.add_argument(
        "--heat-comp-force-min",
        type=float,
        default=None,
        metavar="KCAL",
        help="pycharmm: |F| threshold for heat COMP selection (kcal/mol/Å).",
    )
    parser.add_argument(
        "--heat-comp-force-scale",
        type=float,
        default=None,
        help="pycharmm: scale for forces copied into COMP during heat.",
    )
    parser.add_argument(
        "--skip-energy-show",
        action="store_true",
        help="pycharmm: skip CHARMM energy.show() (MPI/cluster segfault guard)",
    )
    parser.add_argument(
        "--show-energy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pycharmm: print CHARMM energy tables (off by default)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="pycharmm: reduce CHARMM console output",
    )
    parser.add_argument(
        "--dcd-nsavc",
        type=int,
        default=1,
        help="pycharmm: DCD frame every N integration/SD steps",
    )
    parser.add_argument(
        "--dcd-interval-ps",
        type=float,
        default=None,
        metavar="PS",
        help="pycharmm: DCD save interval in ps (overrides --dcd-nsavc when set)",
    )
    parser.add_argument(
        "--dcd-max-frames",
        type=int,
        default=25,
        metavar="N",
        help=(
            "pycharmm: cap DCD output to ~N frames per stage when --dcd-interval-ps "
            "is unset (0 = no cap)"
        ),
    )
    parser.add_argument(
        "--save-forces-npz",
        action="store_true",
        help="pycharmm: write <output-dir>/forces.npz during dynamics",
    )
    parser.add_argument(
        "--forces-npz-interval",
        type=int,
        default=1,
        help="pycharmm: force NPZ save every N steps (default: 1)",
    )
    parser.add_argument(
        "--no-scale-mini-nstep",
        action="store_true",
        help="pycharmm: do not auto-increase --mini-nstep for large clusters",
    )
    parser.add_argument(
        "--no-scale-echeck",
        action="store_true",
        help="pycharmm: do not auto-loosen --echeck for large clusters (default scales with N monomers/atoms)",
    )
    parser.add_argument(
        "--allow-high-grms",
        action="store_true",
        help="pycharmm: start dynamics even if post-min GRMS is high",
    )
    parser.add_argument(
        "--max-grms-before-dyn",
        type=float,
        default=50.0,
        help="pycharmm: abort if post-min GRMS exceeds this (kcal/mol/Å)",
    )
    parser.add_argument(
        "--test-first",
        action="store_true",
        help="pycharmm: CHARMM TEST FIRSt after MLpot SD minimization",
    )
    parser.add_argument(
        "--test-first-tol",
        type=float,
        default=0.005,
        help="pycharmm: TEST FIRSt tolerance (default: 0.005)",
    )
    parser.add_argument(
        "--test-first-step",
        type=float,
        default=1.0e-4,
        help="pycharmm: TEST FIRSt finite-difference step in Å (default: 1e-4)",
    )
    parser.add_argument(
        "--test-first-resids",
        type=str,
        default="",
        help="pycharmm: limit derivative tests to these resids (default: all atoms)",
    )
    parser.add_argument(
        "--test-first-charmm",
        action="store_true",
        help="pycharmm: also run CHARMM TEST FIRSt (ANALYTIC omits MLpot USER energy)",
    )
    parser.add_argument(
        "--test-first-update-nbonds",
        action="store_true",
        help="pycharmm: UPDATE nonbond lists before CHARMM TEST FIRSt",
    )
    parser.add_argument(
        "--ml-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "pycharmm: chunk PhysNet batches (auto: 256 on GPU / 64 on CPU for n>=40; "
            "or MMML_MLPOT_ML_BATCH_SIZE). DCM:90 try 256-512 on one GPU."
        ),
    )
    parser.add_argument(
        "--ml-gpu-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "pycharmm: parallel PhysNet chunks on N local GPUs (default 1; "
            "or MMML_MLPOT_N_GPUS). Set CUDA_VISIBLE_DEVICES to the GPU ids to use."
        ),
    )
    parser.add_argument(
        "--ml-spatial-mpi",
        action="store_true",
        help=(
            "pycharmm: per-rank spatial ML decomposition when MPI size>1 (PBC only; "
            "or MMML_MLPOT_SPATIAL_MPI=1). Use with MMML_MPI_NP>1 and --ml-gpu-count 1."
        ),
    )
    add_ml_compute_dtype_args(parser)
    parser.add_argument(
        "--ml-max-active-dimers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "pycharmm: sparse ML dimer slot cap per step (PBC default max(1000, 6*n_monomers); "
            "free-space default all unique dimers). Run scripts/validate_mlpot_sparse_dimers.py to check."
        ),
    )
    parser.add_argument(
        "--md-stages",
        type=str,
        default=None,
        help="pycharmm: comma-separated mini,heat,nve,equi,prod (default from --setup)",
    )
    parser.add_argument(
        "--md-stage",
        type=str,
        default=None,
        choices=["mini", "heat", "nve", "equi", "prod"],
        help="pycharmm: run one stage only (implies prior artifacts under --output-dir)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="pycharmm: artifact tag for staged outputs (default: from composition)",
    )
    parser.add_argument(
        "--ps-heat",
        type=float,
        default=10.0,
        help="pycharmm: heating length in ps (default: 10)",
    )
    parser.add_argument(
        "--charmm-mm-pretreat",
        action="store_true",
        help=(
            "pycharmm: CGENFF minimize + CHARMM heat before MLpot (no PhysNet); "
            "see --charmm-mm-pretreat-heat-nstep"
        ),
    )
    parser.add_argument(
        "--charmm-mm-pretreat-heat-nstep",
        type=int,
        default=2000,
        metavar="N",
        help="pycharmm: integration steps for pretreat CHARMM heat (default: 2000)",
    )
    parser.add_argument(
        "--charmm-mm-pretreat-mini-sd",
        type=int,
        default=None,
        metavar="N",
        help="pycharmm: pretreat CHARMM SD steps (default: --charmm-sd-steps)",
    )
    parser.add_argument(
        "--charmm-mm-pretreat-mini-abnr",
        type=int,
        default=None,
        metavar="N",
        help="pycharmm: pretreat CHARMM ABNR steps (default: --charmm-abnr-steps)",
    )
    parser.add_argument(
        "--ps-nve",
        type=float,
        default=None,
        help="pycharmm: NVE length in ps (default: --ps)",
    )
    parser.add_argument(
        "--ps-equi",
        type=float,
        default=50.0,
        help="pycharmm: NPT equilibration length in ps (default: 50)",
    )
    parser.add_argument(
        "--ps-prod",
        type=float,
        default=None,
        help="pycharmm: production length in ps (default: --ps)",
    )
    parser.add_argument(
        "--npt-thermostat",
        type=str,
        choices=["hoover", "berendsen"],
        default="hoover",
        help="pycharmm: NPT temperature control for equi/prod (default: hoover)",
    )
    parser.add_argument(
        "--npt-pressure",
        type=float,
        default=1.0,
        help="pycharmm: NPT reference pressure in atm for equi/prod (default: 1.0)",
    )
    parser.add_argument(
        "--npt-pgamma",
        type=float,
        default=5.0,
        help=(
            "pycharmm: CPT barostat Langevin collision frequency in 1/ps "
            "(default: 5; 0 disables barostat coupling)"
        ),
    )
    parser.add_argument(
        "--n-heat-segments",
        type=int,
        default=1,
        help="pycharmm: split heating into short chained restart segments",
    )
    parser.add_argument(
        "--n-equi-segments",
        type=int,
        default=1,
        help="pycharmm: split NPT equilibration into chained restart segments",
    )
    parser.add_argument(
        "--n-prod-segments",
        type=int,
        default=1,
        help="pycharmm: split production into chained restart segments",
    )
    parser.add_argument(
        "--bonded-mm-mini",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pycharmm: bonded-only SD if MM bonded strain exceeds post-MM-pre-min baseline "
            "(default: on; heat always checked when enabled)"
        ),
    )
    parser.add_argument(
        "--bonded-mm-mini-after",
        type=str,
        default="mini,heat",
        help="pycharmm: comma-separated stages to check (default: mini,heat; heat always)",
    )
    parser.add_argument(
        "--bonded-mm-mini-steps",
        type=int,
        default=50,
        help="pycharmm: bonded recovery SD steps (default: 50)",
    )
    parser.add_argument(
        "--bonded-mm-mini-always",
        action="store_true",
        help=(
            "pycharmm: bonded SD after every --bonded-mm-mini-after stage "
            "(ignore strain margins)"
        ),
    )
    parser.add_argument(
        "--bonded-mm-internal-margin",
        type=float,
        default=0.0,
        help="pycharmm: deprecated alias for --bonded-mm-grms-margin (default: 0)",
    )
    parser.add_argument(
        "--bonded-mm-grms-margin",
        type=float,
        default=None,
        help="pycharmm: kcal/mol/Å above baseline GRMS before recovery",
    )
    parser.add_argument(
        "--bonded-mm-internal-energy-margin",
        type=float,
        default=0.0,
        help="pycharmm: kcal/mol above baseline bonded internal before recovery (0=off)",
    )
    parser.add_argument(
        "--bonded-mm-angl-margin",
        type=float,
        default=0.0,
        help="pycharmm: kcal/mol above baseline ANGL before recovery (0=off)",
    )
    parser.add_argument(
        "--bonded-mm-max-angl-kcal",
        type=float,
        default=None,
        help="pycharmm: abort after MM pre-min if ANGL exceeds this (e.g. 15)",
    )
    parser.add_argument(
        "--bonded-mm-max-internal-kcal",
        type=float,
        default=None,
        help="pycharmm: abort after MM pre-min if bonded internal exceeds this",
    )
    parser.add_argument(
        "--allow-high-bonded-strain",
        action="store_true",
        help="pycharmm: continue when max-angl/max-internal pre-min limits exceeded",
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        add_dynamics_overlap_args,
    )

    add_dynamics_overlap_args(parser)
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help="pycharmm: CHARMM .res restart for first dynamics stage",
    )
    parser.add_argument(
        "--from-psf",
        type=Path,
        default=None,
        help="pycharmm: load PSF instead of rebuilding cluster",
    )
    parser.add_argument(
        "--from-crd",
        type=Path,
        default=None,
        help="pycharmm: load CRD with --from-psf",
    )
    parser.add_argument(
        "--skip-cluster-build",
        action="store_true",
        help="pycharmm: skip Packmol/IC; use --from-psf/--from-crd or prior mini artifacts",
    )
    parser.add_argument(
        "--skip-if-crd-exists",
        action="store_true",
        help="pycharmm: skip MLpot SD when mini CRD already exists in --output-dir",
    )
    parser.add_argument(
        "--no-save-vmd-topology",
        action="store_true",
        help="pycharmm: skip cluster_for_vmd PSF/PDB before MLpot registration",
    )
    parser.add_argument(
        "--free-space",
        action="store_true",
        help=(
            "pycharmm: force vacuum (no PBC). free_nve/free_nvt setups are vacuum "
            "by default; use to override when --box-size is also set."
        ),
    )
    parser.add_argument(
        "--mlpot-pbc",
        action="store_true",
        help=(
            "pycharmm: enable ML MIC / periodic dimer lists (default for pbc_* setups). "
            "With free_* + --box-size, CHARMM uses loose PBC unless this flag is set."
        ),
    )
    parser.add_argument(
        "--dyn-inbfrq",
        type=int,
        default=None,
        help="pycharmm: CHARMM inbfrq for dynamics (-1=heuristic, 50=vacuum default)",
    )
    parser.add_argument(
        "--dyn-imgfrq",
        type=int,
        default=None,
        metavar="N",
        help="pycharmm PBC: image/HB list rebuild every N steps (default 50; larger=faster)",
    )
    parser.add_argument(
        "--pre-nve-charmm-update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pycharmm: ENER+UPDATE after mini before vacuum NVE (default: on)",
    )

    # --- lambda_ti (--setup lambda_ti) ----------------------------------------
    parser.add_argument(
        "--lambda-md-mode",
        choices=["free_nve", "free_nvt", "pbc_nve", "pbc_nvt"],
        default="free_nve",
        help="lambda_ti: MD ensemble (vacuum/PBC × NVE/NVT); use --backend ase or jaxmd.",
    )
    parser.add_argument(
        "--couple-residues",
        type=str,
        default="1",
        help="lambda_ti: 1-based residue numbers sharing λ (comma-separated, cluster order).",
    )
    parser.add_argument(
        "--lambda-windows",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="lambda_ti: shared λ values for coupled residues.",
    )
    parser.add_argument("--pre-min-steps", type=int, default=50, help="lambda_ti: MMML BFGS steps per window.")
    parser.add_argument("--pre-min-fmax", type=float, default=0.1, help="lambda_ti: MMML BFGS fmax (eV/Å).")
    parser.add_argument("--min-steps", type=int, default=None, help="lambda_ti: alias for --pre-min-steps.")
    parser.add_argument("--min-fmax", type=float, default=None, help="lambda_ti: alias for --pre-min-fmax.")
    parser.add_argument("--bfgs-maxstep", type=float, default=0.05)
    parser.add_argument(
        "--charmm-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lambda_ti: CHARMM SD/ABNR before MMML BFGS (default on).",
    )
    parser.add_argument(
        "--calculator-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lambda_ti: MMML-calculator BFGS after CHARMM (default on).",
    )
    parser.add_argument("--charmm-sd-steps", type=int, default=25)
    parser.add_argument("--charmm-abnr-steps", type=int, default=100)
    parser.add_argument("--charmm-tolenr", type=float, default=1e-3)
    parser.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    parser.add_argument("--charmm-nbxmod", type=int, default=5)
    parser.add_argument(
        "--rescue-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lambda_ti: ASE FIRE if BFGS fmax stays high.",
    )
    parser.add_argument("--max-fmax-after-min", type=float, default=2.0)
    parser.add_argument("--n-equil", type=int, default=500, help="lambda_ti: equilibration steps per window.")
    parser.add_argument(
        "--save-equil-traj",
        action="store_true",
        help="lambda_ti: write …_eq.traj under trajectories/ during equilibration (debug).",
    )
    parser.add_argument(
        "--equil-traj-interval",
        type=int,
        default=None,
        help="lambda_ti: equil trajectory frame interval (default: --interval).",
    )
    parser.add_argument("--n-prod", type=int, default=2000, help="lambda_ti: production steps per window.")
    parser.add_argument(
        "--repeats-per-window",
        type=int,
        default=1,
        help="lambda_ti: independent repeats per λ window.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="lambda_ti: sample dU/dλ every N production steps.",
    )
    parser.add_argument(
        "--min-com-start-distance",
        type=float,
        default=2.0,
        help="lambda_ti: minimum inter-monomer COM distance after placement (Å).",
    )
    parser.add_argument(
        "--no-fix-com",
        action="store_true",
        help="lambda_ti: disable ASE FixCom (COM position can drift during MD).",
    )
    parser.add_argument(
        "--no-stationary",
        action="store_true",
        help="lambda_ti: skip Stationary/ZeroRotation on velocity init (with --no-fix-com, COM can translate).",
    )
    parser.add_argument("--ml-cutoff", type=float, default=1.0, help="lambda_ti: ML cutoff (Å).")
    parser.add_argument(
        "--ml-switch-width",
        "--ml-cutoff-distance",
        dest="ml_switch_width",
        type=float,
        default=DEFAULT_ML_SWITCH_WIDTH,
        help=(
            "ML taper width (Å) over [mm_switch_on - width, mm_switch_on] for pycharmm/MMML handoff "
            f"(default: {DEFAULT_ML_SWITCH_WIDTH:g}). Does not affect lambda_ti (see --ml-cutoff)."
        ),
    )
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        default=DEFAULT_MM_SWITCH_ON,
        help=(
            f"MM handoff distance (Å); ML→0 / MM→1 at this COM separation "
            f"(default: {DEFAULT_MM_SWITCH_ON:g})."
        ),
    )
    parser.add_argument(
        "--mm-cutoff",
        "--mm-switch-width",
        dest="mm_switch_width",
        type=float,
        default=DEFAULT_MM_SWITCH_WIDTH,
        help=(
            f"MM outer taper width (Å) past mm_switch_on "
            f"(default: {DEFAULT_MM_SWITCH_WIDTH:g})."
        ),
    )
    parser.add_argument(
        "--mlpot-mm-internal-scale",
        type=float,
        default=0.0,
        metavar="W",
        help=(
            "pycharmm: scale CGENFF BOND/ANGL/DIHE on ML atoms during MLpot BLOCK "
            "(0=off, 0.1=10%% internal). ELEC/VDW stay off."
        ),
    )
    parser.add_argument(
        "--residue",
        type=str,
        default="MEOH",
        help=(
            "Single-residue name when --composition is not set "
            "(ignored when --composition is set; lambda_ti default MEOH)."
        ),
    )
    parser.add_argument(
        "--skip-jit-warmup",
        action="store_true",
        help=(
            "Skip JIT/XLA warmup. jaxmd/ase: generic XLA GPU compile and pre-MD hybrid "
            "MMML eval; lambda_ti: skip first MMML energy eval per window."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="lambda_ti: skip completed production trajectories; redo partial prod.traj files.",
    )

    # --- YAML config / campaign / handoff -------------------------------------
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file with md-system options or a campaign (defaults + runs/jobs).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Run one job from a campaign config (--config); honors depends_on chain.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all jobs from a campaign config in dependency order (in-process).",
    )
    parser.add_argument(
        "--resume-campaign",
        action="store_true",
        help="Skip campaign jobs whose output-dir/handoff already exists.",
    )
    parser.add_argument(
        "--campaign-output-dir",
        type=Path,
        default=None,
        help="Directory for campaign_plan.json and campaign_summary.json.",
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        help="Resume from handoff path (.res, .h5, .traj, run_state/, state.npz).",
    )
    parser.add_argument(
        "--continue-from-frame",
        type=int,
        default=-1,
        help="Frame index for .h5/.traj continue-from (default: -1 last).",
    )
    parser.add_argument(
        "--continue-velocities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use velocities from handoff when present (else re-thermalize).",
    )
    parser.add_argument(
        "--handoff-write-res",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write handoff/final.res alongside state.npz after dynamics.",
    )
    parser.add_argument(
        "--handoff-template-res",
        type=Path,
        default=None,
        help="Template CHARMM .res for pure-Python handoff writer.",
    )
    parser.add_argument(
        "--handoff-pre-minimize",
        action="store_true",
        help="Run pre-minimization even when continuing from a handoff.",
    )
    parser.add_argument(
        "--no-stage-summary",
        action="store_true",
        help="Do not write stage_summary.json (campaigns).",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_md_system_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI with optional ``--config`` YAML (CLI overrides file)."""
    from mmml.cli.run.md_config import apply_mapping_to_namespace, load_yaml_config

    if argv is None:
        argv = sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args(argv)
    defaults = vars(parse_args([]))
    if pre_args.config:
        cfg = load_yaml_config(pre_args.config)
        flat = {k: v for k, v in cfg.items() if k not in {"defaults", "runs", "jobs"}}
        tmp = argparse.Namespace(**defaults)
        apply_mapping_to_namespace(tmp, flat, source="config")
        defaults.update(vars(tmp))
        defaults["config"] = pre_args.config
    parser = build_parser()
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


_last_job_stages: list[Any] = []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_job_filename(job_name: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", job_name.strip())
    if not safe:
        raise ValueError("job name must contain at least one alphanumeric character")
    return safe


def resolve_job_name(args: argparse.Namespace) -> str | None:
    """Return the job id used for last-run manifest lookup."""
    explicit = getattr(args, "job_name", None)
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        return None
    stem = Path(output_dir).name
    if not stem or stem in _DEFAULT_OUTPUT_DIR_STEMS:
        return None
    return stem


def _args_for_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in {"extra_args", "jobs_dir"}:
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif value is None or isinstance(value, (str, int, float, bool, list)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def build_run_manifest(
    args: argparse.Namespace,
    *,
    backend: str | None,
    argv: list[str] | None,
    started_at: str,
    finished_at: str,
    exit_code: int,
) -> dict[str, Any]:
    job_name = resolve_job_name(args)
    if job_name is None:
        raise ValueError("resolve_job_name returned None")
    command_parts = list(sys.argv)
    if command_parts and command_parts[0] == "mmml md-system":
        command_parts = ["mmml", "md-system", *command_parts[1:]]
    manifest: dict[str, Any] = {
        "job_name": job_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": int(exit_code),
        "command": " ".join(shlex.quote(part) for part in command_parts),
        "setup": args.setup,
        "backend": backend,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "args": _args_for_manifest(args),
    }
    if argv is not None:
        manifest["backend_argv"] = list(argv)
    return manifest


def save_job_run_manifest(
    jobs_dir: Path,
    job_name: str,
    manifest: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> Path:
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = jobs_dir / f"{_safe_job_filename(job_name)}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if output_dir is not None:
        out_copy = Path(output_dir) / "run_manifest.json"
        out_copy.parent.mkdir(parents=True, exist_ok=True)
        out_copy.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _maybe_save_job_run_manifest(
    args: argparse.Namespace,
    *,
    backend: str | None,
    argv: list[str] | None,
    started_at: str,
    finished_at: str,
    exit_code: int,
) -> Path | None:
    job_name = resolve_job_name(args)
    if job_name is None:
        return None
    manifest = build_run_manifest(
        args,
        backend=backend,
        argv=argv,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
    )
    path = save_job_run_manifest(
        args.jobs_dir,
        job_name,
        manifest,
        output_dir=args.output_dir,
    )
    print(f"mmml md-system: wrote job manifest {path}", flush=True)
    return path


def _append_optional(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_boolean_optional_flag(cmd: list[str], flag: str, value: bool) -> None:
    """Forward ``BooleanOptionalAction`` flags (--flag / --no-flag)."""
    if value:
        cmd.append(flag)
    else:
        cmd.append(f"--no-{flag.removeprefix('--')}")


def _append_if_nonempty(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        cmd.extend([flag, text])


def _composition_molecule_count(spec: str) -> int:
    total = 0
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            _residue, count_str = tok.split(":", 1)
            count = int(count_str)
        else:
            count = 1
        if count <= 0:
            raise ValueError(f"Invalid composition token: {tok!r}")
        total += count
    if total <= 0:
        raise ValueError("Empty composition")
    return total


def _pycharmm_run_summary(args: argparse.Namespace) -> str:
    """One-line summary of the active pycharmm run (for logs)."""
    parts = [f"setup={args.setup}"]
    if args.composition:
        parts.append(f"composition={args.composition}")
    else:
        parts.append(f"residue={args.residue}")
        parts.append(f"n={args.n_molecules}")
    if getattr(args, "box_size", None):
        parts.append(f"box={args.box_size}Å")
    if args.md_stage:
        parts.append(f"stage={args.md_stage}")
    elif args.md_stages:
        parts.append(f"stages={args.md_stages}")
    if not args.no_fix and str(args.fix_resids).strip():
        parts.append(f"fix-resids={args.fix_resids}")
    if str(args.constrain_resids).strip():
        parts.append(f"constrain-resids={args.constrain_resids}")
    return " | ".join(parts)


def _validate_packmol_args(args: argparse.Namespace) -> None:
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_cube_side,
        resolve_packmol_placement_mode,
        resolve_packmol_sphere_radius,
        resolve_packmol_use,
    )

    if not resolve_packmol_use(
        composition=args.composition,
        packmol=getattr(args, "packmol", None),
    ):
        return
    if not args.composition:
        raise ValueError(
            "Packmol placement requires --composition (e.g. ACO:30)."
        )
    placement = resolve_packmol_placement_mode(
        packmol_placement=getattr(args, "packmol_placement", None),
        packmol_sphere=getattr(args, "packmol_sphere", None),
    )
    if placement == "sphere":
        resolve_packmol_sphere_radius(
            getattr(args, "packmol_radius", None),
            args.flat_bottom_radius,
        )
    else:
        resolve_packmol_cube_side(
            box_size=getattr(args, "box_size", None),
            packmol_box_size=getattr(args, "packmol_box_size", None),
            packmol_radius=getattr(args, "packmol_radius", None),
            flat_bottom_radius=args.flat_bottom_radius,
        )


def _validate_packmol_sphere_args(args: argparse.Namespace) -> None:
    """Backward-compatible alias."""
    _validate_packmol_args(args)


def _append_packmol_args(cmd: list[str], args: argparse.Namespace) -> None:
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_placement_mode,
        resolve_packmol_use,
    )

    if not resolve_packmol_use(
        composition=args.composition,
        packmol=getattr(args, "packmol", None),
    ):
        return
    placement = resolve_packmol_placement_mode(
        packmol_placement=getattr(args, "packmol_placement", None),
        packmol_sphere=getattr(args, "packmol_sphere", None),
    )
    if placement == "sphere":
        cmd.append("--packmol-sphere")
    else:
        cmd.extend(["--packmol-placement", "cube"])
    _append_optional(cmd, "--packmol-radius", getattr(args, "packmol_radius", None))
    if args.packmol_center is not None:
        cmd.extend(["--packmol-center", *[str(x) for x in args.packmol_center]])
    cmd.extend(["--packmol-tolerance", str(args.packmol_tolerance)])


def _append_packmol_sphere_args(cmd: list[str], args: argparse.Namespace) -> None:
    """Backward-compatible alias."""
    _append_packmol_args(cmd, args)


def _run_lambda_ti_inline(args: argparse.Namespace) -> int:
    from mmml.cli.run.lambda_dynamics import (
        config_from_namespace,
        print_lambda_summary,
        run_lambda_dynamics,
    )

    if args.interval < 1:
        raise ValueError("--interval must be >= 1")
    if args.repeats_per_window < 1:
        raise ValueError("--repeats-per-window must be >= 1")

    if args.output_dir is None:
        args.output_dir = Path("artifacts/lambda_ti")
    if args.backend == "auto":
        args.backend = "ase"
    args.timestep_fs = args.dt_fs
    args.temperature_K = args.temperature
    args.langevin_friction = getattr(args, "langevin_friction", 0.02)

    cfg = config_from_namespace(args, repo_root=_repo_root())
    print("mmml md-system: starting lambda_ti (in-process)", flush=True)
    summary = run_lambda_dynamics(cfg)
    print_lambda_summary(summary)
    print(f"Wrote {summary['_summary_path']}")
    print(f"Snapshots: {summary['snapshots_npz']}")
    print(f"MBAR: mmml lambda-mbar --run-dir {cfg.output_dir}")
    return 0


def _pycharmm_setups() -> set[str]:
    return {
        "free_nve",
        "free_nvt",
        "pycharmm_minimize",
        "pycharmm_full",
        "pbc_nve",
        "pbc_nvt",
        "pbc_npt",
    }


def _apply_backend_setup_defaults(args: argparse.Namespace) -> None:
    """Align ``--setup`` with ``--backend`` when the CLI default setup is wrong for pycharmm."""
    if args.setup == "lambda_ti":
        return

    pycharmm_setups = _pycharmm_setups()
    if args.backend == "auto":
        if args.setup in pycharmm_setups:
            args.backend = "pycharmm"
        return

    if args.backend != "pycharmm":
        return

    if args.setup in pycharmm_setups:
        return

    if args.setup in ("all",):
        raise ValueError(
            f"--backend pycharmm does not support --setup {args.setup!r}. "
            f"Use one of: {', '.join(sorted(pycharmm_setups))}."
        )

    raise ValueError(
        f"--backend pycharmm does not support --setup {args.setup!r}. "
        f"Use one of: {', '.join(sorted(pycharmm_setups))}."
    )


def build_pycharmm_command(args: argparse.Namespace) -> list[str]:
    _phase_for_setup = {
        "pycharmm_minimize": "minimize",
        "free_nve": "staged",
        "free_nvt": "staged",
        "pycharmm_full": "staged",
        "pbc_nve": "staged",
        "pbc_nvt": "staged",
        "pbc_npt": "staged",
    }
    _default_stages = {
        "pycharmm_minimize": "mini",
        "free_nve": "mini,nve",
        "free_nvt": "mini,heat",
        "pycharmm_full": "mini,heat,nve,equi,prod",
        "pbc_nve": "mini,heat,nve,equi,prod",
        "pbc_nvt": "mini,heat,equi",
        "pbc_npt": "mini,heat,nve,equi,prod",
    }
    phase = _phase_for_setup.get(args.setup, "staged")
    ensemble = "nvt" if args.setup == "free_nvt" else "nve"

    if args.output_dir is None:
        args.output_dir = Path("artifacts/pycharmm_mlpot")

    cmd = [
        "--phase",
        phase,
        "--ensemble",
        ensemble,
        "--setup",
        str(args.setup),
        "--spacing",
        str(args.spacing),
        "--ps",
        str(args.ps),
        "--dt-fs",
        str(args.dt_fs),
        "--temperature",
        str(args.temperature),
        "--mini-nstep",
        str(args.mini_nstep),
        "--nprint",
        str(args.nprint),
        "--dyn-nprint",
        str(args.dyn_nprint),
        "--dyn-iprfrq",
        str(args.dyn_iprfrq),
        "--heat-ihtfrq",
        str(args.heat_ihtfrq),
    ]
    _append_optional(cmd, "--heat-firstt", getattr(args, "heat_firstt", None))
    _append_optional(cmd, "--heat-finalt", getattr(args, "heat_finalt", None))
    _append_optional(cmd, "--heat-hoover-tmass", getattr(args, "heat_hoover_tmass", None))
    _append_optional(cmd, "--nve-boltzmann-temp", getattr(args, "nve_boltzmann_temp", None))
    cmd.extend(["--heat-thermostat", str(getattr(args, "heat_thermostat", "scale"))])
    _append_boolean_optional_flag(
        cmd, "--heat-comp-damp", bool(getattr(args, "heat_comp_damp", False))
    )
    _append_boolean_optional_flag(
        cmd,
        "--heat-comp-hydrogen-only",
        bool(getattr(args, "heat_comp_hydrogen_only", True)),
    )
    _append_optional(cmd, "--heat-comp-force-min", getattr(args, "heat_comp_force_min", None))
    _append_optional(
        cmd, "--heat-comp-force-scale", getattr(args, "heat_comp_force_scale", None)
    )
    cmd.extend(
        [
        "--dcd-nsavc",
        str(args.dcd_nsavc),
        "--echeck",
        str(args.echeck),
        "--ps-heat",
        str(args.ps_heat),
        "--ps-equi",
        str(args.ps_equi),
        "--npt-thermostat",
        str(getattr(args, "npt_thermostat", "hoover")),
        "--npt-pressure",
        str(getattr(args, "npt_pressure", 1.0)),
        "--npt-pgamma",
        str(getattr(args, "npt_pgamma", 5.0)),
        "--n-heat-segments",
        str(getattr(args, "n_heat_segments", 1)),
        "--n-equi-segments",
        str(getattr(args, "n_equi_segments", 1)),
        "--n-prod-segments",
        str(args.n_prod_segments),
        ]
    )
    if args.composition:
        cmd.extend(["--composition", str(args.composition)])
        cmd.extend(["--n-molecules", str(_composition_molecule_count(args.composition))])
    else:
        cmd.extend(["--residue", str(args.residue)])
        cmd.extend(["--n-molecules", str(args.n_molecules)])
    if not args.no_fix:
        _append_if_nonempty(cmd, "--fix-resids", args.fix_resids)
    _append_if_nonempty(cmd, "--constrain-resids", args.constrain_resids)
    if args.md_stage:
        cmd.extend(["--md-stage", str(args.md_stage)])
    elif args.md_stages:
        cmd.extend(["--md-stages", str(args.md_stages)])
    elif args.setup in _default_stages:
        cmd.extend(["--md-stages", _default_stages[args.setup]])
    _append_optional(cmd, "--tag", getattr(args, "tag", None))
    _append_optional(cmd, "--checkpoint", args.checkpoint)
    _append_optional(cmd, "--output-dir", args.output_dir)
    _append_optional(cmd, "--box-size", args.box_size)
    _append_optional(cmd, "--ps-nve", getattr(args, "ps_nve", None))
    _append_optional(cmd, "--ps-prod", getattr(args, "ps_prod", None))
    _append_optional(cmd, "--restart-from", getattr(args, "restart_from", None))
    if getattr(args, "continue_from", None) and not getattr(args, "restart_from", None):
        _append_optional(cmd, "--restart-from", args.continue_from)
    _append_optional(cmd, "--from-psf", getattr(args, "from_psf", None))
    _append_optional(cmd, "--from-crd", getattr(args, "from_crd", None))
    if args.no_fix:
        cmd.append("--no-fix")
    if args.no_pre_minimize:
        cmd.append("--no-pre-minimize")
    if getattr(args, "skip_cluster_build", False):
        cmd.append("--skip-cluster-build")
    if getattr(args, "skip_if_crd_exists", False):
        cmd.append("--skip-if-crd-exists")
    if getattr(args, "rescue_old_dcd", False):
        cmd.append("--rescue-old-dcd")
    if getattr(args, "no_save_vmd_topology", False):
        cmd.append("--no-save-vmd-topology")
    if getattr(args, "free_space", False):
        cmd.append("--free-space")
    if getattr(args, "mlpot_pbc", False):
        cmd.append("--mlpot-pbc")
    _append_optional(cmd, "--dyn-inbfrq", getattr(args, "dyn_inbfrq", None))
    _append_optional(cmd, "--dyn-imgfrq", getattr(args, "dyn_imgfrq", None))
    if getattr(args, "pre_nve_charmm_update", None) is False:
        cmd.append("--no-pre-nve-charmm-update")
    elif getattr(args, "pre_nve_charmm_update", None) is True:
        cmd.append("--pre-nve-charmm-update")
    if getattr(args, "bonded_mm_mini", True):
        cmd.append("--bonded-mm-mini")
        cmd.extend(
            [
                "--bonded-mm-mini-after",
                str(getattr(args, "bonded_mm_mini_after", "mini,heat")),
            ]
        )
        cmd.extend(
            [
                "--bonded-mm-mini-steps",
                str(getattr(args, "bonded_mm_mini_steps", 50)),
            ]
        )
        cmd.extend(
            [
                "--bonded-mm-internal-margin",
                str(getattr(args, "bonded_mm_internal_margin", 0.0)),
            ]
        )
        if getattr(args, "bonded_mm_grms_margin", None) is not None:
            cmd.extend(
                ["--bonded-mm-grms-margin", str(args.bonded_mm_grms_margin)]
            )
        if getattr(args, "bonded_mm_internal_energy_margin", 0.0):
            cmd.extend(
                [
                    "--bonded-mm-internal-energy-margin",
                    str(args.bonded_mm_internal_energy_margin),
                ]
            )
        if getattr(args, "bonded_mm_angl_margin", 0.0):
            cmd.extend(
                ["--bonded-mm-angl-margin", str(args.bonded_mm_angl_margin)]
            )
        if getattr(args, "bonded_mm_max_angl_kcal", None) is not None:
            cmd.extend(
                ["--bonded-mm-max-angl-kcal", str(args.bonded_mm_max_angl_kcal)]
            )
        if getattr(args, "bonded_mm_max_internal_kcal", None) is not None:
            cmd.extend(
                [
                    "--bonded-mm-max-internal-kcal",
                    str(args.bonded_mm_max_internal_kcal),
                ]
            )
        if getattr(args, "bonded_mm_mini_always", False):
            cmd.append("--bonded-mm-mini-always")
    else:
        cmd.append("--no-bonded-mm-mini")
    if getattr(args, "allow_high_bonded_strain", False):
        cmd.append("--allow-high-bonded-strain")
    cmd.extend(["--dynamics-overlap-action", str(args.dynamics_overlap_action)])
    cmd.extend(
        ["--dynamics-overlap-min-distance", str(args.dynamics_overlap_min_distance)]
    )
    cmd.extend(
        [
            "--dynamics-intra-min-distance",
            str(getattr(args, "dynamics_intra_min_distance", 1.0)),
        ]
    )
    if getattr(args, "no_dynamics_intra_exclude_1_3", False):
        cmd.append("--no-dynamics-intra-exclude-1-3")
    _append_optional(
        cmd,
        "--dynamics-intra-rescue-sd-steps",
        getattr(args, "dynamics_intra_rescue_sd_steps", None),
    )
    cmd.extend(
        [
            "--dynamics-overlap-check-interval",
            str(args.dynamics_overlap_check_interval),
        ]
    )
    cmd.extend(
        [
            "--dynamics-overlap-charmm-sd-steps",
            str(getattr(args, "dynamics_overlap_charmm_sd_steps", 200)),
        ]
    )
    cmd.extend(
        [
            "--dynamics-overlap-charmm-abnr-steps",
            str(getattr(args, "dynamics_overlap_charmm_abnr_steps", 400)),
        ]
    )
    if getattr(args, "no_dynamics_overlap_separate", False):
        cmd.append("--no-dynamics-overlap-separate")
    if getattr(args, "dynamics_overlap_memory_handoff", False):
        cmd.append("--dynamics-overlap-memory-handoff")
    cmd.extend(
        [
            "--dynamics-overlap-separate-margin",
            str(getattr(args, "dynamics_overlap_separate_margin", 0.2)),
        ]
    )
    cmd.extend(
        ["--mm-switch-on", str(getattr(args, "mm_switch_on", DEFAULT_MM_SWITCH_ON))]
    )
    cmd.extend(
        [
            "--mm-switch-width",
            str(
                getattr(
                    args,
                    "mm_switch_width",
                    getattr(args, "mm_cutoff", DEFAULT_MM_SWITCH_WIDTH),
                )
            ),
        ]
    )
    cmd.extend(
        ["--ml-switch-width", str(getattr(args, "ml_switch_width", DEFAULT_ML_SWITCH_WIDTH))]
    )
    scale = float(getattr(args, "mlpot_mm_internal_scale", 0.0) or 0.0)
    if scale > 0.0:
        cmd.extend(["--mlpot-mm-internal-scale", str(scale)])
    if args.charmm_pre_minimize is False:
        cmd.append("--no-charmm-pre-minimize")
    cmd.extend(["--charmm-sd-steps", str(args.charmm_sd_steps)])
    cmd.extend(["--charmm-abnr-steps", str(args.charmm_abnr_steps)])
    cmd.extend(["--charmm-tolenr", str(args.charmm_tolenr)])
    cmd.extend(["--charmm-tolgrd", str(args.charmm_tolgrd)])
    if getattr(args, "charmm_mm_pretreat", False):
        cmd.append("--charmm-mm-pretreat")
    _append_optional(
        cmd,
        "--charmm-mm-pretreat-heat-nstep",
        getattr(args, "charmm_mm_pretreat_heat_nstep", None),
    )
    _append_optional(
        cmd,
        "--charmm-mm-pretreat-mini-sd",
        getattr(args, "charmm_mm_pretreat_mini_sd", None),
    )
    _append_optional(
        cmd,
        "--charmm-mm-pretreat-mini-abnr",
        getattr(args, "charmm_mm_pretreat_mini_abnr", None),
    )
    if getattr(args, "ml_batch_size", None) is not None:
        cmd.extend(["--ml-batch-size", str(args.ml_batch_size)])
    if getattr(args, "ml_gpu_count", None) is not None:
        cmd.extend(["--ml-gpu-count", str(args.ml_gpu_count)])
    if getattr(args, "ml_spatial_mpi", False):
        cmd.append("--ml-spatial-mpi")
    if getattr(args, "ml_compute_dtype", None) is not None:
        cmd.extend(["--ml-compute-dtype", str(args.ml_compute_dtype)])
    if getattr(args, "ml_max_active_dimers", None) is not None:
        cmd.extend(["--ml-max-active-dimers", str(args.ml_max_active_dimers)])
    if args.no_echeck:
        cmd.append("--no-echeck")
    if getattr(args, "allow_incomplete_dynamics", False):
        cmd.append("--allow-incomplete-dynamics")
    if getattr(args, "skip_energy_show", False):
        cmd.append("--skip-energy-show")
    if getattr(args, "show_energy", None) is True:
        cmd.append("--show-energy")
    elif getattr(args, "show_energy", None) is False:
        cmd.append("--no-show-energy")
    if args.quiet:
        cmd.append("--quiet")
    if getattr(args, "save_forces_npz", False):
        cmd.append("--save-forces-npz")
        cmd.extend(["--forces-npz-interval", str(args.forces_npz_interval)])
    if getattr(args, "no_scale_mini_nstep", False):
        cmd.append("--no-scale-mini-nstep")
    if getattr(args, "no_scale_echeck", False):
        cmd.append("--no-scale-echeck")
    if getattr(args, "allow_high_grms", False):
        cmd.append("--allow-high-grms")
    cmd.extend(["--max-grms-before-dyn", str(args.max_grms_before_dyn)])
    if getattr(args, "test_first", False):
        cmd.append("--test-first")
        cmd.extend(["--test-first-tol", str(args.test_first_tol)])
        cmd.extend(["--test-first-step", str(args.test_first_step)])
        if getattr(args, "test_first_resids", ""):
            cmd.extend(["--test-first-resids", str(args.test_first_resids)])
        if getattr(args, "test_first_charmm", False):
            cmd.append("--test-first-charmm")
        if getattr(args, "test_first_update_nbonds", False):
            cmd.append("--test-first-update-nbonds")
    if args.flat_bottom_radius is not None:
        cmd.extend(["--fb-rad", str(args.flat_bottom_radius)])
        cmd.extend(["--fb-forc", str(args.flat_bottom_k)])
        cmd.extend(["--fb-selection", str(args.flat_bottom_selection)])
    cmd.extend(["--seed", str(args.seed)])
    _append_packmol_sphere_args(cmd, args)
    _append_boolean_optional_flag(
        cmd, "--reuse-packmol-cache", bool(getattr(args, "reuse_packmol_cache", True))
    )
    if getattr(args, "rebuild_packmol", False):
        cmd.append("--rebuild-packmol")
    _append_optional(cmd, "--packmol-cache-dir", getattr(args, "packmol_cache_dir", None))
    if getattr(args, "save_run_state", False):
        cmd.append("--save-run-state")
    _append_optional(cmd, "--run-state-dir", getattr(args, "run_state_dir", None))
    if getattr(args, "flat_bottom_radius", None) is not None:
        cmd.extend(["--flat-bottom-radius", str(args.flat_bottom_radius)])
    if args.extra_args:
        cmd.extend(_filter_campaign_flags_from_argv(list(args.extra_args)))
    return cmd


def _filter_campaign_flags_from_argv(argv: list[str]) -> list[str]:
    """Drop md-system campaign keys that must not reach ASE/JAX-MD parsers."""
    skip = {
        "--job-id",
        "--job-name",
        "--continue-from",
        "--continue-from-frame",
        "--config",
        "--run-all",
        "--resume-campaign",
        "--campaign-output-dir",
    }
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in skip:
            i += 2 if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-") else 1
            continue
        out.append(tok)
        i += 1
    return out


def build_command(args: argparse.Namespace) -> tuple[str, list[str]]:
    backend = args.backend
    if backend == "auto":
        backend = "jaxmd" if args.setup == "pbc_npt" else "ase"

    if backend == "pycharmm":
        if args.setup not in _pycharmm_setups():
            raise ValueError(
                f"--backend pycharmm supports {_pycharmm_setups()}; got {args.setup!r}"
            )
        return "pycharmm", build_pycharmm_command(args)

    skip_box_size_for_cmd = False
    if backend == "jaxmd":
        jaxmd_setups = {"pbc_nve", "pbc_nvt", "pbc_npt", "free_nve", "free_nvt"}
        if args.setup not in jaxmd_setups:
            raise ValueError(
                f"--backend jaxmd supports pbc_nve, pbc_nvt, pbc_npt, free_nve, free_nvt; got {args.setup!r}"
            )
        if args.setup.endswith("_nvt") and args.nvt_integrator == "langevin":
            raise ValueError(
                "--backend jaxmd uses Nose-Hoover chain NVT; use --backend ase for Langevin NVT"
            )
        if args.setup.startswith("free_"):
            ensemble = args.setup[len("free_") :]
            jaxmd_free = True
        else:
            ensemble = args.setup[len("pbc_") :]
            jaxmd_free = False
        cmd = [
            "--ensemble",
            ensemble,
            "--spacing",
            str(args.spacing),
            "--ps",
            str(args.ps),
            "--dt-fs",
            str(args.dt_fs),
            "--temperature",
            str(args.temperature),
            "--pressure",
            str(args.pressure),
            "--traj-chunk-frames",
            str(args.traj_chunk_frames),
        ]
        if jaxmd_free:
            cmd.append("--free-space")
            skip_box_size_for_cmd = True
        if args.traj_export_molecular_wrap:
            cmd.append("--traj-export-molecular-wrap")
    else:
        if args.setup == "pbc_npt":
            raise ValueError("pbc_npt requires --backend jaxmd or --backend auto")
        cmd = [
            "--spacing",
            str(args.spacing),
            "--ps",
            str(args.ps),
            "--dt-fs",
            str(args.dt_fs),
            "--traj-chunk-frames",
            str(args.traj_chunk_frames),
        ]
        if args.setup == "all":
            cmd.append("--all")
        elif args.setup == "free_nve":
            cmd.extend(["--only", "vac_nve"])
        elif args.setup == "free_nvt":
            if args.nvt_integrator == "auto":
                use_langevin = bool(args.composition)
            else:
                use_langevin = args.nvt_integrator == "langevin"
            cmd.extend(["--only", "vac_nvt_langevin" if use_langevin else "vac_nvt_nhc"])
            cmd.extend(["--nvt-temp-K", str(args.temperature)])
        elif args.setup == "pbc_nve":
            cmd.extend(["--only", "pbc_nve"])
        elif args.setup == "pbc_nvt":
            if args.nvt_integrator == "auto":
                use_langevin = bool(args.composition)
            else:
                use_langevin = args.nvt_integrator == "langevin"
            cmd.extend(["--only", "pbc_nvt_langevin" if use_langevin else "pbc_nvt_nhc"])
            cmd.extend(["--nvt-temp-K", str(args.temperature)])
        else:
            raise ValueError(f"Unsupported setup: {args.setup}")

    if args.composition:
        cmd.extend(["--composition", str(args.composition)])
    else:
        cmd.extend(["--n-molecules", str(args.n_molecules)])
    if not skip_box_size_for_cmd:
        _append_optional(cmd, "--box-size", args.box_size)
    _append_optional(cmd, "--checkpoint", args.checkpoint)
    _append_optional(cmd, "--output-dir", args.output_dir)
    _append_optional(cmd, "--template-pdb", args.template_pdb)
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--min-intermonomer-atom-distance", str(args.min_intermonomer_atom_distance)])
    _append_packmol_sphere_args(cmd, args)
    _append_optional(cmd, "--flat-bottom-radius", args.flat_bottom_radius)
    if args.flat_bottom_radius is not None:
        cmd.extend(["--flat-bottom-k", str(args.flat_bottom_k)])
        cmd.extend(["--flat-bottom-mode", str(args.flat_bottom_mode)])
    if getattr(args, "skip_jit_warmup", False):
        cmd.append("--skip-jit-warmup")
    if args.extra_args:
        cmd.extend(_filter_campaign_flags_from_argv(list(args.extra_args)))
    return backend, cmd


def run_backend(backend: str, argv: list[str], args: argparse.Namespace) -> int:
    """Dispatch one backend in-process; honors handoff context vars."""
    global _last_job_stages
    from mmml.cli.run.md_handoff import (
        get_handoff_in,
        get_handoff_out,
        load_handoff,
        save_handoff,
        set_handoff_in,
    )
    from mmml.cli.run.md_stage_summary import (
        MdJobSummary,
        MdStageSummary,
        build_pycharmm_plan_rows,
        build_single_leg_plan_row,
        dynamics_nstep_from_ps,
        ps_from_nsteps,
        print_stage_summary,
        write_stage_summary_json,
    )

    if backend == "pycharmm":
        print(
            f"mmml md-system: running pycharmm ({_pycharmm_run_summary(args)})",
            flush=True,
        )
        print(f"  {' '.join(argv)}", flush=True)
    else:
        print(
            f"mmml md-system: running {backend} in-process:",
            " ".join(argv),
            flush=True,
        )

    handoff_in = get_handoff_in()
    if handoff_in is None and getattr(args, "continue_from", None):
        handoff_in = load_handoff(
            Path(args.continue_from),
            frame=int(getattr(args, "continue_from_frame", -1)),
        )
        if not getattr(args, "continue_velocities", True) and handoff_in is not None:
            from dataclasses import replace

            handoff_in = replace(handoff_in, velocities=None)
        set_handoff_in(handoff_in)

    if backend == "ase":
        from mmml.cli.run.md_pbc_suite import ase as backend_mod
    elif backend == "pycharmm":
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            charmm_lib_links_mpi,
            maybe_rerun_md_system_under_mpirun,
            prepare_serial_charmm_mpi_env,
            _under_mpirun,
            mpirun_launch_hint,
        )

        prepare_serial_charmm_mpi_env()
        campaign_active = bool(
            getattr(args, "config", None)
            and (getattr(args, "job_id", None) or getattr(args, "run_all", False))
        )
        if not campaign_active:
            rerun_code = maybe_rerun_md_system_under_mpirun(sys.argv[1:])
            if rerun_code is not None:
                return rerun_code
        if charmm_lib_links_mpi() and not _under_mpirun():
            print(
                "mmml: OpenMPI-linked CHARMM — for large MLpot clusters prefer:\n  "
                f"  {_repo_root() / 'scripts/mmml-charmm-mpirun.sh'} md-system ...\n"
                "  (or see mpirun hint below if SD hits domdec/MPI errors)\n  "
                + mpirun_launch_hint("mmml md-system"),
                flush=True,
            )
        from mmml.interfaces.pycharmmInterface.jax_device_policy import (
            apply_mlpot_jax_compilation_cache_env,
        )

        apply_mlpot_jax_compilation_cache_env(quiet=True)
        from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
            apply_jax_compile_xla_flags,
        )

        apply_jax_compile_xla_flags(quiet=True)
        from mmml.cli.run.md_pbc_suite import pycharmm_mlpot as backend_mod
    else:
        from mmml.cli.run.md_pbc_suite import jaxmd as backend_mod

    exit_code = int(backend_mod.main(argv))
    handoff_out = get_handoff_out()
    if handoff_out is not None and args.output_dir is not None:
        save_handoff(
            handoff_out,
            Path(args.output_dir),
            template_res=getattr(args, "handoff_template_res", None),
            write_res=bool(getattr(args, "handoff_write_res", True)),
        )

    job_name = resolve_job_name(args) or "run"
    if backend == "pycharmm":
        from mmml.cli.run.md_stage_summary import (
            pycharmm_stage_dcd_frames,
            pycharmm_trajectory_tag,
        )

        plan: list[MdStageSummary] = build_pycharmm_plan_rows(job_name, args)
        tag = pycharmm_trajectory_tag(args)
        out_dir = Path(args.output_dir) if args.output_dir is not None else None
        for row in plan:
            if row.stage != "mini":
                row.nsteps_completed = row.nsteps_requested
                row.ps_completed = row.ps_requested
                row.status = "complete" if exit_code == 0 else "error"
                if out_dir is not None and row.stage in {"heat", "equi", "nve", "prod"}:
                    row.frames_written = pycharmm_stage_dcd_frames(out_dir, row.stage, tag)
                    row.record_every_steps = max(1, int(row.record_every_steps or 1))
    else:
        row = build_single_leg_plan_row(job_name, args, backend)
        nsteps = dynamics_nstep_from_ps(float(args.ps), float(args.dt_fs))
        row.nsteps_requested = nsteps
        row.nsteps_completed = nsteps if exit_code == 0 else 0
        row.ps_completed = ps_from_nsteps(row.nsteps_completed, float(args.dt_fs))
        row.status = "complete" if exit_code == 0 else "error"
        plan = [row]
    _last_job_stages = plan
    if not getattr(args, "no_stage_summary", False) and args.output_dir is not None:
        js = MdJobSummary(
            job_id=job_name,
            backend=backend,
            setup=str(args.setup),
            stages=plan,
            exit_code=exit_code,
        )
        write_stage_summary_json(js, Path(args.output_dir))
        if not getattr(args, "quiet", False):
            for st in plan:
                print_stage_summary(st)
    return exit_code


def main() -> int:
    args = parse_md_system_args()
    started_at = datetime.now(timezone.utc).isoformat()
    backend: str | None = None
    argv: list[str] | None = None
    exit_code = 2
    try:
        try:
            _validate_packmol_args(args)
        except ValueError as exc:
            print(f"mmml md-system: error: {exc}", file=sys.stderr)
            return 2
        if args.config and (args.job_id or args.run_all):
            from mmml.cli.run.md_campaign import run_campaign

            return int(run_campaign(args))
        if args.setup == "lambda_ti":
            backend = str(args.backend)
            try:
                exit_code = _run_lambda_ti_inline(args)
            except ValueError as exc:
                print(f"mmml md-system: error: {exc}", file=sys.stderr)
                exit_code = 2
            return exit_code
        try:
            _apply_backend_setup_defaults(args)
        except ValueError as exc:
            print(f"mmml md-system: error: {exc}", file=sys.stderr)
            return 2
        try:
            backend, argv = build_command(args)
        except ValueError as exc:
            print(f"mmml md-system: error: {exc}", file=sys.stderr)
            return 2
        exit_code = run_backend(backend, argv, args)
        return exit_code
    finally:
        _maybe_save_job_run_manifest(
            args,
            backend=backend,
            argv=argv,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            exit_code=exit_code,
        )


if __name__ == "__main__":
    raise SystemExit(main())
