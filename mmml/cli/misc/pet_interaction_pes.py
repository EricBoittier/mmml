"""PET-MAD (or any ASE calculator) interaction slices, surfaces, and trimer MBE.

CHARMM-free single-point scans. Default recipe: PET-MAD xs 1.5.0 on water /
ethanol (acetone 1D optional) with ``E_int = E(AB)-E(A)-E(B)``. Orientations
are internal-axis H-bonds (linear OH···O vs acceptor–acceptor), not COM copies.

Example::

    JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \\
      mmml pet-interaction-pes --checkpoint /tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mmml.analysis.interaction_pes import (
    DEFAULT_ETOH_XYZ,
    DEFAULT_N_R_1D,
    DEFAULT_N_R_2D,
    DEFAULT_N_R_TRIMER,
    DEFAULT_N_THETA_2D,
    DEFAULT_R_2D_MAX_A,
    DEFAULT_R_2D_MIN_A,
    DEFAULT_R_MAX_A,
    DEFAULT_R_MIN_A,
    DEFAULT_SURFACE_SYSTEM,
    DEFAULT_THETA_MAX_DEG,
    DEFAULT_THETA_MIN_DEG,
    DEFAULT_WATER_XYZ,
    SCHEMA_VERSION,
    SYSTEM_ACETONE,
    SYSTEM_ETHANOL,
    SYSTEM_WATER,
)

DEFAULT_OUTPUT_DIR = Path("scratch") / "pet_interaction_pes"
DEFAULT_JSON_NAME = "interaction_pes.json"
DEFAULT_NPZ_NAME = "interaction_pes.npz"


def _default_checkpoint() -> Path | None:
    env = os.environ.get("PET_MAD_CKPT", "").strip() or os.environ.get("MMML_CKPT", "").strip()
    if env:
        return Path(env).expanduser()
    model_dir = os.environ.get("MMML_METATOMIC_MODEL_DIR", "").strip()
    if model_dir:
        candidate = Path(model_dir).expanduser() / "pet-mad-xs-v1.5.0.pt"
        if candidate.is_file():
            return candidate
    fallback = Path("/tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt")
    return fallback if fallback.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml pet-interaction-pes",
        description=(
            "Rigid OH···O interaction slices/surfaces and trimer many-body leftover "
            "for a metatomic PET checkpoint (CHARMM-free single points)."
        ),
    )
    inputs = parser.add_argument_group("input")
    inputs.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="TorchScript AtomisticModel (.pt). Default: $PET_MAD_CKPT.",
    )
    inputs.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Replot a saved campaign JSON (no calculator).",
    )
    inputs.add_argument(
        "--water-xyz",
        type=Path,
        default=DEFAULT_WATER_XYZ,
        help=f"Water monomer xyz (default: {DEFAULT_WATER_XYZ.as_posix()}).",
    )
    inputs.add_argument(
        "--ethanol-xyz",
        type=Path,
        default=DEFAULT_ETOH_XYZ,
        help=f"Ethanol monomer xyz (default: {DEFAULT_ETOH_XYZ.as_posix()}).",
    )
    inputs.add_argument(
        "--include-acetone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add acetone 1D slices (C=O acceptor vs methyl–methyl).",
    )
    grids = parser.add_argument_group("scan grid")
    grids.add_argument("--r-min", type=float, default=DEFAULT_R_MIN_A, metavar="ANGSTROM")
    grids.add_argument("--r-max", type=float, default=DEFAULT_R_MAX_A, metavar="ANGSTROM")
    grids.add_argument(
        "--n-r",
        type=int,
        default=DEFAULT_N_R_1D,
        help="Uniform 1D count; 0 uses a piecewise well/far grid.",
    )
    grids.add_argument("--r-2d-min", type=float, default=DEFAULT_R_2D_MIN_A, metavar="ANGSTROM")
    grids.add_argument("--r-2d-max", type=float, default=DEFAULT_R_2D_MAX_A, metavar="ANGSTROM")
    grids.add_argument(
        "--n-r-2d",
        type=int,
        default=DEFAULT_N_R_2D,
        help="Uniform 2D r count; 0 uses the default well window.",
    )
    grids.add_argument("--n-theta", type=int, default=DEFAULT_N_THETA_2D)
    grids.add_argument("--theta-min", type=float, default=DEFAULT_THETA_MIN_DEG, metavar="DEG")
    grids.add_argument("--theta-max", type=float, default=DEFAULT_THETA_MAX_DEG, metavar="DEG")
    grids.add_argument(
        "--n-r-trimer",
        type=int,
        default=DEFAULT_N_R_TRIMER,
        help="Uniform trimer count; 0 uses the default O–O grid.",
    )
    grids.add_argument("--surface-system", default=DEFAULT_SURFACE_SYSTEM)
    outputs = parser.add_argument_group("output")
    outputs.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    outputs.add_argument("--json-out", type=Path, default=None)
    outputs.add_argument("--prefix", default="pet_mad")
    return parser


def _write_report(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _grid_or_default(n: int, start: float, stop: float, default_fn):
    from mmml.analysis.interaction_pes import linspace_angstrom

    if int(n) >= 2:
        return linspace_angstrom(start, stop, int(n))
    return default_fn()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser()
    json_out = (args.json_out or (output_dir / DEFAULT_JSON_NAME)).expanduser()

    if args.from_json is not None:
        from mmml.analysis.interaction_pes import load_interaction_pes_json
        from mmml.analysis.interaction_pes_plot import write_interaction_pes_figures

        document = load_interaction_pes_json(args.from_json)
        figures = write_interaction_pes_figures(document, output_dir, prefix=args.prefix)
        dump = {
            "ok": True,
            "schema": document.get("schema", SCHEMA_VERSION),
            "from_json": str(args.from_json.resolve()),
            "figures": {name: str(path) for name, path in figures.items()},
            "summary": document.get("summary", {}),
        }
        _write_report(output_dir / "report.json", dump)
        return 0

    checkpoint = args.checkpoint or _default_checkpoint()
    if checkpoint is None or not Path(checkpoint).expanduser().is_file():
        payload = {
            "ok": False,
            "error": "missing_checkpoint",
            "checkpoint": str(checkpoint) if checkpoint else "",
        }
        _write_report(output_dir / "report.json", payload)
        if args.json_out is not None:
            _write_report(json_out, payload)
        return 2

    from mmml.analysis.dimer_scans import centered_atoms
    from mmml.analysis.interaction_pes import (
        dump_interaction_pes_json,
        dump_interaction_pes_npz,
        default_dha_deg,
        default_r_1d_angstrom,
        default_r_2d_angstrom,
        default_r_trimer_angstrom,
        load_monomer_xyz,
        linspace_angstrom,
        run_interaction_pes_campaign,
        sha256_file,
    )
    from mmml.analysis.interaction_pes_plot import write_interaction_pes_figures
    from mmml.interfaces.calculators.metatomic import load_metatomic_calculator

    ckpt = Path(checkpoint).expanduser().resolve()
    systems = {
        SYSTEM_WATER: load_monomer_xyz(args.water_xyz),
        SYSTEM_ETHANOL: load_monomer_xyz(args.ethanol_xyz),
    }
    slice_systems = [SYSTEM_WATER, SYSTEM_ETHANOL]
    if args.include_acetone:
        from mmml.distill.acetone_pool import load_acetone_monomer

        systems[SYSTEM_ACETONE] = centered_atoms(load_acetone_monomer(), center="com")
        slice_systems.append(SYSTEM_ACETONE)

    def factory():
        return load_metatomic_calculator(ckpt)

    theta = (
        linspace_angstrom(args.theta_min, args.theta_max, args.n_theta)
        if args.n_theta >= 2
        else default_dha_deg()
    )
    document = run_interaction_pes_campaign(
        calculator_factory=factory,
        systems=systems,
        r_1d=_grid_or_default(args.n_r, args.r_min, args.r_max, default_r_1d_angstrom),
        r_2d=_grid_or_default(args.n_r_2d, args.r_2d_min, args.r_2d_max, default_r_2d_angstrom),
        theta_deg=theta,
        r_trimer=_grid_or_default(args.n_r_trimer, args.r_min, args.r_max, default_r_trimer_angstrom),
        slice_systems=slice_systems,
        surface_system=args.surface_system,
        trimer_systems=(SYSTEM_WATER, SYSTEM_ETHANOL),
        calculator_name="metatomic",
        checkpoint=str(ckpt),
        checkpoint_sha256=sha256_file(ckpt),
    )
    dump_interaction_pes_json(document, json_out)
    dump_interaction_pes_npz(document, json_out.with_name(DEFAULT_NPZ_NAME))
    figures = write_interaction_pes_figures(document, output_dir, prefix=args.prefix)
    _write_report(
        output_dir / "report.json",
        {
            "ok": True,
            "schema": SCHEMA_VERSION,
            "json": str(json_out.resolve()),
            "checkpoint": str(ckpt),
            "checkpoint_sha256": document.get("checkpoint_sha256"),
            "figures": {name: str(path) for name, path in figures.items()},
            "summary": document["summary"],
            "n_cached_energies": document["n_cached_energies"],
        },
    )
    return 0
