#!/usr/bin/env python3
"""Bond/angle/dihedral/RDF plots for one completed mixed_calculator_sweep setting.

Reuses `mmml.utils.plotting.trajectory_structure` (the module written for
exactly this: bonds/angles/dihedrals/RDFs from a trajectory) and
`scripts/plot_trajectory_structure.py`'s `plot_rdfs`/`plot_internal` plotting
functions, applied to ASE `Atoms` reconstructed from `trajectory.npz`.

`JaxmdDriver`/`RigidBodySampler` now save `Z` (and `box`) alongside
`positions`/`energies` specifically so this doesn't need to re-run any
dynamics. For `trajectory.npz` files written *before* that fix, pass
`--workflow-config`/`--repo-root` so this script can quickly rebuild the same
system (deterministic given the same seed; this only reconstructs topology,
it does not re-run the simulation) to recover `Z`/`box`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from ase import Atoms

from mmml.utils.plotting.styles import apply_plot_style
from mmml.utils.plotting.trajectory_structure import (
    element_pair_rdfs,
    internal_coordinate_distributions,
)

_STYLE_NAME = "icml"  # see docs/plot-style-gallery.md


def _load_reference_plotting_module(repo_root: Path):
    """Import scripts/plot_trajectory_structure.py's plot_rdfs/plot_internal by path.

    It's a "scripts/" file (not a package), so importlib-by-path is simpler and
    more robust than sys.path manipulation.
    """
    target = repo_root / "scripts" / "plot_trajectory_structure.py"
    spec = importlib.util.spec_from_file_location("plot_trajectory_structure", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rebuild_topology(workflow_config_path: Path, repo_root: Path, setting: str, seed: int):
    """Recover Z/box for a trajectory.npz written before the topology fix.

    Only rebuilds the *system* (packmol/CHARMM topology, seconds) -- does not
    re-run any dynamics.
    """
    import yaml

    workflow_config = yaml.safe_load(workflow_config_path.read_text(encoding="utf-8"))
    spec = workflow_config["settings"][setting]

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.md.system import SystemSpec

    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

    if spec["system"] == "water_box":
        from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams

        sys_spec = SystemSpec(
            builder="packmol", composition=spec["composition"],
            box_size=float(spec["box_size"]), seed=seed,
        )
        system = build_packmol_system_with_ffparams(sys_spec)
    elif spec["system"] == "peptide_water":
        from mmml.md.assemble import build_system

        sys_spec = SystemSpec(
            builder="peptide_water", n_molecules=int(spec["n_waters"]),
            box_size=float(spec["box_size"]), seed=seed,
        )
        system = build_system(sys_spec)
    else:
        raise ValueError(f"unknown system {spec['system']!r}")
    return np.asarray(system.Z), (None if system.box is None else np.asarray(system.box))


def frames_from_npz(
    npz_path: Path, *, workflow_config: Path | None, repo_root: Path | None,
    setting: str | None, seed: int | None,
) -> list[Atoms]:
    data = np.load(npz_path)
    positions = np.asarray(data["positions"])
    if "Z" in data:
        Z = np.asarray(data["Z"])
        box = np.asarray(data["box"]) if "box" in data else None
    else:
        if workflow_config is None or repo_root is None or setting is None or seed is None:
            raise ValueError(
                f"{npz_path} predates topology-saving; pass --workflow-config/--repo-root/"
                "--setting/--seed so the system can be rebuilt to recover Z/box."
            )
        Z, box = _rebuild_topology(workflow_config, repo_root, setting, seed)
    return [
        Atoms(numbers=Z, positions=pos, cell=box, pbc=box is not None)
        for pos in positions
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--setting", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workflow-config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--rdf-bins", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.results_dir / args.setting / f"seed_{args.seed}" / "trajectory.npz"
    if not npz_path.is_file():
        raise SystemExit(f"no trajectory.npz at {npz_path}")

    frames = frames_from_npz(
        npz_path, workflow_config=args.workflow_config, repo_root=args.repo_root,
        setting=args.setting, seed=args.seed,
    )

    out_dir = args.out_dir or (args.results_dir / args.setting / f"seed_{args.seed}" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style(_STYLE_NAME)
    ref = _load_reference_plotting_module(args.repo_root)

    n_atoms = len(frames[0])
    radii, rdfs = element_pair_rdfs(frames, r_max=args.r_max, bins=args.rdf_bins)
    internal = internal_coordinate_distributions(frames, range(n_atoms))

    rdf_path = ref.plot_rdfs(radii, rdfs, out_dir / "element_pair_rdfs.png")
    internal_path = ref.plot_internal(internal, out_dir / "internal_coordinates.png")
    print(f"wrote {out_dir / 'element_pair_rdfs.png'}")
    print(f"wrote {out_dir / 'internal_coordinates.png'}")


if __name__ == "__main__":
    main()
