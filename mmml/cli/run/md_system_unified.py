"""Opt-in unified-stack path for ``mmml md-system --backend jaxmd``.

Wires the legacy ``md-system`` CLI onto the shared ``mmml.md`` pipeline
(``runconfig_from_md_system_args`` → ``assemble_and_run``) instead of the
legacy ``mmml.cli.run.md_pbc_suite.jaxmd`` inline loop. Opt-in via
``--jaxmd-unified`` (see ``build_parser()`` / ``main()`` in
``mmml.cli.run.md_system``) so the existing default path is untouched until
this one is validated across more of md-system's feature surface.

Deliberately NOT yet supported (raise clearly rather than silently diverge
from the legacy backend):

- ``--builder pyxtal`` / ``--template-pdb`` (only the packmol composition
  builder is wired here).
- Campaign/handoff continuation (``--continue-from``), lambda-TI.

See ``docs/md-cg-unification-design.md`` (§0, §9, §11) and
``docs/md-cg-unification-handoff.md`` for the surrounding architecture.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["check_md_system_args_supported", "build_packmol_system_with_ffparams", "run_unified_jaxmd"]


def check_md_system_args_supported(args: Any) -> None:
    """Raise clearly (before any CHARMM build) for combinations not yet wired."""
    builder = getattr(args, "builder", None)
    if builder not in (None, "packmol"):
        raise NotImplementedError(
            f"--jaxmd-unified only supports the packmol composition builder; got --builder {builder!r}"
        )
    if getattr(args, "template_pdb", None):
        raise NotImplementedError("--jaxmd-unified does not yet support --template-pdb")
    if getattr(args, "continue_from", None):
        raise NotImplementedError("--jaxmd-unified does not yet support --continue-from (handoff)")
    if not getattr(args, "checkpoint", None):
        raise ValueError("--jaxmd-unified requires --checkpoint")


def build_packmol_system_with_ffparams(spec: Any):
    """Build a composition system via packmol and lower it to ``FFParams``.

    ``PackmolSystemBuilder`` (``mmml.md.builders.placement``) does not write a
    PSF file by default, since ``build_packmol_composition_cluster`` never
    persists one — it only leaves the built system live in CHARMM. Reuse the
    same building blocks, then write the live PSF to a scratch file so
    ``FFParams`` can be resolved from it (mirrors ``_lower_optional_psf``).
    """
    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster
    from mmml.md.builders.placement import _box, _composition, _lower_optional_psf, _placement_system

    params = dict(spec.params)
    box = _box(spec, params)
    composition = _composition(spec)
    if "center" not in params:
        if box is None:
            raise ValueError("packmol builder requires center or a box size")
        params["center"] = tuple(np.diag(box) / 2.0)
    params.setdefault("cube_side", spec.box_size)

    z, positions, sizes, residues = build_packmol_composition_cluster(
        composition=composition, seed=spec.seed, **params
    )

    with tempfile.TemporaryDirectory() as tmp:
        import pycharmm.write as write

        psf_path = Path(tmp) / "md_system_unified.psf"
        write.psf_card(str(psf_path))
        system = _placement_system(
            name="packmol", spec=spec, z=z, positions=positions,
            atoms_per_molecule=list(sizes), residue_names=list(residues), box=box,
        )
        system = _lower_optional_psf(system, psf_path=psf_path, prm_paths=())
    return system


def _load_model(checkpoint_path: Path) -> tuple[Any, Any]:
    from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint

    calc = create_calculator_from_checkpoint(str(checkpoint_path))
    model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
    params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
    if model is None or params is None:
        raise ValueError(f"could not extract model/params from checkpoint {checkpoint_path}")
    return model, params


def run_unified_jaxmd(args: Any) -> int:
    """Run ``args`` through the unified ``mmml.md`` pipeline; return an exit code."""
    check_md_system_args_supported(args)

    import jax

    jax.config.update("jax_enable_x64", True)

    from mmml.md.assemble import assemble_and_run
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.lowering import runconfig_from_md_system_args

    run_config = runconfig_from_md_system_args(args)
    system = build_packmol_system_with_ffparams(run_config.system)
    model, params = _load_model(run_config.checkpoint)
    ctx = EnergyContext(model=model, params=params)

    traj = assemble_and_run(run_config, system=system, ctx=ctx)

    energies = traj.metadata.get("energies")
    if energies is not None and len(energies):
        print(
            f"mmml md-system (jaxmd-unified): {traj.n_frames} frames, "
            f"E0={energies[0]:.4f} eV, Efinal={energies[-1]:.4f} eV",
            flush=True,
        )
    if not np.all(np.isfinite(energies)):
        print("mmml md-system: jaxmd-unified produced non-finite energies", file=sys.stderr)
        return 1
    return 0
