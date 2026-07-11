"""Lower the two front-end configs into one :class:`RunConfig`.

Both entry points feed the shared assembly layer (``mmml.md.assemble``) through a
single internal representation (constraint 7). This module holds the pure
mapping functions:

- ``runconfig_from_md_system_args`` — the ``md-system`` argparse ``Namespace``.
- ``runconfig_from_cg_config`` — the ``cg_jaxmd`` Snakemake/JSON config.

The heart of the cg lowering is ``terms_from_cg_config``: the sweep toggles
(``use_ml_intramolecular`` / ``peptide_water_ml`` / …) become an energy-term
selection (doc §8) — no code fork between the two energy modes.

Pure (stdlib only) so it is testable without jax/CHARMM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mmml.md.config import EnsembleSpec, RunConfig
from mmml.md.system import SystemSpec

__all__ = [
    "terms_from_cg_config",
    "runconfig_from_cg_config",
    "ensemble_from_setup",
    "runconfig_from_md_system_args",
]


def _nsteps_from_ps(ps: float, dt_fs: float) -> int:
    return int(round(float(ps) * 1000.0 / float(dt_fs)))


# --- cg_jaxmd JSON -----------------------------------------------------------

_CG_PHASE_ENSEMBLE = {"fire": "min", "nvt": "nvt", "nve": "nve"}


def terms_from_cg_config(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Map cg_jaxmd energy-mode toggles to a registered-term selection (doc §8).

    - ``use_ml_intramolecular`` → ``ml_intra`` (else CHARMM handles intramol).
    - ``mm_nonbonded`` is always present for intermolecular MM.
    - ``peptide_water_ml`` → ``ml_pep_water`` (core–solvent handled by ML, so
      those pairs are excluded from MM); ``+ vdw_core`` when the repulsive wall
      is enabled.
    - ``constrain_phi_psi`` → ``dihedral``; ``smd_enable`` → ``smd``.
    """
    terms: list[str] = []
    if bool(cfg.get("use_ml_intramolecular", True)):
        terms.append("ml_intra")
    terms.append("mm_nonbonded")
    if bool(cfg.get("peptide_water_ml", False)):
        terms.append("ml_pep_water")
        if bool(cfg.get("peptide_water_ml_core_vdw", False)):
            terms.append("vdw_core")
    if bool(cfg.get("constrain_phi_psi", False)):
        terms.append("dihedral")
    if bool(cfg.get("smd_enable", False)):
        terms.append("smd")
    return tuple(terms)


def runconfig_from_cg_config(cfg: Mapping[str, Any], phase: str = "nve") -> RunConfig:
    """Lower one cg_jaxmd JSON config + phase (``fire`` / ``nvt`` / ``nve``)."""
    if phase not in _CG_PHASE_ENSEMBLE:
        raise ValueError(f"unknown cg phase {phase!r}; expected fire, nvt, or nve")
    ensemble_name = _CG_PHASE_ENSEMBLE[phase]
    dt_fs = float(cfg.get("dt_fs", 0.5))
    n_steps = int(cfg.get(f"{phase}_total_steps", cfg.get(f"{phase}_steps", 0)))

    checkpoint = cfg.get("checkpoint") or cfg.get("peptide_checkpoint")
    output_dir = cfg.get("output_dir")

    system = SystemSpec(
        builder="peptide_water",
        n_molecules=cfg.get("n_waters"),
        box_size=cfg.get("box_size"),
        seed=int(cfg.get("seed", 0)),
        params={k: cfg[k] for k in ("sequence", "workdir", "initial_peptide_pdb") if k in cfg},
    )
    ensemble = EnsembleSpec(
        ensemble=ensemble_name,
        space="pbc",
        temperature_K=float(cfg.get("temperature", 300.0)),
        dt_fs=dt_fs,
        n_steps=n_steps,
        params={"block_steps": cfg.get(f"{phase}_block_steps")},
    )
    return RunConfig(
        system=system,
        terms=terms_from_cg_config(cfg),
        ensemble=ensemble,
        backend="jaxmd",
        checkpoint=Path(checkpoint) if checkpoint else None,
        output_dir=Path(output_dir) if output_dir else None,
        seed=int(cfg.get("seed", 0)),
    )


# --- md-system argparse ------------------------------------------------------


def ensemble_from_setup(setup: str) -> tuple[str, str]:
    """Split a jaxmd ``--setup`` into ``(space, ensemble)``.

    ``pbc_nve`` → (``pbc``, ``nve``); ``free_nvt`` → (``free``, ``nvt``);
    ``pbc_npt`` → (``pbc``, ``npt``). ``*_thermalize`` aliases to ``nvt``.
    """
    setup = str(setup)
    if "_" not in setup:
        raise ValueError(f"cannot parse setup {setup!r}; expected '<space>_<ensemble>'")
    space, ensemble = setup.split("_", 1)
    if space not in ("pbc", "free"):
        raise ValueError(f"unsupported space in setup {setup!r}")
    if ensemble == "thermalize":
        ensemble = "nvt"
    if ensemble not in ("nve", "nvt", "npt"):
        raise ValueError(f"unsupported ensemble in setup {setup!r}")
    return space, ensemble


def runconfig_from_md_system_args(args: Any) -> RunConfig:
    """Lower an ``mmml md-system`` argparse ``Namespace`` (jaxmd backend)."""
    space, ensemble_name = ensemble_from_setup(getattr(args, "setup"))
    dt_fs = float(getattr(args, "dt_fs", 1.0))
    ps = float(getattr(args, "ps", 0.0))

    system = SystemSpec(
        builder=getattr(args, "builder", None) or "packmol",
        composition=getattr(args, "composition", None),
        n_molecules=getattr(args, "n_molecules", None),
        box_size=getattr(args, "box_size", None),
        seed=int(getattr(args, "seed", 0)),
    )
    ensemble = EnsembleSpec(
        ensemble=ensemble_name,
        space=space,
        temperature_K=float(getattr(args, "temperature", 300.0)),
        pressure_bar=float(getattr(args, "pressure", 1.0)),
        dt_fs=dt_fs,
        n_steps=_nsteps_from_ps(ps, dt_fs),
    )
    terms = tuple(getattr(args, "terms", ()) or ("ml_intra", "mm_nonbonded"))
    checkpoint = getattr(args, "checkpoint", None)
    output_dir = getattr(args, "output_dir", None)
    return RunConfig(
        system=system,
        terms=terms,
        ensemble=ensemble,
        backend="jaxmd",
        checkpoint=Path(checkpoint) if checkpoint else None,
        output_dir=Path(output_dir) if output_dir else None,
        seed=int(getattr(args, "seed", 0)),
    )
