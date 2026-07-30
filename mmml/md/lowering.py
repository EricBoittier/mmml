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
from mmml.md.temperature import parse_temperature_schedule

__all__ = [
    "terms_from_cg_config",
    "terms_from_md_system_args",
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
    schedule_text = cfg.get("temperature_schedule") or cfg.get("temp_schedule")
    ensemble = EnsembleSpec(
        ensemble=ensemble_name,
        space="pbc",
        temperature_K=float(cfg.get("temperature", 300.0)),
        temperature_schedule=(parse_temperature_schedule(str(schedule_text)) if schedule_text else None),
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


def terms_from_md_system_args(args: Any) -> tuple[str, ...]:
    """Select registered energy terms from ``md-system`` argparse knobs.

    Precedence:
    1. Explicit ``args.terms`` when non-empty.
    2. ``--ff cgenff`` → ``mm_nonbonded`` only.
    3. ``--ff zbl-mbd-multipoles`` → intermolecular ZBL + fixed multipole + fixed C6.
    4. ``--sampler rigid`` with no checkpoint and no ``--ff`` → CGenFF default.
    5. Otherwise hybrid ``ml_intra`` + ``mm_nonbonded``.
    """
    explicit = getattr(args, "terms", None)
    if explicit:
        return tuple(explicit)

    ff = getattr(args, "ff", None)
    sampler = getattr(args, "sampler", "md") or "md"
    checkpoint = getattr(args, "checkpoint", None)

    if ff is None and sampler == "rigid" and not checkpoint:
        ff = "cgenff"

    if ff == "cgenff":
        return ("mm_nonbonded",)
    if ff == "zbl-mbd-multipoles":
        return ("zbl", "mbd", "multipole")
    return ("ml_intra", "mm_nonbonded")


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

    # ``--from-pdb`` loads a prebuilt full-system PDB (e.g. a make-box solvated
    # cell) and needs no composition; without this it fell through to the
    # packmol composition builder and failed on the missing composition.
    from_pdb = getattr(args, "from_pdb", None)
    builder = getattr(args, "builder", None)
    if not builder:
        builder = "from_pdb" if from_pdb else "packmol"
    system = SystemSpec(
        builder=builder,
        composition=getattr(args, "composition", None),
        n_molecules=getattr(args, "n_molecules", None),
        box_size=getattr(args, "box_size", None),
        # Full-system PDB for the ``from_pdb`` builder. Distinct from the
        # ``--template-pdb`` monomer-template flag, which this path rejects.
        template_pdb=Path(from_pdb) if from_pdb else None,
        seed=int(getattr(args, "seed", 0)),
    )
    schedule_text = getattr(args, "temperature_schedule", None)
    ens_params: dict[str, Any] = {
        "seed": int(getattr(args, "seed", 0)),
    }
    # NPT barostat + virial AD is sensitive to float32; prefer float64 for NPT.
    if ensemble_name == "npt":
        ens_params["float64"] = True
    ensemble = EnsembleSpec(
        ensemble=ensemble_name,
        space=space,
        temperature_K=float(getattr(args, "temperature", 300.0)),
        temperature_schedule=(parse_temperature_schedule(schedule_text) if schedule_text else None),
        pressure_bar=float(getattr(args, "pressure", 1.0)),
        dt_fs=dt_fs,
        n_steps=_nsteps_from_ps(ps, dt_fs),
        params=ens_params,
    )
    terms = terms_from_md_system_args(args)
    checkpoint = getattr(args, "checkpoint", None)
    output_dir = getattr(args, "output_dir", None)
    sampler = getattr(args, "sampler", "md") or "md"
    return RunConfig(
        system=system,
        terms=terms,
        ensemble=ensemble,
        backend="jaxmd",
        sampler=str(sampler),
        checkpoint=Path(checkpoint) if checkpoint else None,
        output_dir=Path(output_dir) if output_dir else None,
        seed=int(getattr(args, "seed", 0)),
        params={
            k: v
            for k, v in {
                "ff": getattr(args, "ff", None),
                "mbd_checkpoint": getattr(args, "mbd_checkpoint", None),
                "mbd_weight": getattr(args, "mbd_weight", 1.0),
                "multipole_checkpoint": getattr(args, "multipole_checkpoint", None),
                "interaction_policy": getattr(args, "interaction_policy", None),
            }.items()
            if v is not None
        },
    )
