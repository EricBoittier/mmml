"""Enforce zero CHARMM energy components via PSF/.prm reload (not BLOCK)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import (
    resolve_periodic_charmm_vdw,
)


@dataclass(frozen=True)
class CharmmEnergyTermPolicy:
    """Map CHARMM ENER keys to PSF/.prm remediation actions."""

    name: str
    energy_keys: tuple[str, ...]
    tolerance_kcal: float = 1.0e-4
    zero_bonded_prm: bool = False
    zero_nonbond_prm: bool = False
    zero_ml_charges: bool = False


POLICY_REGISTRY: dict[str, CharmmEnergyTermPolicy] = {
    "vdw": CharmmEnergyTermPolicy(
        name="vdw",
        energy_keys=("VDW", "IMNB"),
        zero_nonbond_prm=True,
    ),
    "elec": CharmmEnergyTermPolicy(
        name="elec",
        energy_keys=("ELEC", "IMEL", "EXTE"),
        zero_ml_charges=True,
    ),
    "bonded": CharmmEnergyTermPolicy(
        name="bonded",
        energy_keys=("BOND", "ANGL", "UREY", "DIHE", "IMPR", "CDIH"),
        zero_bonded_prm=True,
    ),
    "hbond": CharmmEnergyTermPolicy(
        name="hbond",
        energy_keys=("HBON", "IMHB"),
        tolerance_kcal=1.0e-3,
    ),
}


def _parse_term_list(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    out: list[str] = []
    for tok in str(raw).replace(" ", ",").split(","):
        name = tok.strip().lower()
        if name:
            out.append(name)
    return out


def resolve_charmm_energy_term_policies(
    args: argparse.Namespace | None,
) -> list[CharmmEnergyTermPolicy]:
    """Active policies from CLI flags and ``--charmm-zero-energy-terms``."""
    names = _parse_term_list(
        getattr(args, "charmm_zero_energy_terms", None) if args is not None else None
    )
    if args is not None and not resolve_periodic_charmm_vdw(args):
        if "vdw" not in names:
            names.append("vdw")
    policies: list[CharmmEnergyTermPolicy] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        policy = POLICY_REGISTRY.get(name)
        if policy is None:
            known = ", ".join(sorted(POLICY_REGISTRY))
            raise ValueError(
                f"Unknown --charmm-zero-energy-terms entry {name!r}; known: {known}"
            )
        seen.add(name)
        policies.append(policy)
    return policies


def measure_charmm_energy_terms() -> dict[str, float]:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_energy_row

    return dict(charmm_energy_row())


def _policy_violation(
    policy: CharmmEnergyTermPolicy,
    terms: dict[str, float],
) -> tuple[bool, dict[str, float]]:
    hits: dict[str, float] = {}
    for key in policy.energy_keys:
        val = float(terms.get(key, 0.0))
        if abs(val) > float(policy.tolerance_kcal):
            hits[key] = val
    return bool(hits), hits


def _policy_scratch_dir(args: argparse.Namespace | None) -> Path:
    if args is not None:
        out = getattr(args, "output_dir", None)
        if out is not None:
            return Path(out) / "charmm_energy_policy"
    return Path("charmm_energy_policy")


def _zero_ml_atom_charges(ml_selection: Any) -> None:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.psf as psf

    charges = list(psf.get_charges())
    for idx in ml_selection.get_atom_indexes():
        charges[int(idx)] = 0.0
    psf.set_charge(charges)


def _reload_prm_overlay(
    overlay_path: Path,
    *,
    use_pbc: bool,
    cubic_box_side_A: float | None,
    ml_selection: Any,
    zero_ml_charges: bool,
    verbose: bool,
    workflow_args: argparse.Namespace | None = None,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _finalize_pbc_mlpot_exclusions_after_param_read,
        _suspend_pbc_for_cgenff_param_read,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

    if use_pbc:
        _suspend_pbc_for_cgenff_param_read(verbose=verbose)
    read_cgenff_prm(overlay_path, append=True)
    if zero_ml_charges:
        _zero_ml_atom_charges(ml_selection)
    if use_pbc:
        if cubic_box_side_A is None or float(cubic_box_side_A) <= 0.0:
            raise ValueError("PBC energy-policy reload requires cubic_box_side_A")
        _finalize_pbc_mlpot_exclusions_after_param_read(
            ml_selection,
            cubic_box_side_A=float(cubic_box_side_A),
            verbose=verbose,
            workflow_args=workflow_args,
        )
        from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
            run_mlpot_pbc_image_registration_gate,
        )

        run_mlpot_pbc_image_registration_gate(
            cubic_box_side_A=float(cubic_box_side_A),
            workflow_args=workflow_args,
            context="MLpot PBC registration (post energy-policy reload)",
            verbose=verbose,
        )
    else:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.nbonds_config import (
            apply_nbonds_script_kwargs,
            vacuum_nbond_kwargs,
        )

        apply_nbonds_script_kwargs(vacuum_nbond_kwargs(nbxmod=5), rebuild=True)


def _run_silent_ener() -> None:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    with charmm_silent_command():
        pycharmm.lingo.charmm_script("ENER")


def enforce_charmm_energy_term_policies(
    args: argparse.Namespace | None,
    *,
    ml_selection: Any,
    use_pbc: bool,
    cubic_box_side_A: float | None,
    verbose: bool = False,
) -> list[str]:
    """Probe CHARMM ENER; reload PSF/.prm overlays for violated policies."""
    policies = resolve_charmm_energy_term_policies(args)
    if not policies:
        return []

    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        cgenff_prm_path,
    )

    _run_silent_ener()
    terms = measure_charmm_energy_terms()

    violated: list[CharmmEnergyTermPolicy] = []
    for policy in policies:
        bad, hits = _policy_violation(policy, terms)
        if bad:
            violated.append(policy)
            if verbose or not getattr(args, "quiet", False):
                detail = ", ".join(f"{k}={v:.6g}" for k, v in sorted(hits.items()))
                print(
                    f"CHARMM energy policy {policy.name}: non-zero before reload ({detail})",
                    flush=True,
                )

    if not violated:
        return []

    zero_bonded = any(p.zero_bonded_prm for p in violated)
    zero_nonbond = any(p.zero_nonbond_prm for p in violated)
    zero_charges = any(p.zero_ml_charges for p in violated)
    remediable = zero_bonded or zero_nonbond or zero_charges
    if not remediable:
        names = ", ".join(p.name for p in violated)
        raise RuntimeError(
            f"CHARMM energy policies violated ({names}) but no PSF/.prm remediation "
            "is defined for them."
        )

    scratch = _policy_scratch_dir(args)
    scratch.mkdir(parents=True, exist_ok=True)
    policy_tag = "_".join(p.name for p in violated)
    overlay = scratch / f"zeroed_{policy_tag}.prm"
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import write_prm_policy_overlay

    write_prm_policy_overlay(
        cgenff_prm_path(),
        overlay,
        zero_bonded=zero_bonded,
        zero_nonbond=zero_nonbond,
        note=f"policies={policy_tag}",
    )
    if verbose or not getattr(args, "quiet", False):
        print(
            f"CHARMM energy policy: reloading overlay {overlay} "
            f"(bonded={zero_bonded}, nonbond={zero_nonbond}, zero_charges={zero_charges})",
            flush=True,
        )

    _reload_prm_overlay(
        overlay,
        use_pbc=use_pbc,
        cubic_box_side_A=cubic_box_side_A,
        ml_selection=ml_selection,
        zero_ml_charges=zero_charges,
        verbose=verbose,
        workflow_args=args,
    )

    _run_silent_ener()
    terms_after = measure_charmm_energy_terms()
    still_bad: list[str] = []
    for policy in violated:
        bad, hits = _policy_violation(policy, terms_after)
        if bad:
            still_bad.append(policy.name)
            detail = ", ".join(f"{k}={v:.6g}" for k, v in sorted(hits.items()))
            print(
                f"WARN: CHARMM energy policy {policy.name} still non-zero after reload "
                f"({detail})",
                flush=True,
            )
    if still_bad:
        raise RuntimeError(
            "CHARMM energy policy reload failed for: "
            + ", ".join(still_bad)
            + ". Inspect charmm_energy_policy/*.prm and ENER decomposition."
        )

    applied = [p.name for p in violated]
    if verbose or not getattr(args, "quiet", False):
        print(
            f"CHARMM energy policy: reload OK ({', '.join(applied)})",
            flush=True,
        )
    return applied


def summarize_policy_energy_terms(
    policies: Sequence[CharmmEnergyTermPolicy],
    terms: dict[str, float],
) -> dict[str, float]:
    """Subset of *terms* touched by *policies* (for tests/logging)."""
    keys: set[str] = set()
    for policy in policies:
        keys.update(policy.energy_keys)
    return {k: float(terms[k]) for k in sorted(keys) if k in terms and np.isfinite(terms[k])}
