"""Enforce zero CHARMM energy components via PSF/.prm reload (not BLOCK)."""

from __future__ import annotations

import argparse
import os
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
    # CHARMM SKIPE term names: excluded from every later ENER/DYNA so the
    # term is exactly zero even when the .prm overlay does not take.
    skipe_terms: tuple[str, ...] = ()


POLICY_REGISTRY: dict[str, CharmmEnergyTermPolicy] = {
    "vdw": CharmmEnergyTermPolicy(
        name="vdw",
        energy_keys=("VDW", "IMNB"),
        tolerance_kcal=1.0e-8,
        zero_nonbond_prm=True,
        skipe_terms=("VDW", "IMNB"),
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


#: CHARMM nonbond ELEC terms that ``SKIPE`` drops once every live partial charge
#: is zero (all-ML MLpot: registration zeroes all ML charges). With VDW/IMNB
#: already skipped by the ``vdw`` policy, ENBFS8 then returns at its first test
#: instead of looping over the primary + image pair lists every step.
ZERO_CHARGE_SKIPE_TERMS: tuple[str, ...] = ("ELEC", "IMEL")

#: Opt-out for A/B checks: keep CHARMM's (zero) ELEC/IMEL evaluation.
KEEP_CHARMM_ELEC_ENV = "MMML_MLPOT_KEEP_CHARMM_ELEC"

# SKIPE is global and accumulates for the CHARMM process; mirror it here so the
# MLpot eterm router does not push MM buckets into terms CHARMM no longer sums.
_SKIPPED_CHARMM_TERMS: set[str] = set()


def charmm_skipped_terms() -> frozenset[str]:
    """CHARMM energy terms this process removed with ``SKIPE``."""
    return frozenset(_SKIPPED_CHARMM_TERMS)


def _charmm_skipe(terms: Sequence[str]) -> None:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    # eval_charmm_script skips CHARMM's uppercase conversion: commands must be uppercase.
    pycharmm.lingo.charmm_script("SKIPE " + " ".join(terms))
    _SKIPPED_CHARMM_TERMS.update(str(t).upper() for t in terms)


def charmm_elec_redundant(
    policies: Sequence[CharmmEnergyTermPolicy],
    charges: Sequence[float] | np.ndarray | None,
) -> bool:
    """True when CHARMM ELEC/IMEL can only add zero energy and zero force.

    Requires the ``vdw`` policy (JAX owns the intermolecular MM nonbond, CHARMM
    VDW/IMNB are skipped) and an all-zero live charge vector. Every CHARMM
    Coulomb pair term carries ``q_i q_j``, so ELEC and IMEL are then exactly
    zero and skipping them leaves the Hamiltonian unchanged.
    """
    if not any(p.zero_nonbond_prm for p in policies):
        return False
    if charges is None:
        return False
    q = np.asarray(charges, dtype=np.float64).ravel()
    return bool(q.size) and bool(np.all(q == 0.0))


def skip_redundant_charmm_elec(
    policies: Sequence[CharmmEnergyTermPolicy],
    *,
    verbose: bool = False,
) -> list[str]:
    """``SKIPE ELEC IMEL`` when :func:`charmm_elec_redundant` holds; return skipped terms."""
    if (os.environ.get(KEEP_CHARMM_ELEC_ENV) or "").strip().lower() in ("1", "true", "yes", "on"):
        return []
    if not any(p.zero_nonbond_prm for p in policies):
        return []
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm

        charges = list(pycharmm.psf.get_charges())
    except (AttributeError, ImportError, OSError):
        return []
    if not charmm_elec_redundant(policies, charges):
        return []
    terms = list(ZERO_CHARGE_SKIPE_TERMS)
    _charmm_skipe(terms)
    if verbose:
        print(
            f"CHARMM energy policy: SKIPE {' '.join(terms)} (all {len(charges)} "
            "CHARMM charges are zero; JAX computes the MM nonbond)",
            flush=True,
        )
    return terms


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


def _post_remediation_policy(policy: CharmmEnergyTermPolicy) -> CharmmEnergyTermPolicy:
    """Loosen verification after remediation for CHARMM image-list residuals."""
    if policy.name == "vdw":
        return CharmmEnergyTermPolicy(
            name=policy.name,
            energy_keys=policy.energy_keys,
            tolerance_kcal=max(float(policy.tolerance_kcal), 1.0),
            zero_bonded_prm=policy.zero_bonded_prm,
            zero_nonbond_prm=policy.zero_nonbond_prm,
            zero_ml_charges=policy.zero_ml_charges,
        )
    return policy


def _skip_policy_terms(policies: Sequence[CharmmEnergyTermPolicy], *, verbose: bool) -> None:
    """``SKIPE`` every policy term; MLpot USER and the rest stay included.

    The ε=0 ``READ PARAM APPEND`` overlay leaves the live VDW table untouched
    (VDW stays at its CGenFF value). The old libcharmm hid that because its
    latched ``qappend`` wiped the table on the next full read (fixed in
    6b050e2fb). ``SKIPE`` removes the term from the energy instead.
    """
    terms = list(dict.fromkeys(t for p in policies for t in p.skipe_terms))
    if not terms:
        return
    _charmm_skipe(terms)
    if verbose:
        print(f"CHARMM energy policy: SKIPE {' '.join(terms)}", flush=True)


def _zero_scalar_vdw() -> None:
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    with charmm_silent_command():
        pycharmm.lingo.charmm_script("SCALAR VDW SET 0.0 SELE ALL END")
        pycharmm.lingo.charmm_script("SCALAR VDW14 SET 0.0 SELE ALL END")


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
    zero_nonbond: bool = False,
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
    if zero_nonbond:
        _zero_scalar_vdw()
        _skip_policy_terms([POLICY_REGISTRY["vdw"]], verbose=verbose)
        if verbose:
            print("CHARMM energy policy: applied SCALAR VDW/VDW14 SET 0.0 to all atoms (READ PARAM APPEND workaround)", flush=True)

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
    skip_ener_probe: bool | None = None,
    reload_on_violation: bool = True,
) -> list[str]:
    """Probe CHARMM ENER; reload PSF/.prm overlays for violated policies."""
    policies = resolve_charmm_energy_term_policies(args)
    if not policies:
        return []

    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        _mlpot_active_in_charmm,
    )

    if skip_ener_probe is None:
        skip_ener_probe = _mlpot_active_in_charmm()
    if skip_ener_probe:
        if verbose or not getattr(args, "quiet", False):
            print(
                "CHARMM energy policy: deferring ENER probe while MLpot USER is active "
                "(run before MLpot registration instead)",
                flush=True,
            )
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
                stage = (
                    "non-zero before reload"
                    if reload_on_violation
                    else "still non-zero after pre-registration remediation"
                )
                print(
                    f"CHARMM energy policy {policy.name}: {stage} ({detail})",
                    flush=True,
                )

    if not violated:
        return []

    if not reload_on_violation:
        details: list[str] = []
        for policy in violated:
            _bad, hits = _policy_violation(_post_remediation_policy(policy), terms)
            if hits:
                detail = ", ".join(f"{k}={v:.6g}" for k, v in sorted(hits.items()))
                details.append(f"{policy.name} ({detail})")
        if details:
            raise RuntimeError(
                "CHARMM energy policy still non-zero after pre-registration remediation: "
                + ", ".join(details)
            )
        if verbose or not getattr(args, "quiet", False):
            names = ", ".join(policy.name for policy in violated)
            print(
                f"CHARMM energy policy: residual after pre-registration remediation "
                f"within tolerance ({names})",
                flush=True,
            )
        return [policy.name for policy in violated]


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
        zero_nonbond=zero_nonbond,
        workflow_args=args,
    )

    _run_silent_ener()
    terms_after = measure_charmm_energy_terms()
    still_bad: list[str] = []
    for policy in violated:
        bad, hits = _policy_violation(_post_remediation_policy(policy), terms_after)
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


def apply_charmm_energy_term_policies_before_pbc_finalize(
    args: argparse.Namespace | None,
    *,
    ml_selection: Any,
    verbose: bool = False,
) -> list[str]:
    """Apply policy remediations before PBC image/nonbond list finalization.

    This avoids the unsafe sequence seen with all-ML PBC systems:
    finalize image lists -> probe IMNB -> READ PARAM APPEND -> crystal free/hang.

    For VDW/IMNB, ``SCALAR VDW SET 0.0`` alone is not enough after IMAGE lists
    are built (residual ``IMNB`` often ~1–2 kcal/mol).  Load an ε=0 NONBONDED
    APPEND overlay first so CHARMM's VDW table is overwritten while crystal is
    still suspended, then SCALAR as a belt-and-suspenders clear.
    """
    policies = resolve_charmm_energy_term_policies(args)
    if not policies:
        return []

    zero_nonbond = any(p.zero_nonbond_prm for p in policies)
    zero_charges = any(p.zero_ml_charges for p in policies)
    applied: list[str] = []

    if zero_nonbond:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
            zeroed_nonbond_prm_text,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
            cgenff_prm_path,
        )
        from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

        scratch = _policy_scratch_dir(args)
        scratch.mkdir(parents=True, exist_ok=True)
        overlay = scratch / "zeroed_vdw_pre_pbc.prm"
        src = cgenff_prm_path()
        body = zeroed_nonbond_prm_text(
            src.read_text(encoding="utf-8", errors="replace")
        )
        if not body.strip():
            raise RuntimeError(
                f"empty pre-PBC VDW zero overlay from {src}; "
                "cannot enforce IMNB≈0 for jax_mic"
            )
        overlay.write_text(
            "* MMML pre-PBC VDW zero overlay (ε=0 READ PARAM APPEND)\n"
            f"* Source: {src.name}\n"
            "* --------------------------------------------------------------------------  *\n"
            + body,
            encoding="utf-8",
        )
        # Crystal is already suspended by register_mlpot before this hook.
        read_cgenff_prm(overlay, append=True)
        _zero_scalar_vdw()
        _skip_policy_terms(
            [p for p in policies if p.zero_nonbond_prm],
            verbose=verbose or not getattr(args, "quiet", False),
        )
        applied.extend(p.name for p in policies if p.zero_nonbond_prm)
        if verbose or not getattr(args, "quiet", False):
            print(
                "CHARMM energy policy: pre-PBC ε=0 NONBONDED overlay + "
                f"SCALAR VDW/VDW14 ({overlay.name})",
                flush=True,
            )

    if zero_charges:
        _zero_ml_atom_charges(ml_selection)
        applied.extend(p.name for p in policies if p.zero_ml_charges)

    if zero_nonbond:
        # After the charge edits above: all-ML registration has zeroed every
        # charge, so CHARMM ELEC/IMEL are identically zero (JAX owns MM).
        skip_redundant_charmm_elec(
            policies, verbose=verbose or not getattr(args, "quiet", False)
        )

    return list(dict.fromkeys(applied))


def summarize_policy_energy_terms(
    policies: Sequence[CharmmEnergyTermPolicy],
    terms: dict[str, float],
) -> dict[str, float]:
    """Subset of *terms* touched by *policies* (for tests/logging)."""
    keys: set[str] = set()
    for policy in policies:
        keys.update(policy.energy_keys)
    return {k: float(terms[k]) for k in sorted(keys) if k in terms and np.isfinite(terms[k])}
