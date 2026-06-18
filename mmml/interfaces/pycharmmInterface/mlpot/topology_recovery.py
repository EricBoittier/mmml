"""Safe PSF / topology recovery without ``DELETE ATOM`` when composition is unchanged."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import BondedMmMiniConfig
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

PathLike = str | Path


class BondedRecoveryMode(str, Enum):
    """BLOCK mode for inplace bonded recovery (MLpot detached)."""

    BONDED_ONLY = "bonded_only"
    BONDED_VDW = "bonded_vdw"
    FULL_MM = "full_mm"


@dataclass(frozen=True)
class TopologyFingerprint:
    """Lightweight composition identity for pre-MLpot cluster topology."""

    natom: int
    nres: int
    nseg: int
    atom_names: tuple[str, ...]
    resids: tuple[int, ...]

    def matches_live(self) -> bool:
        live = capture_topology_fingerprint_from_charmm()
        return (
            live.natom == self.natom
            and live.nres == self.nres
            and live.nseg == self.nseg
            and live.atom_names == self.atom_names
            and live.resids == self.resids
        )


def topology_fingerprint_path(psf_path: PathLike) -> Path:
    """Sidecar JSON path for a ``cluster_for_vmd_*.psf`` fingerprint."""
    p = Path(psf_path).expanduser()
    if p.suffix.lower() == ".psf":
        return p.with_suffix(".topology.json")
    return p.parent / f"{p.name}.topology.json"


def allow_psf_delete_reload() -> bool:
    """True when deprecated ``DELETE ATOM`` + ``read.psf_card`` reload is allowed."""
    return (os.environ.get("MMML_ALLOW_PSF_DELETE_RELOAD") or "").strip().lower() in (
        "1",
        "yes",
        "true",
    )


def capture_topology_fingerprint_from_charmm() -> TopologyFingerprint:
    """Snapshot current CHARMM PSF composition (atom names, residue IDs, counts)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.psf as psf

    natom = int(psf.get_natom())
    nres = int(psf.get_nres())
    nseg = int(psf.get_nseg())
    atom_names = tuple(str(x) for x in psf.get_atype())
    resids = tuple(int(x) for x in psf.get_resid())
    if len(atom_names) != natom or len(resids) != natom:
        raise RuntimeError(
            f"PSF composition mismatch: natom={natom}, "
            f"len(atom_names)={len(atom_names)}, len(resids)={len(resids)}"
        )
    return TopologyFingerprint(
        natom=natom,
        nres=nres,
        nseg=nseg,
        atom_names=atom_names,
        resids=resids,
    )


def save_topology_fingerprint(path: PathLike, fingerprint: TopologyFingerprint) -> Path:
    return save_topology_sidecar(path, fingerprint)


def save_topology_sidecar(
    path: PathLike,
    fingerprint: TopologyFingerprint,
    *,
    pre_mlpot_iblo: list[int] | None = None,
    pre_mlpot_inb: list[int] | None = None,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(fingerprint)
    if pre_mlpot_iblo is not None:
        payload["pre_mlpot_iblo"] = [int(x) for x in pre_mlpot_iblo]
    if pre_mlpot_inb is not None:
        payload["pre_mlpot_inb"] = [int(x) for x in pre_mlpot_inb]
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_topology_fingerprint(path: PathLike) -> TopologyFingerprint | None:
    sidecar = load_topology_sidecar(path)
    return sidecar.fingerprint if sidecar is not None else None


@dataclass(frozen=True)
class TopologySidecar:
    fingerprint: TopologyFingerprint
    pre_mlpot_iblo: tuple[int, ...] | None = None
    pre_mlpot_inb: tuple[int, ...] | None = None


def load_topology_sidecar(path: PathLike) -> TopologySidecar | None:
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    fp = TopologyFingerprint(
        natom=int(raw["natom"]),
        nres=int(raw["nres"]),
        nseg=int(raw["nseg"]),
        atom_names=tuple(str(x) for x in raw["atom_names"]),
        resids=tuple(int(x) for x in raw["resids"]),
    )
    iblo_raw = raw.get("pre_mlpot_iblo")
    inb_raw = raw.get("pre_mlpot_inb")
    return TopologySidecar(
        fingerprint=fp,
        pre_mlpot_iblo=tuple(int(x) for x in iblo_raw) if iblo_raw is not None else None,
        pre_mlpot_inb=tuple(int(x) for x in inb_raw) if inb_raw is not None else None,
    )


def attach_topology_recovery_state(
    ctx: "MlpotContext",
    topology_psf: PathLike | None,
) -> None:
    """Load composition fingerprint (+ optional pre-MLpot iblo/inb) onto ``ctx``."""
    if topology_psf is None:
        return
    psf = Path(topology_psf).expanduser()
    if not psf.is_file():
        return
    ctx.topology_psf_path = psf.resolve()
    sidecar = load_topology_sidecar(topology_fingerprint_path(psf))
    if sidecar is None:
        return
    ctx.topology_fingerprint = sidecar.fingerprint
    if sidecar.pre_mlpot_iblo is not None:
        ctx.pre_mlpot_iblo = list(sidecar.pre_mlpot_iblo)
    if sidecar.pre_mlpot_inb is not None:
        ctx.pre_mlpot_inb = list(sidecar.pre_mlpot_inb)


def resolve_topology_fingerprint(
    topology_psf: PathLike | None,
) -> TopologyFingerprint | None:
    """Load fingerprint sidecar for ``topology_psf``, if present."""
    if topology_psf is None:
        return None
    psf = Path(topology_psf).expanduser()
    if not psf.is_file():
        return None
    sidecar = load_topology_sidecar(topology_fingerprint_path(psf))
    return sidecar.fingerprint if sidecar is not None else None


def ensure_composition_unchanged(
    fingerprint: TopologyFingerprint | None,
    *,
    topology_psf: PathLike | None = None,
    context: str = "topology recovery",
) -> None:
    """Raise when live CHARMM composition no longer matches the saved fingerprint."""
    fp = fingerprint
    if fp is None and topology_psf is not None:
        fp = resolve_topology_fingerprint(topology_psf)
    if fp is None:
        return
    if fp.matches_live():
        return
    live = capture_topology_fingerprint_from_charmm()
    raise RuntimeError(
        f"{context}: CHARMM composition changed since pre-MLpot topology "
        f"(expected natom={fp.natom}, live natom={live.natom}). "
        "Restart from a prior segment .res or rebuild the cluster; "
        "do not use DELETE ATOM PSF reload mid-run."
    )


def prepare_rescue_lists_safe(
    ctx: "MlpotContext",
    *,
    context: str = "bonded-MM rescue",
    use_recovery_nbxmod: bool = False,
) -> None:
    """Refresh pair lists with ``UPDATE`` only (no ``upinb`` / ``update_bnbnd``)."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        assert_bonded_mm_energy_active,
    )

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    if use_recovery_nbxmod:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import RECOVERY_NBXMOD
        from mmml.interfaces.pycharmmInterface.nbonds_config import vacuum_nbond_kwargs

        pycharmm.UpdateNonBondedScript(
            **vacuum_nbond_kwargs(nbxmod=RECOVERY_NBXMOD)
        ).run()
    with charmm_relaxed_bomlev():
        pycharmm.lingo.charmm_script("UPDATE")
    try:
        assert_bonded_mm_energy_active(context=context)
    except RuntimeError:
        pre_iblo = getattr(ctx, "pre_mlpot_iblo", None)
        pre_inb = getattr(ctx, "pre_mlpot_inb", None)
        if pre_iblo is not None and pre_inb is not None:
            import pycharmm.psf as psf

            psf.set_iblo_inb_no_update(pre_iblo, pre_inb)
            with charmm_relaxed_bomlev():
                pycharmm.lingo.charmm_script("UPDATE")
            assert_bonded_mm_energy_active(context=f"{context} (restored pre-MLpot iblo/inb)")
        else:
            raise


def _apply_recovery_block(mode: BondedRecoveryMode) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_mm_only_block,
        apply_bonded_vdw_recovery_block,
        apply_charmm_mm_block,
    )

    if mode == BondedRecoveryMode.BONDED_ONLY:
        apply_bonded_mm_only_block()
    elif mode == BondedRecoveryMode.BONDED_VDW:
        apply_bonded_vdw_recovery_block()
    elif mode == BondedRecoveryMode.FULL_MM:
        apply_charmm_mm_block()
    else:
        raise ValueError(f"unknown BondedRecoveryMode: {mode!r}")


def run_bonded_recovery_inplace(
    ctx: "MlpotContext",
    mode: BondedRecoveryMode,
    config: "BondedMmMiniConfig",
    *,
    topology_psf: PathLike | None = None,
    run_sd: bool = True,
    rescue: Any | None = None,
) -> float | None:
    """Bonded recovery via BLOCK toggle + MLpot detach/reattach (no PSF DELETE)."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _bonded_recovery_sd_kwargs,
        _import_pycharmm_modules,
        _with_mlpot_detached,
    )

    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        charmm_quiet_output,
        run_charmm_script_quiet,
    )

    fingerprint = getattr(ctx, "topology_fingerprint", None)
    if fingerprint is None and topology_psf is not None:
        fingerprint = resolve_topology_fingerprint(topology_psf)
    ensure_composition_unchanged(
        fingerprint,
        topology_psf=topology_psf,
        context="inplace bonded recovery",
    )

    use_nbxmod = mode == BondedRecoveryMode.BONDED_VDW
    if config.verbose:
        print(
            f"bonded recovery: inplace {mode.value} "
            "(BLOCK toggle, UPDATE-only lists; no DELETE ATOM PSF reload)",
            flush=True,
        )

    def _work() -> float | None:
        _apply_recovery_block(mode)
        prepare_rescue_lists_safe(
            ctx,
            context=f"inplace {mode.value} recovery",
            use_recovery_nbxmod=use_nbxmod,
        )
        pycharmm, cons_fix, *_ = _import_pycharmm_modules()
        minimize = _import_pycharmm_modules()[3]
        run_charmm_script_quiet("ENER")
        grms_before = float(charmm_grms())
        if config.verbose and run_sd:
            print(
                f"inplace recovery start: GRMS={grms_before:.4f} kcal/mol/Å",
                flush=True,
            )
        if run_sd and int(config.nstep_sd) > 0:
            with charmm_quiet_output():
                minimize.run_sd(**_bonded_recovery_sd_kwargs(ctx, config))
        if rescue is not None and int(getattr(rescue, "nstep_abnr", 0) or 0) > 0:
            with charmm_quiet_output():
                minimize.run_abnr(
                    nstep=int(rescue.nstep_abnr),
                    tolenr=float(rescue.tolenr),
                    tolgrd=float(rescue.tolgrd),
                )
        run_charmm_script_quiet("ENER")
        grms_after = float(charmm_grms())
        if config.verbose and run_sd:
            print(
                f"inplace recovery end: GRMS={grms_after:.4f} kcal/mol/Å",
                flush=True,
            )
        cons_fix.turn_off()
        return grms_after

    return _with_mlpot_detached(ctx, _work)


def measure_mm_strain_inplace(
    ctx: "MlpotContext",
    *,
    topology_psf: PathLike | None = None,
) -> "MmStrainBaseline":
    """Measure MM bonded strain with full MM BLOCK and MLpot detached (no PSF reload)."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import MmStrainBaseline
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _with_mlpot_detached,
        charmm_bonded_term_kcalmol,
        charmm_internal_energy_kcalmol,
    )

    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    fingerprint = getattr(ctx, "topology_fingerprint", None)
    if fingerprint is None and topology_psf is not None:
        fingerprint = resolve_topology_fingerprint(topology_psf)
    ensure_composition_unchanged(
        fingerprint,
        topology_psf=topology_psf,
        context="MM strain measurement",
    )

    def _measure() -> MmStrainBaseline:
        _apply_recovery_block(BondedRecoveryMode.FULL_MM)
        prepare_rescue_lists_safe(ctx, context="MM strain measurement")
        run_charmm_script_quiet("ENER")
        return MmStrainBaseline(
            grms_kcalmol_A=float(charmm_grms()),
            internal_kcalmol=charmm_internal_energy_kcalmol(),
            angl_kcalmol=charmm_bonded_term_kcalmol("ANGL"),
        )

    return _with_mlpot_detached(ctx, _measure)
