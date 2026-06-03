"""CHARMM restraints for non-PBC MLpot workflows (MMFP flat-bottom sphere, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MMFP_GEO_ACTIVE = False
_DROFF_MARGIN_A = 1.0e-3
_ENERGY_VERIFY_TOL_KCAL = 1.0e-4
_DROFF_TUNE_MAX_ATTEMPTS = 8


@dataclass
class FlatBottomSphereConfig:
    """Flat-bottom spherical MMFP wall; inside ``radius`` has no restraint."""

    radius: float = 20.0
    force: float = 1.0
    xref: float = 0.0
    yref: float = 0.0
    zref: float = 0.0
    selection: str = "all"


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm

    return pycharmm


def center_cluster_at_origin(*, orient: bool = True) -> None:
    """``coor orient`` and translate so the cluster COM is at the origin (non-PBC)."""
    pycharmm = _import_pycharmm()
    if orient:
        pycharmm.lingo.charmm_script("coor orient sele all end")
    pycharmm.lingo.charmm_script(
        """
coor stat sele all end
coor translate xdir -?xave ydir -?yave zdir -?zave sele all end
"""
    )


def setup_flat_bottom_sphere_mmfp(config: FlatBottomSphereConfig) -> None:
    """Install CHARMM MMFP flat-bottom sphere restraint (inside ``radius``: no force).

    Matches::

        MMFP
        GEO sphere harm -
            xref … yref … zref … -
            droff <radius> force <force> -
            sele all end
        END
    """
    if config.radius <= 0:
        raise ValueError(f"flat-bottom radius must be > 0, got {config.radius}")
    if config.force <= 0:
        raise ValueError(f"flat-bottom force must be > 0, got {config.force}")

    clear_mmfp_restraints()
    pycharmm = _import_pycharmm()
    sel = config.selection.strip() or "all"
    script = f"""
MMFP
GEO sphere harm -
    xref {float(config.xref):.6f} yref {float(config.yref):.6f} zref {float(config.zref):.6f} -
    droff {float(config.radius):.6f} force {float(config.force):.6f} -
    sele {sel} end
END
"""
    pycharmm.lingo.charmm_script(script)
    global _MMFP_GEO_ACTIVE
    _MMFP_GEO_ACTIVE = True


def _selected_max_radius(selection: str, *, xref: float, yref: float, zref: float) -> float | None:
    """Conservative max selected distance using CHARMM's own selection parser."""
    pycharmm = _import_pycharmm()
    sel = (selection or "all").strip() or "all"
    try:
        pycharmm.lingo.charmm_script(f"coor stat sele {sel} end")
        xmin = float(pycharmm.lingo.get_energy_value("XMIN"))
        xmax = float(pycharmm.lingo.get_energy_value("XMAX"))
        ymin = float(pycharmm.lingo.get_energy_value("YMIN"))
        ymax = float(pycharmm.lingo.get_energy_value("YMAX"))
        zmin = float(pycharmm.lingo.get_energy_value("ZMIN"))
        zmax = float(pycharmm.lingo.get_energy_value("ZMAX"))
    except Exception as exc:
        print(
            f"WARN: could not estimate MMFP droff from selection {sel!r}: {exc}",
            flush=True,
        )
        return None

    dx = max(abs(xmin - float(xref)), abs(xmax - float(xref)))
    dy = max(abs(ymin - float(yref)), abs(ymax - float(yref)))
    dz = max(abs(zmin - float(zref)), abs(zmax - float(zref)))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _current_charmm_energy_kcalmol() -> float | None:
    try:
        import pycharmm
        import pycharmm.energy as energy

        pycharmm.lingo.charmm_script("ENER")
        row = energy.get_energy().iloc[0].to_dict()
        for key in ("ENER", "ENERgy", "ENERGY"):
            if key in row:
                return float(row[key])
    except Exception as exc:
        print(
            f"WARN: could not verify MMFP zero-energy install: {exc}",
            flush=True,
        )
    return None


def _energy_delta_after_install(before: float | None) -> float | None:
    if before is None:
        return None
    after = _current_charmm_energy_kcalmol()
    if after is None:
        return None
    return after - before


def _next_droff_increment(radius: float, attempt: int) -> float:
    base = max(0.05, 0.01 * float(radius))
    return base * (2 ** max(0, attempt - 1))


def clear_mmfp_restraints() -> None:
    """Remove MMFP terms (safe to call if none were defined)."""
    global _MMFP_GEO_ACTIVE
    if not _MMFP_GEO_ACTIVE:
        return
    pycharmm = _import_pycharmm()
    pycharmm.lingo.charmm_script(
        """
MMFP
GEO RESET
END
"""
    )
    _MMFP_GEO_ACTIVE = False


def apply_flat_bottom_workflow(
    *,
    radius: float | None,
    force: float = 1.0,
    center_at_origin: bool = True,
    xref: float = 0.0,
    yref: float = 0.0,
    zref: float = 0.0,
    selection: str = "all",
) -> FlatBottomSphereConfig | None:
    """Optionally center the cluster and set up MMFP flat-bottom sphere."""
    if radius is None or radius <= 0:
        return None
    if center_at_origin:
        center_cluster_at_origin()
    energy_before = _current_charmm_energy_kcalmol()
    requested_radius = float(radius)
    current_radius = _selected_max_radius(selection, xref=xref, yref=yref, zref=zref)
    effective_radius = requested_radius
    if current_radius is not None:
        effective_radius = max(requested_radius, current_radius + _DROFF_MARGIN_A)
        if effective_radius > requested_radius:
            print(
                "MMFP flat-bottom droff adjusted "
                f"{requested_radius:.3f} -> {effective_radius:.3f} Å "
                f"so initial {selection!r} wall energy is zero",
                flush=True,
            )
    cfg = FlatBottomSphereConfig(
        radius=effective_radius,
        force=float(force),
        xref=xref,
        yref=yref,
        zref=zref,
        selection=selection,
    )
    for attempt in range(1, _DROFF_TUNE_MAX_ATTEMPTS + 1):
        setup_flat_bottom_sphere_mmfp(cfg)
        delta = _energy_delta_after_install(energy_before)
        if delta is None:
            return cfg
        if abs(delta) <= _ENERGY_VERIFY_TOL_KCAL:
            print(
                f"MMFP flat-bottom zero-energy check OK: ΔE={delta:+.6f} kcal/mol",
                flush=True,
            )
            return cfg
        if attempt == _DROFF_TUNE_MAX_ATTEMPTS:
            print(
                "WARN: MMFP flat-bottom changed energy at install "
                f"by {delta:+.6f} kcal/mol after {attempt} droff tuning attempt(s) "
                f"(droff={cfg.radius:.6f} Å)",
                flush=True,
            )
            return cfg
        old_radius = cfg.radius
        cfg.radius = old_radius + _next_droff_increment(old_radius, attempt)
        print(
            "MMFP flat-bottom ΔE not zero "
            f"({delta:+.6f} kcal/mol); increasing droff "
            f"{old_radius:.3f} -> {cfg.radius:.3f} Å and retrying",
            flush=True,
        )
    return cfg
