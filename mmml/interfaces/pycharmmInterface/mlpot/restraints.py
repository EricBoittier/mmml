"""CHARMM restraints for non-PBC MLpot workflows (MMFP flat-bottom sphere, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

_MMFP_GEO_ACTIVE = False


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
            droff <radius> force <force> outside -
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
    droff {float(config.radius):.6f} force {float(config.force):.6f} outside -
    sele {sel} end
END
"""
    pycharmm.lingo.charmm_script(script)
    global _MMFP_GEO_ACTIVE
    _MMFP_GEO_ACTIVE = True


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
    cfg = FlatBottomSphereConfig(
        radius=float(radius),
        force=float(force),
        xref=xref,
        yref=yref,
        zref=zref,
        selection=selection,
    )
    setup_flat_bottom_sphere_mmfp(cfg)
    return cfg
