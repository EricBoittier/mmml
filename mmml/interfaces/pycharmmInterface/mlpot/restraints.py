"""CHARMM restraints for non-PBC MLpot workflows (MMFP flat-bottom sphere, etc.)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlatBottomSphereConfig:
    """Flat-bottom spherical MMFP (``GEO sphere quartic``), as in production ``dyna.inp``."""

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
        GEO sphere quartic -
            xref … yref … zref … -
            droff <radius> force <force> -
            sele all end
        END
    """
    if config.radius <= 0:
        raise ValueError(f"flat-bottom radius must be > 0, got {config.radius}")
    if config.force <= 0:
        raise ValueError(f"flat-bottom force must be > 0, got {config.force}")

    pycharmm = _import_pycharmm()
    sel = config.selection.strip() or "all"
    script = f"""
MMFP
GEO sphere quartic -
    xref {float(config.xref):.6f} yref {float(config.yref):.6f} zref {float(config.zref):.6f} -
    droff {float(config.radius):.6f} force {float(config.force):.6f} p1 {float(0.8*config.radius):.6f} -
    sele {sel} end
END
"""
    pycharmm.lingo.charmm_script(script)


def clear_mmfp_restraints() -> None:
    """Remove MMFP terms (safe to call if none were defined)."""
    pycharmm = _import_pycharmm()
    pycharmm.lingo.charmm_script("MMFP CLEAR")


def apply_flat_bottom_workflow(
    *,
    radius: float | None,
    force: float = 1.0,
    center_at_origin: bool = True,
    xref: float = 0.0,
    yref: float = 0.0,
    zref: float = 0.0,
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
    )
    setup_flat_bottom_sphere_mmfp(cfg)
    return cfg
